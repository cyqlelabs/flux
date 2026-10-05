use axum::{
    body::{to_bytes, Body},
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    response::IntoResponse,
    Json,
};
use flux_core::{
    config::FluxConfig,
    plan::{EngineKind, Plan},
};
use flux_serve::{journal::Status, AppState, Replanner};
use serde_json::{json, Value};
use std::{path::Path, sync::Arc, time::Duration};
use tower::ServiceExt;

fn plan(mode: &str) -> Plan {
    serde_json::from_value(json!({
        "schema":1,"id":"audit","created":"2026-10-04T00:00:00Z",
        "key":{"model_identity":"audit","topology":"audit","backend_revision":"audit","backend_build":"audit","driver":"audit","ctx_bucket":4096,"concurrency":2},
        "model_files":["fixture.gguf"],"architecture":mode,"engine":"native",
        "workload":{"n_ctx_seq":4096,"concurrency":2,"objective":"interactive"},
        "placement":{"devices":[],"layer_device":[],"output_device":"CPU","overrides":[],"n_gpu_layers":0,"tensor_split":[]},
        "runtime":{"n_batch":512,"n_ubatch":128,"n_threads":1,"n_threads_batch":1,"flash_attn":true,"type_k":"f16","type_v":"f16","mmap":true,"mlock":false,"op_offload":true,"speculation":null},
        "budgets":[],"host":{"capacity":0,"available_at_plan":0,"os_reserve":0,"resident_weights":0,"mirrored_weights":0,"pinned_buffers":0,"state_and_scratch":0},
        "quality":{"kind":"exact"},"decisions":[],"validation":null
    })).unwrap()
}

async fn app(root: &Path, p: Plan, replanner: Option<Replanner>) -> Arc<AppState> {
    let mut cfg = FluxConfig { data_dir: root.into(), ..Default::default() };
    cfg.serve.worker_timeouts.rpc_s = 1;
    cfg.serve.decode_timeout_s = 1;
    cfg.serve.replan_timeout_s = 1;
    cfg.serve.client_write_timeout_s = 1;
    flux_serve::build(cfg, p, replanner).await.unwrap()
}

async fn completion(st: &Arc<AppState>, id: &str, stream: bool) -> axum::response::Response {
    let mut headers = HeaderMap::new();
    headers.insert("x-request-id", id.parse().unwrap());
    flux_serve::openai::completions(State(st.clone()), headers, Json(json!({"prompt":[1,2],"max_tokens":80,"stream":stream}))).await
}

async fn response_json(response: axum::response::Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap()
}

// A single test owns FLUX_WORKER; it never races other tests changing process environment.
#[tokio::test]
async fn lifecycle_faults_and_output_integrity() {
    tokio::time::timeout(Duration::from_secs(30), scenarios()).await.expect("lifecycle test hung");
}

async fn scenarios() {
    let dir = tempfile::tempdir().unwrap();
    let worker = dir.path().join("worker");
    std::fs::write(&worker, include_str!("fixtures/worker.py")).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&worker, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::env::set_var("FLUX_WORKER", worker);

    let mut context_plan = plan("normal");
    context_plan.workload.n_ctx_seq = 16384;
    context_plan.key.ctx_bucket = 16384;
    let st = app(dir.path(), context_plan, None).await;
    let model_info = response_json(flux_serve::openai::models(State(st.clone())).await.into_response()).await;
    for field in ["context_length", "max_model_len"] {
        assert_eq!(model_info["data"][0][field], 16384);
    }
    assert_eq!(model_info["data"][0]["meta"]["n_ctx"], 16384);
    let overflow = flux_serve::openai::completions(State(st.clone()), HeaderMap::new(), Json(json!({"prompt":vec![1;16284],"max_tokens":32768}))).await;
    assert_eq!(overflow.status(), StatusCode::BAD_REQUEST);
    let overflow = response_json(overflow).await;
    assert_eq!(overflow["error"]["type"], "invalid_request_error");
    assert_eq!(overflow["error"]["param"], "prompt");
    assert_eq!(overflow["error"]["code"], "context_length_exceeded");
    assert_eq!(overflow["error"]["message"], "This model's maximum context length is 16384 tokens. However, your messages resulted in 16284 tokens.");
    assert_eq!(st.journal.running(), 0);
    assert_eq!(st.admission.in_use(2), 0);
    let small_reply = flux_serve::openai::completions(State(st.clone()), HeaderMap::new(), Json(json!({"prompt":vec![1;16284],"max_tokens":100}))).await;
    assert_eq!(small_reply.status(), StatusCode::OK);
    for body in [
        json!({"prompt":vec![1;16285],"max_tokens":100}),
        json!({"prompt":vec![1;16384],"max_tokens":1}),
        json!({"prompt":vec![1;16284]}),
        json!({"prompt":vec![1;12289],"max_tokens":32768}),
    ] {
        let r = flux_serve::openai::completions(State(st.clone()), HeaderMap::new(), Json(body)).await;
        assert_eq!(r.status(), StatusCode::BAD_REQUEST);
        assert_eq!(response_json(r).await["error"]["code"], "context_length_exceeded");
    }
    let precedence = flux_serve::openai::completions(
        State(st.clone()),
        HeaderMap::new(),
        Json(json!({"prompt":vec![1;16284],"max_tokens":32768,"max_completion_tokens":100})),
    )
    .await;
    assert_eq!(precedence.status(), StatusCode::OK);
    let body = response_json(completion(&st, "tail", false).await).await;
    assert_eq!(body["choices"][0]["text"], "hello </");
    assert_eq!(st.journal.get("tail").unwrap().texts.concat(), "hello </");
    assert!(st.journal.event("tail", 1).unwrap().0.unwrap().contains("</"));
    let terminal: Value = serde_json::from_str(&st.journal.event("tail", 2).unwrap().0.unwrap()).unwrap();
    assert_eq!(terminal["terminal"]["usage"]["completion_tokens"], 1);
    let resumed =
        flux_serve::router(st.clone()).oneshot(Request::builder().uri("/flux/requests/tail/stream?after=1").body(Body::empty()).unwrap()).await.unwrap();
    let resumed = String::from_utf8(to_bytes(resumed.into_body(), 1 << 20).await.unwrap().to_vec()).unwrap();
    assert!(resumed.contains("id: 1") && resumed.contains("</") && resumed.contains("terminal"));
    st.supervisor.current().await.1.kill().await;
    let health = flux_serve::router(st.clone()).oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(health.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body = response_json(
        flux_serve::openai::chat(State(st.clone()), HeaderMap::new(), Json(json!({"messages":[{"role":"user","content":"hi"}],"max_tokens":80}))).await,
    )
    .await;
    assert!(body.get("error").is_none(), "{body}");
    assert_eq!(st.supervisor.current().await.0, 1);
    let w = st.supervisor.current().await.1;
    assert!(w.tokenize("HANG", true).await.is_err());
    assert!(!w.is_alive().await);
    let mut monitors = flux_serve::monitor::spawn(st.clone());
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if st.supervisor.current().await.0 == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    monitors.abort_all();
    while monitors.join_next().await.is_some() {}
    st.supervisor.shutdown().await;

    let worker = flux_core::worker::Worker::spawn(&EngineKind::Native, None).await.unwrap();
    let mut mismatched = plan("normal");
    mismatched.key.backend_build = "stale".into();
    assert!(worker.load(&mismatched).await.unwrap_err().to_string().contains("identity"));
    worker.shutdown().await;

    for mode in ["crash", "stall", "malformed"] {
        let st = app(dir.path(), plan(mode), None).await;
        let response = completion(&st, mode, false).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR, "{mode}");
        assert!(matches!(st.journal.get(mode).unwrap().status, Status::Failed { .. }));
        if mode != "malformed" {
            assert_eq!(st.journal.get(mode).unwrap().tokens.len(), 1);
        }
        st.supervisor.shutdown().await;
    }

    let st = app(dir.path(), plan("backpressure"), None).await;
    let response = completion(&st, "slow", true).await;
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert!(matches!(st.journal.get("slow").unwrap().status, Status::Failed { .. }));
    assert_eq!(st.admission.in_use(2), 0);
    drop(response);
    st.supervisor.shutdown().await;

    let replanner: Replanner = Arc::new(|_| Box::pin(std::future::pending()));
    let st = app(dir.path(), plan("normal"), Some(replanner)).await;
    assert!(tokio::time::timeout(Duration::from_millis(50), flux_serve::replan_now(&st)).await.is_err());
    tokio::time::sleep(Duration::from_millis(1200)).await;
    assert!(st.admission.closed_reason().is_none());
    assert!(st.supervisor.current().await.1.is_alive().await);
    st.supervisor.shutdown().await;

    let replanner: Replanner = Arc::new(|p| Box::pin(async { Ok(p) }));
    let st = app(dir.path(), plan("normal"), Some(replanner)).await;
    let (a, b) = tokio::join!(flux_serve::replan_now(&st), flux_serve::replan_now(&st));
    assert!(a.is_ok() && b.is_ok());
    assert_eq!(st.supervisor.current().await.0, 2);
    st.admission.block("pressure", "memory pressure");
    flux_serve::replan_now(&st).await.unwrap();
    assert!(st.admission.blocked_by("pressure"));
    st.supervisor.shutdown().await;

    let replanner: Replanner = Arc::new(|mut p| {
        Box::pin(async move {
            p.architecture = "load_error".into();
            Ok(p)
        })
    });
    let st = app(dir.path(), plan("normal"), Some(replanner)).await;
    assert!(flux_serve::replan_now(&st).await.is_err());
    assert!(st.supervisor.current().await.1.is_alive().await);
    assert!(st.admission.closed_reason().is_none());
    st.supervisor.shutdown().await;

    let mut p = plan("normal");
    p.engine = EngineKind::External("fixture".into());
    let st = app(dir.path(), p, None).await;
    let body = response_json(flux_serve::openai::chat(State(st.clone()), HeaderMap::new(), Json(json!({"messages":[]}))).await).await;
    assert_eq!(body["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "lookup");
    assert_eq!(body["choices"][0]["message"]["reasoning_content"], "thinking");
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(st.journal.running(), 0);
    let id = body["id"].as_str().unwrap();
    assert!(st.journal.event(id, 0).unwrap().0.unwrap().contains("call_audit"));
    st.supervisor.shutdown().await;
    std::env::remove_var("FLUX_WORKER");
}
