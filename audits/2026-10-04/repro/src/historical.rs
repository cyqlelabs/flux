//! Historical reproductions against f18ac23. Not compiled: the maintained regression suite covers the fixes.
use axum::{extract::State, http::HeaderMap, Json};
use flux_core::{config::FluxConfig, plan::Plan, worker::Worker};
use flux_serve::{admission::{Admission, Rejection}, journal::{Journal, Status}};
use serde_json::json;
use std::{sync::Arc, time::Duration};

#[path = "../../../../crates/flux-worker/src/text.rs"]
mod text;

fn fixture_plan() -> Plan {
    serde_json::from_value(json!({
        "schema": 1, "id": "audit", "created": "2026-10-04T00:00:00Z",
        "key": {"model_identity":"audit", "topology":"audit", "backend_revision":"audit",
            "backend_build":"audit", "driver":"audit", "ctx_bucket":4096, "concurrency":2},
        "model_files": ["fixture.gguf"], "architecture":"fixture", "engine":"native",
        "workload":{"n_ctx_seq":3072, "concurrency":2, "objective":"interactive"},
        "placement":{"devices":[], "layer_device":[], "output_device":"CPU", "overrides":[],
            "n_gpu_layers":0, "tensor_split":[]},
        "runtime":{"n_batch":512,"n_ubatch":128,"n_threads":1,"n_threads_batch":1,
            "flash_attn":true,"type_k":"f16","type_v":"f16","mmap":true,"mlock":false,
            "op_offload":true,"speculation":null},
        "budgets":[], "host":{"capacity":0,"available_at_plan":0,"os_reserve":0,
            "resident_weights":0,"mirrored_weights":0,"pinned_buffers":0,"state_and_scratch":0},
        "quality":{"kind":"exact"}, "decisions":[], "validation":null
    })).unwrap()
}

async fn admission_checks() {
    let a = Admission::new(1, 1);
    let running = a.admit().await.unwrap();
    {
        let mut queued = Box::pin(a.admit());
        assert!(futures::poll!(&mut queued).is_pending());
        assert_eq!(a.waiting(), 1);
    }
    assert_eq!(a.waiting(), 1);
    assert_eq!(a.admit().await.unwrap_err(), Rejection::QueueFull);
    drop(running);
    println!("CONFIRMED: dropped admission future leaks waiting count and rejects future queueing");

    let a = Admission::new(1, 1);
    let running = a.admit().await.unwrap();
    let mut queued = Box::pin(a.admit());
    assert!(futures::poll!(&mut queued).is_pending());
    a.close("memory pressure");
    drop(running);
    assert!(queued.await.is_ok());
    println!("CONFIRMED: request already queued is admitted after memory-pressure closure");

    let a = Admission::new(2, 0);
    let p1 = a.admit().await.unwrap();
    let p2 = a.admit().await.unwrap();
    a.close("replanning");
    let mut d1 = Box::pin(a.drain(2));
    let mut d2 = Box::pin(a.drain(2));
    assert!(futures::poll!(&mut d1).is_pending());
    assert!(futures::poll!(&mut d2).is_pending());
    drop((p1, p2));
    assert!(futures::poll!(&mut d1).is_pending());
    assert!(futures::poll!(&mut d2).is_pending());
    assert_eq!(a.in_use(2), 2);
    assert!(tokio::time::timeout(Duration::from_millis(50), async {
        tokio::join!(d1, d2)
    }).await.is_err());
    println!("CONFIRMED: two concurrent drains each hold one permit and deadlock");
}

fn text_checks() {
    let mut before = text::TextStream::new(vec!["</end>".into()]);
    assert!(matches!(before.push(b"hello </"), text::Pushed::Text(t) if t == "hello "));
    let mut after = text::TextStream::new(vec!["</end>".into()]);
    assert!(matches!(after.push(b"end>"), text::Pushed::Text(t) if t == "end>"));
    assert!(matches!(before.push(b"end>"), text::Pushed::Stop(t) if t.is_empty()));
    println!("CONFIRMED: rebuilding TextStream after token replay loses pending stop prefix");

    let mut before = text::TextStream::new(vec![]);
    assert!(matches!(before.push(&[0xe2]), text::Pushed::Text(t) if t.is_empty()));
    let mut after = text::TextStream::new(vec![]);
    assert!(matches!(after.push(&[0x82, 0xac]), text::Pushed::Text(t) if t == "��"));
    assert!(matches!(before.push(&[0x82, 0xac]), text::Pushed::Text(t) if t == "€"));
    assert!(before.finish().is_empty());
    println!("CONFIRMED: rebuilding TextStream after token replay corrupts split UTF-8");
}

async fn journal_checks() {
    let j = Journal::new(Duration::from_millis(1));
    j.begin("old", vec![1]);
    j.finish("old", Status::Done { finish_reason: "stop".into() });
    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(j.get("old").is_some());
    j.begin("new", vec![2]);
    assert!(j.get("old").is_none());
    println!("CONFIRMED: journal TTL does not evict until a subsequent begin()");

    for n in [1000, 2000, 4000] {
        let j = Journal::new(Duration::from_secs(600));
        j.begin("bench", vec![1; 1024]);
        for _ in 0..n { j.push("bench", 42, "word"); }
        let t = std::time::Instant::now();
        // resume() calls get() once for every output event.
        for _ in 0..n { std::hint::black_box(j.get("bench")); }
        println!("MEASURED: journal replay clones n={n}, elapsed_ms={:.1}", t.elapsed().as_secs_f64() * 1000.0);
    }
}

fn plan_checks() {
    let dir = tempfile::tempdir().unwrap();
    let store = flux_plan::store::PlanStore::new(dir.path());
    let saved = fixture_plan();
    store.save(&saved).unwrap();
    let mut requested_key = saved.key.clone();
    requested_key.ctx_bucket = flux_core::plan::ctx_bucket(4096);
    let reused = store.lookup(&requested_key).unwrap();
    assert_eq!(reused.workload.n_ctx_seq, 3072);
    assert!(reused.workload.n_ctx_seq < 4096);
    println!("CONFIRMED: a 4096-context cache lookup returns a 3072-context plan");
}

async fn worker_checks() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = FluxConfig { data_dir: dir.path().to_owned(), ..Default::default() };
    let st = flux_serve::build(cfg, fixture_plan(), None).await.unwrap();
    let w = st.supervisor.current().await.1;
    assert!(tokio::time::timeout(Duration::from_millis(100), w.tokenize("HANG", true)).await.is_err());
    assert_eq!(w.tokenize("normal", true).await.unwrap(), vec![1, 2]);
    println!("CONFIRMED: unresponsive worker RPC requires a caller-supplied timeout");

    let response = flux_serve::openai::completions(
        State(st.clone()), HeaderMap::new(), Json(json!({"prompt":[1,2],"max_tokens":1}))
    ).await;
    assert!(response.status().is_success());
    let body = axum::body::to_bytes(response.into_body(), 100_000).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["choices"][0]["text"], "hello </");
    let id = body["id"].as_str().unwrap();
    assert_eq!(st.journal.get(id).unwrap().texts.concat(), "hello ");
    println!("CONFIRMED: completed HTTP reply includes tail that the resume journal omits");

    w.kill().await;
    for _ in 0..2 {
        let response = flux_serve::openai::chat(
            State(st.clone()), HeaderMap::new(), Json(json!({"messages":[{"role":"user","content":"hi"}]}))
        ).await;
        assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST);
    }
    assert_eq!(st.supervisor.current().await.0, 0);
    assert!(st.admission.closed_reason().is_none());
    println!("CONFIRMED: idle worker death leaves repeated chat failures, generation=0, admission open");

    let w = Arc::new(Worker::spawn(&flux_core::plan::EngineKind::Native, None).await.unwrap());
    w.kill().await;

    let cfg = FluxConfig { data_dir: dir.path().to_owned(), ..Default::default() };
    let replanner: flux_serve::Replanner = Arc::new(|_| Box::pin(std::future::pending()));
    let st = flux_serve::build(cfg, fixture_plan(), Some(replanner)).await.unwrap();
    assert!(tokio::time::timeout(Duration::from_millis(100), flux_serve::replan_now(&st)).await.is_err());
    assert_eq!(st.admission.closed_reason().as_deref(), Some("replanning"));
    assert_eq!(st.supervisor.current().await.0, 0);
    assert!(st.supervisor.current().await.1.tokenize("normal", true).await.is_err());
    println!("CONFIRMED: cancelled replan leaves admission closed and the old worker stopped");

    let cfg = FluxConfig { data_dir: dir.path().to_owned(), ..Default::default() };
    let mut plan = fixture_plan();
    plan.engine = flux_core::plan::EngineKind::External("audit".into());
    let st = flux_serve::build(cfg, plan, None).await.unwrap();
    let response = flux_serve::openai::chat(State(st.clone()), HeaderMap::new(),
        Json(json!({"messages":[{"role":"user","content":"hi"}]}))).await;
    assert!(response.status().is_success());
    let body = axum::body::to_bytes(response.into_body(), 100_000).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(body["choices"][0]["message"]["tool_calls"].is_null());
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    let id = body["id"].as_str().unwrap();
    assert_eq!(st.journal.get(id).unwrap().status, Status::Running);
    assert_eq!(st.journal.running(), 1);
    println!("CONFIRMED: external chat loses nonstream tool calls and leaves journal Running");
    st.supervisor.shutdown().await;
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("mock-worker.py");
    std::env::set_var("FLUX_WORKER", fixture);
    admission_checks().await;
    text_checks();
    journal_checks().await;
    plan_checks();
    tokio::time::timeout(Duration::from_secs(10), worker_checks()).await.expect("fixture timed out");
}
