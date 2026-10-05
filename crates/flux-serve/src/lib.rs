//! flux-serve: the HTTP service in front of a plan's worker. Asynchronous web work stays here;
//! the kernel scheduling loop lives in the worker process.

pub mod admission;
pub mod generate;
pub mod journal;
pub mod live;
pub mod monitor;
pub mod openai;
pub mod supervisor;

use admission::Admission;
use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use flux_core::config::FluxConfig;
use flux_core::plan::Plan;
use futures::future::BoxFuture;
use journal::Journal;
use serde_json::json;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use supervisor::Supervisor;

/// Produces a new plan for the same model and workload (supplied by the CLI, which owns ingest and probing).
pub type Replanner = Arc<dyn Fn(Plan) -> BoxFuture<'static, Result<Plan>> + Send + Sync>;

pub struct AppState {
    pub cfg: FluxConfig,
    pub supervisor: Supervisor,
    pub admission: Admission,
    pub journal: Journal,
    pub pressure_closed: AtomicBool,
    /// Admission closes when MemAvailable falls below this reserve; adjustable at run time.
    pub host_reserve_bytes: std::sync::atomic::AtomicU64,
    pub drift: Mutex<monitor::Drift>,
    pub live: live::Live,
    pub model_name: String,
    /// `tokens` or `chat`, from the worker's hello.
    pub level: String,
    pub concurrency: usize,
    pub started: Instant,
    pub replanner: Option<Replanner>,
    pub replan_lock: tokio::sync::Mutex<()>,
    pub control_slots: tokio::sync::Semaphore,
}

fn retune_cost_s(p: &Plan, load_ms: f64) -> f64 {
    p.validation.as_ref().map_or(0.0, |v| v.tuning_seconds) + load_ms / 1e3
}

/// The chosen candidate's short-prompt decode rate and, where measured, its long-prompt measurement.
fn validated_rates(p: &Plan) -> monitor::Rates {
    let Some(c) = p.validation.as_ref().and_then(|v| v.candidates.iter().find(|c| c.label == v.chosen)) else { return monitor::Rates::default() };
    let m = c.validation.as_ref().or(c.calibration.as_ref());
    monitor::Rates { prose_tps: m.map(|m| m.decode_tps.p50), agent_tps: m.and_then(|m| m.agent_decode_tps.as_ref()).map(|a| a.p50), depth: c.depth.clone() }
}

pub async fn build(cfg: FluxConfig, plan: Plan, replanner: Option<Replanner>) -> Result<Arc<AppState>> {
    anyhow::ensure!(!plan.model_files.is_empty() && plan.workload.concurrency > 0, "plan needs a model and positive concurrency");
    let logs = cfg.data_dir.join("logs");
    let total = monitor::read_kib(&std::fs::read_to_string("/proc/meminfo")?, "MemTotal:").ok_or_else(|| anyhow::anyhow!("cannot read total host memory"))?;
    let host_reserve_bytes = cfg.host_reserve_bytes(total.saturating_mul(1024));
    std::fs::create_dir_all(&logs)?;
    let model_name = plan.model_files[0].file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let concurrency = plan.workload.concurrency as usize;
    let rates = validated_rates(&plan);
    let tuning = plan.clone();
    let supervisor = Supervisor::start_with(plan, logs.join("serve-worker.log"), cfg.serve.worker_timeouts.clone()).await?;
    supervisor.current().await.1.send(&flux_core::protocol::Request::Admission { host_reserve_bytes, min_reply: cfg.serve.min_reply }).await?;
    let drift = monitor::Drift::new(rates, retune_cost_s(&tuning, supervisor.loaded().await.load_ms));
    let level = supervisor.current().await.1.level.clone();
    Ok(Arc::new(AppState {
        admission: Admission::new(concurrency, cfg.serve.queue_depth),
        journal: Journal::with_limit(Duration::from_secs(cfg.serve.journal_ttl_s), cfg.serve.journal_max_bytes),
        pressure_closed: AtomicBool::new(false),
        host_reserve_bytes: std::sync::atomic::AtomicU64::new(host_reserve_bytes),
        drift: Mutex::new(drift),
        live: live::Live::default(),
        model_name,
        level,
        concurrency,
        started: Instant::now(),
        replanner,
        replan_lock: tokio::sync::Mutex::new(()),
        control_slots: tokio::sync::Semaphore::new(4),
        cfg,
        supervisor,
    }))
}

pub fn router(st: Arc<AppState>) -> Router {
    let limit = st.cfg.serve.max_body_bytes;
    Router::new()
        .route("/health", get(health))
        .route("/live", get(|| async { StatusCode::OK }))
        .route("/v1/models", get(openai::models))
        .route("/v1/chat/completions", post(openai::chat))
        .route("/v1/completions", post(openai::completions))
        .route("/flux/plan", get(plan))
        .route("/flux/stats", get(stats))
        .route("/flux/replan", post(replan))
        .route("/flux/tokenize", post(tokenize))
        .route("/flux/admission", post(admission))
        .route("/flux/requests/{id}", get(openai::request_entry))
        .route("/flux/requests/{id}/stream", get(openai::resume))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(limit))
        // the extractors' own 2 MB default would refuse long contexts below the configured limit
        .layer(axum::extract::DefaultBodyLimit::max(limit))
        .with_state(st)
}

/// Serves until interrupted; binds the configured local address by default.
pub async fn run(st: Arc<AppState>, host: &str, port: u16) -> Result<()> {
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    let mut monitors = monitor::spawn(st.clone());
    tracing::info!("serving plan {} on http://{}", st.supervisor.plan().await.id, listener.local_addr()?);
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    let result = axum::serve(listener, router(st.clone())).with_graceful_shutdown(shutdown).await;
    monitors.abort_all();
    while monitors.join_next().await.is_some() {}
    let _exclusive = st.replan_lock.lock().await;
    st.supervisor.shutdown().await;
    result?;
    Ok(())
}

async fn health(State(st): State<Arc<AppState>>) -> Response {
    let closed = st.admission.closed_reason().or(if st.supervisor.current().await.1.is_alive().await { None } else { Some("worker unavailable".into()) });
    let status = if closed.is_some() { StatusCode::SERVICE_UNAVAILABLE } else { StatusCode::OK };
    (status, Json(json!({"status": if closed.is_some() { "closed" } else { "ok" }, "reason": closed, "plan": st.supervisor.plan().await.id}))).into_response()
}

/// Adjusts the host-memory reserve in bytes; the server binds locally.
async fn admission(State(st): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> Response {
    match body.get("host_reserve_bytes").and_then(|v| v.as_u64()) {
        Some(m) => {
            // stored first: requests resend the stored reserve, and one that read the old value must not undo this
            st.host_reserve_bytes.store(m, std::sync::atomic::Ordering::Relaxed);
            if let Err(e) = st
                .supervisor
                .current()
                .await
                .1
                .send(&flux_core::protocol::Request::Admission { host_reserve_bytes: m, min_reply: st.cfg.serve.min_reply })
                .await
            {
                return openai::error(StatusCode::SERVICE_UNAVAILABLE, "worker_unavailable", e.to_string());
            }
            Json(json!({"host_reserve_bytes": m})).into_response()
        }
        None => openai::error(StatusCode::BAD_REQUEST, "invalid_request_error", "expected {\"host_reserve_bytes\": N}"),
    }
}

async fn tokenize(State(st): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> Response {
    let Ok(_slot) = st.control_slots.try_acquire() else { return openai::error(StatusCode::TOO_MANY_REQUESTS, "control_busy", "control queue full") };
    if let Some(reason) = st.admission.closed_reason() {
        return openai::error(StatusCode::SERVICE_UNAVAILABLE, "admission_closed", reason);
    }
    let text = body.get("content").and_then(|v| v.as_str()).unwrap_or_default();
    let add_special = body.get("add_special").and_then(|v| v.as_bool()).unwrap_or(true);
    match st.supervisor.current().await.1.tokenize(text, add_special).await {
        Ok(tokens) => Json(json!({"tokens": tokens})).into_response(),
        Err(e) => openai::error(StatusCode::BAD_REQUEST, "tokenize_failed", e.to_string()),
    }
}

async fn plan(State(st): State<Arc<AppState>>) -> Json<Plan> {
    Json(st.supervisor.plan().await)
}

async fn stats(State(st): State<Arc<AppState>>) -> Response {
    let Ok(_slot) = st.control_slots.try_acquire() else { return openai::error(StatusCode::TOO_MANY_REQUESTS, "control_busy", "control queue full") };
    let (gen, w) = st.supervisor.current().await;
    let worker = if st.level == "tokens" || w.engine != "native" { w.stats().await.ok() } else { None };
    let host_weights = st.supervisor.plan().await.host.resident_weights;
    // KV pages live in VRAM up to each GPU's budget and in pinned RAM past it
    let memory = worker.as_ref().map(|w| {
        let kv_ram: u64 = w.kv_pages.as_object().map_or(0, |classes| classes.values().filter_map(|c| c["ram_bytes"].as_u64()).sum());
        let vram: Vec<_> = w
            .memory
            .iter()
            .filter(|m| m.device != "CPU")
            .map(|m| json!({"device": m.device, "weights": m.model, "kv_and_state": m.context, "compute": m.compute}))
            .collect();
        json!({"ram": {"host_weights": host_weights, "kv_pages": kv_ram, "worker_rss": w.rss_bytes}, "vram": vram})
    });
    let d = st.drift.lock().unwrap();
    Json(json!({
        "uptime_s": st.started.elapsed().as_secs(),
        "worker_generation": gen,
        "engine": w.engine,
        "admission": {
            "host_reserve_bytes": st.host_reserve_bytes.load(std::sync::atomic::Ordering::Relaxed),
            "min_reply": st.cfg.serve.min_reply,
            "closed": st.admission.closed_reason(),
            "running": st.admission.in_use(st.concurrency),
            "waiting": st.admission.waiting(),
            "admitted": st.admission.admitted.load(std::sync::atomic::Ordering::Relaxed),
            "rejected": st.admission.rejected.load(std::sync::atomic::Ordering::Relaxed),
        },
        "journal_running": st.journal.running(),
        "drift": {"baseline_tps": d.baseline_tps, "agent_baseline_tps": d.agent_tps, "recent_tps": d.recent_tps, "recent_vs_expected": d.recent_ratio, "drifted": d.drifted},
        "requests": st.live.snapshot(),
        "memory": memory,
        "worker": worker,
    }))
    .into_response()
}

/// Closes admission, drains running requests, frees the devices, plans again and switches
/// at this request boundary. The previous plan is restored if the new one fails to load.
async fn replan(State(st): State<Arc<AppState>>) -> Response {
    match replan_now(&st).await {
        Ok(p) => Json(json!({"plan": p.id, "placement": p.placement.describe()})).into_response(),
        Err(e) => openai::error(StatusCode::INTERNAL_SERVER_ERROR, "replan_failed", format!("{e:#}")),
    }
}

pub async fn replan_now(st: &Arc<AppState>) -> Result<Plan> {
    let st = st.clone();
    tokio::spawn(async move { replan_inner(&st).await }).await?
}

async fn replan_inner(st: &Arc<AppState>) -> Result<Plan> {
    use futures::FutureExt;
    let _exclusive = st.replan_lock.lock().await;
    let replanner = st.replanner.clone().ok_or_else(|| anyhow::anyhow!("this server was started without a replanner"))?;
    st.admission.block("replan", "replanning");
    let deadline = Duration::from_secs(st.cfg.serve.replan_timeout_s);
    let permits = match tokio::time::timeout(deadline, st.admission.drain(st.concurrency)).await {
        Ok(p) => p,
        Err(_) => {
            st.admission.unblock("replan");
            anyhow::bail!("draining requests timed out")
        }
    };
    let current = st.supervisor.plan().await;
    st.supervisor.shutdown().await;
    let result = async {
        let p = tokio::time::timeout(deadline, std::panic::AssertUnwindSafe(async { replanner(current.clone()).await }).catch_unwind())
            .await
            .map_err(|_| anyhow::anyhow!("replanner timed out"))?
            .map_err(|_| anyhow::anyhow!("replanner panicked"))??;
        anyhow::ensure!(
            p.workload.concurrency == current.workload.concurrency && p.engine == current.engine,
            "live replanning cannot change concurrency or engine"
        );
        st.supervisor.switch(p.clone()).await?;
        Ok::<Plan, anyhow::Error>(p)
    }
    .await;
    let mut result = result;
    match &result {
        Ok(p) => {
            let load_ms = st.supervisor.loaded().await.load_ms;
            *st.drift.lock().unwrap() = monitor::Drift::new(validated_rates(p), retune_cost_s(p, load_ms));
        }
        // A planning failure leaves the devices free: bring the previous plan back.
        Err(_) => {
            if !st.supervisor.current().await.1.is_alive().await {
                if let Err(e) = st.supervisor.switch(current).await {
                    result = Err(anyhow::anyhow!("replanning failed; restoring previous plan failed: {e:#}"));
                }
            }
        }
    }
    drop(permits);
    if st.supervisor.current().await.1.is_alive().await {
        st.admission.unblock("worker");
    } else {
        st.admission.block("worker", "worker unavailable after replanning");
    }
    st.admission.unblock("replan");
    result
}
