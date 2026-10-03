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
    /// Admission closes when MemAvailable falls below this (MiB); adjustable at run time.
    pub min_available_mib: std::sync::atomic::AtomicU64,
    pub drift: Mutex<monitor::Drift>,
    pub live: live::Live,
    pub model_name: String,
    /// `tokens` or `chat`, from the worker's hello.
    pub level: String,
    pub concurrency: usize,
    pub started: Instant,
    pub replanner: Option<Replanner>,
}

fn retune_cost_s(p: &Plan, load_ms: f64) -> f64 {
    p.validation.as_ref().map_or(0.0, |v| v.tuning_seconds) + load_ms / 1e3
}

fn validated_rate(p: &Plan) -> Option<f64> {
    let v = p.validation.as_ref()?;
    v.candidates.iter().find(|c| c.label == v.chosen).and_then(|c| c.validation.as_ref().or(c.calibration.as_ref())).map(|m| m.decode_tps.p50)
}

pub async fn build(cfg: FluxConfig, plan: Plan, replanner: Option<Replanner>) -> Result<Arc<AppState>> {
    let logs = cfg.data_dir.join("logs");
    std::fs::create_dir_all(&logs)?;
    let model_name = plan.model_files[0].file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let concurrency = plan.workload.concurrency as usize;
    let rate = validated_rate(&plan);
    let tuning = plan.clone();
    let supervisor = Supervisor::start(plan, logs.join("serve-worker.log")).await?;
    let drift = monitor::Drift::new(rate, retune_cost_s(&tuning, supervisor.loaded().await.load_ms));
    let level = supervisor.current().await.1.level.clone();
    Ok(Arc::new(AppState {
        admission: Admission::new(concurrency, cfg.serve.queue_depth),
        journal: Journal::new(Duration::from_secs(cfg.serve.journal_ttl_s)),
        pressure_closed: AtomicBool::new(false),
        min_available_mib: std::sync::atomic::AtomicU64::new(cfg.serve.min_available_mib),
        drift: Mutex::new(drift),
        live: live::Live::default(),
        model_name,
        level,
        concurrency,
        started: Instant::now(),
        replanner,
        cfg,
        supervisor,
    }))
}

pub fn router(st: Arc<AppState>) -> Router {
    let limit = st.cfg.serve.max_body_bytes;
    Router::new()
        .route("/health", get(health))
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
        .with_state(st)
}

/// Serves until interrupted; binds the configured local address by default.
pub async fn run(st: Arc<AppState>, host: &str, port: u16) -> Result<()> {
    monitor::spawn(st.clone());
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    tracing::info!("serving plan {} on http://{}", st.supervisor.plan().await.id, listener.local_addr()?);
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    axum::serve(listener, router(st.clone())).with_graceful_shutdown(shutdown).await?;
    st.supervisor.shutdown().await;
    Ok(())
}

async fn health(State(st): State<Arc<AppState>>) -> Response {
    let closed = st.admission.closed_reason();
    let status = if closed.is_some() { StatusCode::SERVICE_UNAVAILABLE } else { StatusCode::OK };
    (status, Json(json!({"status": if closed.is_some() { "closed" } else { "ok" }, "reason": closed, "plan": st.supervisor.plan().await.id}))).into_response()
}

/// Adjusts the host-memory admission threshold (`{"min_available_mib": N}`); the server binds locally.
async fn admission(State(st): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> Response {
    match body.get("min_available_mib").and_then(|v| v.as_u64()) {
        Some(m) => {
            st.min_available_mib.store(m, std::sync::atomic::Ordering::Relaxed);
            Json(json!({"min_available_mib": m})).into_response()
        }
        None => openai::error(StatusCode::BAD_REQUEST, "invalid_request_error", "expected {\"min_available_mib\": N}"),
    }
}

async fn tokenize(State(st): State<Arc<AppState>>, Json(body): Json<serde_json::Value>) -> Response {
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
    let (gen, w) = st.supervisor.current().await;
    let worker = if st.level == "tokens" || w.engine != "native" { w.stats().await.ok() } else { None };
    let d = st.drift.lock().unwrap();
    Json(json!({
        "uptime_s": st.started.elapsed().as_secs(),
        "worker_generation": gen,
        "engine": w.engine,
        "admission": {
            "closed": st.admission.closed_reason(),
            "running": st.admission.in_use(st.concurrency),
            "waiting": st.admission.waiting(),
            "admitted": st.admission.admitted.load(std::sync::atomic::Ordering::Relaxed),
            "rejected": st.admission.rejected.load(std::sync::atomic::Ordering::Relaxed),
        },
        "journal_running": st.journal.running(),
        "drift": {"baseline_tps": d.baseline_tps, "recent_tps": d.recent_tps, "drifted": d.drifted},
        "requests": st.live.snapshot(),
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
    let replanner = st.replanner.clone().ok_or_else(|| anyhow::anyhow!("this server was started without a replanner"))?;
    st.admission.close("replanning");
    let permits = st.admission.drain(st.concurrency).await;
    let current = st.supervisor.plan().await;
    st.supervisor.shutdown().await;
    let result = async {
        let p = replanner(current.clone()).await?;
        st.supervisor.switch(p.clone()).await?;
        Ok::<Plan, anyhow::Error>(p)
    }
    .await;
    match &result {
        Ok(p) => {
            let load_ms = st.supervisor.loaded().await.load_ms;
            *st.drift.lock().unwrap() = monitor::Drift::new(validated_rate(p), retune_cost_s(p, load_ms));
        }
        // A planning failure leaves the devices free: bring the previous plan back.
        Err(_) => {
            if st.supervisor.current().await.1.stats().await.is_err() {
                let _ = st.supervisor.switch(current).await;
            }
        }
    }
    drop(permits);
    st.admission.open();
    result
}
