//! Supervisor side of the worker protocol: spawns a flux-worker process and routes its events.

use crate::plan::{EngineKind, Plan};
use crate::protocol::{ErrorCode, Event, FinishReason, Request, Sampling, WorkerStats, PROTOCOL_VERSION};
use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot};

/// flux-worker next to the current executable, unless `FLUX_WORKER` names another.
pub fn worker_binary() -> PathBuf {
    if let Some(p) = std::env::var_os("FLUX_WORKER") {
        return p.into();
    }
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.parent().map(Path::to_path_buf).unwrap_or_default();
    // Test binaries live one level deeper, in target/<profile>/deps.
    [dir.join("flux-worker"), dir.join("../flux-worker")].into_iter().find(|p| p.exists()).unwrap_or_else(|| dir.join("flux-worker"))
}

/// Worker subcommand serving `engine`.
pub fn worker_args(engine: &EngineKind) -> Vec<String> {
    match engine {
        EngineKind::Native => vec!["serve".into()],
        EngineKind::LlamaServer => vec!["external".into(), "--api".into(), "llama-server".into()],
        EngineKind::External(_) => vec!["external".into(), "--api".into(), "openai-chat".into()],
    }
}

#[derive(Default)]
struct Routes {
    by_req: HashMap<String, mpsc::UnboundedSender<Event>>,
    by_id: HashMap<u64, oneshot::Sender<Event>>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Timeouts {
    pub hello_s: u64,
    pub load_s: u64,
    pub rpc_s: u64,
    pub write_s: u64,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self { hello_s: 30, load_s: 1800, rpc_s: 120, write_s: 30 }
    }
}

struct CallRoute<'a> {
    routes: &'a Mutex<Routes>,
    id: u64,
}
impl Drop for CallRoute<'_> {
    fn drop(&mut self) {
        self.routes.lock().unwrap().by_id.remove(&self.id);
    }
}

// Cancelling a partial JSON-line write must retire that protocol connection.
struct WriteGuard<'a> {
    alive: &'a AtomicBool,
    complete: bool,
}
impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        if !self.complete {
            self.alive.store(false, Ordering::Release);
        }
    }
}

pub struct Worker {
    stdin: tokio::sync::Mutex<ChildStdin>,
    routes: Arc<Mutex<Routes>>,
    control: tokio::sync::Mutex<mpsc::UnboundedReceiver<Event>>,
    child: tokio::sync::Mutex<Child>,
    next_id: AtomicU64,
    alive: Arc<AtomicBool>,
    timeouts: Timeouts,
    pub engine: String,
    pub backend_revision: String,
    pub backend_build: String,
    /// `tokens` or `chat`.
    pub level: String,
}

#[derive(Debug, Clone)]
pub struct Loaded {
    pub load_ms: f64,
    pub n_ctx_seq: u32,
    pub n_seq: u32,
    pub memory: Vec<crate::protocol::DeviceMemory>,
}

#[derive(Debug, Clone)]
pub struct Templated {
    pub prompt: String,
    pub preserved_tokens: Vec<i32>,
    pub additional_stops: Vec<String>,
    /// How to split the reply into reasoning, content and tool calls (`ChatOptions::parser`).
    pub parser: Option<serde_json::Value>,
    /// Prompt positions worth checkpointing for reuse (`ChatOptions::checkpoints`).
    pub checkpoints: Vec<u32>,
}

/// Chat extras of a request: reply parsing and the prompt positions worth checkpointing for reuse.
#[derive(Debug, Clone, Default)]
pub struct ChatOptions {
    pub parser: Option<serde_json::Value>,
    /// Reply text delivered before a worker restart; parsing resumes after it.
    pub prefix: String,
    pub checkpoints: Vec<u32>,
}

impl Worker {
    /// Starts a worker for `engine`; its stderr goes to `log` (or is inherited).
    pub async fn spawn(engine: &EngineKind, log: Option<&Path>) -> Result<Worker> {
        Self::spawn_with(engine, log, Timeouts::default()).await
    }

    pub async fn spawn_with(engine: &EngineKind, log: Option<&Path>, timeouts: Timeouts) -> Result<Worker> {
        let bin = worker_binary();
        let stderr = match log {
            Some(p) => Stdio::from(std::fs::File::options().create(true).append(true).open(p).with_context(|| format!("opening {}", p.display()))?),
            None => Stdio::inherit(),
        };
        let mut child = Command::new(&bin)
            .args(worker_args(engine))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("starting {}", bin.display()))?;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let routes = Arc::new(Mutex::new(Routes::default()));
        let (ctl_tx, mut ctl_rx) = mpsc::unbounded_channel();
        let alive = Arc::new(AtomicBool::new(true));
        tokio::spawn(read_events(stdout, routes.clone(), ctl_tx, alive.clone()));

        let Some(Event::Hello { protocol, engine, backend_revision, backend_build, level, .. }) =
            tokio::time::timeout(Duration::from_secs(timeouts.hello_s), ctl_rx.recv()).await.context("worker hello timed out")?
        else {
            bail!("worker exited before its hello");
        };
        if protocol != PROTOCOL_VERSION {
            bail!("worker speaks protocol {protocol}, this build expects {PROTOCOL_VERSION}");
        }
        let w = Worker {
            stdin: tokio::sync::Mutex::new(stdin),
            routes,
            control: tokio::sync::Mutex::new(ctl_rx),
            child: tokio::sync::Mutex::new(child),
            next_id: AtomicU64::new(1),
            alive,
            timeouts,
            engine,
            backend_revision,
            backend_build,
            level,
        };
        w.send(&Request::Hello { protocol: PROTOCOL_VERSION }).await?;
        Ok(w)
    }

    pub async fn send(&self, r: &Request) -> Result<()> {
        anyhow::ensure!(self.alive.load(Ordering::Acquire), "worker unavailable");
        let mut line = serde_json::to_vec(r)?;
        line.push(b'\n');
        let mut guard = WriteGuard { alive: &self.alive, complete: false };
        tokio::time::timeout(Duration::from_secs(self.timeouts.write_s), async {
            let mut s = self.stdin.lock().await;
            s.write_all(&line).await.context("worker stdin closed")?;
            s.flush().await?;
            Ok::<_, anyhow::Error>(())
        })
        .await
        .context("worker write timed out")??;
        guard.complete = true;
        Ok(())
    }

    async fn control_event(&self) -> Result<Event> {
        self.control.lock().await.recv().await.ok_or_else(|| anyhow!("worker exited"))
    }

    pub async fn load(&self, plan: &Plan) -> Result<Loaded> {
        self.load_with(plan, false).await
    }

    /// Loads with per-node time attribution installed (for `trace`).
    pub async fn load_with(&self, plan: &Plan, trace: bool) -> Result<Loaded> {
        let result = tokio::time::timeout(Duration::from_secs(self.timeouts.load_s), self.load_inner(plan, trace)).await;
        if result.is_err() {
            self.kill().await;
        }
        result.context("worker load timed out")?
    }

    async fn load_inner(&self, plan: &Plan, trace: bool) -> Result<Loaded> {
        anyhow::ensure!(
            plan.key.backend_build == self.backend_build && plan.key.backend_revision == self.backend_revision,
            "plan backend identity differs from the running worker; replan with this build"
        );
        self.send(&Request::Load { plan: Box::new(plan.clone()), trace }).await?;
        loop {
            match self.control_event().await? {
                Event::Loaded { load_ms, n_ctx_seq, n_seq, memory } => return Ok(Loaded { load_ms, n_ctx_seq, n_seq, memory }),
                Event::Error { message, .. } => bail!("load failed: {message}"),
                _ => {}
            }
        }
    }

    async fn call(&self, make: impl FnOnce(u64) -> Request) -> Result<Event> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.routes.lock().unwrap().by_id.insert(id, tx);
        let _route = CallRoute { routes: &self.routes, id };
        self.send(&make(id)).await?;
        let result = tokio::time::timeout(Duration::from_secs(self.timeouts.rpc_s), rx).await;
        if result.is_err() {
            self.kill().await;
        }
        match result.context("worker RPC timed out")?.map_err(|_| anyhow!("worker exited"))? {
            Event::Error { message, .. } => bail!("{message}"),
            ev => Ok(ev),
        }
    }

    pub async fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        match self.call(|id| Request::Tokenize { id, text: text.into(), add_special }).await? {
            Event::Tokens { tokens, .. } => Ok(tokens),
            ev => bail!("unexpected reply {ev:?}"),
        }
    }

    pub async fn apply_template(&self, messages: serde_json::Value, tools: Option<serde_json::Value>) -> Result<Templated> {
        match self.call(|id| Request::ApplyTemplate { id, messages, tools, add_generation_prompt: true }).await? {
            Event::Templated { prompt, preserved_tokens, additional_stops, parser, checkpoints, .. } => {
                Ok(Templated { prompt, preserved_tokens, additional_stops, parser, checkpoints })
            }
            ev => bail!("unexpected reply {ev:?}"),
        }
    }

    pub async fn stats(&self) -> Result<WorkerStats> {
        match self.call(|id| Request::Stats { id }).await? {
            Event::Stats { stats, .. } => Ok(stats),
            ev => bail!("unexpected reply {ev:?}"),
        }
    }

    pub async fn trace(&self, prompt: Vec<i32>, steps: u32, per_op: bool, routes: bool) -> Result<serde_json::Value> {
        match self.call(|id| Request::Trace { id, prompt, steps, per_op, routes }).await? {
            Event::Traced { report, .. } => Ok(report),
            ev => bail!("unexpected reply {ev:?}"),
        }
    }

    /// Registers the request's event stream, then sends `Prefill`; grant tokens with `credit`.
    #[allow(clippy::too_many_arguments)]
    pub async fn start(
        &self,
        req: &str,
        prompt: Vec<i32>,
        sampling: Sampling,
        stop: Vec<String>,
        max_tokens: u32,
        render_special: Vec<i32>,
        chat: ChatOptions,
    ) -> Result<mpsc::UnboundedReceiver<Event>> {
        let rx = self.route(req);
        let ChatOptions { parser, prefix, checkpoints } = chat;
        if let Err(e) = self
            .send(&Request::Prefill { req: req.into(), prompt, sampling, stop, max_tokens, render_special, chat: parser, chat_prefix: prefix, checkpoints })
            .await
        {
            self.routes.lock().unwrap().by_req.remove(req);
            return Err(e);
        }
        Ok(rx)
    }

    pub async fn chat(&self, req: &str, body: serde_json::Value) -> Result<mpsc::UnboundedReceiver<Event>> {
        let rx = self.route(req);
        if let Err(e) = self.send(&Request::Chat { req: req.into(), body }).await {
            self.routes.lock().unwrap().by_req.remove(req);
            return Err(e);
        }
        Ok(rx)
    }

    fn route(&self, req: &str) -> mpsc::UnboundedReceiver<Event> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.routes.lock().unwrap().by_req.insert(req.to_string(), tx);
        rx
    }

    pub async fn credit(&self, req: &str, n: u32) -> Result<()> {
        self.send(&Request::Decode { req: req.into(), n }).await
    }

    pub async fn cancel(&self, req: &str) -> Result<()> {
        let result = self.send(&Request::Cancel { req: req.into() }).await;
        self.routes.lock().unwrap().by_req.remove(req);
        result
    }

    pub async fn shutdown(&self) {
        if self.send(&Request::Shutdown).await.is_err() {
            self.kill().await;
            return;
        }
        let mut child = self.child.lock().await;
        if tokio::time::timeout(std::time::Duration::from_secs(10), child.wait()).await.is_err() {
            let _ = child.kill().await;
        }
    }

    /// Kills the process, e.g. for fault injection or after a hang.
    pub async fn kill(&self) {
        self.alive.store(false, Ordering::Release);
        let _ = self.child.lock().await.kill().await;
    }

    pub async fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Acquire) && matches!(self.child.lock().await.try_wait(), Ok(None))
    }

    pub async fn pid(&self) -> Option<u32> {
        self.child.lock().await.id()
    }
}

async fn read_events(stdout: tokio::process::ChildStdout, routes: Arc<Mutex<Routes>>, control: mpsc::UnboundedSender<Event>, alive: Arc<AtomicBool>) {
    let mut lines = BufReader::new(stdout).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let Ok(ev) = serde_json::from_str::<Event>(&line) else { break };
        let mut r = routes.lock().unwrap();
        let (req, id) = match &ev {
            Event::Prefilled { req, .. } | Event::Prefilling { req, .. } | Event::Token { req, .. } | Event::ChatChunk { req, .. } | Event::Paused { req } => {
                (Some(req.clone()), None)
            }
            Event::Finished { req, .. } => (Some(req.clone()), None),
            Event::Tokens { id, .. } | Event::Templated { id, .. } | Event::Stats { id, .. } | Event::Traced { id, .. } => (None, Some(*id)),
            Event::Error { req, id, .. } => (req.clone(), *id),
            _ => (None, None),
        };
        let finished = matches!(ev, Event::Finished { .. });
        if let Some(id) = id {
            if let Some(tx) = r.by_id.remove(&id) {
                let _ = tx.send(ev);
            }
        } else if let Some(req) = req {
            if let Some(tx) = r.by_req.get(&req) {
                let _ = tx.send(ev);
            }
            if finished {
                r.by_req.remove(&req);
            }
        } else {
            let _ = control.send(ev);
        }
    }
    // The worker is gone: every open stream ends with an explicit error.
    alive.store(false, Ordering::Release);
    let mut r = routes.lock().unwrap();
    for (req, tx) in r.by_req.drain() {
        let _ = tx.send(Event::Error { req: Some(req.clone()), id: None, code: ErrorCode::Backend, message: "worker process exited".into() });
        let _ = tx.send(Event::Finished { req, reason: FinishReason::Error, n_prompt: 0, n_decoded: 0, tail: String::new(), deltas: vec![], message: None });
    }
    r.by_id.clear();
}

/// Runs a one-shot native job (`measure`, `probe <kind>`) in a fresh worker process.
pub async fn oneshot_job(args: &[&str], input: &serde_json::Value) -> Result<serde_json::Value> {
    tokio::time::timeout(Duration::from_secs(1800), oneshot_inner(args, input)).await.context("worker job timed out after 1800 seconds")?
}

async fn oneshot_inner(args: &[&str], input: &serde_json::Value) -> Result<serde_json::Value> {
    let mut child = Command::new(worker_binary())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("starting flux-worker")?;
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(input.to_string().as_bytes()).await?;
    drop(stdin);
    let out = child.wait_with_output().await?;
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).with_context(|| {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let lines: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
        format!("worker {args:?} exited with {} and no result:\n{}", out.status, lines[lines.len().saturating_sub(10)..].join("\n"))
    })?;
    if let Some(e) = v.get("error").and_then(|e| e.as_str()) {
        bail!("{e}");
    }
    Ok(v)
}
