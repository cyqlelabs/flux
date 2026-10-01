//! Contestants: each runs as its own server process, started cold and stopped after its trial.

use crate::client::{wait_healthy, Api};
use anyhow::{Context, Result};
use flux_core::config::FluxConfig;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Kind {
    /// `flux serve <plan>`.
    Flux { plan: String },
    /// The pinned llama-server with explicit flags (host and port are added).
    LlamaServer { args: Vec<String> },
    /// An engine from flux.toml `[engines]`.
    External { engine: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contestant {
    pub label: String,
    pub kind: Kind,
}

impl Contestant {
    pub fn api(&self) -> Api {
        match self.kind {
            Kind::Flux { .. } => Api::FluxCompletion,
            Kind::LlamaServer { .. } => Api::LlamaCompletion,
            Kind::External { .. } => Api::Chat,
        }
    }

    pub fn tokenize_path(&self) -> Option<&'static str> {
        match self.kind {
            Kind::Flux { .. } => Some("/flux/tokenize"),
            Kind::LlamaServer { .. } => Some("/tokenize"),
            Kind::External { .. } => None,
        }
    }
}

pub struct Running {
    child: tokio::process::Child,
    pub base: String,
    pub pid: u32,
    pub load_s: f64,
    /// Bytes the server tree read from storage while loading.
    pub load_read_bytes: u64,
}

fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

/// How to launch one contestant: program, arguments, environment, working directory, health path, startup timeout.
type Launch = (PathBuf, Vec<String>, Vec<(String, String)>, Option<PathBuf>, String, u64);

pub async fn start(cfg: &FluxConfig, c: &Contestant, model: &Path, ctx_total: u32, log: &Path) -> Result<Running> {
    let port = free_port()?;
    let (program, args, env, cwd, health, timeout): Launch = match &c.kind {
        Kind::Flux { plan } => {
            (std::env::current_exe()?, vec!["serve".into(), plan.clone(), "--port".into(), port.to_string()], vec![], None, "/health".into(), 1800)
        }
        Kind::LlamaServer { args } => {
            let mut a = args.clone();
            a.extend(["--host", "127.0.0.1", "--port", &port.to_string(), "--no-webui"].map(String::from));
            (cfg.llama_bin("llama-server"), a, vec![], None, "/health".into(), 1800)
        }
        Kind::External { engine } => {
            let e = cfg.engines.get(engine).with_context(|| format!("engine {engine} is not in flux.toml"))?;
            let subst =
                |s: &str| s.replace("{port}", &port.to_string()).replace("{model}", &model.display().to_string()).replace("{ctx}", &ctx_total.to_string());
            let mut cmd: Vec<String> = e.command.iter().chain(&e.args).map(|s| subst(s)).collect();
            let program = PathBuf::from(cmd.remove(0));
            (program, cmd, e.env.clone().into_iter().collect(), e.cwd.clone(), e.health_path.clone(), e.startup_timeout_s)
        }
    };
    let logf = std::fs::File::options().create(true).append(true).open(log)?;
    let mut cmd = tokio::process::Command::new(&program);
    // Its own process group: engines started through wrappers (Strata's server.py) must stop with them.
    cmd.args(&args).envs(env).stdin(Stdio::null()).stdout(Stdio::from(logf.try_clone()?)).stderr(Stdio::from(logf)).kill_on_drop(true).process_group(0);
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let t0 = Instant::now();
    let mut child = cmd.spawn().with_context(|| format!("starting {}", program.display()))?;
    let pid = child.id().context("server pid")?;
    let base = format!("http://127.0.0.1:{port}");
    let http = reqwest::Client::new();
    let exited = std::sync::atomic::AtomicBool::new(false);
    let alive = || {
        if exited.load(std::sync::atomic::Ordering::Relaxed) {
            return false;
        }
        let ok =
            std::path::Path::new(&format!("/proc/{pid}")).exists() && !std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default().contains(") Z");
        if !ok {
            exited.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        ok
    };
    if let Err(e) = wait_healthy(&http, &base, &health, timeout, alive).await {
        kill_group(pid, libc::SIGKILL);
        let _ = child.wait().await;
        return Err(e.context(format!("{} failed to start (log: {})", c.label, log.display())));
    }
    Ok(Running { load_s: t0.elapsed().as_secs_f64(), load_read_bytes: crate::proc::tree_io_read(pid), child, base, pid })
}

fn kill_group(pid: u32, signal: i32) {
    unsafe {
        libc::kill(-(pid as i32), signal);
    }
}

impl Running {
    /// SIGTERM to the server's process group, then SIGKILL after 15 s.
    pub async fn stop(mut self) {
        kill_group(self.pid, libc::SIGTERM);
        if tokio::time::timeout(std::time::Duration::from_secs(15), self.child.wait()).await.is_err() {
            kill_group(self.pid, libc::SIGKILL);
            let _ = self.child.wait().await;
        }
        // Children that outlive the group leader (a wrapper's engine) still get the signal.
        kill_group(self.pid, libc::SIGKILL);
    }
}
