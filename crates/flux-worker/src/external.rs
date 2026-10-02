//! Engines that keep their own scheduler behind HTTP. The worker owns the engine process and
//! translates the Flux protocol to it: token level for llama-server's API, chat level otherwise.

use crate::out::Out;
use anyhow::{bail, Context, Result};
use flux_core::config::FluxConfig;
use flux_core::plan::{EngineKind, Plan};
use flux_core::protocol::{ErrorCode, Event, FinishReason, Request, Sampling, WorkerStats, PROTOCOL_VERSION};
use futures::StreamExt;
use std::collections::HashMap;
use std::os::fd::AsFd;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncBufReadExt;
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Api {
    LlamaServer,
    OpenAiChat,
}

struct Active {
    credit: Arc<Semaphore>,
    task: JoinHandle<()>,
}

struct State {
    out: Out,
    http: reqwest::Client,
    base: Option<String>,
    child: Option<tokio::process::Child>,
    api: Api,
    active: Arc<Mutex<HashMap<String, Active>>>,
    start: Instant,
}

pub async fn run(out: Out, api: Api) -> Result<()> {
    let mut st = State { out, http: reqwest::Client::new(), base: None, child: None, api, active: Arc::new(Mutex::new(HashMap::new())), start: Instant::now() };
    let mut lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                st.error(None, None, ErrorCode::Protocol, format!("bad request line: {e}"));
                continue;
            }
        };
        if matches!(req, Request::Shutdown) {
            break;
        }
        st.handle(req).await;
    }
    for (_, a) in st.active.lock().unwrap().drain() {
        a.task.abort();
    }
    if let Some(mut c) = st.child.take() {
        let _ = c.kill().await;
    }
    st.out.send(&Event::Bye);
    Ok(())
}

impl State {
    fn error(&self, req: Option<&str>, id: Option<u64>, code: ErrorCode, message: impl Into<String>) {
        self.out.send(&Event::Error { req: req.map(str::to_string), id, code, message: message.into() });
    }

    /// Refuses a generation request; it still ends with exactly one Finished.
    fn reject(&self, r: &Request, code: ErrorCode, message: impl Into<String>) {
        match r {
            Request::Prefill { req, .. } | Request::Chat { req, .. } => {
                self.error(Some(req), None, code, message);
                self.out.send(&Event::Finished { req: req.clone(), reason: FinishReason::Error, n_prompt: 0, n_decoded: 0, tail: String::new() });
            }
            Request::Tokenize { id, .. } | Request::ApplyTemplate { id, .. } | Request::Stats { id } | Request::Trace { id, .. } => {
                self.error(None, Some(*id), code, message)
            }
            _ => self.error(None, None, code, message),
        }
    }

    async fn handle(&mut self, r: Request) {
        if self.base.is_none() && !matches!(r, Request::Hello { .. } | Request::Load { .. }) {
            return self.reject(&r, ErrorCode::NotLoaded, "engine not started");
        }
        match r {
            Request::Hello { protocol } if protocol != PROTOCOL_VERSION => {
                self.error(None, None, ErrorCode::Protocol, format!("worker speaks protocol {PROTOCOL_VERSION}, supervisor sent {protocol}"))
            }
            Request::Hello { .. } | Request::Shutdown => {}
            Request::Load { plan, .. } => {
                let t0 = Instant::now();
                match self.launch(&plan).await {
                    Ok(()) => self.out.send(&Event::Loaded {
                        load_ms: t0.elapsed().as_secs_f64() * 1e3,
                        n_ctx_seq: plan.workload.n_ctx_seq,
                        n_seq: plan.workload.concurrency,
                        memory: vec![],
                    }),
                    Err(e) => self.error(None, None, ErrorCode::LoadFailed, format!("{e:#}")),
                }
            }
            Request::Tokenize { id, text, add_special } if self.api == Api::LlamaServer => {
                match self.post("/tokenize", serde_json::json!({"content": text, "add_special": add_special, "parse_special": true})).await {
                    Ok(v) => self.out.send(&Event::Tokens { id, tokens: serde_json::from_value(v["tokens"].clone()).unwrap_or_default() }),
                    Err(e) => self.error(None, Some(id), ErrorCode::Backend, e.to_string()),
                }
            }
            Request::ApplyTemplate { id, messages, tools, .. } if self.api == Api::LlamaServer => {
                match self.post("/apply-template", serde_json::json!({"messages": messages, "tools": tools})).await {
                    Ok(v) => self.out.send(&Event::Templated {
                        id,
                        prompt: v["prompt"].as_str().unwrap_or_default().to_string(),
                        preserved_tokens: vec![],
                        additional_stops: vec![],
                    }),
                    Err(e) => self.error(None, Some(id), ErrorCode::Backend, e.to_string()),
                }
            }
            Request::Prefill { req, prompt, sampling, stop, max_tokens, .. } if self.api == Api::LlamaServer => {
                let body = completion_body(&prompt, &sampling, &stop, max_tokens);
                self.spawn_stream(req, "/completion", body, prompt.len() as u32);
            }
            Request::Chat { req, mut body } if self.api == Api::OpenAiChat => {
                body["stream"] = serde_json::Value::Bool(true);
                self.spawn_stream(req, "/v1/chat/completions", body, 0);
            }
            Request::Decode { req, n } => match self.active.lock().unwrap().get(&req) {
                Some(a) => a.credit.add_permits(n as usize),
                None => self.error(Some(&req), None, ErrorCode::BadRequest, "unknown request"),
            },
            Request::Cancel { req } => {
                if let Some(a) = self.active.lock().unwrap().remove(&req) {
                    // Dropping the HTTP stream makes the engine cancel its task.
                    a.task.abort();
                    self.out.send(&Event::Finished { req, reason: FinishReason::Cancelled, n_prompt: 0, n_decoded: 0, tail: String::new() });
                }
            }
            Request::Stats { id } => {
                let stats = WorkerStats { active: self.active.lock().unwrap().len() as u32, rss_bytes: crate::rss_bytes(), ..Default::default() };
                self.out.send(&Event::Stats { id, stats });
            }
            other => {
                let level = if self.api == Api::LlamaServer { "token" } else { "chat" };
                self.reject(&other, ErrorCode::BadRequest, format!("{} is not available on a {level}-level engine", op_name(&other)));
            }
        }
    }

    async fn post(&self, path: &str, body: serde_json::Value) -> Result<serde_json::Value> {
        let url = format!("{}{path}", self.base.as_ref().unwrap());
        Ok(self.http.post(url).json(&body).send().await?.error_for_status()?.json().await?)
    }

    async fn launch(&mut self, plan: &Plan) -> Result<()> {
        if self.child.is_some() {
            bail!("engine already started; start a new worker");
        }
        let cfg = FluxConfig::load()?;
        let port = free_port()?;
        let (program, args, env, cwd, health, timeout) = match &plan.engine {
            EngineKind::LlamaServer => {
                let mut args = plan.backend_params().llama_server_args();
                args.extend(["--host", "127.0.0.1", "--port", &port.to_string(), "--jinja", "--no-webui"].map(String::from));
                (cfg.llama_bin("llama-server").display().to_string(), args, Default::default(), None, "/health".to_string(), 600)
            }
            EngineKind::External(name) => {
                let e = cfg.engines.get(name).with_context(|| format!("engine `{name}` is not in flux.toml"))?;
                let subst = |s: &str| {
                    s.replace("{port}", &port.to_string())
                        .replace("{model}", &plan.model_files[0].display().to_string())
                        .replace("{ctx}", &plan.workload.n_ctx_seq.to_string())
                };
                let mut cmd: Vec<String> = e.command.iter().chain(&e.args).map(|s| subst(s)).collect();
                let program = cmd.remove(0);
                (program, cmd, e.env.clone(), e.cwd.clone(), e.health_path.clone(), e.startup_timeout_s)
            }
            EngineKind::Native => bail!("the native engine runs in `flux-worker serve`"),
        };
        let stderr = std::io::stderr().as_fd().try_clone_to_owned()?;
        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&args).envs(&env).stdin(Stdio::null()).stdout(Stdio::from(stderr)).stderr(Stdio::inherit()).kill_on_drop(true);
        if let Some(d) = cwd {
            cmd.current_dir(d);
        }
        let mut child = cmd.spawn().with_context(|| format!("starting {program}"))?;
        let base = format!("http://127.0.0.1:{port}");
        let deadline = Instant::now() + Duration::from_secs(timeout);
        loop {
            if let Some(status) = child.try_wait()? {
                bail!("{program} exited with {status} during startup (see worker log)");
            }
            if let Ok(r) = self.http.get(format!("{base}{health}")).timeout(Duration::from_secs(2)).send().await {
                if r.status().is_success() {
                    break;
                }
            }
            if Instant::now() > deadline {
                let _ = child.kill().await;
                bail!("{program} did not become healthy within {timeout} s");
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        self.child = Some(child);
        self.base = Some(base);
        Ok(())
    }

    fn spawn_stream(&self, req: String, path: &str, body: serde_json::Value, n_prompt: u32) {
        if self.active.lock().unwrap().contains_key(&req) {
            return self.error(Some(&req), None, ErrorCode::BadRequest, "request id already active");
        }
        let credit = Arc::new(Semaphore::new(0));
        let (out, http, active, api, start) = (self.out.clone(), self.http.clone(), self.active.clone(), self.api, self.start);
        let url = format!("{}{path}", self.base.as_ref().unwrap());
        let sem = credit.clone();
        let r = req.clone();
        let task = tokio::spawn(async move {
            let result = stream(&out, &http, &url, &body, &r, n_prompt, &sem, api, start).await;
            if let Err(e) = result {
                out.send(&Event::Error { req: Some(r.clone()), id: None, code: ErrorCode::Backend, message: format!("{e:#}") });
                out.send(&Event::Finished { req: r.clone(), reason: FinishReason::Error, n_prompt, n_decoded: 0, tail: String::new() });
            }
            active.lock().unwrap().remove(&r);
        });
        self.active.lock().unwrap().insert(req, Active { credit, task });
    }
}

#[allow(clippy::too_many_arguments)]
async fn stream(
    out: &Out,
    http: &reqwest::Client,
    url: &str,
    body: &serde_json::Value,
    req: &str,
    n_prompt: u32,
    credit: &Semaphore,
    api: Api,
    start: Instant,
) -> Result<()> {
    let t0 = Instant::now();
    let resp = http.post(url).json(body).send().await?.error_for_status()?;
    let mut bytes = resp.bytes_stream();
    let mut buf: Vec<u8> = vec![];
    let (mut first, mut emitted, mut finish) = (true, 0u32, None::<(FinishReason, u32)>);
    while let Some(chunk) = bytes.next().await {
        buf.extend_from_slice(&chunk?);
        while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=nl).collect();
            let Some(data) = std::str::from_utf8(&line)?.trim().strip_prefix("data:").map(str::trim) else { continue };
            if data == "[DONE]" {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(data)?;
            if first {
                first = false;
                out.send(&Event::Prefilled { req: req.into(), n_prompt, ms: t0.elapsed().as_secs_f64() * 1e3 });
            }
            match api {
                Api::LlamaServer => {
                    let tokens: Vec<i32> = serde_json::from_value(v["tokens"].clone()).unwrap_or_default();
                    let content = v["content"].as_str().unwrap_or_default();
                    for (k, &tok) in tokens.iter().enumerate() {
                        take_credit(out, req, credit).await?;
                        let text = if k + 1 == tokens.len() { content.to_string() } else { String::new() };
                        let t_us = start.elapsed().as_micros() as u64;
                        out.send(&Event::Token { req: req.into(), i: emitted, token: tok, text, t_us, alt: None });
                        emitted += 1;
                    }
                    if v["stop"].as_bool() == Some(true) {
                        let reason = match v["stop_type"].as_str() {
                            Some("eos") => FinishReason::Eog,
                            Some("word") => FinishReason::Stop,
                            Some("limit") => FinishReason::Length,
                            _ => FinishReason::Error,
                        };
                        finish = Some((reason, v["tokens_evaluated"].as_u64().unwrap_or(n_prompt as u64) as u32));
                    }
                }
                Api::OpenAiChat => {
                    take_credit(out, req, credit).await?;
                    let t_us = start.elapsed().as_micros() as u64;
                    if let Some(fr) = v["choices"][0]["finish_reason"].as_str() {
                        let reason = if fr == "length" { FinishReason::Length } else { FinishReason::Eog };
                        finish = Some((reason, v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32));
                    }
                    out.send(&Event::ChatChunk { req: req.into(), chunk: v, t_us });
                    emitted += 1;
                }
            }
        }
    }
    let (reason, n_prompt) = finish.context("engine stream ended without a finish marker")?;
    out.send(&Event::Finished { req: req.into(), reason, n_prompt, n_decoded: emitted, tail: String::new() });
    Ok(())
}

/// Waits for one unit of credit, announcing the pause when none is left.
async fn take_credit(out: &Out, req: &str, credit: &Semaphore) -> Result<()> {
    match credit.try_acquire() {
        Ok(p) => p.forget(),
        Err(_) => {
            out.send(&Event::Paused { req: req.into() });
            credit.acquire().await?.forget();
        }
    }
    Ok(())
}

fn completion_body(prompt: &[i32], s: &Sampling, stop: &[String], max_tokens: u32) -> serde_json::Value {
    let mut body = serde_json::to_value(s).unwrap();
    let o = body.as_object_mut().unwrap();
    o.insert("prompt".into(), serde_json::json!(prompt));
    o.insert("n_predict".into(), max_tokens.into());
    o.insert("stop".into(), serde_json::json!(stop));
    o.insert("stream".into(), true.into());
    o.insert("return_tokens".into(), true.into());
    o.insert("cache_prompt".into(), false.into());
    body
}

fn free_port() -> Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

fn op_name(r: &Request) -> &'static str {
    match r {
        Request::Tokenize { .. } => "tokenize",
        Request::ApplyTemplate { .. } => "apply_template",
        Request::Prefill { .. } => "prefill",
        Request::Chat { .. } => "chat",
        Request::Trace { .. } => "trace",
        _ => "this request",
    }
}
