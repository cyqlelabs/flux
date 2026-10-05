//! Soak with fault injection against `flux serve`: overload, client cancellations, worker kills and
//! a bounded low-memory episode. Passing means no corruption, no hang, no crash of the server.

use crate::servers::{start, Contestant, Kind};
use anyhow::Result;
use flux_core::config::FluxConfig;
use futures::StreamExt;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct SoakOptions {
    pub duration: Duration,
    /// Client loops; more than the plan's concurrency exercises the queue and its bound.
    pub clients: usize,
    pub cancel_fraction: f64,
    pub kill_every: Option<Duration>,
    /// Periodically allocate a 2 GiB ballast with the admission threshold set just below free memory.
    pub pressure_every: Option<Duration>,
    pub request_timeout: Duration,
    pub seed: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SoakReport {
    pub seconds: f64,
    pub requests: u64,
    pub completed: u64,
    pub cancelled: u64,
    /// Explicit partial-reply failures verified against the journal during an injected kill.
    #[serde(default)]
    pub interrupted_by_fault: u64,
    pub rejected_queue_full: u64,
    pub rejected_closed: u64,
    /// Refusals for memory pressure while no ballast was held: the gate closing on its own.
    pub rejected_closed_outside_episodes: u64,
    pub errors: BTreeMap<String, u64>,
    pub worker_kills: u64,
    pub pressure_episodes: u64,
    pub admission_closed_seen: bool,
    pub hung: u64,
    /// Streams whose delivered tokens disagree with the journal.
    pub corrupted: u64,
    pub max_latency_s: f64,
    pub server_alive: bool,
    pub pass: bool,
    pub reasons: Vec<String>,
}

/// A direct child named `name`, whichever of the process's threads spawned it.
fn child_named(pid: u32, name: &str) -> Option<u32> {
    std::fs::read_dir(format!("/proc/{pid}/task")).ok()?.flatten().find_map(|t| {
        let kids = std::fs::read_to_string(t.path().join("children")).ok()?;
        kids.split_whitespace()
            .filter_map(|k| k.parse::<u32>().ok())
            .find(|k| std::fs::read_to_string(format!("/proc/{k}/comm")).is_ok_and(|c| c.trim() == name))
    })
}

fn mem_available_mib() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:"))?.split_whitespace().nth(1)?.parse::<u64>().ok())
        .unwrap_or(0)
        >> 10
}

enum Outcome {
    Completed {
        tokens: Vec<i32>,
        text: String,
        finish: String,
        id: String,
    },
    Cancelled,
    Interrupted {
        tokens: Vec<i32>,
        text: String,
        id: String,
    },
    /// A refusal, with the server's Retry-After in seconds.
    Status(u16, Option<u64>),
    Error(String),
}

async fn one_request(http: &reqwest::Client, base: &str, id: &str, prompt: &[i32], max_tokens: u32, cancel_after: Option<usize>) -> Outcome {
    let body = json!({"prompt": prompt, "max_tokens": max_tokens, "stream": true, "temperature": 0.7, "seed": 7});
    let resp = match http.post(format!("{base}/v1/completions")).header("x-request-id", id).json(&body).send().await {
        Ok(r) => r,
        Err(e) => return Outcome::Error(format!("connect: {e}")),
    };
    if !resp.status().is_success() {
        let retry = resp.headers().get("retry-after").and_then(|v| v.to_str().ok()?.parse().ok());
        return Outcome::Status(resp.status().as_u16(), retry);
    }
    let mut bytes = resp.bytes_stream();
    let mut buf = vec![];
    let mut streamed = 0usize;
    let mut tokens = vec![];
    let mut text = String::new();
    let mut finish = None;
    while let Some(c) = bytes.next().await {
        let Ok(c) = c else { return Outcome::Error("stream interrupted".into()) };
        buf.extend_from_slice(&c);
        while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=nl).collect();
            let Some(d) = std::str::from_utf8(&line).unwrap_or("").trim().strip_prefix("data:").map(str::trim) else {
                continue;
            };
            if d == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(d) else {
                return Outcome::Error("bad chunk".into());
            };
            if let Some(e) = v.get("error") {
                if e.to_string().contains("worker interrupted a partial reply") {
                    return Outcome::Interrupted { tokens, text, id: id.into() };
                }
                return Outcome::Error(e.to_string());
            }
            if let Some(reason) = v["choices"][0]["finish_reason"].as_str() {
                finish = Some(reason.to_string());
            }
            if let Some(t) = v["flux_token"].as_i64() {
                tokens.push(t as i32);
            }
            if let Some(t) = v["choices"][0]["text"].as_str() {
                text.push_str(t);
            }
            if v["choices"][0]["finish_reason"].is_null() {
                streamed += 1;
                if cancel_after.is_some_and(|k| streamed >= k) {
                    return Outcome::Cancelled;
                }
            }
        }
    }
    let Some(finish) = finish else { return Outcome::Error("stream ended without finish marker".into()) };
    if finish != "length" && finish != "stop" {
        return Outcome::Error(format!("unexpected finish: {finish}"));
    }
    Outcome::Completed { tokens, text, finish, id: id.into() }
}

pub async fn soak(
    cfg: &FluxConfig,
    plan_id: &str,
    prompts: Vec<Vec<i32>>,
    opts: &SoakOptions,
    logdir: &Path,
    log: &(dyn Fn(&str) + Sync),
) -> Result<SoakReport> {
    let contestant = Contestant { label: "flux".into(), kind: Kind::Flux { plan: plan_id.into() } };
    let server = start(cfg, &contestant, Path::new(""), 0, &logdir.join("soak-server.log")).await?;
    let http = reqwest::Client::new();
    let report = Arc::new(Mutex::new(SoakReport::default()));
    let seq = Arc::new(AtomicU64::new(0));
    let t0 = Instant::now();
    let deadline = t0 + opts.duration;
    let prompts = Arc::new(prompts);
    let in_episode = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let fault_epoch = Arc::new(AtomicU64::new(0));
    let recovering = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let mut tasks = vec![];
    for c in 0..opts.clients {
        let (http, base, report, seq, prompts, opts, in_episode) =
            (http.clone(), server.base.clone(), report.clone(), seq.clone(), prompts.clone(), opts.clone(), in_episode.clone());
        let (fault_epoch, recovering) = (fault_epoch.clone(), recovering.clone());
        tasks.push(tokio::spawn(async move {
            let mut rng = rand::rngs::StdRng::seed_from_u64(opts.seed + c as u64);
            while Instant::now() < deadline {
                let n = seq.fetch_add(1, Ordering::Relaxed);
                let id = format!("soak-{n}");
                let prompt = &prompts[rng.random_range(0..prompts.len())];
                let max_tokens = rng.random_range(16..=256);
                let cancel = (rng.random::<f64>() < opts.cancel_fraction).then(|| rng.random_range(1..=max_tokens as usize));
                let started = Instant::now();
                let epoch = fault_epoch.load(Ordering::Acquire);
                let began_recovering = recovering.load(Ordering::Acquire);
                let out = tokio::time::timeout(opts.request_timeout, one_request(&http, &base, &id, prompt, max_tokens, cancel)).await;
                let during_fault = began_recovering || recovering.load(Ordering::Acquire) || epoch != fault_epoch.load(Ordering::Acquire);
                let latency = started.elapsed().as_secs_f64();
                let verify = match &out {
                    Ok(Outcome::Completed { tokens, text, finish, id }) => {
                        let j: Option<serde_json::Value> =
                            async { http.get(format!("{base}/flux/requests/{id}")).timeout(opts.request_timeout).send().await.ok()?.json().await.ok() }.await;
                        Some(j.is_none_or(|j| {
                            j["tokens"] != json!(tokens) || j["text"] != *text || j["status"]["state"] != "done" || j["status"]["finish_reason"] != *finish
                        }))
                    }
                    Ok(Outcome::Interrupted { tokens, text, id }) => {
                        let j: Option<serde_json::Value> =
                            async { http.get(format!("{base}/flux/requests/{id}")).timeout(opts.request_timeout).send().await.ok()?.json().await.ok() }.await;
                        Some(j.is_none_or(|j| {
                            let recorded: Vec<i32> = serde_json::from_value(j["tokens"].clone()).unwrap_or_default();
                            !recorded.starts_with(tokens) || !j["text"].as_str().unwrap_or_default().starts_with(text) || j["status"]["state"] != "failed"
                        }))
                    }
                    _ => None,
                };
                {
                    let mut r = report.lock().unwrap();
                    r.requests += 1;
                    r.max_latency_s = r.max_latency_s.max(latency);
                    match &out {
                        Err(_) => r.hung += 1,
                        Ok(Outcome::Completed { .. }) => r.completed += 1,
                        Ok(Outcome::Cancelled) => r.cancelled += 1,
                        Ok(Outcome::Interrupted { .. }) if during_fault => r.interrupted_by_fault += 1,
                        Ok(Outcome::Interrupted { .. }) => *r.errors.entry("worker interrupted outside injected fault".into()).or_default() += 1,
                        Ok(Outcome::Status(429, _)) => r.rejected_queue_full += 1,
                        Ok(Outcome::Status(503, _)) if in_episode.load(Ordering::Relaxed) || during_fault => r.rejected_closed += 1,
                        Ok(Outcome::Status(503, _)) => r.rejected_closed_outside_episodes += 1,
                        Ok(Outcome::Status(s, _)) => *r.errors.entry(format!("HTTP {s}")).or_default() += 1,
                        Ok(Outcome::Error(e)) => *r.errors.entry(e.chars().take(80).collect()).or_default() += 1,
                    }
                    if verify == Some(true) {
                        r.corrupted += 1;
                    }
                }
                if let Ok(Outcome::Status(_, Some(secs))) = out {
                    tokio::time::sleep(Duration::from_secs(secs)).await;
                } else if matches!(rng.random_range(0..10), 0) {
                    tokio::time::sleep(Duration::from_millis(rng.random_range(10..500))).await;
                }
            }
        }));
    }

    let mut next_kill = opts.kill_every.map(|d| t0 + d);
    let mut next_pressure = opts.pressure_every.map(|d| t0 + d);
    while Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if next_kill.is_some_and(|t| Instant::now() >= t) {
            if let Some(w) = child_named(server.pid, "flux-worker") {
                log(&format!("fault: killing worker {w}"));
                recovering.store(true, Ordering::Release);
                fault_epoch.fetch_add(1, Ordering::AcqRel);
                unsafe {
                    libc::kill(w as i32, libc::SIGKILL);
                }
                report.lock().unwrap().worker_kills += 1;
            }
            next_kill = opts.kill_every.map(|d| Instant::now() + d);
        }
        if recovering.load(Ordering::Acquire) {
            if let Ok(r) = http.get(format!("{}/health", server.base)).timeout(Duration::from_secs(2)).send().await {
                if r.status().is_success() {
                    recovering.store(false, Ordering::Release);
                }
            }
        }
        if next_pressure.is_some_and(|t| Instant::now() >= t) {
            // Threshold 1 GiB below what is free now, then a 2 GiB allocation crosses it.
            let threshold = mem_available_mib().saturating_sub(1024);
            let set = |m: u64| http.post(format!("{}/flux/admission", server.base)).json(&json!({"min_available_mib": m})).send();
            let _ = set(threshold).await;
            log(&format!("fault: admission threshold {threshold} MiB, allocating a 2 GiB ballast"));
            report.lock().unwrap().pressure_episodes += 1;
            in_episode.store(true, Ordering::Relaxed);
            let mut ballast: Vec<u8> = vec![0u8; 2 << 30];
            ballast.iter_mut().step_by(4096).for_each(|b| *b = 1);
            for _ in 0..8 {
                tokio::time::sleep(Duration::from_millis(500)).await;
                if let Ok(r) = http.get(format!("{}/health", server.base)).send().await {
                    if r.status().as_u16() == 503 {
                        report.lock().unwrap().admission_closed_seen = true;
                    }
                }
            }
            drop(std::hint::black_box(ballast));
            let _ = set(cfg.serve.min_available_mib).await;
            // Let the monitor observe recovery before refusals count against the server again.
            tokio::time::sleep(Duration::from_secs(3)).await;
            in_episode.store(false, Ordering::Relaxed);
            next_pressure = opts.pressure_every.map(|d| Instant::now() + d);
        }
    }
    for t in tasks {
        let _ = tokio::time::timeout(opts.request_timeout, t).await;
    }
    let alive = http.get(format!("{}/live", server.base)).timeout(opts.request_timeout).send().await.is_ok_and(|r| r.status().is_success());
    let ready = http.get(format!("{}/health", server.base)).timeout(opts.request_timeout).send().await.is_ok_and(|r| r.status().is_success());
    server.stop().await;

    let mut r = report.lock().unwrap().clone();
    r.seconds = t0.elapsed().as_secs_f64();
    r.server_alive = alive;
    if !alive {
        r.reasons.push("server died".into());
    }
    if !ready {
        r.reasons.push("server did not recover readiness".into());
    }
    if r.hung > 0 {
        r.reasons.push(format!("{} request(s) exceeded {:?}", r.hung, opts.request_timeout));
    }
    if r.corrupted > 0 {
        r.reasons.push(format!("{} stream(s) disagree with the journal", r.corrupted));
    }
    if r.rejected_closed_outside_episodes > 0 {
        r.reasons.push(format!("{} request(s) refused for memory pressure outside injected episodes", r.rejected_closed_outside_episodes));
    }
    if !r.errors.is_empty() {
        r.reasons.push(format!("unexpected errors: {:?}", r.errors));
    }
    if opts.pressure_every.is_some() && r.pressure_episodes > 0 && !r.admission_closed_seen {
        r.reasons.push("memory pressure never closed admission".into());
    }
    if opts.clients > 0 && r.completed == 0 {
        r.reasons.push("no request completed".into());
    }
    r.pass = r.reasons.is_empty();
    Ok(r)
}
