//! Mixed workload: Poisson arrivals with varied prompt and output lengths, plus one long-running chat
//! whose context grows every turn. Every contestant gets the identical, seeded schedule over the chat API.

use crate::servers::{start, Contestant};
use anyhow::Result;
use flux_core::config::FluxConfig;
use flux_core::corpus::{Corpus, Role};
use flux_core::stats::Summary;
use futures::StreamExt;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ArrivalOptions {
    pub duration: Duration,
    /// Mean arrivals per second.
    pub rate: f64,
    pub chat_turns: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arrival {
    pub at_s: f64,
    pub prompt_chars: usize,
    pub max_tokens: u32,
    pub corpus_index: usize,
}

/// The same schedule for every contestant: exponential gaps, log-uniform prompt sizes.
pub fn schedule(o: &ArrivalOptions) -> Vec<Arrival> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(o.seed);
    let mut t = 0.0;
    let mut out = vec![];
    while t < o.duration.as_secs_f64() {
        t += -rng.random::<f64>().max(1e-9).ln() / o.rate;
        out.push(Arrival {
            at_s: t,
            prompt_chars: (200.0 * 80f64.powf(rng.random::<f64>())) as usize,
            max_tokens: rng.random_range(32..=384),
            corpus_index: rng.random_range(0..500),
        });
    }
    out.retain(|a| a.at_s < o.duration.as_secs_f64());
    out
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestResult {
    pub ttft_s: Option<f64>,
    pub tokens: usize,
    pub total_s: f64,
    pub decode_tps: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrivalReport {
    pub contestant: String,
    pub requests: usize,
    pub failures: usize,
    pub tokens_per_s: f64,
    pub ttft_ms: Option<Summary>,
    pub decode_tps: Option<Summary>,
    pub chat_turn_ttft_ms: Vec<f64>,
    pub chat_turn_decode_tps: Vec<Option<f64>>,
}

async fn chat(http: &reqwest::Client, base: &str, messages: &Value, max_tokens: u32) -> (RequestResult, String) {
    let t0 = Instant::now();
    let mut r = RequestResult::default();
    let mut text = String::new();
    let mut times = vec![];
    let body = json!({"messages": messages, "max_tokens": max_tokens, "stream": true, "temperature": 0.7, "seed": 11});
    let out: Result<()> = async {
        let resp = http.post(format!("{base}/v1/chat/completions")).json(&body).send().await?.error_for_status()?;
        let mut bytes = resp.bytes_stream();
        let mut buf = vec![];
        while let Some(c) = bytes.next().await {
            buf.extend_from_slice(&c?);
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let Some(d) = std::str::from_utf8(&line)?.trim().strip_prefix("data:").map(str::trim) else { continue };
                if d == "[DONE]" {
                    continue;
                }
                let v: Value = serde_json::from_str(d)?;
                let delta = &v["choices"][0]["delta"];
                for k in ["content", "reasoning_content"] {
                    if let Some(s) = delta.get(k).and_then(Value::as_str) {
                        times.push(t0.elapsed().as_secs_f64());
                        if k == "content" {
                            text.push_str(s);
                        }
                    }
                }
            }
        }
        Ok(())
    }
    .await;
    r.error = out.err().map(|e| format!("{e:#}"));
    r.total_s = t0.elapsed().as_secs_f64();
    r.ttft_s = times.first().copied();
    r.tokens = times.len();
    r.decode_tps = flux_core::stats::interactive_rate(&times);
    (r, text)
}

pub async fn run(cfg: &FluxConfig, c: &Contestant, model: &Path, ctx_total: u32, corpus: &Corpus, o: &ArrivalOptions, logdir: &Path) -> Result<ArrivalReport> {
    let server = start(cfg, c, model, ctx_total, &logdir.join(format!("arrivals-{}.log", c.label.replace(|ch: char| !ch.is_alphanumeric(), "_")))).await?;
    let http = reqwest::Client::new();
    let base = server.base.clone();
    let t0 = Instant::now();
    let arrivals = schedule(o);
    let tasks = arrivals.iter().map(|a| {
        let (http, base) = (http.clone(), base.clone());
        let text: String = corpus.text(Role::HeldOut, a.corpus_index, a.prompt_chars).chars().take(a.prompt_chars).collect();
        let (at, max_tokens) = (a.at_s, a.max_tokens);
        async move {
            tokio::time::sleep_until(tokio::time::Instant::from_std(t0 + Duration::from_secs_f64(at))).await;
            let msgs = json!([{"role": "user", "content": format!("Summarize the following text.\n\n{text}")}]);
            chat(&http, &base, &msgs, max_tokens).await.0
        }
    });
    // The long-running chat runs alongside the arrivals, one turn after another.
    let chat_session = async {
        let mut messages = vec![json!({"role": "system", "content": "You are a helpful assistant. Answer in a few sentences."})];
        let mut turns = vec![];
        for i in 0..o.chat_turns {
            let q = corpus.text(Role::HeldOut, 600 + i, 300).chars().take(300).collect::<String>();
            messages.push(json!({"role": "user", "content": format!("Explain this passage briefly: {q}")}));
            let (r, answer) = chat(&http, &base, &Value::Array(messages.clone()), 160).await;
            messages.push(json!({"role": "assistant", "content": answer}));
            turns.push(r);
        }
        turns
    };
    let (results, turns) = tokio::join!(futures::future::join_all(tasks), chat_session);
    let wall = t0.elapsed().as_secs_f64();
    server.stop().await;
    let ok: Vec<&RequestResult> = results.iter().filter(|r| r.error.is_none()).collect();
    Ok(ArrivalReport {
        contestant: c.label.clone(),
        requests: results.len(),
        failures: results.len() - ok.len() + turns.iter().filter(|t| t.error.is_some()).count(),
        tokens_per_s: (results.iter().map(|r| r.tokens).sum::<usize>() + turns.iter().map(|t| t.tokens).sum::<usize>()) as f64 / wall,
        ttft_ms: Summary::of(&ok.iter().filter_map(|r| r.ttft_s).map(|t| t * 1e3).collect::<Vec<_>>()),
        decode_tps: Summary::of(&ok.iter().filter_map(|r| r.decode_tps).collect::<Vec<_>>()),
        chat_turn_ttft_ms: turns.iter().map(|t| t.ttft_s.unwrap_or(f64::NAN) * 1e3).collect(),
        chat_turn_decode_tps: turns.iter().map(|t| t.decode_tps).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_seeded_and_bounded() {
        let o = ArrivalOptions { duration: Duration::from_secs(60), rate: 0.5, chat_turns: 0, seed: 3 };
        let a = schedule(&o);
        assert_eq!(a.len(), schedule(&o).len());
        assert!(a.iter().all(|x| x.at_s < 60.0 && (32..=384).contains(&x.max_tokens)));
        assert!(a.len() > 10 && a.len() < 60, "{}", a.len());
    }
}
