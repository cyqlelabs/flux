//! Executes prompts against a loaded worker and measures what the client observes.

use anyhow::{Context, Result};
use flux_core::corpus::{Corpus, Role};
use flux_core::protocol::{Event, FinishReason, Sampling};
use flux_core::stats::{interactive_rate, Summary};
use flux_core::worker::Worker;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct StreamResult {
    pub n_prompt: usize,
    /// Request sent → first token received.
    pub ttft_s: f64,
    /// Arrival time of every token, seconds since the request was sent.
    pub token_times_s: Vec<f64>,
    pub tokens: Vec<i32>,
    /// Runner-up of each token's row, where the engine reports it.
    pub alts: Vec<Option<i32>>,
    pub finish: Option<FinishReason>,
    pub error: Option<String>,
    pub total_s: f64,
}

impl StreamResult {
    pub fn decode_rate(&self) -> Option<f64> {
        interactive_rate(&self.token_times_s)
    }
}

/// Greedy, fixed-length generation: identical work for every engine and placement.
pub fn bench_sampling() -> Sampling {
    Sampling { temperature: Some(0.0), ignore_eos: Some(true), seed: Some(1), ..Default::default() }
}

/// Exactly `n_tokens` prompt tokens from the corpus role, starting at article `index`.
pub async fn corpus_prompt(worker: &Worker, corpus: &Corpus, role: Role, index: usize, n_tokens: usize) -> Result<Vec<i32>> {
    let mut chars = n_tokens * 5;
    loop {
        let mut toks = worker.tokenize(&corpus.text(role, index, chars), true).await?;
        if toks.len() >= n_tokens {
            toks.truncate(n_tokens);
            return Ok(toks);
        }
        chars *= 2;
    }
}

/// A chat request of about `n_tokens` tokens in the model's own template: a corpus passage the user asks
/// the model to explain. Its reply is chat-shaped generation (reasoning, prose), which routes experts and
/// accepts drafts like interactive traffic rather than like a continuation of encyclopedia text.
pub async fn chat_prompt(worker: &Worker, corpus: &Corpus, role: Role, index: usize, n_tokens: usize) -> Result<Vec<i32>> {
    const ASK: &str = "Read the passage below. Explain its main points in your own words, note anything surprising, and say what questions it leaves open.\n\n";
    let passage: String = corpus.text(role, index, n_tokens * 4).chars().take(n_tokens * 4).collect();
    let messages = serde_json::json!([{"role": "user", "content": format!("{ASK}{passage}")}]);
    let templated = worker.apply_template(messages, None).await?;
    worker.tokenize(&templated.prompt, true).await
}

/// Coding-agent conversations (system prompt, tools, a task and usually one tool round-trip) whose replies
/// reason about code, call tools and write code: agent clients route experts unlike prose chat.
const AGENT_CALIBRATION: &str = include_str!("agent_calibration.json");

/// Agent conversation `index` of `of` to spread over, in the model's own template with its tools.
pub async fn agent_prompt(worker: &Worker, index: usize, of: usize) -> Result<Vec<i32>> {
    let set: serde_json::Value = serde_json::from_str(AGENT_CALIBRATION)?;
    let convs = set["conversations"].as_array().map(Vec::as_slice).unwrap_or_default();
    let c = &convs[index * convs.len() / of.max(1) % convs.len()];
    let mut messages = vec![serde_json::json!({"role": "system", "content": set["system"]}), serde_json::json!({"role": "user", "content": c["task"]})];
    if !c["call"].is_null() {
        let call =
            serde_json::json!({"id": "call_0", "type": "function", "function": {"name": c["call"]["name"], "arguments": c["call"]["arguments"].to_string()}});
        messages.push(serde_json::json!({"role": "assistant", "content": "", "tool_calls": [call]}));
        messages.push(serde_json::json!({"role": "tool", "tool_call_id": "call_0", "content": c["result"]}));
    }
    let templated = worker.apply_template(serde_json::Value::Array(messages), Some(set["tools"].clone())).await?;
    worker.tokenize(&templated.prompt, true).await
}

/// Runs one request to completion, granting all credit up front.
pub async fn run_stream(worker: &Worker, req: &str, prompt: Vec<i32>, max_tokens: u32, sampling: Sampling) -> Result<StreamResult> {
    let n_prompt = prompt.len();
    let t0 = Instant::now();
    let mut rx = worker.start(req, prompt, sampling, vec![], max_tokens, vec![], Default::default()).await?;
    worker.credit(req, max_tokens).await?;
    let mut r = StreamResult { n_prompt, ttft_s: 0.0, token_times_s: vec![], tokens: vec![], alts: vec![], finish: None, error: None, total_s: 0.0 };
    while let Some(ev) = tokio::time::timeout(std::time::Duration::from_secs(300), rx.recv()).await.context("measurement made no progress for 300 seconds")? {
        match ev {
            Event::Token { .. } | Event::ChatChunk { .. } => {
                if let Event::Token { token, alt, .. } = ev {
                    r.tokens.push(token);
                    r.alts.push(alt);
                }
                let t = t0.elapsed().as_secs_f64();
                if r.token_times_s.is_empty() {
                    r.ttft_s = t;
                }
                r.token_times_s.push(t);
            }
            Event::Error { message, .. } => r.error = Some(message),
            Event::Finished { reason, n_decoded, .. } => {
                if n_decoded as usize != r.tokens.len() {
                    r.error = Some(format!("completion usage {n_decoded} differs from {} emitted tokens", r.tokens.len()));
                }
                r.finish = Some(reason);
                break;
            }
            _ => {}
        }
    }
    if r.error.is_none() && (r.finish != Some(FinishReason::Length) || r.tokens.len() != max_tokens as usize) {
        r.error = Some(format!("incomplete measurement: {:?}, {} of {max_tokens} tokens", r.finish, r.tokens.len()));
    }
    r.total_s = t0.elapsed().as_secs_f64();
    Ok(r)
}

/// Runs `prompts` with up to `concurrency` streams in flight; results keep prompt order.
pub async fn run_all(worker: &Worker, tag: &str, prompts: Vec<Vec<i32>>, max_tokens: u32, concurrency: usize, sampling: Sampling) -> Result<Vec<StreamResult>> {
    let mut out: Vec<Option<StreamResult>> = vec![None; prompts.len()];
    let mut pending = prompts.into_iter().enumerate();
    let mut inflight: Vec<Inflight> = vec![];
    loop {
        while inflight.len() < concurrency.max(1) {
            let Some((i, p)) = pending.next() else { break };
            let req = format!("{tag}-{i}");
            let s = sampling.clone();
            inflight.push(Box::pin(async move { (i, run_stream(worker, &req, p, max_tokens, s).await) }));
        }
        if inflight.is_empty() {
            break;
        }
        let ((i, r), _, rest) = futures::future::select_all(inflight).await;
        inflight = rest;
        out[i] = Some(r?);
    }
    Ok(out.into_iter().map(|r| r.expect("every prompt ran")).collect())
}

/// Chat engines must expose per-token logprobs and completion usage to participate in token timing.
pub async fn run_chats(worker: &Worker, prompts: Vec<String>, max_tokens: u32, concurrency: usize) -> Result<Vec<StreamResult>> {
    use futures::{stream, StreamExt};
    stream::iter(prompts.into_iter().enumerate().map(|(i, text)| async move {
        let req = format!("chat-cal-{i}");
        let t0 = Instant::now();
        let body = serde_json::json!({"messages":[{"role":"user","content":text}],"stream":true,
            "stream_options":{"include_usage":true},"logprobs":true,"max_tokens":max_tokens,"temperature":0,"ignore_eos":true});
        let mut rx = worker.chat(&req, body).await?;
        worker.credit(&req, 16).await?;
        let mut r = StreamResult { n_prompt: 0, ttft_s: 0.0, token_times_s: vec![], tokens: vec![], alts: vec![], finish: None, error: None, total_s: 0.0 };
        while let Some(ev) = tokio::time::timeout(std::time::Duration::from_secs(300), rx.recv()).await.context("chat measurement stalled")? {
            match ev {
                Event::ChatChunk { chunk, .. } => {
                    let count = chunk["choices"][0]["logprobs"]["content"].as_array().map_or(0, Vec::len);
                    let t = t0.elapsed().as_secs_f64();
                    if count > 0 && r.token_times_s.is_empty() {
                        r.ttft_s = t;
                    }
                    r.token_times_s.extend(std::iter::repeat_n(t, count));
                    worker.credit(&req, 1).await?;
                }
                Event::Finished { reason, n_prompt, n_decoded, .. } => {
                    r.finish = Some(reason);
                    r.n_prompt = n_prompt as usize;
                    if reason != FinishReason::Length || n_decoded != max_tokens || r.token_times_s.len() != max_tokens as usize {
                        r.error = Some("chat measurement needs complete fixed-length output, per-token logprobs, and matching usage".into());
                    }
                    break;
                }
                Event::Error { message, .. } => r.error = Some(message),
                _ => {}
            }
        }
        if r.finish.is_none() {
            r.error = Some("chat measurement ended without finish".into());
        }
        r.total_s = t0.elapsed().as_secs_f64();
        Ok(r)
    }))
    .buffered(concurrency.max(1))
    .collect::<Vec<Result<StreamResult>>>()
    .await
    .into_iter()
    .collect()
}

type Inflight<'a> = std::pin::Pin<Box<dyn std::future::Future<Output = (usize, Result<StreamResult>)> + Send + 'a>>;

/// Aggregates of a set of streams.
#[derive(Debug, Clone)]
pub struct RunSummary {
    pub ttft_ms: Summary,
    pub decode_tps: Summary,
    pub token_ms: Summary,
    /// Total emitted tokens over wall time of the whole set.
    pub aggregate_tps: f64,
    pub failures: usize,
    /// Ids of every emitted token.
    pub tokens: Vec<i32>,
}

pub fn summarize(results: &[StreamResult], wall_s: f64) -> Option<RunSummary> {
    let ok: Vec<&StreamResult> = results.iter().filter(|r| r.error.is_none() && r.finish == Some(FinishReason::Length) && r.token_times_s.len() > 1).collect();
    let gaps: Vec<f64> = ok.iter().flat_map(|r| r.token_times_s.windows(2).map(|w| (w[1] - w[0]) * 1e3)).collect();
    Some(RunSummary {
        ttft_ms: Summary::of(&ok.iter().map(|r| r.ttft_s * 1e3).collect::<Vec<_>>())?,
        decode_tps: Summary::of(&ok.iter().filter_map(|r| r.decode_rate()).collect::<Vec<_>>())?,
        token_ms: Summary::of(&gaps)?,
        aggregate_tps: ok.iter().map(|r| r.token_times_s.len()).sum::<usize>() as f64 / wall_s,
        failures: results.len() - ok.len(),
        tokens: ok.iter().flat_map(|r| r.tokens.iter().copied()).collect(),
    })
}
