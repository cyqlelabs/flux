//! Executes prompts against a loaded worker and measures what the client observes.

use anyhow::Result;
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

/// Runs one request to completion, granting all credit up front.
pub async fn run_stream(worker: &Worker, req: &str, prompt: Vec<i32>, max_tokens: u32, sampling: Sampling) -> Result<StreamResult> {
    let n_prompt = prompt.len();
    let t0 = Instant::now();
    let mut rx = worker.start(req, prompt, sampling, vec![], max_tokens, vec![], Default::default()).await?;
    worker.credit(req, max_tokens).await?;
    let mut r = StreamResult { n_prompt, ttft_s: 0.0, token_times_s: vec![], tokens: vec![], alts: vec![], finish: None, error: None, total_s: 0.0 };
    while let Some(ev) = rx.recv().await {
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
            Event::Finished { reason, .. } => {
                r.finish = Some(reason);
                break;
            }
            _ => {}
        }
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
    let ok: Vec<&StreamResult> = results.iter().filter(|r| r.error.is_none() && r.token_times_s.len() > 1).collect();
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
