//! One request against the worker: credit-based streaming, journaling, cancellation when the
//! client goes away, and replay of committed tokens after a worker failure.

use crate::journal::Status;
use crate::AppState;
use flux_core::protocol::{Event, FinishReason, Sampling};
use flux_core::worker::ChatOptions;
use serde_json::Value;
use tokio::sync::mpsc;

/// Credit granted per top-up: generation runs at most this far ahead of the client.
const CREDIT: u32 = 16;
const MAX_RESTARTS: u32 = 2;

pub enum Piece {
    /// `deltas` are the OpenAI chat deltas of a parsed request; `text` is the raw reply text either way.
    Text {
        token: i32,
        text: String,
        deltas: Vec<Value>,
    },
    Chunk(Value),
}

#[derive(Debug, Clone)]
pub struct Outcome {
    pub reason: Option<FinishReason>,
    pub n_prompt: u32,
    pub n_completion: u32,
    pub error: Option<String>,
    /// R = (N-1)/(t_last - t_first) over this request's tokens, comparable to the plan's validated rate.
    pub decode_tps: Option<f64>,
    /// The parsed assistant message (reasoning, content, tool calls) of a parsed chat request.
    pub message: Option<Value>,
}

pub struct TokenJob {
    pub id: String,
    pub prompt: Vec<i32>,
    pub sampling: Sampling,
    pub stop: Vec<String>,
    pub max_tokens: u32,
    pub render_special: Vec<i32>,
    pub chat: ChatOptions,
}

enum Ended {
    Finished(Outcome),
    WorkerLost,
    ClientGone,
}

/// Runs a token-level job, sending text to `tx`; restarts the worker and replays on failure.
pub async fn run_tokens(st: &AppState, job: TokenJob, tx: mpsc::Sender<Piece>) -> Outcome {
    let mut times: Vec<std::time::Instant> = vec![];
    let mut o = run_tokens_inner(st, job, tx, &mut times).await;
    if times.len() > 1 && o.error.is_none() {
        o.decode_tps = Some((times.len() - 1) as f64 / (times[times.len() - 1] - times[0]).as_secs_f64().max(1e-9));
    }
    o
}

async fn run_tokens_inner(st: &AppState, job: TokenJob, tx: mpsc::Sender<Piece>, times: &mut Vec<std::time::Instant>) -> Outcome {
    let mut restarts = 0;
    loop {
        let (gen, w) = st.supervisor.current().await;
        let (committed, delivered) = st.journal.get(&job.id).map(|e| (e.tokens, e.texts.concat())).unwrap_or_default();
        let mut prompt = job.prompt.clone();
        prompt.extend(&committed);
        let remaining = job.max_tokens.saturating_sub(committed.len() as u32);
        let wreq = format!("{}#{restarts}", job.id);
        let chat = ChatOptions { prefix: delivered, ..job.chat.clone() };
        let ended = match w.start(&wreq, prompt, job.sampling.clone(), job.stop.clone(), remaining, job.render_special.clone(), chat).await {
            Err(_) => Ended::WorkerLost,
            Ok(mut rx) => {
                let _ = w.credit(&wreq, CREDIT).await;
                let mut outstanding = CREDIT;
                let mut error = None;
                loop {
                    let ev = tokio::select! {
                        ev = rx.recv() => ev,
                        _ = tx.closed() => {
                            let _ = w.cancel(&wreq).await;
                            break Ended::ClientGone;
                        }
                    };
                    match ev {
                        Some(Event::Prefilling { done, reused, ms, .. }) => st.live.prompt(&job.id, done, reused, ms),
                        Some(Event::Prefilled { n_prompt, reused, ms, .. }) => st.live.prompt(&job.id, n_prompt, reused, ms),
                        Some(Event::Token { token, text, deltas, .. }) => {
                            times.push(std::time::Instant::now());
                            st.live.token(&job.id);
                            st.journal.push(&job.id, token, &text);
                            if tx.send(Piece::Text { token, text, deltas }).await.is_err() {
                                let _ = w.cancel(&wreq).await;
                                break Ended::ClientGone;
                            }
                            outstanding -= 1;
                            if outstanding <= CREDIT / 2 {
                                let _ = w.credit(&wreq, CREDIT).await;
                                outstanding += CREDIT;
                            }
                        }
                        Some(Event::Error { message, .. }) if message == "worker process exited" => {}
                        Some(Event::Error { message, .. }) => error = Some(message),
                        Some(Event::Finished { reason: FinishReason::Error, .. }) if error.is_none() => break Ended::WorkerLost,
                        Some(Event::Finished { reason, n_decoded, tail, deltas, message, .. }) => {
                            if !tail.is_empty() || !deltas.is_empty() {
                                let _ = tx.send(Piece::Text { token: -1, text: tail, deltas }).await;
                            }
                            break Ended::Finished(Outcome {
                                reason: Some(reason),
                                n_prompt: job.prompt.len() as u32,
                                n_completion: committed.len() as u32 + n_decoded,
                                error,
                                decode_tps: None,
                                message,
                            });
                        }
                        Some(_) => {}
                        None => break Ended::WorkerLost,
                    }
                }
            }
        };
        match ended {
            Ended::Finished(o) => {
                let status = match (&o.error, o.reason) {
                    (Some(e), _) => Status::Failed { message: e.clone() },
                    (None, r) => Status::Done { finish_reason: finish_name(r) },
                };
                st.journal.finish(&job.id, status);
                return o;
            }
            Ended::ClientGone => {
                st.journal.finish(&job.id, Status::Done { finish_reason: "cancelled".into() });
                return Outcome {
                    reason: Some(FinishReason::Cancelled),
                    n_prompt: job.prompt.len() as u32,
                    n_completion: 0,
                    error: None,
                    decode_tps: None,
                    message: None,
                };
            }
            Ended::WorkerLost if restarts < MAX_RESTARTS => {
                restarts += 1;
                tracing::warn!(request = %job.id, "worker lost mid-request; restarting and replaying committed tokens");
                if let Err(e) = st.supervisor.restart_if(gen).await {
                    let message = format!("worker restart failed: {e:#}");
                    st.journal.finish(&job.id, Status::Failed { message: message.clone() });
                    return Outcome {
                        reason: Some(FinishReason::Error),
                        n_prompt: job.prompt.len() as u32,
                        n_completion: 0,
                        error: Some(message),
                        decode_tps: None,
                        message: None,
                    };
                }
            }
            Ended::WorkerLost => {
                let message = "worker failed repeatedly; request aborted".to_string();
                st.journal.finish(&job.id, Status::Failed { message: message.clone() });
                return Outcome {
                    reason: Some(FinishReason::Error),
                    n_prompt: job.prompt.len() as u32,
                    n_completion: 0,
                    error: Some(message),
                    decode_tps: None,
                    message: None,
                };
            }
        }
    }
}

/// Relays a chat-level engine's stream; such engines are restarted but not replayed.
pub async fn run_chat(st: &AppState, id: &str, body: serde_json::Value, tx: mpsc::Sender<Piece>) -> Outcome {
    let (gen, w) = st.supervisor.current().await;
    let fail = |m: String| Outcome { reason: Some(FinishReason::Error), n_prompt: 0, n_completion: 0, error: Some(m), decode_tps: None, message: None };
    let mut rx = match w.chat(id, body).await {
        Ok(rx) => rx,
        Err(e) => return fail(e.to_string()),
    };
    let _ = w.credit(id, CREDIT).await;
    let mut outstanding = CREDIT;
    loop {
        let ev = tokio::select! {
            ev = rx.recv() => ev,
            _ = tx.closed() => {
                let _ = w.cancel(id).await;
                return Outcome { reason: Some(FinishReason::Cancelled), n_prompt: 0, n_completion: 0, error: None, decode_tps: None, message: None };
            }
        };
        match ev {
            Some(Event::ChatChunk { chunk, .. }) => {
                st.live.token(id);
                if tx.send(Piece::Chunk(chunk)).await.is_err() {
                    let _ = w.cancel(id).await;
                    return Outcome { reason: Some(FinishReason::Cancelled), n_prompt: 0, n_completion: 0, error: None, decode_tps: None, message: None };
                }
                outstanding -= 1;
                if outstanding <= CREDIT / 2 {
                    let _ = w.credit(id, CREDIT).await;
                    outstanding += CREDIT;
                }
            }
            Some(Event::Finished { reason, n_prompt, n_decoded, .. }) => {
                return Outcome { reason: Some(reason), n_prompt, n_completion: n_decoded, error: None, decode_tps: None, message: None };
            }
            Some(Event::Error { message, .. }) => {
                if message == "worker process exited" {
                    let _ = st.supervisor.restart_if(gen).await;
                }
                return fail(message);
            }
            Some(_) => {}
            None => return fail("worker stream closed".into()),
        }
    }
}

pub fn finish_name(r: Option<FinishReason>) -> String {
    match r {
        Some(FinishReason::Length) => "length",
        Some(FinishReason::Cancelled) => "cancelled",
        Some(FinishReason::Error) => "error",
        _ => "stop",
    }
    .into()
}
