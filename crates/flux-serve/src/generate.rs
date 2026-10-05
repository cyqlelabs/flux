//! Bounded streaming and request lifecycle. Partial replies fail explicitly on worker loss:
//! token IDs cannot restore detokenization, parser IDs, or sampler state.
use crate::{journal::Status, AppState};
use flux_core::protocol::{ErrorCode, Event, FinishReason, Request, Sampling};
use flux_core::worker::ChatOptions;
use serde_json::{json, Value};
use std::{collections::BTreeMap, time::Duration};
use tokio::sync::{mpsc, oneshot};

const CREDIT: u32 = 16;
const MAX_RESTARTS: u32 = 2;

pub enum Piece {
    Text { token: i32, text: String, deltas: Vec<Value> },
    Chunk(Value),
}

#[derive(Debug, Clone, Default)]
pub struct Outcome {
    pub reason: Option<FinishReason>,
    pub n_prompt: u32,
    pub n_completion: u32,
    pub error: Option<String>,
    pub error_code: Option<ErrorCode>,
    pub decode_tps: Option<f64>,
    pub message: Option<Value>,
    pub finish_reason: Option<String>,
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

fn failure(message: impl Into<String>) -> Outcome {
    Outcome { reason: Some(FinishReason::Error), error: Some(message.into()), ..Default::default() }
}

fn finish(st: &AppState, id: &str, o: &mut Outcome) {
    if !st.journal.terminal(
        id,
        json!({"finish_reason":crate::openai::finish_reason(o),"message":o.message,
        "usage":{"prompt_tokens":o.n_prompt,"completion_tokens":o.n_completion},"error":o.error}),
    ) && o.error.is_none()
    {
        o.error = Some("request journal capacity exceeded at completion".into());
        o.reason = Some(FinishReason::Error);
    }
    let status = match &o.error {
        Some(message) => Status::Failed { message: message.clone() },
        None => Status::Done { finish_reason: crate::openai::finish_reason(o) },
    };
    st.journal.finish(id, status);
}

async fn deliver(st: &AppState, tx: &mpsc::Sender<Piece>, p: Piece, measured: &mut bool) -> bool {
    match tx.try_send(p) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Closed(_)) => false,
        Err(mpsc::error::TrySendError::Full(p)) => {
            *measured = false;
            matches!(tokio::time::timeout(Duration::from_secs(st.cfg.serve.client_write_timeout_s), tx.send(p)).await, Ok(Ok(())))
        }
    }
}

pub async fn run_tokens(st: &AppState, job: TokenJob, tx: mpsc::Sender<Piece>, admission: oneshot::Sender<Result<(), Outcome>>) -> Outcome {
    let mut admission = Some(admission);
    let mut outcome = token_inner(st, &job, &tx, &mut admission).await;
    outcome.n_prompt = job.prompt.len() as u32;
    finish(st, &job.id, &mut outcome);
    if let Some(admission) = admission {
        let _ = admission.send(Err(outcome.clone()));
    }
    outcome
}

/// Lets the HTTP handler send its headers: past this point a failure streams instead of returning a status.
fn admit(admission: &mut Option<oneshot::Sender<Result<(), Outcome>>>) {
    if let Some(admission) = admission.take() {
        let _ = admission.send(Ok(()));
    }
}

async fn token_inner(st: &AppState, job: &TokenJob, tx: &mpsc::Sender<Piece>, admission: &mut Option<oneshot::Sender<Result<(), Outcome>>>) -> Outcome {
    let mut emitted = 0;
    for attempt in 0..=MAX_RESTARTS {
        let (generation, w) = st.supervisor.current().await;
        let req = format!("{}#{attempt}", job.id);
        let mut measured = attempt == 0 && st.concurrency == 1;
        let mut times = Vec::new();
        let mut error = None;
        let mut prefilling = true;
        // A restarted worker has lost the run-time reserve, so every attempt sends it again.
        let admission_control =
            Request::Admission { host_reserve_bytes: st.host_reserve_bytes.load(std::sync::atomic::Ordering::Relaxed), min_reply: st.cfg.serve.min_reply };
        let started = match w.send(&admission_control).await {
            Ok(()) => {
                w.start(&req, job.prompt.clone(), job.sampling.clone(), job.stop.clone(), job.max_tokens, job.render_special.clone(), job.chat.clone()).await
            }
            Err(e) => Err(e),
        };
        if let Ok(mut rx) = started {
            // Only the native worker rejects at admission; an external engine's first event comes after its prefill.
            if w.engine != "native" {
                admit(admission);
            }
            let mut credit = CREDIT;
            if w.credit(&req, CREDIT).await.is_ok() {
                loop {
                    let secs = if prefilling { st.cfg.serve.prefill_timeout_s } else { st.cfg.serve.decode_timeout_s };
                    let ev = tokio::select! {
                        ev = tokio::time::timeout(Duration::from_secs(secs), rx.recv()) => match ev {
                            Ok(ev) => ev,
                            Err(_) => { w.kill().await; break; }
                        },
                        _ = tx.closed() => {
                            let _ = w.cancel(&req).await;
                            return Outcome { reason: Some(FinishReason::Cancelled), n_completion: emitted, ..Default::default() };
                        }
                    };
                    if matches!(ev, Some(Event::Prefilling { .. } | Event::Prefilled { .. } | Event::Token { .. })) {
                        admit(admission);
                    }
                    match ev {
                        Some(Event::Prefilling { done, reused, ms, .. }) => st.live.prompt(&job.id, done, reused, ms),
                        Some(Event::Prefilled { n_prompt, reused, ms, .. }) => {
                            prefilling = false;
                            st.live.prompt(&job.id, n_prompt, reused, ms);
                        }
                        Some(Event::Token { token, text, deltas, t_us, .. }) => {
                            prefilling = false;
                            emitted += 1;
                            times.push(t_us);
                            st.live.token(&job.id);
                            if !st.journal.record(&job.id, Some(token), &text, &deltas, None) {
                                let _ = w.cancel(&req).await;
                                return failure("request journal capacity exceeded");
                            }
                            if !deliver(st, tx, Piece::Text { token, text, deltas }, &mut measured).await {
                                let _ = w.cancel(&req).await;
                                return failure("client disconnected or stopped reading");
                            }
                            credit -= 1;
                            if credit <= CREDIT / 2 {
                                if w.credit(&req, CREDIT).await.is_err() {
                                    break;
                                }
                                credit += CREDIT;
                            }
                        }
                        Some(Event::Error { message, .. }) if message == "worker process exited" => {}
                        Some(Event::Error { code: ErrorCode::ContextFull, message, .. }) if emitted == 0 => {
                            return Outcome { error_code: Some(ErrorCode::ContextFull), ..failure(message) };
                        }
                        Some(Event::Error { message, .. }) => error = Some(message),
                        Some(Event::Finished { reason: FinishReason::Error, .. }) if error.is_none() => break,
                        Some(Event::Finished { reason, n_decoded, tail, deltas, message, .. }) => {
                            if !tail.is_empty() || !deltas.is_empty() {
                                if !st.journal.record(&job.id, None, &tail, &deltas, None) {
                                    return failure("request journal capacity exceeded at completion");
                                }
                                if !deliver(st, tx, Piece::Text { token: -1, text: tail, deltas }, &mut measured).await {
                                    return failure("client disconnected or stopped reading");
                                }
                            }
                            let tps = if measured && error.is_none() && times.len() > 1 && times.last() > times.first() {
                                Some((times.len() - 1) as f64 * 1e6 / (times[times.len() - 1] - times[0]) as f64)
                            } else {
                                None
                            };
                            return Outcome { reason: Some(reason), n_completion: n_decoded, error, decode_tps: tps, message, ..Default::default() };
                        }
                        Some(Event::Paused { .. }) => measured = false,
                        Some(_) => {}
                        None => break,
                    }
                }
            }
        }
        // Empty text can still represent a committed UTF-8 fragment or partial stop string.
        if emitted > 0 {
            w.kill().await;
            return Outcome { n_completion: emitted, ..failure("worker interrupted a partial reply; start a new request (retained events remain resumable)") };
        }
        if attempt == MAX_RESTARTS {
            return failure("worker failed repeatedly before producing output");
        }
        if let Err(e) = st.supervisor.restart_if(generation).await {
            return failure(format!("worker restart failed: {e:#}"));
        }
    }
    unreachable!()
}

#[derive(Default)]
struct ChatReply {
    message: Value,
    calls: BTreeMap<u64, Value>,
    finish: Option<String>,
}

impl ChatReply {
    fn push(&mut self, chunk: &Value) {
        if !self.message.is_object() {
            self.message = json!({"role":"assistant", "content":""});
        }
        let choice = &chunk["choices"][0];
        if let Some(s) = choice["finish_reason"].as_str() {
            self.finish = Some(s.into());
        }
        if let Some(delta) = choice["delta"].as_object() {
            for (key, value) in delta {
                if key == "tool_calls" {
                    for call in value.as_array().into_iter().flatten() {
                        let idx = call["index"].as_u64().unwrap_or(0);
                        let dst = self.calls.entry(idx).or_insert_with(|| json!({"type":"function","function":{"name":"","arguments":""}}));
                        for k in ["id", "type"] {
                            if let Some(v) = call.get(k) {
                                dst[k] = v.clone();
                            }
                        }
                        for k in ["name", "arguments"] {
                            append(&mut dst["function"][k], &call["function"][k]);
                        }
                    }
                } else if key != "role" {
                    append(&mut self.message[key], value);
                }
            }
        }
    }

    fn message(mut self) -> Value {
        if !self.message.is_object() {
            self.message = json!({"role":"assistant", "content":""});
        }
        if !self.calls.is_empty() {
            self.message["tool_calls"] = Value::Array(self.calls.into_values().collect());
        }
        self.message
    }
}

fn append(to: &mut Value, value: &Value) {
    if let Some(s) = value.as_str() {
        let mut text = to.as_str().unwrap_or_default().to_string();
        text.push_str(s);
        *to = Value::String(text);
    } else if !value.is_null() {
        *to = value.clone();
    }
}

pub async fn run_chat(st: &AppState, id: &str, body: Value, tx: mpsc::Sender<Piece>) -> Outcome {
    let mut outcome = chat_inner(st, id, body, tx).await;
    finish(st, id, &mut outcome);
    outcome
}

async fn chat_inner(st: &AppState, id: &str, body: Value, tx: mpsc::Sender<Piece>) -> Outcome {
    let (_, w) = st.supervisor.current().await;
    let mut rx = match w.chat(id, body).await {
        Ok(rx) => rx,
        Err(e) => return failure(e.to_string()),
    };
    if let Err(e) = w.credit(id, CREDIT).await {
        return failure(e.to_string());
    }
    let mut outstanding = CREDIT;
    let mut reply = ChatReply::default();
    let mut measured = false;
    let mut first = true;
    loop {
        let secs = if first { st.cfg.serve.prefill_timeout_s } else { st.cfg.serve.decode_timeout_s };
        let ev = tokio::select! {
            ev = tokio::time::timeout(Duration::from_secs(secs), rx.recv()) => match ev {
                Ok(ev) => ev,
                Err(_) => { w.kill().await; return failure("chat worker progress timed out"); }
            },
            _ = tx.closed() => {
                let _ = w.cancel(id).await;
                return Outcome { reason: Some(FinishReason::Cancelled), ..Default::default() };
            }
        };
        match ev {
            Some(Event::ChatChunk { chunk, .. }) => {
                first = false;
                reply.push(&chunk);
                let text = chunk["choices"][0]["delta"]["content"].as_str().unwrap_or_default();
                if !st.journal.record(id, None, text, &[], Some(&chunk)) {
                    let _ = w.cancel(id).await;
                    return failure("request journal capacity exceeded");
                }
                if !deliver(st, &tx, Piece::Chunk(chunk), &mut measured).await {
                    let _ = w.cancel(id).await;
                    return failure("client disconnected or stopped reading");
                }
                outstanding -= 1;
                if outstanding <= CREDIT / 2 {
                    if let Err(e) = w.credit(id, CREDIT).await {
                        return failure(e.to_string());
                    }
                    outstanding += CREDIT;
                }
            }
            Some(Event::Finished { reason, n_prompt, n_decoded, .. }) => {
                if reason == FinishReason::Error {
                    return failure("external engine failed");
                }
                let finish_reason = reply.finish.clone();
                return Outcome {
                    reason: Some(reason),
                    n_prompt,
                    n_completion: n_decoded,
                    message: Some(reply.message()),
                    finish_reason,
                    ..Default::default()
                };
            }
            Some(Event::Error { message, .. }) => return failure(message),
            Some(_) => {}
            None => return failure("worker stream closed"),
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chat_accumulates_tools_reasoning_and_finish() {
        let mut reply = ChatReply::default();
        reply.push(
            &json!({"choices":[{"delta":{"reasoning_content":"think", "tool_calls":[{"index":0,"id":"call_1","function":{"name":"look","arguments":"{"}}]}}]}),
        );
        reply.push(&json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"up","arguments":"}"}}]},"finish_reason":"tool_calls"}]}));
        assert_eq!(reply.finish.as_deref(), Some("tool_calls"));
        let message = reply.message();
        assert_eq!(message["reasoning_content"], "think");
        assert_eq!(message["tool_calls"][0]["function"], json!({"name":"lookup","arguments":"{}"}));
        assert_eq!(message["tool_calls"][0]["id"], "call_1");
    }
}
