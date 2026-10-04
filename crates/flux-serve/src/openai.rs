//! OpenAI-compatible endpoints over the plan's worker.

use crate::admission::Rejection;
use crate::generate::{finish_name, run_chat, run_tokens, Outcome, Piece, TokenJob};
use crate::AppState;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as Sse, KeepAlive, Sse as SseResponse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use flux_core::protocol::Sampling;
use flux_core::worker::ChatOptions;
use futures::stream::{self, Stream, StreamExt};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

pub fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"message": message.into(), "type": kind, "code": status.as_u16()}}))).into_response()
}

fn rejection(r: Rejection) -> Response {
    match r {
        Rejection::QueueFull => error(StatusCode::TOO_MANY_REQUESTS, "queue_full", "the request queue is full; retry later"),
        Rejection::Closed(why) => {
            let mut resp = error(StatusCode::SERVICE_UNAVAILABLE, "admission_closed", why);
            resp.headers_mut().insert("retry-after", "5".parse().unwrap());
            resp
        }
    }
}

fn sampling(b: &Value) -> Sampling {
    let f = |k: &str| b.get(k).and_then(Value::as_f64).map(|v| v as f32);
    Sampling {
        temperature: f("temperature"),
        top_k: b.get("top_k").and_then(Value::as_i64).map(|v| v as i32),
        top_p: f("top_p"),
        min_p: f("min_p"),
        repeat_penalty: f("repeat_penalty"),
        repeat_last_n: b.get("repeat_last_n").and_then(Value::as_i64).map(|v| v as i32),
        presence_penalty: f("presence_penalty"),
        frequency_penalty: f("frequency_penalty"),
        seed: b.get("seed").and_then(Value::as_u64).map(|v| v as u32),
        ignore_eos: b.get("ignore_eos").and_then(Value::as_bool),
        runner_up: None,
    }
}

fn stops(b: &Value) -> Vec<String> {
    match b.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        _ => vec![],
    }
}

fn request_id(h: &HeaderMap, prefix: &str) -> String {
    h.get("x-request-id").and_then(|v| v.to_str().ok()).map(str::to_string).unwrap_or_else(|| {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        format!("{prefix}-{n:x}")
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Api {
    Chat,
    Completion,
}

fn chunk(api: Api, id: &str, model: &str, created: i64, text: Option<&str>, finish: Option<&str>, usage: Option<Value>) -> Value {
    let choice = match api {
        Api::Chat => json!({"index": 0, "delta": text.map_or(json!({}), |t| json!({"content": t})), "finish_reason": finish}),
        Api::Completion => json!({"index": 0, "text": text.unwrap_or(""), "finish_reason": finish}),
    };
    let object = if api == Api::Chat { "chat.completion.chunk" } else { "text_completion" };
    let mut c = json!({"id": id, "object": object, "created": created, "model": model, "choices": [choice]});
    if let Some(u) = usage {
        c["usage"] = u;
    }
    c
}

/// One streamed chunk carrying a parsed delta (reasoning, content or a tool call).
fn delta_chunk(id: &str, model: &str, created: i64, delta: Value) -> Value {
    json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": null}]})
}

/// "tool_calls" when a parsed reply ends by calling tools, as OpenAI reports it.
fn finish_reason(o: &Outcome) -> String {
    let calls = o.message.as_ref().and_then(|m| m["tool_calls"].as_array()).is_some_and(|c| !c.is_empty());
    match finish_name(o.reason) {
        f if calls && f == "stop" => "tool_calls".into(),
        f => f,
    }
}

fn usage(o: &Outcome) -> Value {
    json!({"prompt_tokens": o.n_prompt, "completion_tokens": o.n_completion, "total_tokens": o.n_prompt + o.n_completion})
}

/// Shared tail of token-level requests: admit, validate context, generate, stream or collect.
#[allow(clippy::too_many_arguments)]
async fn token_request(
    st: Arc<AppState>,
    api: Api,
    id: String,
    prompt: Vec<i32>,
    body: Value,
    render_special: Vec<i32>,
    extra_stops: Vec<String>,
    chat: ChatOptions,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let n_ctx = st.supervisor.loaded().await.n_ctx_seq as usize;
    let requested = body.get("max_completion_tokens").or_else(|| body.get("max_tokens")).and_then(Value::as_u64).map(|v| v as usize);
    // max_tokens is a ceiling, as in llama-server: agents ask for 32K or more on every turn, so a reply stops
    // ("length") where the context ends instead of the request failing once the conversation grows.
    let room = n_ctx.saturating_sub(prompt.len());
    let max_tokens = requested.unwrap_or(room).min(room);
    if prompt.is_empty() || requested == Some(0) {
        return error(StatusCode::BAD_REQUEST, "invalid_request_error", "the prompt is empty or max_tokens is 0");
    }
    if room == 0 {
        return error(
            StatusCode::BAD_REQUEST,
            "context_length_exceeded",
            format!("prompt ({} tokens) leaves no room in the plan's {n_ctx} tokens per sequence; Flux never truncates a prompt", prompt.len()),
        );
    }
    if st.journal.begin(&id, prompt.clone()).is_none() {
        return error(StatusCode::CONFLICT, "duplicate_request", format!("request {id} exists; resume it at /flux/requests/{id}/stream"));
    }
    let mut stop = stops(&body);
    stop.extend(extra_stops);
    st.live.begin(&id, prompt.len() as u32);
    let parsed = chat.parser.is_some();
    let agent = body.get("tools").and_then(Value::as_array).is_some_and(|t| !t.is_empty());
    let job = TokenJob { id: id.clone(), prompt, sampling: sampling(&body), stop, max_tokens: max_tokens as u32, render_special, chat };
    let (tx, rx) = mpsc::channel(32);
    let (done_tx, done_rx) = oneshot::channel();
    let (st2, id2) = (st.clone(), id.clone());
    tokio::spawn(async move {
        let o = run_tokens(&st2, job, tx).await;
        drop(permit);
        st2.live.finish(&id2, &o);
        if let Some(tps) = o.decode_tps {
            crate::monitor::observe_rate(&st2, tps, o.n_prompt, agent);
        }
        let _ = done_tx.send(o);
    });
    respond(st, api, id, body.get("stream").and_then(Value::as_bool).unwrap_or(false), parsed, rx, done_rx).await
}

/// `parsed`: the reply arrives as chat deltas and a final message instead of plain text.
async fn respond(st: Arc<AppState>, api: Api, id: String, stream: bool, parsed: bool, rx: mpsc::Receiver<Piece>, done: oneshot::Receiver<Outcome>) -> Response {
    let model = st.model_name.clone();
    let created = chrono::Utc::now().timestamp();
    if stream {
        let body = sse(api, id, model, created, parsed, rx, done);
        return SseResponse::new(body).keep_alive(KeepAlive::default()).into_response();
    }
    let mut rx = rx;
    let mut text = String::new();
    let mut chunks = vec![];
    while let Some(p) = rx.recv().await {
        match p {
            Piece::Text { text: t, .. } => text.push_str(&t),
            Piece::Chunk(c) => chunks.push(c),
        }
    }
    let o = done.await.unwrap_or(Outcome { error: Some("generation task failed".into()), ..failed() });
    if let Some(e) = &o.error {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "engine_error", e.clone());
    }
    if !chunks.is_empty() {
        text = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    }
    let finish = finish_reason(&o);
    let message = match &o.message {
        Some(m) => {
            let mut m = m.clone();
            m["role"] = json!("assistant");
            m
        }
        None => json!({"role": "assistant", "content": text}),
    };
    let choice = match api {
        Api::Chat => json!({"index": 0, "message": message, "finish_reason": finish}),
        Api::Completion => json!({"index": 0, "text": text, "finish_reason": finish}),
    };
    let object = if api == Api::Chat { "chat.completion" } else { "text_completion" };
    Json(json!({"id": id, "object": object, "created": created, "model": model, "choices": [choice], "usage": usage(&o)})).into_response()
}

fn failed() -> Outcome {
    Outcome { reason: None, n_prompt: 0, n_completion: 0, error: None, decode_tps: None, message: None }
}

fn sse(
    api: Api,
    id: String,
    model: String,
    created: i64,
    parsed: bool,
    rx: mpsc::Receiver<Piece>,
    done: oneshot::Receiver<Outcome>,
) -> impl Stream<Item = Result<Sse, Infallible>> {
    let (id2, model2) = (id.clone(), model.clone());
    let pieces = stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|p| (p, rx)) }).flat_map(move |p| {
        let values = match p {
            Piece::Text { deltas, .. } if parsed => deltas.into_iter().map(|d| delta_chunk(&id, &model, created, d)).collect(),
            Piece::Text { text, .. } => vec![chunk(api, &id, &model, created, Some(&text), None, None)],
            Piece::Chunk(c) => vec![c],
        };
        stream::iter(values.into_iter().map(|v| Ok(Sse::default().data(v.to_string()))))
    });
    let tail = stream::once(async move {
        let o = done.await.unwrap_or(Outcome { error: Some("generation task failed".into()), ..failed() });
        let mut last = chunk(api, &id2, &model2, created, None, Some(&finish_reason(&o)), Some(usage(&o)));
        if let Some(e) = o.error {
            last["error"] = json!({"message": e});
        }
        stream::iter(vec![Ok(Sse::default().data(last.to_string())), Ok(Sse::default().data("[DONE]"))])
    })
    .flatten();
    pieces.chain(tail)
}

pub async fn chat(State(st): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let id = request_id(&headers, "chatcmpl");
    let permit = match st.admission.admit().await {
        Ok(p) => p,
        Err(r) => return rejection(r),
    };
    if st.level == "chat" {
        if st.journal.begin(&id, vec![]).is_none() {
            return error(StatusCode::CONFLICT, "duplicate_request", format!("request {id} exists"));
        }
        let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let (tx, rx) = mpsc::channel(32);
        let (done_tx, done_rx) = oneshot::channel();
        let (st2, id2) = (st.clone(), id.clone());
        st.live.begin(&id, 0);
        tokio::spawn(async move {
            let o = run_chat(&st2, &id2, body, tx).await;
            drop(permit);
            st2.live.finish(&id2, &o);
            let _ = done_tx.send(o);
        });
        return respond(st, Api::Chat, id, stream, false, rx, done_rx).await;
    }
    let (_, w) = st.supervisor.current().await;
    let messages = body.get("messages").cloned().unwrap_or(Value::Null);
    let templated = match w.apply_template(messages, body.get("tools").cloned()).await {
        Ok(t) => t,
        Err(e) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", format!("chat template: {e}")),
    };
    let prompt = match w.tokenize(&templated.prompt, true).await {
        Ok(p) => p,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "engine_error", e.to_string()),
    };
    let chat = ChatOptions { parser: templated.parser, prefix: String::new(), checkpoints: templated.checkpoints };
    token_request(st, Api::Chat, id, prompt, body, templated.preserved_tokens, templated.additional_stops, chat, permit).await
}

pub async fn completions(State(st): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    if st.level == "chat" {
        return error(StatusCode::BAD_REQUEST, "unsupported", "this engine only accepts chat requests");
    }
    let id = request_id(&headers, "cmpl");
    let permit = match st.admission.admit().await {
        Ok(p) => p,
        Err(r) => return rejection(r),
    };
    let prompt = match body.get("prompt") {
        Some(Value::String(s)) => {
            let (_, w) = st.supervisor.current().await;
            match w.tokenize(s, true).await {
                Ok(p) => p,
                Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "engine_error", e.to_string()),
            }
        }
        Some(Value::Array(a)) if a.iter().all(Value::is_i64) => a.iter().map(|v| v.as_i64().unwrap() as i32).collect(),
        _ => return error(StatusCode::BAD_REQUEST, "invalid_request_error", "prompt must be a string or an array of token ids"),
    };
    token_request(st, Api::Completion, id, prompt, body, vec![], vec![], ChatOptions::default(), permit).await
}

pub async fn models(State(st): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": st.model_name, "object": "model", "owned_by": "flux"}]}))
}

pub async fn request_entry(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match st.journal.get(&id) {
        Some(e) => Json(json!({
            "id": id,
            "status": e.status.to_json(),
            "prompt_tokens": e.prompt.len(),
            "tokens": e.tokens,
            "text": e.texts.concat(),
        }))
        .into_response(),
        None => error(StatusCode::NOT_FOUND, "not_found", format!("no request {id} in the journal")),
    }
}

#[derive(serde::Deserialize)]
pub struct After {
    #[serde(default)]
    after: usize,
}

/// Replays a request's committed tokens after index `after`, then follows it live: a client that
/// lost its stream resumes without duplicated tokens.
pub async fn resume(State(st): State<Arc<AppState>>, Path(id): Path<String>, Query(q): Query<After>) -> Response {
    let Some(watch) = st.journal.watch(&id) else {
        return error(StatusCode::NOT_FOUND, "not_found", format!("no request {id} in the journal"));
    };
    let st2 = st.clone();
    let s = stream::unfold((q.after, false), move |(next, done)| {
        let st = st2.clone();
        let id = id.clone();
        let mut w = watch.clone();
        async move {
            if done {
                return None;
            }
            loop {
                let e = st.journal.get(&id)?;
                if next < e.texts.len() {
                    let v = json!({"index": next, "token": e.tokens[next], "text": e.texts[next]});
                    return Some((Ok::<_, Infallible>(Sse::default().data(v.to_string())), (next + 1, false)));
                }
                if e.status != crate::journal::Status::Running {
                    return Some((Ok(Sse::default().data(json!({"status": e.status.to_json()}).to_string())), (next, true)));
                }
                if w.changed().await.is_err() {
                    return None;
                }
            }
        }
    });
    SseResponse::new(s).keep_alive(KeepAlive::default()).into_response()
}
