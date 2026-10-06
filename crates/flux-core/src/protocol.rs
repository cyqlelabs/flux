//! Versioned control protocol between flux-serve and a flux-worker process.
//! One JSON object per line: requests on the worker's stdin, events on its stdout.

use crate::plan::Plan;
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 2;

/// Unset fields keep the backend's defaults, so every engine samples the same way for the same request.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    /// Bans end-of-generation tokens so benchmarks get exactly `max_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
    /// Reports each undrafted token's runner-up (`Token.alt`), which certification compares against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runner_up: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Sets the RAM reserve before the next step: admission's threshold and, plus the reopen margin, the RAM the native page ledger leaves free.
    Admission {
        host_reserve_bytes: u64,
        min_reply: u32,
    },
    Hello {
        protocol: u32,
    },
    Load {
        plan: Box<Plan>,
        /// Install per-node time attribution (slows traced steps; see `Trace`).
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        trace: bool,
    },
    Tokenize {
        id: u64,
        text: String,
        add_special: bool,
    },
    /// Renders chat messages (OpenAI shape) with the model's own template.
    ApplyTemplate {
        id: u64,
        messages: serde_json::Value,
        #[serde(default)]
        tools: Option<serde_json::Value>,
        add_generation_prompt: bool,
    },
    /// Creates a sequence and processes its prompt. Decoding starts only when credit arrives via `Decode`.
    Prefill {
        req: String,
        prompt: Vec<i32>,
        sampling: Sampling,
        stop: Vec<String>,
        max_tokens: u32,
        /// Special tokens to render as text (the chat template's preserved tokens); others render empty.
        #[serde(default)]
        render_special: Vec<i32>,
        /// The `parser` spec from `Templated`: replies are split into reasoning, content and tool calls.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chat: Option<serde_json::Value>,
        /// Reply text already delivered before a restart; the parser starts after it.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        chat_prefix: String,
        /// Prompt positions worth a recurrent-state checkpoint for later requests' reuse (`Templated`).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        checkpoints: Vec<u32>,
    },
    /// Grants credit for `n` more tokens; bounds how far generation may run ahead of the consumer.
    Decode {
        req: String,
        n: u32,
    },
    Cancel {
        req: String,
    },
    /// Whole OpenAI chat request for engines that only accept messages (e.g. Strata).
    /// Tokens arrive as `Token` events with `token = -1`; credit applies as for `Prefill`.
    Chat {
        req: String,
        body: serde_json::Value,
    },
    Stats {
        id: u64,
    },
    /// Decodes `steps` tokens after `prompt` untraced, then traced, while no request is active.
    Trace {
        id: u64,
        prompt: Vec<i32>,
        steps: u32,
        /// Observe every computing node (heavy synchronization) instead of one point per device split.
        #[serde(default)]
        per_op: bool,
        /// Count expert selections per MoE layer instead of timing.
        #[serde(default)]
        routes: bool,
    },
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// End-of-generation token.
    Eog,
    /// A stop string matched; its text is not emitted.
    Stop,
    /// `max_tokens` reached.
    Length,
    Cancelled,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    Protocol,
    NotLoaded,
    LoadFailed,
    /// Prompt plus `max_tokens` exceeds the planned context per sequence.
    ContextFull,
    /// All planned sequences are in use.
    Busy,
    BadRequest,
    Backend,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceMemory {
    pub device: String,
    pub model: u64,
    pub context: u64,
    /// Reserved address space; paged caches back only the active stream prefixes.
    #[serde(default)]
    pub context_capacity: u64,
    /// Page-rounded KV capacities by placement class, from the backend.
    #[serde(default)]
    pub kv_full_read: u64,
    #[serde(default)]
    pub kv_dense: u64,
    #[serde(default)]
    pub kv_sparse: u64,
    /// Full-read capacity plus the attention floor on this device.
    #[serde(default)]
    pub kv_floor: u64,
    pub compute: u64,
    /// The part of `compute` that stages op-offloaded expert weights during prefill; an expert cache at least
    /// this large on the device holds it instead.
    #[serde(default)]
    pub staging: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkerStats {
    pub active: u32,
    pub steps: u64,
    pub prefilled_tokens: u64,
    pub decoded_tokens: u64,
    pub step_ms_p50: f64,
    pub step_ms_p95: f64,
    pub memory: Vec<DeviceMemory>,
    pub rss_bytes: u64,
    #[serde(default)]
    pub kv_pages: serde_json::Value,
    /// Host RAM holding parked conversations.
    #[serde(default)]
    pub parked_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "ev", rename_all = "snake_case")]
pub enum Event {
    Hello {
        protocol: u32,
        worker: String,
        engine: String,
        backend_revision: String,
        #[serde(default)]
        backend_build: String,
        /// `tokens` (Tokenize/ApplyTemplate/Prefill) or `chat` (Chat only).
        level: String,
    },
    Loaded {
        load_ms: f64,
        n_ctx_seq: u32,
        n_seq: u32,
        memory: Vec<DeviceMemory>,
    },
    Tokens {
        id: u64,
        tokens: Vec<i32>,
    },
    Templated {
        id: u64,
        prompt: String,
        preserved_tokens: Vec<i32>,
        additional_stops: Vec<String>,
        /// How to split this request's reply (passed back in `Prefill.chat`).
        #[serde(default)]
        parser: Option<serde_json::Value>,
        /// Prompt tokens shared with other requests: up to the end of the system prompt and of the last message.
        #[serde(default)]
        checkpoints: Vec<u32>,
    },
    Prefilled {
        req: String,
        n_prompt: u32,
        ms: f64,
        /// Prompt tokens served from the slot's cached prefix instead of being processed again.
        #[serde(default)]
        reused: u32,
    },
    /// Prompt progress after each chunk short of the end; `Prefilled` follows the last one.
    Prefilling {
        req: String,
        done: u32,
        total: u32,
        reused: u32,
        ms: f64,
    },
    /// `text` is the valid UTF-8 completed by this token; it may be empty while a character is split across tokens.
    Token {
        req: String,
        i: u32,
        token: i32,
        text: String,
        t_us: u64,
        /// The runner-up of the row this token was sampled from (native engine, single-row steps).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        alt: Option<i32>,
        /// OpenAI chat deltas this token completes, when the request is parsed (`Prefill.chat`).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        deltas: Vec<serde_json::Value>,
    },
    /// One streamed chunk from a chat-level engine, relayed verbatim (OpenAI chat.completion.chunk).
    ChatChunk {
        req: String,
        chunk: serde_json::Value,
        t_us: u64,
    },
    /// Credit exhausted; waiting for `Decode`.
    Paused {
        req: String,
    },
    Finished {
        req: String,
        reason: FinishReason,
        n_prompt: u32,
        n_decoded: u32,
        /// Text held back while it could still become a stop string, released at the end.
        tail: String,
        /// Parsed requests: the deltas the tail completes and the whole assistant message.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        deltas: Vec<serde_json::Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<serde_json::Value>,
    },
    Stats {
        id: u64,
        stats: WorkerStats,
    },
    Traced {
        id: u64,
        report: serde_json::Value,
    },
    Error {
        #[serde(skip_serializing_if = "Option::is_none")]
        req: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<u64>,
        code: ErrorCode,
        message: String,
    },
    Bye,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_format_is_tagged() {
        let r = Request::Decode { req: "r1".into(), n: 8 };
        assert_eq!(serde_json::to_string(&r).unwrap(), r#"{"op":"decode","req":"r1","n":8}"#);
        let e: Event = serde_json::from_str(r#"{"ev":"paused","req":"r1"}"#).unwrap();
        assert_eq!(e, Event::Paused { req: "r1".into() });
        let s = Sampling { temperature: Some(0.0), ..Default::default() };
        assert_eq!(serde_json::to_string(&s).unwrap(), r#"{"temperature":0.0}"#);
    }
}
