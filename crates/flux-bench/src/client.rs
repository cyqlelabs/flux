//! Streaming clients for the three request shapes the contestants accept. Every contestant is
//! measured the same way: client-side timestamps of each streamed token.

use anyhow::{bail, Context, Result};
use futures::StreamExt;
use serde_json::{json, Value};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Api {
    /// llama-server `/completion` with token-id prompts.
    LlamaCompletion,
    /// flux-serve `/v1/completions` with token-id prompts.
    FluxCompletion,
    /// OpenAI `/v1/chat/completions` with a text message (chat-level engines).
    Chat,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Stream {
    pub ttft_s: f64,
    pub token_times_s: Vec<f64>,
    pub total_s: f64,
    pub prompt_tokens: Option<u64>,
    pub error: Option<String>,
}

impl Stream {
    pub fn decode_rate(&self) -> Option<f64> {
        flux_core::stats::interactive_rate(&self.token_times_s)
    }
}

pub enum Prompt<'a> {
    Tokens(&'a [i32]),
    Text(&'a str),
}

fn body(api: Api, prompt: &Prompt, max_tokens: u32) -> Result<(String, Value)> {
    Ok(match (api, prompt) {
        (Api::LlamaCompletion, Prompt::Tokens(t)) => (
            "/completion".into(),
            json!({"prompt": t, "n_predict": max_tokens, "stream": true, "temperature": 0, "ignore_eos": true, "cache_prompt": false, "return_tokens": true}),
        ),
        (Api::FluxCompletion, Prompt::Tokens(t)) => {
            ("/v1/completions".into(), json!({"prompt": t, "max_tokens": max_tokens, "stream": true, "temperature": 0, "ignore_eos": true}))
        }
        (Api::Chat, Prompt::Text(s)) => (
            "/v1/chat/completions".into(),
            json!({"messages": [{"role": "user", "content": s}], "max_tokens": max_tokens, "stream": true, "temperature": 0, "ignore_eos": true}),
        ),
        _ => bail!("prompt kind does not match the API"),
    })
}

/// Tokens carried by one SSE payload.
fn tokens_in(api: Api, v: &Value) -> usize {
    match api {
        Api::LlamaCompletion => v["tokens"].as_array().map_or(0, |a| a.len()),
        Api::FluxCompletion => usize::from(v["choices"][0]["finish_reason"].is_null()),
        Api::Chat => {
            let d = &v["choices"][0]["delta"];
            usize::from(d.get("content").is_some_and(|c| !c.is_null()) || d.get("reasoning_content").is_some_and(|c| !c.is_null()))
        }
    }
}

pub async fn stream(http: &reqwest::Client, base: &str, api: Api, prompt: Prompt<'_>, max_tokens: u32) -> Stream {
    let t0 = Instant::now();
    let mut s = Stream::default();
    let r: Result<()> = async {
        let (path, b) = body(api, &prompt, max_tokens)?;
        let resp = http.post(format!("{base}{path}")).json(&b).send().await?;
        let status = resp.status();
        if !status.is_success() {
            bail!("HTTP {status}: {}", resp.text().await.unwrap_or_default());
        }
        let mut bytes = resp.bytes_stream();
        let mut buf = Vec::new();
        while let Some(chunk) = bytes.next().await {
            buf.extend_from_slice(&chunk.context("stream interrupted")?);
            while let Some(nl) = buf.iter().position(|&c| c == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let Some(data) = std::str::from_utf8(&line)?.trim().strip_prefix("data:").map(str::trim) else { continue };
                if data == "[DONE]" {
                    continue;
                }
                let v: Value = serde_json::from_str(data)?;
                if let Some(e) = v.get("error") {
                    bail!("engine error: {e}");
                }
                let t = t0.elapsed().as_secs_f64();
                for _ in 0..tokens_in(api, &v) {
                    if s.token_times_s.is_empty() {
                        s.ttft_s = t;
                    }
                    s.token_times_s.push(t);
                }
                if let Some(p) = v["usage"]["prompt_tokens"].as_u64().or_else(|| v["tokens_evaluated"].as_u64()) {
                    s.prompt_tokens = Some(p);
                }
            }
        }
        Ok(())
    }
    .await;
    s.total_s = t0.elapsed().as_secs_f64();
    s.error = r.err().map(|e| format!("{e:#}"));
    s
}

pub async fn tokenize(http: &reqwest::Client, base: &str, path: &str, text: &str) -> Result<Vec<i32>> {
    let v: Value = http.post(format!("{base}{path}")).json(&json!({"content": text, "add_special": true})).send().await?.error_for_status()?.json().await?;
    serde_json::from_value(v["tokens"].clone()).context("tokenize response")
}

pub async fn wait_healthy(http: &reqwest::Client, base: &str, path: &str, timeout_s: u64, alive: impl Fn() -> bool) -> Result<()> {
    let deadline = Instant::now() + std::time::Duration::from_secs(timeout_s);
    loop {
        if !alive() {
            bail!("server exited during startup");
        }
        if let Ok(r) = http.get(format!("{base}{path}")).timeout(std::time::Duration::from_secs(2)).send().await {
            if r.status().is_success() {
                return Ok(());
            }
        }
        if Instant::now() > deadline {
            bail!("server not healthy after {timeout_s} s");
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}
