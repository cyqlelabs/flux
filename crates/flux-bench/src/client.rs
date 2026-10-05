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
            json!({"messages": [{"role": "user", "content": s}], "max_tokens": max_tokens, "stream": true, "stream_options":{"include_usage":true}, "logprobs":true, "temperature": 0, "ignore_eos": true}),
        ),
        _ => bail!("prompt kind does not match the API"),
    })
}

/// Tokens carried by one SSE payload.
fn tokens_in(api: Api, v: &Value) -> usize {
    match api {
        Api::LlamaCompletion => v["tokens"].as_array().map_or(0, |a| a.len()),
        Api::FluxCompletion => usize::from(v["flux_token"].as_i64().is_some_and(|t| t >= 0)),
        Api::Chat => v["choices"][0]["logprobs"]["content"].as_array().map_or(0, Vec::len),
    }
}

pub async fn stream(http: &reqwest::Client, base: &str, api: Api, prompt: Prompt<'_>, max_tokens: u32) -> Stream {
    let t0 = Instant::now();
    let mut s = Stream::default();
    let r: Result<()> = async {
        let (path, b) = body(api, &prompt, max_tokens)?;
        let resp = tokio::time::timeout(std::time::Duration::from_secs(300), http.post(format!("{base}{path}")).json(&b).send())
            .await
            .context("request headers timed out")??;
        let status = resp.status();
        if !status.is_success() {
            bail!("HTTP {status}: {}", resp.text().await.unwrap_or_default());
        }
        let mut bytes = resp.bytes_stream();
        let mut buf = Vec::new();
        let mut terminal = false;
        let mut completion_tokens = None;
        while let Some(chunk) = tokio::time::timeout(std::time::Duration::from_secs(300), bytes.next()).await.context("stream made no progress")? {
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
                if v["stop"].as_bool() == Some(true) {
                    anyhow::ensure!(v["stop_type"] == "limit", "benchmark stopped before its token limit");
                    terminal = true;
                }
                if let Some(reason) = v["choices"][0]["finish_reason"].as_str() {
                    anyhow::ensure!(reason == "length", "benchmark finished with {reason}, expected length");
                    terminal = true;
                }
                completion_tokens = v["usage"]["completion_tokens"].as_u64().or_else(|| v["tokens_predicted"].as_u64()).or(completion_tokens);
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
        validate_completion(api, terminal, completion_tokens, s.token_times_s.len(), max_tokens)?;
        Ok(())
    }
    .await;
    s.total_s = t0.elapsed().as_secs_f64();
    s.error = r.err().map(|e| format!("{e:#}"));
    s
}

fn validate_completion(api: Api, terminal: bool, usage: Option<u64>, counted: usize, requested: u32) -> Result<()> {
    anyhow::ensure!(terminal, "stream ended without a terminal finish marker");
    anyhow::ensure!(counted == requested as usize, "expected {requested} token timestamps, got {counted}; chat benchmarks require per-token logprobs");
    if let Some(n) = usage {
        anyhow::ensure!(n == requested as u64, "completion usage {n} differs from requested {requested}");
    }
    anyhow::ensure!(api != Api::Chat || usage.is_some(), "chat benchmark omitted completion usage");
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn real_http_streams_require_complete_output() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        async fn wire(body: String, truncate: bool) -> Stream {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                loop {
                    let mut bytes = [0; 4096];
                    let n = socket.read(&mut bytes).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&bytes[..n]);
                    if let Some(end) = request.windows(4).position(|s| s == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&request[..end]).to_lowercase();
                        let len: usize = headers.lines().find_map(|s| s.strip_prefix("content-length:").map(str::trim)).unwrap().parse().unwrap();
                        if request.len() >= end + 4 + len {
                            break;
                        }
                    }
                }
                let len = body.len() + if truncate { 100 } else { 0 };
                socket
                    .write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n").as_bytes())
                    .await
                    .unwrap();
                socket.write_all(body.as_bytes()).await.unwrap();
                socket.shutdown().await.unwrap();
            });
            let result = stream(&reqwest::Client::new(), &base, Api::FluxCompletion, Prompt::Tokens(&[1]), 2).await;
            server.await.unwrap();
            result
        }
        let tokens = "data: {\"flux_token\":1}\n\ndata: {\"flux_token\":2}\n\n";
        let finished = format!("{tokens}data: {{\"choices\":[{{\"finish_reason\":\"length\"}}],\"usage\":{{\"completion_tokens\":2}}}}\n\ndata: [DONE]\n\n");
        assert!(wire(tokens.into(), false).await.error.unwrap().contains("finish marker"));
        assert!(wire(finished.clone(), true).await.error.is_some());
        assert!(wire(finished, false).await.error.is_none());
    }

    #[test]
    fn truncation_and_chunk_counts_cannot_pass() {
        assert!(validate_completion(Api::FluxCompletion, false, Some(8), 8, 8).is_err());
        assert!(validate_completion(Api::Chat, true, Some(8), 2, 8).is_err());
        assert!(validate_completion(Api::Chat, true, None, 8, 8).is_err());
        assert!(validate_completion(Api::Chat, true, Some(8), 8, 8).is_ok());
        assert_eq!(tokens_in(Api::FluxCompletion, &json!({"flux_token":null,"choices":[{"text":"tail"}]})), 0);
        assert_eq!(tokens_in(Api::Chat, &json!({"choices":[{"delta":{"content":"several words"}}]})), 0);
    }
}
