//! Native engine: one llama.cpp context, continuous batching across planned sequences.
//! Each step decodes one token for every sequence that has credit, plus one prompt chunk.

use crate::out::Out;
use crate::text::{Pushed, TextStream};
use anyhow::Result;
use flux_core::protocol::{DeviceMemory, ErrorCode, Event, FinishReason, Request, Sampling, WorkerStats, PROTOCOL_VERSION};
use flux_native::{Engine, Sampler};
use std::collections::{HashSet, VecDeque};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Instant;

struct Seq {
    req: String,
    slot: i32,
    prompt: Vec<i32>,
    prefilled: usize,
    /// Next KV position to write.
    pos: i32,
    sampler: Sampler,
    max_tokens: u32,
    emitted: u32,
    credit: u32,
    /// Emitted token not yet written to the KV cache; fed by the next decode step.
    next: Option<i32>,
    /// Sampled token waiting for credit.
    held: Option<i32>,
    text: TextStream,
    render_special: HashSet<i32>,
    t_start: Instant,
    paused_sent: bool,
}

impl Seq {
    fn prefill_done(&self) -> bool {
        self.prefilled == self.prompt.len()
    }

    fn wants_decode(&self) -> bool {
        self.prefill_done() && self.held.is_none() && self.credit > 0 && self.next.is_some()
    }
}

pub struct NativeWorker {
    out: Out,
    engine: Option<Engine>,
    n_ctx_seq: u32,
    n_batch: usize,
    seqs: Vec<Seq>,
    free: Vec<i32>,
    start: Instant,
    steps: u64,
    prefilled: u64,
    decoded: u64,
    step_ms: VecDeque<f64>,
}

impl NativeWorker {
    pub fn new(out: Out) -> NativeWorker {
        NativeWorker {
            out,
            engine: None,
            n_ctx_seq: 0,
            n_batch: 0,
            seqs: vec![],
            free: vec![],
            start: Instant::now(),
            steps: 0,
            prefilled: 0,
            decoded: 0,
            step_ms: VecDeque::new(),
        }
    }

    pub fn run(&mut self, rx: Receiver<Request>) -> Result<()> {
        loop {
            let busy = self.seqs.iter().any(|s| !s.prefill_done() || s.wants_decode());
            let req = if busy {
                match rx.try_recv() {
                    Ok(r) => Some(r),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => return Ok(()),
                }
            } else {
                match rx.recv() {
                    Ok(r) => Some(r),
                    Err(_) => return Ok(()),
                }
            };
            match req {
                Some(Request::Shutdown) => {
                    self.out.send(&Event::Bye);
                    return Ok(());
                }
                Some(r) => self.handle(r),
                None => self.step(),
            }
        }
    }

    fn error(&self, req: Option<&str>, id: Option<u64>, code: ErrorCode, message: impl Into<String>) {
        self.out.send(&Event::Error { req: req.map(str::to_string), id, code, message: message.into() });
    }

    /// Refuses a request; every Prefill ends with exactly one Finished, even when rejected.
    fn reject(&self, req: &str, code: ErrorCode, message: impl Into<String>) {
        self.error(Some(req), None, code, message);
        self.out.send(&Event::Finished { req: req.into(), reason: FinishReason::Error, n_prompt: 0, n_decoded: 0, tail: String::new() });
    }

    fn handle(&mut self, r: Request) {
        if self.engine.is_none() && !matches!(r, Request::Hello { .. } | Request::Load { .. } | Request::Shutdown) {
            let (req, id) = request_ids(&r);
            return match (req, &r) {
                (Some(req), Request::Prefill { .. } | Request::Chat { .. }) => self.reject(&req, ErrorCode::NotLoaded, "no model loaded"),
                (req, _) => self.error(req.as_deref(), id, ErrorCode::NotLoaded, "no model loaded"),
            };
        }
        match r {
            Request::Hello { protocol } if protocol != PROTOCOL_VERSION => {
                self.error(None, None, ErrorCode::Protocol, format!("worker speaks protocol {PROTOCOL_VERSION}, supervisor sent {protocol}"))
            }
            Request::Hello { .. } => {}
            Request::Load { plan, trace } => self.load(&plan, trace),
            Request::Tokenize { id, text, add_special } => {
                let tokens = self.engine.as_ref().unwrap().tokenize(&text, add_special);
                self.out.send(&Event::Tokens { id, tokens });
            }
            Request::ApplyTemplate { id, messages, tools, add_generation_prompt } => {
                let req = serde_json::json!({"messages": messages, "tools": tools, "add_generation_prompt": add_generation_prompt});
                match self.engine.as_ref().unwrap().apply_template(&req) {
                    Ok(v) => self.out.send(&Event::Templated {
                        id,
                        prompt: v["prompt"].as_str().unwrap_or_default().to_string(),
                        preserved_tokens: serde_json::from_value(v["preserved_tokens"].clone()).unwrap_or_default(),
                        additional_stops: serde_json::from_value(v["additional_stops"].clone()).unwrap_or_default(),
                    }),
                    Err(e) => self.error(None, Some(id), ErrorCode::BadRequest, e.to_string()),
                }
            }
            Request::Prefill { req, prompt, sampling, stop, max_tokens, render_special } => {
                self.prefill(req, prompt, &sampling, stop, max_tokens, render_special)
            }
            Request::Decode { req, n } => match self.seqs.iter().position(|s| s.req == req) {
                Some(i) => {
                    self.seqs[i].credit += n;
                    self.seqs[i].paused_sent = false;
                    self.emit_held(i);
                }
                None => self.error(Some(&req), None, ErrorCode::BadRequest, "unknown request"),
            },
            Request::Cancel { req } => {
                if let Some(i) = self.seqs.iter().position(|s| s.req == req) {
                    self.finish(i, FinishReason::Cancelled);
                }
            }
            Request::Stats { id } => self.out.send(&Event::Stats { id, stats: self.stats() }),
            Request::Trace { id, prompt, steps, per_op, routes } => {
                if !self.seqs.is_empty() {
                    return self.error(None, Some(id), ErrorCode::Busy, "tracing needs an idle worker");
                }
                let req = serde_json::json!({"prompt": prompt, "steps": steps, "per_op": per_op, "seq": self.free.last().copied().unwrap_or(0)});
                let engine = self.engine.as_mut().unwrap();
                match if routes { engine.route_stats(&req) } else { engine.trace(&req) } {
                    Ok(report) => self.out.send(&Event::Traced { id, report }),
                    Err(e) => self.error(None, Some(id), ErrorCode::Backend, e.to_string()),
                }
            }
            Request::Chat { req, .. } => self.reject(&req, ErrorCode::BadRequest, "the native engine takes token-level requests"),
            Request::Shutdown => unreachable!("handled by run"),
        }
    }

    fn load(&mut self, plan: &flux_core::plan::Plan, trace: bool) {
        if self.engine.is_some() {
            return self.error(None, None, ErrorCode::LoadFailed, "a model is already loaded; start a new worker");
        }
        let params = plan.backend_params();
        if params.speculation.is_some() {
            return self.error(None, None, ErrorCode::LoadFailed, "speculation is only planned for the llama-server engine");
        }
        let t0 = Instant::now();
        let mut json = serde_json::to_value(&params).expect("params serialize");
        json["trace"] = trace.into();
        match Engine::load(&json) {
            Ok(e) => {
                let info = e.info().unwrap_or_default();
                self.n_ctx_seq = info["n_ctx_seq"].as_u64().unwrap_or(params.n_ctx_seq as u64) as u32;
                self.n_batch = info["n_batch"].as_u64().unwrap_or(params.n_batch as u64) as usize;
                let n_seq = info["n_seq"].as_u64().unwrap_or(params.n_seq as u64) as u32;
                self.free = (0..n_seq as i32).rev().collect();
                self.engine = Some(e);
                self.out.send(&Event::Loaded { load_ms: t0.elapsed().as_secs_f64() * 1e3, n_ctx_seq: self.n_ctx_seq, n_seq, memory: memory_of(&info) });
            }
            Err(e) => self.error(None, None, ErrorCode::LoadFailed, e.to_string()),
        }
    }

    fn prefill(&mut self, req: String, prompt: Vec<i32>, sampling: &Sampling, stop: Vec<String>, max_tokens: u32, render_special: Vec<i32>) {
        if prompt.is_empty() || max_tokens == 0 {
            return self.reject(&req, ErrorCode::BadRequest, "prompt and max_tokens must be non-empty");
        }
        if self.seqs.iter().any(|s| s.req == req) {
            return self.error(Some(&req), None, ErrorCode::BadRequest, "request id already active");
        }
        let need = prompt.len() as u64 + max_tokens as u64;
        if need > self.n_ctx_seq as u64 {
            return self.reject(
                &req,
                ErrorCode::ContextFull,
                format!("prompt ({}) + max_tokens ({max_tokens}) exceeds the planned context of {} per sequence", prompt.len(), self.n_ctx_seq),
            );
        }
        let Some(slot) = self.free.pop() else {
            return self.reject(&req, ErrorCode::Busy, "all planned sequences are in use");
        };
        let engine = self.engine.as_ref().unwrap();
        let mut sampler = match Sampler::new(engine, &serde_json::to_value(sampling).unwrap()) {
            Ok(s) => s,
            Err(e) => {
                self.free.push(slot);
                return self.reject(&req, ErrorCode::BadRequest, e.to_string());
            }
        };
        prompt.iter().for_each(|&t| sampler.accept_prompt(t));
        self.seqs.push(Seq {
            req,
            slot,
            prompt,
            prefilled: 0,
            pos: 0,
            sampler,
            max_tokens,
            emitted: 0,
            credit: 0,
            next: None,
            held: None,
            text: TextStream::new(stop),
            render_special: render_special.into_iter().collect(),
            t_start: Instant::now(),
            paused_sent: false,
        });
    }

    fn step(&mut self) {
        let (mut tokens, mut pos, mut seqid, mut logits) = (vec![], vec![], vec![], vec![]);
        // (sequence index, batch row) pairs whose logits are sampled after the step.
        let mut sample_rows: Vec<(usize, i32)> = vec![];
        let mut decode_rows: Vec<usize> = vec![];
        for (i, s) in self.seqs.iter().enumerate() {
            if s.wants_decode() {
                sample_rows.push((i, tokens.len() as i32));
                decode_rows.push(i);
                tokens.push(s.next.unwrap());
                pos.push(s.pos);
                seqid.push(s.slot);
                logits.push(1i8);
            }
        }
        let budget = self.n_batch.saturating_sub(tokens.len());
        let mut chunk: Option<(usize, usize)> = None;
        if budget > 0 {
            if let Some(i) = self.seqs.iter().position(|s| !s.prefill_done()) {
                let s = &self.seqs[i];
                let take = budget.min(s.prompt.len() - s.prefilled);
                for j in 0..take {
                    let p = s.prefilled + j;
                    tokens.push(s.prompt[p]);
                    pos.push(p as i32);
                    seqid.push(s.slot);
                    logits.push((p + 1 == s.prompt.len()) as i8);
                }
                if s.prefilled + take == s.prompt.len() {
                    sample_rows.push((i, tokens.len() as i32 - 1));
                }
                chunk = Some((i, take));
            }
        }
        if tokens.is_empty() {
            return;
        }

        let t0 = Instant::now();
        let engine = self.engine.as_mut().unwrap();
        if let Err(e) = engine.decode(&tokens, &pos, &seqid, &logits) {
            let mut involved: Vec<usize> = decode_rows.clone();
            involved.extend(chunk.map(|c| c.0));
            involved.sort_unstable();
            involved.dedup();
            for &i in involved.iter().rev() {
                self.error(Some(&self.seqs[i].req.clone()), None, ErrorCode::Backend, e.to_string());
                self.finish(i, FinishReason::Error);
            }
            return;
        }
        self.steps += 1;
        self.decoded += decode_rows.len() as u64;

        for &i in &decode_rows {
            let s = &mut self.seqs[i];
            s.pos += 1;
            s.next = None;
        }
        if let Some((i, take)) = chunk {
            let s = &mut self.seqs[i];
            s.prefilled += take;
            s.pos = s.prefilled as i32;
            self.prefilled += take as u64;
            if s.prefill_done() {
                let ev = Event::Prefilled { req: s.req.clone(), n_prompt: s.prompt.len() as u32, ms: s.t_start.elapsed().as_secs_f64() * 1e3 };
                self.out.send(&ev);
            }
        }
        // Sample every row before any sequence finishes, so indices stay valid.
        let sampled: Vec<(usize, i32)> = sample_rows
            .iter()
            .map(|&(i, row)| {
                let engine = self.engine.as_mut().unwrap();
                (i, self.seqs[i].sampler.sample(engine, row))
            })
            .collect();
        self.step_ms.push_back(t0.elapsed().as_secs_f64() * 1e3);
        if self.step_ms.len() > 1024 {
            self.step_ms.pop_front();
        }
        let mut done: Vec<(usize, FinishReason)> = vec![];
        for (i, tok) in sampled {
            if self.engine.as_ref().unwrap().is_eog(tok) {
                // Counted like llama-server's tokens_predicted: the step that sampled it was real work.
                let s = &mut self.seqs[i];
                s.emitted += 1;
                let t_us = self.start.elapsed().as_micros() as u64;
                self.out.send(&Event::Token { req: s.req.clone(), i: s.emitted - 1, token: tok, text: String::new(), t_us });
                done.push((i, FinishReason::Eog));
            } else {
                self.seqs[i].held = Some(tok);
                if let Some(r) = self.emit_one(i) {
                    done.push((i, r));
                }
            }
        }
        done.sort_by_key(|d| std::cmp::Reverse(d.0));
        for (i, r) in done {
            self.finish(i, r);
        }
    }

    /// Emits a held token if credit allows; returns a finish reason if the sequence is complete.
    fn emit_one(&mut self, i: usize) -> Option<FinishReason> {
        let t_us = self.start.elapsed().as_micros() as u64;
        let engine = self.engine.as_ref().unwrap();
        let s = &mut self.seqs[i];
        let tok = s.held?;
        if s.credit == 0 {
            if !s.paused_sent {
                s.paused_sent = true;
                self.out.send(&Event::Paused { req: s.req.clone() });
            }
            return None;
        }
        s.held = None;
        s.credit -= 1;
        s.emitted += 1;
        s.next = Some(tok);
        let bytes = engine.token_bytes(tok, s.render_special.contains(&tok));
        let (text, stopped) = match s.text.push(&bytes) {
            Pushed::Text(t) => (t, false),
            Pushed::Stop(t) => (t, true),
        };
        self.out.send(&Event::Token { req: s.req.clone(), i: s.emitted - 1, token: tok, text, t_us });
        if stopped {
            Some(FinishReason::Stop)
        } else if s.emitted >= s.max_tokens {
            Some(FinishReason::Length)
        } else {
            if s.credit == 0 {
                s.paused_sent = true;
                self.out.send(&Event::Paused { req: s.req.clone() });
            }
            None
        }
    }

    fn emit_held(&mut self, i: usize) {
        if let Some(r) = self.emit_one(i) {
            self.finish(i, r);
        }
    }

    fn finish(&mut self, i: usize, reason: FinishReason) {
        let mut s = self.seqs.remove(i);
        let tail = if reason == FinishReason::Stop { String::new() } else { s.text.finish() };
        self.out.send(&Event::Finished { req: s.req.clone(), reason, n_prompt: s.prompt.len() as u32, n_decoded: s.emitted, tail });
        self.engine.as_mut().unwrap().seq_clear(s.slot);
        self.free.push(s.slot);
    }

    fn stats(&self) -> WorkerStats {
        let mut ms: Vec<f64> = self.step_ms.iter().copied().collect();
        ms.sort_by(f64::total_cmp);
        let pct = |q: f64| if ms.is_empty() { 0.0 } else { flux_core::stats::percentile_sorted(&ms, q) };
        WorkerStats {
            active: self.seqs.len() as u32,
            steps: self.steps,
            prefilled_tokens: self.prefilled,
            decoded_tokens: self.decoded,
            step_ms_p50: pct(0.5),
            step_ms_p95: pct(0.95),
            memory: self.engine.as_ref().and_then(|e| e.info().ok()).map(|i| memory_of(&i)).unwrap_or_default(),
            rss_bytes: crate::rss_bytes(),
        }
    }
}

fn request_ids(r: &Request) -> (Option<String>, Option<u64>) {
    match r {
        Request::Tokenize { id, .. } | Request::ApplyTemplate { id, .. } | Request::Stats { id } | Request::Trace { id, .. } => (None, Some(*id)),
        Request::Prefill { req, .. } | Request::Decode { req, .. } | Request::Cancel { req } | Request::Chat { req, .. } => (Some(req.clone()), None),
        _ => (None, None),
    }
}

pub fn memory_of(info: &serde_json::Value) -> Vec<DeviceMemory> {
    serde_json::from_value(info["memory"].clone()).unwrap_or_default()
}
