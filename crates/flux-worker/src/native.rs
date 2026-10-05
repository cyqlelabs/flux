//! Native engine: one llama.cpp context, continuous batching across planned sequences.
//! Each step decodes one token for every sequence that has credit, plus one prompt chunk.

use crate::out::Out;
use crate::text::{Pushed, TextStream};
use anyhow::Result;
use flux_core::protocol::{DeviceMemory, ErrorCode, Event, FinishReason, Request, Sampling, WorkerStats, PROTOCOL_VERSION};
use flux_native::{ChatParser, Engine, Sampler};
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Instant;

struct Seq {
    req: String,
    slot: i32,
    prompt: Vec<i32>,
    prefilled: usize,
    /// Prompt tokens the slot already held from its last request.
    reused: usize,
    /// Prompt positions where recurrent models checkpoint their state for later reuse, ascending.
    marks: Vec<usize>,
    /// Next KV position to write.
    pos: i32,
    /// Tokens written to the slot's KV, kept for prompt reuse when the sequence finishes.
    kv: Vec<i32>,
    sampler: Sampler,
    max_tokens: u32,
    emitted: u32,
    credit: u32,
    /// Emitted token not yet written to the KV cache; fed by the next decode step.
    next: Option<i32>,
    /// Sampled token waiting for credit, and the runner-up of its row.
    held: Option<i32>,
    held_alt: Option<i32>,
    /// The request asked for runner-ups (a scan of the whole vocabulary per undrafted token).
    runner_up: bool,
    text: TextStream,
    render_special: HashSet<i32>,
    t_start: Instant,
    paused_sent: bool,
    /// Splits the reply into OpenAI deltas (reasoning, content, tool calls) when the request asked for it.
    parser: Option<ChatParser>,
}

impl Seq {
    /// The deltas `text` completes, and the whole message when `last`. A parser failure drops to raw content.
    fn parse(&mut self, text: &str, last: bool) -> (Vec<Value>, Option<Value>) {
        let Some(p) = self.parser.as_mut() else { return (vec![], None) };
        match p.push(text, last) {
            Ok(mut v) => (serde_json::from_value(v["deltas"].take()).unwrap_or_default(), v.get_mut("message").map(Value::take)),
            Err(e) => {
                eprintln!("{}: chat parser failed, streaming raw content from here: {e}", self.req);
                self.parser = None;
                (if text.is_empty() { vec![] } else { vec![json!({"content": text})] }, None)
            }
        }
    }

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
    min_reply: u32,
    n_batch: usize,
    /// Draft tokens per step when the plan speculates (0 otherwise).
    spec_n_max: usize,
    /// Recurrent state: prompt reuse restores it from checkpoints taken at each sequence's marks.
    recurrent: bool,
    paged: bool,
    seqs: Vec<Seq>,
    free: Vec<i32>,
    /// Tokens each idle slot still holds from its last request, reused by the next prompt sharing them.
    cached: Vec<Vec<i32>>,
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
            min_reply: 4096,
            n_batch: 0,
            spec_n_max: 0,
            recurrent: false,
            paged: false,
            seqs: vec![],
            free: vec![],
            cached: vec![],
            start: Instant::now(),
            steps: 0,
            prefilled: 0,
            decoded: 0,
            step_ms: VecDeque::new(),
        }
    }

    pub fn run(&mut self, rx: Receiver<Request>) -> Result<()> {
        let mut commands = 0;
        loop {
            let busy = self.seqs.iter().any(|s| !s.prefill_done() || s.wants_decode());
            if busy && commands >= 8 {
                self.step();
                commands = 0;
                continue;
            }
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
                Some(r) => {
                    self.handle(r);
                    commands += 1;
                }
                None => {
                    self.step();
                    commands = 0;
                }
            }
        }
    }

    fn error(&self, req: Option<&str>, id: Option<u64>, code: ErrorCode, message: impl Into<String>) {
        self.out.send(&Event::Error { req: req.map(str::to_string), id, code, message: message.into() });
    }

    /// Refuses a request; every Prefill ends with exactly one Finished, even when rejected.
    fn reject(&self, req: &str, code: ErrorCode, message: impl Into<String>) {
        self.error(Some(req), None, code, message);
        self.out.send(&Event::Finished {
            req: req.into(),
            reason: FinishReason::Error,
            n_prompt: 0,
            n_decoded: 0,
            tail: String::new(),
            deltas: vec![],
            message: None,
        });
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
            Request::Admission { host_reserve_bytes, min_reply } => {
                self.min_reply = min_reply.max(1);
                self.engine.as_mut().unwrap().host_reserve(host_reserve_bytes.saturating_add(flux_core::config::reopen_margin(host_reserve_bytes)));
            }
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
                        parser: v.get("parser").cloned(),
                        checkpoints: serde_json::from_value(v["checkpoints"].clone()).unwrap_or_default(),
                    }),
                    Err(e) => self.error(None, Some(id), ErrorCode::BadRequest, e.to_string()),
                }
            }
            Request::Prefill { req, prompt, sampling, stop, max_tokens, render_special, chat, chat_prefix, checkpoints } => {
                let parser = chat.and_then(|spec| match ChatParser::new(&spec) {
                    Ok(mut p) => {
                        if !chat_prefix.is_empty() && p.push(&chat_prefix, false).is_err() {
                            return None;
                        }
                        Some(p)
                    }
                    Err(e) => {
                        eprintln!("{req}: no chat parser, streaming raw content: {e}");
                        None
                    }
                });
                self.prefill(req, prompt, &sampling, stop, max_tokens, render_special, parser, checkpoints)
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
                let seq = self.free.last().copied().unwrap_or(0);
                let req = serde_json::json!({"prompt": prompt, "steps": steps, "per_op": per_op, "seq": seq});
                let engine = self.engine.as_mut().unwrap();
                engine.seq_clear(seq);
                if let Some(c) = self.cached.get_mut(seq as usize) {
                    c.clear();
                }
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
        let t0 = Instant::now();
        let mut json = serde_json::to_value(&params).expect("params serialize");
        json["trace"] = trace.into();
        match Engine::load(&json) {
            Ok(e) => {
                let info = e.info().unwrap_or_default();
                self.n_ctx_seq = info["n_ctx_seq"].as_u64().unwrap_or(params.n_ctx_seq as u64) as u32;
                self.n_batch = info["n_batch"].as_u64().unwrap_or(params.n_batch as u64) as usize;
                self.spec_n_max = info["spec_n_max"].as_u64().unwrap_or(0) as usize;
                self.recurrent = info["recurrent"].as_bool().unwrap_or(false);
                self.paged = params.kv_paging.is_some();
                let n_seq = info["n_seq"].as_u64().unwrap_or(params.n_seq as u64) as u32;
                self.free = (0..n_seq as i32).rev().collect();
                self.cached = vec![vec![]; n_seq as usize];
                self.engine = Some(e);
                self.out.send(&Event::Loaded { load_ms: t0.elapsed().as_secs_f64() * 1e3, n_ctx_seq: self.n_ctx_seq, n_seq, memory: memory_of(&info) });
            }
            Err(e) => self.error(None, None, ErrorCode::LoadFailed, e.to_string()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &mut self,
        req: String,
        prompt: Vec<i32>,
        sampling: &Sampling,
        stop: Vec<String>,
        max_tokens: u32,
        render_special: Vec<i32>,
        parser: Option<ChatParser>,
        checkpoints: Vec<u32>,
    ) {
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
        // The idle slot holding the longest prefix of this prompt; at least the last prompt token is decoded
        // again, since its logits pick the first output token.
        let common = |c: &Vec<i32>| c.iter().zip(&prompt).take_while(|(a, b)| a == b).count().min(prompt.len() - 1);
        let Some(at) = (0..self.free.len()).max_by_key(|&k| (common(&self.cached[self.free[k] as usize]), k)) else {
            return self.reject(&req, ErrorCode::Busy, "all planned sequences are in use");
        };
        let slot = self.free.remove(at);
        let reuse = common(&self.cached[slot as usize]);
        let kept = self.engine.as_mut().unwrap().seq_keep(slot, reuse);
        self.cached[slot as usize].clear();
        let reserve = (prompt.len() as u32).saturating_add(max_tokens.min(self.min_reply)).saturating_add(self.spec_n_max as u32).min(self.n_ctx_seq);
        if !self.engine.as_mut().unwrap().seq_reserve(slot, reserve) {
            let mut low = kept as u32;
            let mut high = reserve;
            while low + 1 < high {
                let mid = low + (high - low) / 2;
                if self.engine.as_mut().unwrap().seq_reserve(slot, mid) {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            self.engine.as_mut().unwrap().seq_clear(slot);
            self.engine.as_mut().unwrap().seq_release(slot);
            self.free.push(slot);
            return self.reject(
                &req,
                ErrorCode::ContextFull,
                format!("This model's maximum context length is {low} tokens. However, your messages resulted in {} tokens. The available memory cannot reserve the prompt and reply floor.", prompt.len()),
            );
        }
        let engine = self.engine.as_ref().unwrap();
        let runner_up = sampling.runner_up.unwrap_or(false);
        let mut sampler = match Sampler::new(engine, &serde_json::to_value(sampling).unwrap()) {
            Ok(s) => s,
            Err(e) => {
                self.engine.as_mut().unwrap().seq_release(slot);
                self.free.push(slot);
                return self.reject(&req, ErrorCode::BadRequest, e.to_string());
            }
        };
        prompt.iter().for_each(|&t| sampler.accept_prompt(t));
        // The template's boundaries (end of the system prompt, end of the last message) and one short of the end: a
        // side request reuses the first, a new user turn the second, a continuation after tool results the third.
        let mut marks: Vec<usize> = checkpoints.iter().map(|&c| c as usize).chain([prompt.len() - 1]).filter(|&m| m > kept && m < prompt.len()).collect();
        marks.sort_unstable();
        marks.dedup();
        self.seqs.push(Seq {
            req,
            slot,
            kv: prompt[..kept].to_vec(),
            prompt,
            prefilled: kept,
            reused: kept,
            marks,
            pos: kept as i32,
            sampler,
            max_tokens,
            emitted: 0,
            credit: 0,
            next: None,
            held: None,
            held_alt: None,
            runner_up,
            text: TextStream::new(stop),
            render_special: render_special.into_iter().collect(),
            t_start: Instant::now(),
            paused_sent: false,
            parser,
        });
    }

    fn step(&mut self) {
        let mut undrafted = HashSet::new();
        // Reserve before either drafter or target runs. Admission already holds the reply floor.
        for i in (0..self.seqs.len()).rev() {
            let s = &self.seqs[i];
            if s.wants_decode() {
                let cells = (s.pos as u32).saturating_add(1 + self.spec_n_max as u32).min(self.n_ctx_seq);
                if !self.engine.as_mut().unwrap().seq_reserve(s.slot, cells) {
                    if self.engine.as_mut().unwrap().seq_reserve(s.slot, (s.pos as u32).saturating_add(1).min(self.n_ctx_seq)) {
                        undrafted.insert(s.slot);
                    } else {
                        self.finish(i, FinishReason::Length);
                    }
                }
            }
        }
        // Rotate batch priority so a batch smaller than the active set cannot starve later slots.
        if self.seqs.len() > 1 {
            self.seqs.rotate_left(1);
        }
        // Separate paged streams keep their own depth: a mixed graph reads every stream through the deepest
        // stream's padded KV prefix, so only decoding streams within twice each other's depth share a step.
        let selected = self.paged.then(|| self.seqs.iter().position(|s| s.wants_decode() || !s.prefill_done())).flatten();
        let lead = selected.filter(|&i| self.seqs[i].wants_decode()).map(|i| self.seqs[i].pos.max(1));
        let joins = |s: &Seq| lead.is_some_and(|d| s.pos.max(1) <= 2 * d && d <= 2 * s.pos.max(1));
        let (mut tokens, mut pos, mut seqid, mut logits) = (vec![], vec![], vec![], vec![]);
        // (sequence index, batch row, draft) for the rows sampled after the step; the draft follows the row.
        let mut sample_rows: Vec<(usize, i32, Vec<i32>)> = vec![];
        let mut decode_rows: Vec<usize> = vec![];
        for i in 0..self.seqs.len() {
            if tokens.len() == self.n_batch {
                break;
            }
            if !self.seqs[i].wants_decode() {
                continue;
            }
            if self.paged && selected != Some(i) && !joins(&self.seqs[i]) {
                continue;
            }
            let s = &self.seqs[i];
            // Every drafted token must be emittable: within credit, max_tokens and the planned context.
            let room = (s.credit as usize).min((s.max_tokens - s.emitted) as usize).min((self.n_ctx_seq as i32 - s.pos) as usize);
            let n_draft =
                if undrafted.contains(&s.slot) { 0 } else { self.spec_n_max.min(room.saturating_sub(1)).min(self.n_batch.saturating_sub(tokens.len() + 1)) };
            let draft = if n_draft > 0 { self.engine.as_mut().unwrap().spec_draft(s.slot, s.pos, s.next.unwrap(), &s.kv, n_draft) } else { vec![] };
            sample_rows.push((i, tokens.len() as i32, draft.clone()));
            decode_rows.push(i);
            for (k, &t) in std::iter::once(&s.next.unwrap()).chain(&draft).enumerate() {
                tokens.push(t);
                pos.push(s.pos + k as i32);
                seqid.push(s.slot);
                logits.push(1i8);
            }
        }
        let budget = self.n_batch.saturating_sub(tokens.len());
        let mut chunk: Option<(usize, usize)> = None;
        if budget > 0 {
            if let Some(i) = self.seqs.iter().enumerate().position(|(i, s)| !s.prefill_done() && (!self.paged || selected == Some(i))) {
                let s = &self.seqs[i];
                // Recurrent models stop at each mark first: the state there is checkpointed for reuse.
                let end = if self.recurrent { s.marks.iter().copied().find(|&m| m > s.prefilled).unwrap_or(s.prompt.len()) } else { s.prompt.len() };
                let take = budget.min(end - s.prefilled);
                for j in 0..take {
                    let p = s.prefilled + j;
                    tokens.push(s.prompt[p]);
                    pos.push(p as i32);
                    seqid.push(s.slot);
                    logits.push((p + 1 == s.prompt.len()) as i8);
                }
                if s.prefilled + take == s.prompt.len() {
                    sample_rows.push((i, tokens.len() as i32 - 1, vec![]));
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

        for &i in &decode_rows {
            let s = &mut self.seqs[i];
            s.pos += 1;
            s.kv.extend(s.next.take());
        }
        if let Some((i, take)) = chunk {
            let s = &mut self.seqs[i];
            s.kv.extend_from_slice(&s.prompt[s.prefilled..s.prefilled + take]);
            s.prefilled += take;
            s.pos = s.prefilled as i32;
            self.prefilled += take as u64;
            if self.recurrent && s.marks.contains(&s.prefilled) {
                let slot = s.slot;
                if !self.engine.as_mut().unwrap().seq_checkpoint(slot) {
                    eprintln!("checkpoint of slot {slot} failed: its prompt will not be reused");
                }
            }
            let (req, ms, reused) = (s.req.clone(), s.t_start.elapsed().as_secs_f64() * 1e3, s.reused as u32);
            let ev = match s.prefill_done() {
                true => Event::Prefilled { req, n_prompt: s.prompt.len() as u32, ms, reused },
                false => Event::Prefilling { req, done: s.prefilled as u32, total: s.prompt.len() as u32, reused, ms },
            };
            self.out.send(&ev);
        }
        // Sample every row before any sequence finishes, so indices stay valid. A drafted row keeps the
        // accepted drafts plus the token sampled after them, and the sequence drops the rejected rest.
        let mut sampled: Vec<(usize, Vec<i32>, Option<i32>)> = vec![];
        let mut done: Vec<(usize, FinishReason)> = vec![];
        for (i, row, draft) in sample_rows {
            let engine = self.engine.as_mut().unwrap();
            let s = &mut self.seqs[i];
            let toks = match if draft.is_empty() { s.sampler.sample(engine, row).map(|t| vec![t]) } else { s.sampler.sample_draft(engine, row, &draft) } {
                Ok(t) => t,
                Err(e) => {
                    let req = s.req.clone();
                    self.error(Some(&req), None, ErrorCode::Backend, e.to_string());
                    done.push((i, FinishReason::Error));
                    continue;
                }
            };
            if draft.is_empty() {
                let alt = if s.runner_up { engine.runner_up(row, toks[0]) } else { None };
                sampled.push((i, toks, alt));
                continue;
            }
            // s.pos already moved past the verified row; accepted drafts extend the sequence.
            s.pos += toks.len() as i32 - 1;
            s.kv.extend_from_slice(&draft[..toks.len() - 1]);
            if let Err(e) = engine.spec_accept(s.slot, s.pos, toks.len() - 1) {
                let req = s.req.clone();
                self.error(Some(&req), None, ErrorCode::Backend, e.to_string());
                done.push((i, FinishReason::Error));
                continue;
            }
            sampled.push((i, toks, None));
        }
        self.decoded += sampled.iter().filter(|(i, ..)| decode_rows.contains(i)).map(|(_, t, _)| t.len() as u64).sum::<u64>();
        self.step_ms.push_back(t0.elapsed().as_secs_f64() * 1e3);
        if self.step_ms.len() > 1024 {
            self.step_ms.pop_front();
        }
        for (i, toks, alt) in sampled {
            for tok in toks {
                if self.engine.as_ref().unwrap().is_eog(tok) {
                    // Counted like llama-server's tokens_predicted: the step that sampled it was real work.
                    let s = &mut self.seqs[i];
                    s.emitted += 1;
                    let t_us = self.start.elapsed().as_micros() as u64;
                    self.out.send(&Event::Token { req: s.req.clone(), i: s.emitted - 1, token: tok, text: String::new(), t_us, alt, deltas: vec![] });
                    done.push((i, FinishReason::Eog));
                    break;
                }
                self.seqs[i].held = Some(tok);
                self.seqs[i].held_alt = alt;
                if let Some(r) = self.emit_one(i) {
                    done.push((i, r));
                    break;
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
        let deltas = if text.is_empty() { vec![] } else { s.parse(&text, false).0 };
        self.out.send(&Event::Token { req: s.req.clone(), i: s.emitted - 1, token: tok, text, t_us, alt: s.held_alt.take(), deltas });
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
        let (deltas, message) = s.parse(&tail, true);
        self.out.send(&Event::Finished { req: s.req.clone(), reason, n_prompt: s.prompt.len() as u32, n_decoded: s.emitted, tail, deltas, message });
        // A failed step may leave the cache inconsistent with the tokens; anything else is kept for reuse.
        if reason == FinishReason::Error {
            self.engine.as_mut().unwrap().seq_clear(s.slot);
        } else {
            self.cached[s.slot as usize] = std::mem::take(&mut s.kv);
        }
        self.engine.as_mut().unwrap().seq_release(s.slot);
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
            kv_pages: self.engine.as_ref().and_then(|e| e.info().ok()).and_then(|i| i.get("kv_pages").cloned()).unwrap_or_default(),
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
