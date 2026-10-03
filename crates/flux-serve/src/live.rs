//! Requests in flight, for live progress: listed by `/flux/stats` and logged by `flux serve` every few
//! seconds while the prompt is processed and while tokens are generated, then once when the request ends.

use crate::generate::Outcome;
use flux_core::protocol::FinishReason;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Progress lines per request come at most this often.
const LOG_EVERY: Duration = Duration::from_secs(5);

struct Entry {
    started: Instant,
    prompt_total: u32,
    prompt_done: u32,
    reused: u32,
    /// Worker time spent on the prompt so far; zero until the first progress report.
    prompt_ms: f64,
    first_token: Option<Instant>,
    generated: u32,
    logged: Instant,
}

impl Entry {
    fn prompt_tps(&self) -> Option<f64> {
        (self.prompt_ms > 0.0).then(|| self.prompt_done.saturating_sub(self.reused) as f64 / (self.prompt_ms / 1e3))
    }

    fn decode_tps(&self) -> Option<f64> {
        let t = self.first_token?.elapsed().as_secs_f64();
        (self.generated > 1 && t > 0.0).then(|| (self.generated - 1) as f64 / t)
    }

    fn due(&mut self) -> bool {
        let due = self.logged.elapsed() >= LOG_EVERY;
        if due {
            self.logged = Instant::now();
        }
        due
    }
}

#[derive(Default)]
pub struct Live {
    entries: Mutex<BTreeMap<String, Entry>>,
}

impl Live {
    pub fn begin(&self, id: &str, prompt_total: u32) {
        let now = Instant::now();
        let e = Entry { started: now, prompt_total, prompt_done: 0, reused: 0, prompt_ms: 0.0, first_token: None, generated: 0, logged: now };
        self.entries.lock().unwrap().insert(id.to_string(), e);
    }

    pub fn prompt(&self, id: &str, done: u32, reused: u32, ms: f64) {
        let mut m = self.entries.lock().unwrap();
        let Some(e) = m.get_mut(id) else { return };
        (e.prompt_done, e.reused, e.prompt_ms) = (done, reused, ms);
        if done < e.prompt_total && e.due() {
            let rate = e.prompt_tps().map_or(String::new(), |r| format!(", {r:.0} tok/s"));
            tracing::info!("{id}: prompt {done} of {} tokens{rate}", e.prompt_total);
        }
    }

    pub fn token(&self, id: &str) {
        let mut m = self.entries.lock().unwrap();
        let Some(e) = m.get_mut(id) else { return };
        e.first_token.get_or_insert_with(Instant::now);
        e.prompt_done = e.prompt_total;
        e.generated += 1;
        if let Some(r) = e.decode_tps().filter(|_| e.due()) {
            tracing::info!("{id}: {} generated, {r:.1} tok/s", e.generated);
        }
    }

    /// Removes the request and logs its totals; the decode rate comes from `o`, the prompt rate from the worker.
    pub fn finish(&self, id: &str, o: &Outcome) {
        let Some(e) = self.entries.lock().unwrap().remove(id) else { return };
        if let Some(err) = &o.error {
            tracing::warn!("{id}: failed after {} generated tokens: {err}", o.n_completion);
            return;
        }
        if o.reason == Some(FinishReason::Cancelled) {
            tracing::info!("{id}: cancelled by the client after {} generated tokens", e.generated);
            return;
        }
        // Chat-level engines report no prompt timing: the prompt took what the decode span leaves.
        let prompt_s = match e.prompt_ms > 0.0 {
            true => e.prompt_ms / 1e3,
            false => {
                let decode_s = o.decode_tps.filter(|&r| r > 0.0).map_or(0.0, |r| o.n_completion.saturating_sub(1) as f64 / r);
                (e.started.elapsed().as_secs_f64() - decode_s).max(1e-3)
            }
        };
        let processed = o.n_prompt.saturating_sub(e.reused);
        let reused = if e.reused > 0 { format!(" ({} reused)", e.reused) } else { String::new() };
        let decode = o.decode_tps.map_or(String::new(), |r| format!(" at {r:.1} tok/s"));
        tracing::info!(
            "{id}: {} prompt tokens{reused} in {prompt_s:.1} s ({:.0} tok/s), {} generated{decode}",
            o.n_prompt,
            processed as f64 / prompt_s,
            o.n_completion
        );
    }

    /// Every request in flight with its phase, progress and current rate.
    pub fn snapshot(&self) -> Value {
        let m = self.entries.lock().unwrap();
        m.iter()
            .map(|(id, e)| {
                json!({
                    "id": id,
                    "phase": if e.first_token.is_some() { "generating" } else { "prompt" },
                    "elapsed_s": e.started.elapsed().as_secs_f64(),
                    "prompt_done": e.prompt_done,
                    "prompt_total": e.prompt_total,
                    "prompt_reused": e.reused,
                    "prompt_tps": e.prompt_tps(),
                    "generated": e.generated,
                    "decode_tps": e.decode_tps(),
                })
            })
            .collect()
    }
}
