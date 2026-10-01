//! Committed-token journal: what each request has already emitted, so a retry or a worker restart
//! never emits a token twice, and a client can resume a stream it lost.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tokio::sync::watch;

#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Running,
    Done { finish_reason: String },
    Failed { message: String },
}

impl Status {
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Status::Running => serde_json::json!({"state": "running"}),
            Status::Done { finish_reason } => serde_json::json!({"state": "done", "finish_reason": finish_reason}),
            Status::Failed { message } => serde_json::json!({"state": "failed", "message": message}),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub prompt: Vec<i32>,
    pub tokens: Vec<i32>,
    /// Text delta emitted with each token (same length as `tokens`).
    pub texts: Vec<String>,
    pub status: Status,
    pub updated: Instant,
}

pub struct Journal {
    entries: Mutex<HashMap<String, (Entry, watch::Sender<usize>)>>,
    ttl: Duration,
}

impl Journal {
    pub fn new(ttl: Duration) -> Journal {
        Journal { entries: Mutex::new(HashMap::new()), ttl }
    }

    /// Registers a request; `None` if the id is already known (an idempotent retry).
    pub fn begin(&self, id: &str, prompt: Vec<i32>) -> Option<watch::Receiver<usize>> {
        let mut m = self.entries.lock().unwrap();
        let ttl = self.ttl;
        m.retain(|_, (e, _)| e.status == Status::Running || e.updated.elapsed() < ttl);
        if m.contains_key(id) {
            return None;
        }
        let (tx, rx) = watch::channel(0);
        m.insert(id.to_string(), (Entry { prompt, tokens: vec![], texts: vec![], status: Status::Running, updated: Instant::now() }, tx));
        Some(rx)
    }

    pub fn push(&self, id: &str, token: i32, text: &str) {
        if let Some((e, tx)) = self.entries.lock().unwrap().get_mut(id) {
            e.tokens.push(token);
            e.texts.push(text.to_string());
            e.updated = Instant::now();
            let _ = tx.send(e.tokens.len());
        }
    }

    pub fn finish(&self, id: &str, status: Status) {
        if let Some((e, tx)) = self.entries.lock().unwrap().get_mut(id) {
            e.status = status;
            e.updated = Instant::now();
            let n = e.tokens.len();
            let _ = tx.send(n);
        }
    }

    pub fn get(&self, id: &str) -> Option<Entry> {
        self.entries.lock().unwrap().get(id).map(|(e, _)| e.clone())
    }

    pub fn watch(&self, id: &str) -> Option<watch::Receiver<usize>> {
        self.entries.lock().unwrap().get(id).map(|(_, tx)| tx.subscribe())
    }

    pub fn running(&self) -> usize {
        self.entries.lock().unwrap().values().filter(|(e, _)| e.status == Status::Running).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_with_same_id_does_not_restart() {
        let j = Journal::new(Duration::from_secs(60));
        assert!(j.begin("r1", vec![1, 2]).is_some());
        j.push("r1", 5, "a");
        assert!(j.begin("r1", vec![1, 2]).is_none());
        assert_eq!(j.get("r1").unwrap().tokens, vec![5]);
        j.finish("r1", Status::Done { finish_reason: "stop".into() });
        assert_eq!(j.running(), 0);
    }
}
