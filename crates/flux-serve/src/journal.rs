//! Bounded retained output, including text tails, structured deltas, and terminal metadata.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
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
    /// Text fragments, including tails that do not have a corresponding token.
    pub texts: Vec<String>,
    pub status: Status,
    pub updated: Instant,
    pub events: Vec<Arc<str>>,
    bytes: usize,
}

pub struct Journal {
    entries: Mutex<Entries>,
    ttl: Duration,
    max_bytes: usize,
}

#[derive(Default)]
struct Entries {
    map: HashMap<String, (Entry, watch::Sender<usize>)>,
    bytes: usize,
}

impl Entries {
    fn remove(&mut self, id: &str) {
        if let Some((e, _)) = self.map.remove(id) {
            self.bytes -= e.bytes;
        }
    }

    fn make_room(&mut self, bytes: usize, limit: usize) -> bool {
        while self.bytes.saturating_add(bytes) > limit || self.map.len() >= 4096 {
            let oldest = self.map.iter().filter(|(_, (e, _))| e.status != Status::Running).min_by_key(|(_, (e, _))| e.updated).map(|(id, _)| id.clone());
            let Some(id) = oldest else { return false };
            self.remove(&id);
        }
        true
    }
}

impl Journal {
    pub fn new(ttl: Duration) -> Journal {
        Self::with_limit(ttl, 256 << 20)
    }

    pub fn with_limit(ttl: Duration, max_bytes: usize) -> Journal {
        Journal { entries: Mutex::new(Entries::default()), ttl, max_bytes }
    }

    /// Registers a request; `None` if the id is already known (an idempotent retry).
    pub fn begin(&self, id: &str, prompt: Vec<i32>) -> Option<watch::Receiver<usize>> {
        self.try_begin(id, prompt).ok()
    }

    pub fn try_begin(&self, id: &str, prompt: Vec<i32>) -> Result<watch::Receiver<usize>, &'static str> {
        let mut m = self.entries.lock().unwrap();
        self.expire_one(&mut m, id);
        if m.map.contains_key(id) {
            return Err("duplicate_request");
        }
        let bytes = prompt.capacity() * 4 + id.len() * 2 + 512;
        if !m.make_room(bytes, self.max_bytes) {
            return Err("journal_full");
        }
        let (tx, rx) = watch::channel(0);
        m.map.insert(
            id.to_string(),
            (Entry { prompt, tokens: vec![], texts: vec![], status: Status::Running, updated: Instant::now(), events: vec![], bytes }, tx),
        );
        m.bytes += bytes;
        Ok(rx)
    }

    pub fn push(&self, id: &str, token: i32, text: &str) -> bool {
        self.record(id, Some(token), text, &[], None)
    }

    pub fn record(&self, id: &str, token: Option<i32>, text: &str, deltas: &[serde_json::Value], chunk: Option<&serde_json::Value>) -> bool {
        self.record_value(id, token, text, serde_json::json!({"token":token,"text":text,"deltas":deltas,"chunk":chunk}))
    }

    pub fn terminal(&self, id: &str, value: serde_json::Value) -> bool {
        self.record_value(id, None, "", serde_json::json!({"terminal":value}))
    }

    fn record_value(&self, id: &str, token: Option<i32>, text: &str, mut value: serde_json::Value) -> bool {
        let mut m = self.entries.lock().unwrap();
        let Some((e, _)) = m.map.get(id) else { return false };
        if e.status != Status::Running {
            return false;
        }
        value["version"] = 1.into();
        value["index"] = e.events.len().into();
        let event = value.to_string();
        let bytes = event.len() + 2 * text.len() + 256;
        if !m.make_room(bytes, self.max_bytes) {
            return false;
        }
        let (e, tx) = m.map.get_mut(id).unwrap();
        if let Some(t) = token {
            e.tokens.push(t);
        }
        e.texts.push(text.into());
        e.events.push(Arc::from(event));
        e.bytes += bytes;
        e.updated = Instant::now();
        tx.send_replace(e.events.len());
        m.bytes += bytes;
        true
    }

    pub fn finish(&self, id: &str, status: Status) {
        if let Some((e, tx)) = self.entries.lock().unwrap().map.get_mut(id) {
            e.status = status;
            e.updated = Instant::now();
            tx.send_replace(e.events.len());
        }
    }

    pub fn get(&self, id: &str) -> Option<Entry> {
        let mut m = self.entries.lock().unwrap();
        self.expire_one(&mut m, id);
        m.map.get(id).map(|(e, _)| e.clone())
    }

    /// Reads only the requested event; prompt and prior output are never cloned.
    pub fn event(&self, id: &str, index: usize) -> Option<(Option<Arc<str>>, Status)> {
        let mut m = self.entries.lock().unwrap();
        self.expire_one(&mut m, id);
        m.map.get(id).map(|(e, _)| (e.events.get(index).cloned(), e.status.clone()))
    }

    fn expire_one(&self, m: &mut Entries, id: &str) {
        if m.map.get(id).is_some_and(|(e, _)| e.status != Status::Running && e.updated.elapsed() >= self.ttl) {
            m.remove(id);
        }
    }

    pub fn expire(&self) {
        let mut m = self.entries.lock().unwrap();
        let old: Vec<String> =
            m.map.iter().filter(|(_, (e, _))| e.status != Status::Running && e.updated.elapsed() >= self.ttl).map(|(id, _)| id.clone()).collect();
        for id in old {
            m.remove(&id);
        }
    }

    pub fn watch(&self, id: &str) -> Option<watch::Receiver<usize>> {
        let mut m = self.entries.lock().unwrap();
        self.expire_one(&mut m, id);
        m.map.get(id).map(|(_, tx)| tx.subscribe())
    }

    pub fn running(&self) -> usize {
        self.entries.lock().unwrap().map.values().filter(|(e, _)| e.status == Status::Running).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_retention_and_constant_size_event_reads() {
        let j = Journal::with_limit(Duration::from_secs(60), 2048);
        j.begin("a", vec![1]).unwrap();
        assert!(j.record("a", Some(2), "hello", &[], None));
        assert!(j.record("a", None, " tail", &[], None));
        let (event, _) = j.event("a", 1).unwrap();
        assert!(event.unwrap().contains(" tail"));
        assert!(!j.push("a", 3, &"x".repeat(2048)));
        assert_eq!(j.get("a").unwrap().tokens, vec![2]);
        j.finish("a", Status::Done { finish_reason: "length".into() });
        assert!(j.try_begin("b", vec![0; 300]).is_ok());
        assert!(j.get("a").is_none());
    }

    #[test]
    fn ttl_applies_without_new_requests() {
        let j = Journal::new(Duration::ZERO);
        j.begin("a", vec![]).unwrap();
        j.finish("a", Status::Done { finish_reason: "stop".into() });
        assert!(j.get("a").is_none());
        assert!(j.watch("a").is_none());
    }

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
