//! Admission against the plan: at most `concurrency` sequences run, a bounded queue waits,
//! everything else is refused immediately. Under memory pressure or while draining, admission closes.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, Clone, PartialEq)]
pub enum Rejection {
    QueueFull,
    Closed(String),
}

pub struct Admission {
    slots: Arc<Semaphore>,
    waiting: AtomicUsize,
    queue_depth: usize,
    closed: Mutex<Option<String>>,
    pub admitted: AtomicUsize,
    pub rejected: AtomicUsize,
}

impl Admission {
    pub fn new(concurrency: usize, queue_depth: usize) -> Admission {
        Admission {
            slots: Arc::new(Semaphore::new(concurrency)),
            waiting: AtomicUsize::new(0),
            queue_depth,
            closed: Mutex::new(None),
            admitted: AtomicUsize::new(0),
            rejected: AtomicUsize::new(0),
        }
    }

    pub async fn admit(&self) -> Result<OwnedSemaphorePermit, Rejection> {
        if let Some(r) = self.closed.lock().unwrap().clone() {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(Rejection::Closed(r));
        }
        if let Ok(p) = self.slots.clone().try_acquire_owned() {
            self.admitted.fetch_add(1, Ordering::Relaxed);
            return Ok(p);
        }
        if self.waiting.fetch_add(1, Ordering::SeqCst) >= self.queue_depth {
            self.waiting.fetch_sub(1, Ordering::SeqCst);
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(Rejection::QueueFull);
        }
        let p = self.slots.clone().acquire_owned().await;
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        let p = p.expect("semaphore never closes");
        self.admitted.fetch_add(1, Ordering::Relaxed);
        Ok(p)
    }

    pub fn close(&self, reason: &str) {
        *self.closed.lock().unwrap() = Some(reason.to_string());
    }

    pub fn open(&self) {
        *self.closed.lock().unwrap() = None;
    }

    pub fn closed_reason(&self) -> Option<String> {
        self.closed.lock().unwrap().clone()
    }

    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }

    pub fn in_use(&self, concurrency: usize) -> usize {
        concurrency - self.slots.available_permits()
    }

    /// Waits until every running sequence has finished (admission should be closed first).
    pub async fn drain(&self, concurrency: usize) -> Vec<OwnedSemaphorePermit> {
        let mut held = vec![];
        for _ in 0..concurrency {
            held.push(self.slots.clone().acquire_owned().await.expect("semaphore never closes"));
        }
        held
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bounded_queue_and_closing() {
        let a = Arc::new(Admission::new(1, 1));
        let p1 = a.admit().await.unwrap();
        let a2 = a.clone();
        let queued = tokio::spawn(async move { a2.admit().await });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(a.admit().await.unwrap_err(), Rejection::QueueFull);
        drop(p1);
        assert!(queued.await.unwrap().is_ok());
        a.close("memory pressure");
        assert_eq!(a.admit().await.unwrap_err(), Rejection::Closed("memory pressure".into()));
    }
}
