//! Admission against the plan: at most `concurrency` sequences run, a bounded queue waits,
//! everything else is refused immediately. Under memory pressure or while draining, admission closes.

use std::collections::BTreeMap;
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
    closed: Mutex<BTreeMap<String, String>>,
    pub admitted: AtomicUsize,
    pub rejected: AtomicUsize,
}

impl Admission {
    pub fn new(concurrency: usize, queue_depth: usize) -> Admission {
        Admission {
            slots: Arc::new(Semaphore::new(concurrency)),
            waiting: AtomicUsize::new(0),
            queue_depth,
            closed: Mutex::new(BTreeMap::new()),
            admitted: AtomicUsize::new(0),
            rejected: AtomicUsize::new(0),
        }
    }

    pub async fn admit(&self) -> Result<OwnedSemaphorePermit, Rejection> {
        if let Some(r) = self.closed_reason() {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(Rejection::Closed(r));
        }
        if let Ok(p) = self.slots.clone().try_acquire_owned() {
            return self.accept(p);
        }
        if self.waiting.fetch_add(1, Ordering::SeqCst) >= self.queue_depth {
            self.waiting.fetch_sub(1, Ordering::SeqCst);
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(Rejection::QueueFull);
        }
        let _waiting = Waiting(&self.waiting);
        let p = self.slots.clone().acquire_owned().await;
        let p = p.expect("semaphore never closes");
        self.accept(p)
    }

    fn accept(&self, p: OwnedSemaphorePermit) -> Result<OwnedSemaphorePermit, Rejection> {
        if let Some(reason) = self.closed_reason() {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            return Err(Rejection::Closed(reason));
        }
        self.admitted.fetch_add(1, Ordering::Relaxed);
        Ok(p)
    }

    pub fn close(&self, reason: &str) {
        self.block("manual", reason);
    }

    pub fn open(&self) {
        self.unblock("manual");
    }

    pub fn block(&self, owner: &str, reason: &str) {
        self.closed.lock().unwrap().insert(owner.into(), reason.into());
    }

    pub fn unblock(&self, owner: &str) {
        self.closed.lock().unwrap().remove(owner);
    }

    pub fn blocked_by(&self, owner: &str) -> bool {
        self.closed.lock().unwrap().contains_key(owner)
    }

    pub fn closed_reason(&self) -> Option<String> {
        let reasons = self.closed.lock().unwrap();
        (!reasons.is_empty()).then(|| reasons.values().cloned().collect::<Vec<_>>().join("; "))
    }

    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }

    pub fn in_use(&self, concurrency: usize) -> usize {
        concurrency - self.slots.available_permits()
    }

    /// Waits until every running sequence has finished (admission should be closed first).
    pub async fn drain(&self, concurrency: usize) -> OwnedSemaphorePermit {
        self.slots.clone().acquire_many_owned(concurrency as u32).await.expect("semaphore never closes")
    }
}

struct Waiting<'a>(&'a AtomicUsize);
impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn cancellation_and_closure_release_waiters() {
        let a = Admission::new(1, 1);
        let running = a.admit().await.unwrap();
        {
            let mut queued = Box::pin(a.admit());
            assert!(futures::poll!(&mut queued).is_pending());
        }
        assert_eq!(a.waiting(), 0);
        let mut queued = Box::pin(a.admit());
        assert!(futures::poll!(&mut queued).is_pending());
        a.block("pressure", "memory pressure");
        a.block("replan", "replanning");
        a.unblock("pressure");
        drop(running);
        assert_eq!(queued.await.unwrap_err(), Rejection::Closed("replanning".into()));
        assert_eq!(a.waiting(), 0);
    }

    #[tokio::test]
    async fn concurrent_drains_do_not_split_permits() {
        let a = Admission::new(2, 0);
        let p1 = a.admit().await.unwrap();
        let p2 = a.admit().await.unwrap();
        a.close("draining");
        let mut d1 = Box::pin(a.drain(2));
        let mut d2 = Box::pin(a.drain(2));
        assert!(futures::poll!(&mut d1).is_pending());
        assert!(futures::poll!(&mut d2).is_pending());
        drop((p1, p2));
        drop(d1.await);
        drop(d2.await);
    }

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
