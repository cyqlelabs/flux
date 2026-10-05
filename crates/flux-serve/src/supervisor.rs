//! Owns the worker process for the active plan. A failed worker is replaced at a request boundary;
//! a GPU reset is a worker restart, not a transparent continuation.

use anyhow::Result;
use flux_core::plan::Plan;
use flux_core::worker::{Loaded, Timeouts, Worker};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

pub struct Supervisor {
    plan: RwLock<Plan>,
    current: RwLock<(u64, Arc<Worker>, Loaded)>,
    restart: Mutex<Option<std::time::Instant>>,
    timeouts: Timeouts,
    log: PathBuf,
}

async fn start(plan: &Plan, log: &Path, timeouts: &Timeouts) -> Result<(Arc<Worker>, Loaded)> {
    let w = Worker::spawn_with(&plan.engine, Some(log), timeouts.clone()).await?;
    let loaded = w.load(plan).await?;
    Ok((Arc::new(w), loaded))
}

impl Supervisor {
    pub async fn start(plan: Plan, log: PathBuf) -> Result<Supervisor> {
        Self::start_with(plan, log, Timeouts::default()).await
    }

    pub async fn start_with(plan: Plan, log: PathBuf, timeouts: Timeouts) -> Result<Supervisor> {
        let (w, loaded) = start(&plan, &log, &timeouts).await?;
        Ok(Supervisor { plan: RwLock::new(plan), current: RwLock::new((0, w, loaded)), restart: Mutex::new(None), timeouts, log })
    }

    pub async fn ready(&self) -> Result<Arc<Worker>> {
        let (generation, w) = self.current().await;
        if w.is_alive().await {
            Ok(w)
        } else {
            self.restart_if(generation).await
        }
    }

    pub async fn plan(&self) -> Plan {
        self.plan.read().await.clone()
    }

    /// The live worker and its generation (incremented on every replacement).
    pub async fn current(&self) -> (u64, Arc<Worker>) {
        let c = self.current.read().await;
        (c.0, c.1.clone())
    }

    pub async fn loaded(&self) -> Loaded {
        self.current.read().await.2.clone()
    }

    /// Replaces the worker if it is still generation `seen`; concurrent callers share one restart.
    pub async fn restart_if(&self, seen: u64) -> Result<Arc<Worker>> {
        let mut last = self.restart.lock().await;
        {
            let c = self.current.read().await;
            if c.0 != seen {
                return Ok(c.1.clone());
            }
        }
        if let Some(t) = *last {
            anyhow::ensure!(t.elapsed() >= std::time::Duration::from_secs(1), "worker restart cooling down; retry shortly");
        }
        *last = Some(std::time::Instant::now());
        let plan = self.plan().await;
        tracing::warn!(plan = %plan.id, "worker failed; restarting");
        let old = self.current.read().await.1.clone();
        old.kill().await;
        let (w, loaded) = start(&plan, &self.log, &self.timeouts).await?;
        let mut c = self.current.write().await;
        *c = (seen + 1, w.clone(), loaded);
        Ok(w)
    }

    /// Loads another plan; callers drain running requests first. On failure the previous plan is restored.
    pub async fn switch(&self, plan: Plan) -> Result<()> {
        let _g = self.restart.lock().await;
        let previous = self.plan().await;
        self.current.read().await.1.shutdown().await;
        let gen = self.current.read().await.0 + 1;
        match start(&plan, &self.log, &self.timeouts).await {
            Ok((w, loaded)) => {
                *self.current.write().await = (gen, w, loaded);
                *self.plan.write().await = plan;
                Ok(())
            }
            Err(e) => {
                let (w, loaded) = start(&previous, &self.log, &self.timeouts)
                    .await
                    .map_err(|restore| anyhow::anyhow!("new plan failed: {e:#}; previous plan restoration failed: {restore:#}"))?;
                *self.current.write().await = (gen, w, loaded);
                Err(e.context("the new plan failed to load; the previous plan is running again"))
            }
        }
    }

    pub async fn shutdown(&self) {
        self.current.read().await.1.shutdown().await;
    }
}
