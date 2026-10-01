//! Owns the worker process for the active plan. A failed worker is replaced at a request boundary;
//! a GPU reset is a worker restart, not a transparent continuation.

use anyhow::Result;
use flux_core::plan::Plan;
use flux_core::worker::{Loaded, Worker};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

pub struct Supervisor {
    plan: RwLock<Plan>,
    current: RwLock<(u64, Arc<Worker>, Loaded)>,
    restart: Mutex<()>,
    log: PathBuf,
}

async fn start(plan: &Plan, log: &Path) -> Result<(Arc<Worker>, Loaded)> {
    let w = Worker::spawn(&plan.engine, Some(log)).await?;
    let loaded = w.load(plan).await?;
    Ok((Arc::new(w), loaded))
}

impl Supervisor {
    pub async fn start(plan: Plan, log: PathBuf) -> Result<Supervisor> {
        let (w, loaded) = start(&plan, &log).await?;
        Ok(Supervisor { plan: RwLock::new(plan), current: RwLock::new((0, w, loaded)), restart: Mutex::new(()), log })
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
        let _g = self.restart.lock().await;
        {
            let c = self.current.read().await;
            if c.0 != seen {
                return Ok(c.1.clone());
            }
        }
        let plan = self.plan().await;
        tracing::warn!(plan = %plan.id, "worker failed; restarting");
        let old = self.current.read().await.1.clone();
        old.kill().await;
        let (w, loaded) = start(&plan, &self.log).await?;
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
        match start(&plan, &self.log).await {
            Ok((w, loaded)) => {
                *self.current.write().await = (gen, w, loaded);
                *self.plan.write().await = plan;
                Ok(())
            }
            Err(e) => {
                let (w, loaded) = start(&previous, &self.log).await?;
                *self.current.write().await = (gen, w, loaded);
                Err(e.context("the new plan failed to load; the previous plan is running again"))
            }
        }
    }

    pub async fn shutdown(&self) {
        self.current.read().await.1.shutdown().await;
    }
}
