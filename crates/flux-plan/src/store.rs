//! Saved plans keyed by profile: model hashes, hardware topology, backend build, driver,
//! context bucket and concurrency. A changed key means reprofiling, never silent reuse.

use anyhow::Result;
use flux_core::fsutil::{read_json, write_json_atomic};
use flux_core::plan::{Plan, ProfileKey};
use std::path::{Path, PathBuf};

pub struct PlanStore {
    dir: PathBuf,
}

impl PlanStore {
    pub fn new(dir: &Path) -> PlanStore {
        PlanStore { dir: dir.to_path_buf() }
    }

    fn path(&self, key: &ProfileKey) -> PathBuf {
        self.dir.join(format!("{}.json", key.digest()))
    }

    pub fn save(&self, plan: &Plan) -> Result<PathBuf> {
        let p = self.path(&plan.key);
        write_json_atomic(&p, plan)?;
        Ok(p)
    }

    /// The saved plan for exactly this profile key, if any.
    pub fn lookup(&self, key: &ProfileKey) -> Option<Plan> {
        read_json::<Plan>(&self.path(key)).ok().filter(|p| &p.key == key)
    }

    pub fn by_id(&self, id: &str) -> Option<Plan> {
        self.list().into_iter().find(|p| p.id == id || p.id.starts_with(id))
    }

    pub fn list(&self) -> Vec<Plan> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return vec![] };
        let mut v: Vec<Plan> = rd.filter_map(|e| read_json(&e.ok()?.path()).ok()).collect();
        v.sort_by_key(|p: &Plan| std::cmp::Reverse(p.created));
        v
    }
}

/// Retune only when the expected time saved over the remaining work exceeds the cost of retuning
/// (including reloading or relocating state).
pub fn should_retune(current_tps: f64, expected_tps: f64, remaining_tokens: f64, retune_cost_s: f64) -> bool {
    if current_tps <= 0.0 || expected_tps <= current_tps {
        return false;
    }
    let saved_s = remaining_tokens / current_tps - remaining_tokens / expected_tps;
    saved_s > retune_cost_s
}

/// Rolling decode-rate monitor with hysteresis: drift is reported only after the rate stays below
/// `(1 - drop)` of the plan's validated rate for `windows` consecutive windows.
pub struct DriftMonitor {
    baseline_tps: f64,
    drop: f64,
    windows: u32,
    below: u32,
}

impl DriftMonitor {
    pub fn new(baseline_tps: f64, drop: f64, windows: u32) -> DriftMonitor {
        DriftMonitor { baseline_tps, drop, windows, below: 0 }
    }

    /// Feeds one window's measured rate; true when drift is established.
    pub fn observe(&mut self, tps: f64) -> bool {
        if tps < self.baseline_tps * (1.0 - self.drop) {
            self.below += 1;
        } else {
            self.below = 0;
        }
        self.below >= self.windows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retune_pays_only_when_savings_exceed_cost() {
        // 10k tokens at 20 → 25 tok/s saves 100 s.
        assert!(should_retune(20.0, 25.0, 10_000.0, 60.0));
        assert!(!should_retune(20.0, 25.0, 10_000.0, 120.0));
        assert!(!should_retune(25.0, 20.0, 1e9, 0.0));
    }

    #[test]
    fn drift_needs_consecutive_windows() {
        let mut d = DriftMonitor::new(50.0, 0.15, 3);
        assert!(!d.observe(40.0));
        assert!(!d.observe(41.0));
        assert!(!d.observe(48.0));
        assert!(!d.observe(40.0));
        assert!(!d.observe(40.0));
        assert!(d.observe(40.0));
    }
}
