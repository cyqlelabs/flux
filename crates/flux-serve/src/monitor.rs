//! Host memory pressure closes admission before the OS starts paging uncontrollably;
//! decode-rate drift against the plan's validated rate is tracked with hysteresis.

use crate::AppState;
use flux_core::plan::DepthMeasurement;
use flux_plan::store::DriftMonitor;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemSample {
    pub available: u64,
    /// Pages swapped in since the previous sample.
    pub swapped_in: u64,
}

fn read_kib(meminfo: &str, key: &str) -> Option<u64> {
    meminfo.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse().ok()
}

fn pswpin() -> u64 {
    std::fs::read_to_string("/proc/vmstat")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("pswpin "))?.split_whitespace().nth(1)?.parse().ok())
        .unwrap_or(0)
}

/// Background apps touching old swapped pages are normal; sustained swap-in above this is not.
const SWAP_IN_PAGES_PER_S: u64 = 256;

/// Closed when available memory is below `min`, or when the system swaps in while less than twice
/// `min` is available (with plenty free, swap-in is old pages being touched, not pressure); reopens
/// only once `min` plus a margin (half of `min`, at most 1 GiB) is available again.
pub fn pressure(s: MemSample, min: u64, currently_closed: bool) -> Option<String> {
    let need = if currently_closed { min + (min / 2).min(1 << 30) } else { min };
    let short = if currently_closed { s.available <= need } else { s.available < need };
    if short {
        Some(format!("host memory pressure: {} MiB available, {} MiB required", s.available >> 20, need >> 20))
    } else if s.swapped_in > SWAP_IN_PAGES_PER_S && s.available < 2 * min {
        Some(format!("host is swapping ({} pages in)", s.swapped_in))
    } else {
        None
    }
}

pub fn spawn(st: Arc<AppState>) -> tokio::task::JoinSet<()> {
    let mut tasks = tokio::task::JoinSet::new();
    let health = st.clone();
    tasks.spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            if let Ok(_guard) = health.replan_lock.try_lock() {
                match health.supervisor.ready().await {
                    Ok(_) => health.admission.unblock("worker"),
                    Err(e) => health.admission.block("worker", &format!("worker unavailable: {e}")),
                }
            }
        }
    });
    tasks.spawn(async move {
        let mut last_swap = pswpin();
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            st.journal.expire();
            let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
            let swap = pswpin();
            let s = MemSample { available: read_kib(&meminfo, "MemAvailable:").unwrap_or(0) * 1024, swapped_in: swap.saturating_sub(last_swap) };
            last_swap = swap;
            let closed_by_us = st.pressure_closed.load(Ordering::Relaxed);
            match pressure(s, st.min_available_mib.load(Ordering::Relaxed) << 20, closed_by_us) {
                Some(reason) => {
                    if !closed_by_us {
                        tracing::warn!("{reason}: admission closed");
                    }
                    st.admission.block("pressure", &reason);
                    st.pressure_closed.store(true, Ordering::Relaxed);
                }
                None if closed_by_us => {
                    tracing::info!("memory pressure cleared: admission open");
                    st.admission.unblock("pressure");
                    st.pressure_closed.store(false, Ordering::Relaxed);
                }
                None => {}
            }
        }
    });
    tasks
}

/// Feeds a finished request's decode rate at `depth` prompt tokens. The rate is compared with what the plan
/// measured at that depth, so long-context requests do not read as drift, and an `agent` request (one carrying
/// tools) with the plan's agent-conversation rate, as agent clients route experts unlike prose. Once drift is
/// established, replans automatically when the configured horizon makes it pay (`should_retune`), otherwise logs once.
pub fn observe_rate(st: &Arc<AppState>, tps: f64, depth: u32, agent: bool) {
    let mut d = st.drift.lock().unwrap();
    d.recent_tps = if d.recent_tps == 0.0 { tps } else { 0.8 * d.recent_tps + 0.2 * tps };
    let at_depth = d.depth.as_ref().map_or(d.baseline_tps, |m| m.decode_tps_at(d.baseline_tps, depth));
    let expected = if agent && d.agent_tps > 0.0 && d.baseline_tps > 0.0 { at_depth * d.agent_tps / d.baseline_tps } else { at_depth };
    let ratio = if expected > 0.0 { tps / expected } else { 1.0 };
    d.recent_ratio = if d.recent_ratio == 0.0 { ratio } else { 0.8 * d.recent_ratio + 0.2 * ratio };
    let baseline = d.baseline_tps;
    let Some(m) = d.monitor.as_mut() else { return };
    if !m.observe(ratio * baseline) || d.drifted {
        return;
    }
    d.drifted = true;
    let horizon = st.cfg.serve.retune_horizon_tokens as f64;
    let recent = d.recent_ratio * d.baseline_tps;
    let pays = horizon > 0.0 && st.replanner.is_some() && flux_plan::store::should_retune(recent, d.baseline_tps, horizon, d.retune_cost_s);
    tracing::warn!(
        tps = recent,
        baseline = d.baseline_tps,
        retune_cost_s = d.retune_cost_s,
        pays,
        "decode rate drifted below the plan's validated rate at the requests' depth"
    );
    if pays {
        let st = st.clone();
        tokio::spawn(async move {
            match crate::replan_now(&st).await {
                Ok(p) => tracing::info!(plan = %p.id, "replanned after drift"),
                Err(e) => tracing::error!("automatic replan failed: {e:#}"),
            }
        });
    }
}

/// What the plan measured: decode on prose and on agent conversations, and on one long prompt.
#[derive(Default)]
pub struct Rates {
    pub prose_tps: Option<f64>,
    pub agent_tps: Option<f64>,
    pub depth: Option<DepthMeasurement>,
}

pub struct Drift {
    pub monitor: Option<DriftMonitor>,
    pub baseline_tps: f64,
    /// Decode rate on the plan's agent conversations, 0 when the plan predates that measurement.
    pub agent_tps: f64,
    /// The plan's long-prompt measurement, which sets the expected rate at depth.
    pub depth: Option<DepthMeasurement>,
    pub recent_tps: f64,
    /// Recent rates over the rate the plan measured at the same depth (1 = as validated).
    pub recent_ratio: f64,
    /// Planning plus loading time, what a retune is expected to cost.
    pub retune_cost_s: f64,
    pub drifted: bool,
}

impl Drift {
    pub fn new(rates: Rates, retune_cost_s: f64) -> Drift {
        Drift {
            monitor: rates.prose_tps.map(|b| DriftMonitor::new(b, 0.15, 5)),
            baseline_tps: rates.prose_tps.unwrap_or(0.0),
            agent_tps: rates.agent_tps.unwrap_or(0.0),
            depth: rates.depth,
            recent_tps: 0.0,
            recent_ratio: 0.0,
            retune_cost_s,
            drifted: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_has_hysteresis() {
        let gib = 1u64 << 30;
        assert!(pressure(MemSample { available: gib, swapped_in: 0 }, 2 * gib, false).is_some());
        assert!(pressure(MemSample { available: 5 * gib / 2, swapped_in: 0 }, 2 * gib, false).is_none());
        assert!(pressure(MemSample { available: 5 * gib / 2, swapped_in: 0 }, 2 * gib, true).is_some());
        assert!(pressure(MemSample { available: 51 * gib, swapped_in: 0 }, 50 * gib, true).is_some());
        assert!(pressure(MemSample { available: 51 * gib + 1, swapped_in: 0 }, 50 * gib, true).is_none());
        assert!(pressure(MemSample { available: 4 * gib, swapped_in: 3 }, 2 * gib, false).is_none());
        assert!(pressure(MemSample { available: 3 * gib, swapped_in: 5000 }, 2 * gib, false).is_some());
        assert!(pressure(MemSample { available: 40 * gib, swapped_in: 5000 }, 2 * gib, false).is_none());
    }
}
