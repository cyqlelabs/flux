//! Host memory pressure closes admission before the OS starts paging uncontrollably;
//! decode-rate drift against the plan's validated rate is tracked with hysteresis.

use crate::AppState;
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

pub fn spawn(st: Arc<AppState>) {
    tokio::spawn(async move {
        let mut last_swap = pswpin();
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
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
                    st.admission.close(&reason);
                    st.pressure_closed.store(true, Ordering::Relaxed);
                }
                None if closed_by_us => {
                    tracing::info!("memory pressure cleared: admission open");
                    st.admission.open();
                    st.pressure_closed.store(false, Ordering::Relaxed);
                }
                None => {}
            }
        }
    });
}

/// Feeds a finished request's decode rate. Once drift is established, replans automatically when
/// the configured horizon makes it pay (`should_retune`), otherwise logs once.
pub fn observe_rate(st: &Arc<AppState>, tps: f64) {
    let mut d = st.drift.lock().unwrap();
    d.recent_tps = if d.recent_tps == 0.0 { tps } else { 0.8 * d.recent_tps + 0.2 * tps };
    let Some(m) = d.monitor.as_mut() else { return };
    if !m.observe(tps) || d.drifted {
        return;
    }
    d.drifted = true;
    let horizon = st.cfg.serve.retune_horizon_tokens as f64;
    let pays = horizon > 0.0 && st.replanner.is_some() && flux_plan::store::should_retune(d.recent_tps, d.baseline_tps, horizon, d.retune_cost_s);
    tracing::warn!(tps = d.recent_tps, baseline = d.baseline_tps, retune_cost_s = d.retune_cost_s, pays, "decode rate drifted below the plan's validated rate");
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

pub struct Drift {
    pub monitor: Option<DriftMonitor>,
    pub baseline_tps: f64,
    pub recent_tps: f64,
    /// Planning plus loading time, what a retune is expected to cost.
    pub retune_cost_s: f64,
    pub drifted: bool,
}

impl Drift {
    pub fn new(baseline_tps: Option<f64>, retune_cost_s: f64) -> Drift {
        Drift {
            monitor: baseline_tps.map(|b| DriftMonitor::new(b, 0.15, 5)),
            baseline_tps: baseline_tps.unwrap_or(0.0),
            recent_tps: 0.0,
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
