use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellTrials {
    pub cell: String,
    /// Paired by index: `None` means that run failed.
    pub flux: Vec<Option<f64>>,
    pub baseline: Vec<Option<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellSpeedup {
    pub cell: String,
    pub speedup: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub pairs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CellStatus {
    Compared,
    FluxOnly,
    BaselineOnly,
    Unsupported,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedGate {
    pub geomean: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub cells: Vec<CellSpeedup>,
    pub statuses: Vec<(String, CellStatus)>,
    pub slower_than_3pct: Vec<String>,
    pub flux_failed: Vec<String>,
    pub pass: bool,
    pub reasons: Vec<String>,
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn paired_logs(c: &CellTrials) -> Vec<f64> {
    c.flux
        .iter()
        .zip(&c.baseline)
        .filter_map(|(flux, baseline)| match (flux, baseline) {
            (Some(flux), Some(baseline)) => {
                let ratio = flux / baseline;
                Some(if ratio.is_finite() && ratio > 0.0 { ratio.ln() } else { flux.ln() - baseline.ln() })
            }
            _ => None,
        })
        .collect()
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let position = quantile * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64)
}

// Each cell has equal weight, regardless of its number of successful pairs.
fn bootstrap_ci(cells: &[Vec<f64>], resamples: usize, seed: u64) -> (f64, f64) {
    if cells.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    if cells.iter().all(|pairs| pairs.len() == 1) {
        let point = mean(&cells.iter().map(|pairs| pairs[0]).collect::<Vec<_>>()).exp();
        return (point, point);
    }
    if resamples == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut total = 0.0;
        for pairs in cells {
            let sum = (0..pairs.len()).map(|_| pairs[rng.random_range(0..pairs.len())]).sum::<f64>();
            total += sum / pairs.len() as f64;
        }
        samples.push((total / cells.len() as f64).exp());
    }
    samples.sort_by(f64::total_cmp);
    (percentile(&samples, 0.025), percentile(&samples, 0.975))
}

/// Geomean of paired ratios, with a 95% percentile bootstrap CI.
/// One pair gives a point CI; zero resamples with multiple pairs gives a NaN CI.
pub fn cell_speedup(c: &CellTrials, resamples: usize, seed: u64) -> Option<CellSpeedup> {
    let logs = paired_logs(c);
    if logs.is_empty() {
        return None;
    }
    let speedup = mean(&logs).exp();
    let pairs = logs.len();
    let (ci_low, ci_high) = bootstrap_ci(&[logs], resamples, seed);
    Some(CellSpeedup { cell: c.cell.clone(), speedup, ci_low, ci_high, pairs })
}

/// Resamples paired trials independently within each cell, then equally weights
/// cell geomeans. CI endpoints are interpolated 2.5/97.5 percentiles.
/// Any baseline success paired with a flux failure fails the gate.
pub fn speed_gate(cells: &[CellTrials], resamples: usize, seed: u64) -> SpeedGate {
    let mut compared = Vec::new();
    let mut logs = Vec::new();
    let mut statuses = Vec::new();
    let mut slower_than_3pct = Vec::new();
    let mut flux_failed = Vec::new();
    let mut reasons = Vec::new();

    for (index, cell) in cells.iter().enumerate() {
        let flux_succeeded = cell.flux.iter().any(Option::is_some);
        let baseline_succeeded = cell.baseline.iter().any(Option::is_some);
        let status = match (flux_succeeded, baseline_succeeded) {
            (true, true) => CellStatus::Compared,
            (true, false) => CellStatus::FluxOnly,
            (false, true) => CellStatus::BaselineOnly,
            (false, false) => CellStatus::Unsupported,
        };
        statuses.push((cell.cell.clone(), status));

        let failures = cell.baseline.iter().enumerate().filter(|(i, baseline)| baseline.is_some() && cell.flux.get(*i).is_none_or(Option::is_none)).count();
        if failures > 0 {
            flux_failed.push(cell.cell.clone());
            reasons.push(format!("Cell {}: flux failed in {failures} trials where baseline succeeded; requires 0.", cell.cell));
        }
        if cell.flux.len() != cell.baseline.len() {
            reasons.push(format!(
                "Cell {} has mismatched trial counts: flux {}, baseline {}; requires equal counts.",
                cell.cell,
                cell.flux.len(),
                cell.baseline.len()
            ));
        }
        if cell.flux.iter().chain(&cell.baseline).flatten().any(|value| !value.is_finite() || *value <= 0.0) {
            reasons.push(format!("Cell {} has invalid measurements; requires finite metrics greater than 0.", cell.cell));
        }

        if let Some(result) = cell_speedup(cell, resamples, seed.wrapping_add(index as u64)) {
            if result.speedup < 0.97 {
                slower_than_3pct.push(cell.cell.clone());
                reasons.push(format!("Cell {} speedup is {:.6}; requires >= 0.97 (no more than 3% slower).", cell.cell, result.speedup));
            }
            logs.push(paired_logs(cell));
            compared.push(result);
        } else if flux_succeeded && baseline_succeeded {
            reasons.push(format!("Cell {} has 0 paired successes; requires at least 1 to compare.", cell.cell));
        }
    }

    let geomean = mean(&logs.iter().map(|pairs| mean(pairs)).collect::<Vec<_>>()).exp();
    let (ci_low, ci_high) = bootstrap_ci(&logs, resamples, seed);
    if compared.is_empty() {
        reasons.push("No compared cells: 0 cells have paired successes; requires at least 1.".into());
    }
    if geomean.is_nan() || geomean < 1.10 {
        reasons.push(format!("Geometric mean speedup is {geomean:.6}; requires >= 1.10."));
    }
    if ci_low.is_nan() || ci_low <= 1.00 {
        reasons.push(format!("95% paired-bootstrap lower bound is {ci_low:.6}; requires > 1.00 (resamples: {resamples})."));
    }
    SpeedGate { geomean, ci_low, ci_high, cells: compared, statuses, slower_than_3pct, flux_failed, pass: reasons.is_empty(), reasons }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGate {
    pub ppl_ratio: Option<f64>,
    pub kl_mean: Option<f64>,
    pub top1_agreement: Option<f64>,
    pub task_delta_pp: Option<f64>,
    pub exact_profile: bool,
    pub pass: bool,
    pub reasons: Vec<String>,
}

/// How far an accepted configuration (e.g. llama.cpp's own GPU execution) is from the same reference.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Divergence {
    pub ppl_ratio: Option<f64>,
    pub kl_mean: Option<f64>,
    pub top1_agreement: Option<f64>,
}

/// Placement-only profiles require at least one of ppl/kl/top1; changed
/// profiles require ppl. Every supplied metric relevant to the profile is checked.
/// With `noise`, placement-only limits widen to the divergence an accepted configuration already shows
/// from the reference (kernels differ between devices), never tightening the fixed limits.
pub fn quality_gate(
    exact_profile: bool,
    ppl_ratio: Option<f64>,
    kl_mean: Option<f64>,
    top1_agreement: Option<f64>,
    task_delta_pp: Option<f64>,
    noise: Option<Divergence>,
) -> QualityGate {
    let mut reasons = Vec::new();
    if exact_profile {
        if ppl_ratio.is_none() && kl_mean.is_none() && top1_agreement.is_none() {
            for name in ["ppl_ratio", "kl_mean", "top1_agreement"] {
                reasons.push(format!("not measured: {name}"));
            }
        }
    } else if ppl_ratio.is_none() {
        reasons.push("not measured: ppl_ratio".into());
    }

    let n = noise.unwrap_or_default();
    let checks = if exact_profile {
        vec![
            ("ppl_ratio", ppl_ratio, n.ppl_ratio.map_or(1.005, |r| (r + 0.002).max(1.005)), true),
            ("kl_mean", kl_mean, n.kl_mean.map_or(0.01, |k| (1.5 * k).max(0.01)), true),
            ("top1_agreement", top1_agreement, n.top1_agreement.map_or(0.98, |t| (t - 0.01).min(0.98)), false),
        ]
    } else {
        vec![("ppl_ratio", ppl_ratio, 1.02, true), ("task_delta_pp", task_delta_pp, -1.0, false)]
    };
    for (name, measurement, limit, upper_bound) in checks {
        if let Some(value) = measurement {
            let accepted = value.is_finite() && if upper_bound { value <= limit } else { value >= limit };
            if !accepted {
                let operator = if upper_bound { "<=" } else { ">=" };
                reasons.push(format!("{name} is {value}; requires {operator} {limit}."));
            }
        }
    }
    QualityGate { ppl_ratio, kl_mean, top1_agreement, task_delta_pp, exact_profile, pass: reasons.is_empty(), reasons }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(name: &str, ratios: &[f64]) -> CellTrials {
        CellTrials { cell: name.into(), flux: ratios.iter().map(|ratio| Some(100.0 * ratio)).collect(), baseline: vec![Some(100.0); ratios.len()] }
    }

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn noisy_three_cell_speedup_passes() {
        let cells: Vec<_> = (0..3)
            .map(|i| {
                let ratios: Vec<_> = (0..10).map(|trial| 1.2 + (trial as f64 - 4.5) * 0.004 + i as f64 * 0.002).collect();
                cell(&format!("model-{i}"), &ratios)
            })
            .collect();
        let gate = speed_gate(&cells, 2_000, 42);
        assert!(gate.pass, "{:?}", gate.reasons);
        assert!(gate.geomean >= 1.10);
        assert!(gate.ci_low > 1.00 && gate.ci_low < gate.ci_high);
        assert_eq!(gate.cells.len(), 3);
        assert!(gate.cells.iter().all(|cell| cell.pairs == 10));
    }

    #[test]
    fn slow_cell_fails_even_with_fast_overall_geomean() {
        let gate = speed_gate(&[cell("fast", &[1.5; 10]), cell("slow", &[0.95; 10])], 500, 42);
        assert!(gate.geomean > 1.10);
        assert!(!gate.pass);
        assert_eq!(gate.slower_than_3pct, ["slow"]);
        assert!(gate.reasons.iter().any(|reason| reason.contains("0.97")));
    }

    #[test]
    fn flux_failure_fails_gate() {
        let mut failed = cell("failed", &[1.2; 10]);
        failed.flux.fill(None);
        let gate = speed_gate(&[cell("fast", &[1.2; 10]), failed], 500, 42);
        assert!(!gate.pass);
        assert_eq!(gate.flux_failed, ["failed"]);
        assert_eq!(gate.statuses[1].1, CellStatus::BaselineOnly);
        assert_eq!(gate.cells.len(), 1);
    }

    #[test]
    fn partial_flux_failure_is_not_dropped() {
        let mut partial = cell("partial", &[1.2; 10]);
        partial.flux[3] = None;
        let gate = speed_gate(&[partial], 500, 42);
        assert!(!gate.pass);
        assert_eq!(gate.flux_failed, ["partial"]);
        assert_eq!(gate.statuses[0].1, CellStatus::Compared);
        assert_eq!(gate.cells[0].pairs, 9);
    }

    #[test]
    fn baseline_failure_is_flux_only_and_excluded() {
        let mut flux_only = cell("flux-only", &[0.01; 10]);
        flux_only.baseline.fill(None);
        let gate = speed_gate(&[cell("fast", &[1.2; 10]), flux_only], 500, 42);
        assert!(gate.pass, "{:?}", gate.reasons);
        assert_eq!(gate.statuses[1].1, CellStatus::FluxOnly);
        assert_eq!(gate.cells.len(), 1);
        close(gate.geomean, 1.2);
    }

    #[test]
    fn unsupported_and_empty_inputs_cannot_pass() {
        let unsupported = CellTrials { cell: "unsupported".into(), flux: vec![None; 3], baseline: vec![None; 3] };
        let gate = speed_gate(&[unsupported], 500, 42);
        assert!(!gate.pass);
        assert_eq!(gate.statuses[0].1, CellStatus::Unsupported);
        assert!(gate.cells.is_empty());
        assert!(!speed_gate(&[], 500, 42).pass);
    }

    #[test]
    fn uses_trial_pairs_and_single_pair_has_point_ci() {
        let trials =
            CellTrials { cell: "paired".into(), flux: vec![Some(120.0), None, Some(500.0), None], baseline: vec![Some(100.0), Some(10.0), None, None] };
        let result = cell_speedup(&trials, 500, 42).unwrap();
        assert_eq!(result.pairs, 1);
        close(result.speedup, 1.2);
        assert_eq!(result.ci_low, result.speedup);
        assert_eq!(result.ci_high, result.speedup);
        let gate = speed_gate(&[cell("single", &[1.2])], 500, 42);
        assert_eq!(gate.statuses[0].1, CellStatus::Compared);
        assert!(gate.pass);
    }

    #[test]
    fn bootstrap_is_deterministic() {
        let trials = cell("noisy", &[1.1, 1.2, 1.3, 1.25, 1.15]);
        let first = cell_speedup(&trials, 1_000, 123).unwrap();
        let second = cell_speedup(&trials, 1_000, 123).unwrap();
        assert_eq!((first.ci_low, first.ci_high), (second.ci_low, second.ci_high));
        let cells = [trials, cell("other", &[1.05, 1.4, 1.2])];
        let first = speed_gate(&cells, 1_000, 123);
        let second = speed_gate(&cells, 1_000, 123);
        assert_eq!((first.ci_low, first.ci_high), (second.ci_low, second.ci_high));
    }

    #[test]
    fn cells_are_equally_weighted_and_ratios_are_geometric() {
        let gate = speed_gate(&[cell("small", &[1.0]), cell("large", &[1.44; 20])], 500, 42);
        close(gate.geomean, 1.2);
        let result = cell_speedup(&cell("ratios", &[0.5, 2.0]), 500, 42).unwrap();
        close(result.speedup, 1.0);
    }

    #[test]
    fn bootstrap_samples_cells_independently() {
        let gate = speed_gate(&[cell("first", &[1.0, 4.0]), cell("second", &[4.0, 1.0])], 2_000, 42);
        close(gate.geomean, 2.0);
        close(gate.ci_low, 1.0);
        close(gate.ci_high, 4.0);
    }

    #[test]
    fn confidence_lower_bound_can_fail_despite_fast_point_estimate() {
        let gate = speed_gate(&[cell("uncertain", &[0.5, 3.0])], 2_000, 42);
        assert!(gate.geomean >= 1.10);
        assert!(gate.ci_low <= 1.00);
        assert!(gate.slower_than_3pct.is_empty());
        assert!(!gate.pass);
        assert_eq!(gate.reasons.len(), 1);
    }

    #[test]
    fn speed_threshold_boundaries() {
        assert!(speed_gate(&[cell("minimum", &[1.10; 10])], 500, 42).pass);
        let gate = speed_gate(&[cell("fast", &[1.5; 10]), cell("allowed", &[0.97; 10])], 500, 42);
        assert!(gate.pass, "{:?}", gate.reasons);
        assert!(gate.slower_than_3pct.is_empty());
        let gate = speed_gate(&[cell("equal", &[1.0])], 500, 42);
        assert!(!gate.pass);
        assert_eq!(gate.ci_low, 1.0);
        assert!(gate.reasons.iter().any(|reason| reason.contains("> 1.00")));
    }

    #[test]
    fn reports_all_failed_speed_conditions() {
        let mut trials = cell("slow-failure", &[0.95; 3]);
        trials.flux[0] = None;
        let gate = speed_gate(&[trials], 500, 42);
        assert!(!gate.pass);
        assert_eq!(gate.reasons.len(), 4);
        assert!(gate.reasons.iter().any(|reason| reason.contains(">= 1.10")));
        assert!(gate.reasons.iter().any(|reason| reason.contains("> 1.00")));
    }

    #[test]
    fn zero_resamples_and_unpaired_successes_fail() {
        let gate = speed_gate(&[cell("unmeasured-ci", &[1.1, 1.3])], 0, 42);
        assert!(!gate.pass);
        assert!(gate.ci_low.is_nan());
        let unpaired = CellTrials { cell: "unpaired".into(), flux: vec![Some(100.0), None], baseline: vec![None, Some(100.0)] };
        assert!(cell_speedup(&unpaired, 500, 42).is_none());
        let gate = speed_gate(&[unpaired], 500, 42);
        assert!(!gate.pass);
        assert_eq!(gate.flux_failed, ["unpaired"]);
    }

    #[test]
    fn malformed_trials_fail_without_panicking() {
        let mut trials = cell("invalid", &[1.2; 3]);
        trials.flux.pop();
        assert!(!speed_gate(&[trials], 100, 42).pass);
        let mut trials = cell("invalid", &[1.2; 3]);
        trials.flux[0] = Some(f64::NAN);
        assert!(!speed_gate(&[trials], 100, 42).pass);
    }

    #[test]
    fn exact_quality_thresholds_and_optional_measurements() {
        assert!(quality_gate(true, Some(1.005), Some(0.01), Some(0.98), None, None).pass);
        assert!(quality_gate(true, None, Some(0.005), None, None, None).pass);
        assert!(quality_gate(true, None, None, Some(0.99), None, None).pass);
        let gate = quality_gate(true, Some(1.006), Some(0.011), Some(0.979), None, None);
        assert!(!gate.pass);
        assert_eq!(gate.reasons.len(), 3);
        let missing = quality_gate(true, None, None, None, Some(0.0), None);
        assert!(!missing.pass);
        assert_eq!(missing.reasons, ["not measured: ppl_ratio", "not measured: kl_mean", "not measured: top1_agreement"]);
    }

    #[test]
    fn changed_quality_thresholds_and_required_ppl() {
        assert!(quality_gate(false, Some(1.02), None, None, Some(-1.0), None).pass);
        assert!(quality_gate(false, Some(1.01), None, None, None, None).pass);
        assert!(quality_gate(false, Some(1.01), Some(1.0), Some(0.5), None, None).pass);
        let gate = quality_gate(false, Some(1.021), None, None, Some(-1.01), None);
        assert!(!gate.pass);
        assert_eq!(gate.reasons.len(), 2);
        let missing = quality_gate(false, None, None, None, Some(0.0), None);
        assert!(!missing.pass);
        assert_eq!(missing.reasons, ["not measured: ppl_ratio"]);
    }

    #[test]
    fn nonfinite_quality_measurements_fail() {
        assert!(!quality_gate(true, Some(f64::NAN), None, None, None, None).pass);
        assert!(!quality_gate(true, None, None, Some(f64::INFINITY), None, None).pass);
        assert!(!quality_gate(false, Some(1.01), None, None, Some(f64::NAN), None).pass);
    }

    #[test]
    fn exact_limits_widen_to_accepted_cross_device_divergence() {
        let noise = Divergence { ppl_ratio: Some(1.0004), kl_mean: Some(0.004), top1_agreement: Some(0.965) };
        assert!(!quality_gate(true, Some(1.0004), Some(0.0039), Some(0.966), None, None).pass);
        assert!(quality_gate(true, Some(1.0004), Some(0.0039), Some(0.966), None, Some(noise)).pass);
        assert!(!quality_gate(true, Some(1.0004), Some(0.02), Some(0.966), None, Some(noise)).pass);
    }
}
