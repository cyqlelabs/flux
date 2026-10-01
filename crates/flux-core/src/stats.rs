use serde::{Deserialize, Serialize};

/// Distribution of repeated measurements. Percentiles interpolate linearly between order statistics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub stddev: f64,
    pub min: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

impl Summary {
    pub fn of(samples: &[f64]) -> Option<Summary> {
        if samples.is_empty() {
            return None;
        }
        let mut s = samples.to_vec();
        s.sort_by(f64::total_cmp);
        let n = s.len();
        let mean = s.iter().sum::<f64>() / n as f64;
        let var = if n > 1 { s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64 } else { 0.0 };
        Some(Summary {
            n,
            mean,
            stddev: var.sqrt(),
            min: s[0],
            p50: percentile_sorted(&s, 0.50),
            p95: percentile_sorted(&s, 0.95),
            p99: percentile_sorted(&s, 0.99),
            max: s[n - 1],
        })
    }
}

pub fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Steady decode rate of one stream: R = (N - 1) / (t_last - t_first), defined for N > 1.
pub fn interactive_rate(emission_times_s: &[f64]) -> Option<f64> {
    let (first, last) = (emission_times_s.first()?, emission_times_s.last()?);
    (emission_times_s.len() > 1 && last > first).then(|| (emission_times_s.len() - 1) as f64 / (last - first))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_percentiles() {
        let s = Summary::of(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(s.p50, 3.0);
        assert!((s.p95 - 4.8).abs() < 1e-9);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
    }

    #[test]
    fn rate_excludes_first_token() {
        assert_eq!(interactive_rate(&[10.0, 10.5, 11.0]), Some(2.0));
        assert_eq!(interactive_rate(&[10.0]), None);
    }
}
