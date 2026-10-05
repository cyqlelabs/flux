//! Predicted step times from measured probes. Used to rank and prune candidates; the final choice
//! is always made on measured runs.

use crate::layers::{Block, ExpertTensor, Layout, Weight};
use flux_core::ggml_type::GgmlType;
use flux_core::hardware::{CopyCurve, CopyDirection, ProbeReport};
use std::collections::HashMap;

pub const CPU: &str = "CPU";

#[derive(Debug, Clone, Default)]
struct DeviceRates {
    /// Effective weight-read GB/s of single-token products, per encoding.
    decode_gbps: HashMap<GgmlType, f64>,
    /// Multiply-accumulates per second (x1e9) of prompt-sized products, per encoding.
    prefill_gmacs: HashMap<GgmlType, f64>,
    /// Fixed cost of one small product: launch or thread-wake overhead.
    op_overhead_s: f64,
}

impl DeviceRates {
    fn gbps(&self, t: GgmlType) -> f64 {
        self.decode_gbps.get(&t).copied().unwrap_or_else(|| median(self.decode_gbps.values()))
    }

    fn gmacs(&self, t: GgmlType) -> f64 {
        self.prefill_gmacs.get(&t).copied().unwrap_or_else(|| median(self.prefill_gmacs.values()))
    }

    /// Streaming bandwidth for cache reads: the best measured weight-read rate.
    fn mem_gbps(&self) -> f64 {
        self.decode_gbps.values().copied().fold(1.0, f64::max)
    }
}

/// A routed-expert tensor as a product: all experts' bytes, k selected experts' work per token.
fn expert_weight(l: &Layout, e: &ExpertTensor) -> Weight {
    Weight { ggml_type: e.ggml_type, bytes: e.bytes, macs: e.k * e.n * l.n_expert_used as u64 }
}

fn median<'a>(v: impl Iterator<Item = &'a f64>) -> f64 {
    let mut v: Vec<f64> = v.copied().collect();
    if v.is_empty() {
        return 1.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Workload shape the costs are evaluated at.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    /// Sequences decoding together in one step.
    pub batch: u32,
    /// Cached positions per sequence when decoding.
    pub ctx: u64,
    /// Cells per sequence the state was sized for.
    pub n_ctx_seq: u64,
    pub n_seq: u64,
    /// Prompt chunk processed per prefill step.
    pub ubatch: u32,
    pub op_offload: bool,
}

#[derive(Debug, Clone)]
pub struct CostModel {
    devices: HashMap<String, DeviceRates>,
    h2d: HashMap<String, CopyCurve>,
    d2h: HashMap<String, CopyCurve>,
    d2d: HashMap<(String, String), CopyCurve>,
    /// Host-side work per step (sampling, scheduling, launches outside layers).
    pub step_overhead_s: f64,
}

impl CostModel {
    /// Builds rates from the probe report; `cpu_threads` selects the CPU probes to use.
    pub fn from_report(r: &ProbeReport, cpu_threads: u32) -> CostModel {
        let mut devices: HashMap<String, DeviceRates> = HashMap::new();
        let mut overhead: HashMap<String, Vec<f64>> = HashMap::new();
        let mut decode: HashMap<(String, GgmlType), Vec<(f64, f64)>> = HashMap::new();
        let mut prefill: HashMap<(String, GgmlType), Vec<(f64, f64)>> = HashMap::new();
        for k in &r.kernels {
            if k.device == CPU && k.threads != cpu_threads {
                continue;
            }
            let Some(t) = GgmlType::from_name(&k.ggml_type) else { continue };
            let bytes = t.row_bytes(k.k).unwrap_or(0) * k.n;
            let secs = k.micros.p50 * 1e-6;
            if k.k * k.n <= 256 * 256 {
                overhead.entry(k.device.clone()).or_default().push(secs);
                continue;
            }
            if k.batch == 1 {
                decode.entry((k.device.clone(), t)).or_default().push((bytes as f64, secs));
            } else {
                prefill.entry((k.device.clone(), t)).or_default().push(((k.k * k.n * k.batch) as f64, secs));
            }
        }
        // Size-weighted rates: total work over total time across probed shapes.
        for ((dev, t), v) in decode {
            let (w, s): (f64, f64) = v.iter().fold((0.0, 0.0), |a, x| (a.0 + x.0, a.1 + x.1));
            devices.entry(dev).or_default().decode_gbps.insert(t, w / s / 1e9);
        }
        for ((dev, t), v) in prefill {
            let (w, s): (f64, f64) = v.iter().fold((0.0, 0.0), |a, x| (a.0 + x.0, a.1 + x.1));
            devices.entry(dev).or_default().prefill_gmacs.insert(t, w / s / 1e9);
        }
        for (dev, v) in overhead {
            devices.entry(dev).or_default().op_overhead_s = median(v.iter());
        }
        let mut h2d = HashMap::new();
        let mut d2h = HashMap::new();
        let mut d2d = HashMap::new();
        for c in r.copies.iter().filter(|c| c.pinned || c.direction == CopyDirection::DeviceToDevice) {
            match (c.direction, &c.peer) {
                (CopyDirection::HostToDevice, _) => {
                    h2d.insert(c.device.clone(), c.clone());
                }
                (CopyDirection::DeviceToHost, _) => {
                    d2h.insert(c.device.clone(), c.clone());
                }
                (CopyDirection::DeviceToDevice, Some(p)) => {
                    d2d.insert((c.device.clone(), p.clone()), c.clone());
                }
                _ => {}
            }
        }
        CostModel { devices, h2d, d2h, d2d, step_overhead_s: 2e-4 }
    }

    pub fn has_device(&self, dev: &str) -> bool {
        self.devices.get(dev).is_some_and(|d| !d.decode_gbps.is_empty())
    }

    fn rates(&self, dev: &str) -> &DeviceRates {
        static EMPTY: std::sync::OnceLock<DeviceRates> = std::sync::OnceLock::new();
        self.devices.get(dev).unwrap_or_else(|| EMPTY.get_or_init(DeviceRates::default))
    }

    /// Seconds to move `bytes` between devices (host staging when no peer path was measured).
    pub fn copy_s(&self, from: &str, to: &str, bytes: u64) -> f64 {
        if from == to || bytes == 0 {
            return 0.0;
        }
        let curve_s = |c: Option<&CopyCurve>| c.map_or(bytes as f64 / 8e9 + 1e-5, |c| c.seconds_for(bytes));
        match (from == CPU, to == CPU) {
            (true, false) => curve_s(self.h2d.get(to)),
            (false, true) => curve_s(self.d2h.get(from)),
            _ => match self.d2d.get(&(from.to_string(), to.to_string())) {
                Some(c) => c.seconds_for(bytes),
                None => curve_s(self.d2h.get(from)) + curve_s(self.h2d.get(to)),
            },
        }
    }

    /// One product: bandwidth- or compute-bound, plus its fixed overhead.
    fn product_s(&self, dev: &str, w: &Weight, tokens: u64, read_fraction: f64) -> f64 {
        let r = self.rates(dev);
        let read = w.bytes as f64 * read_fraction / (r.gbps(w.ggml_type) * 1e9);
        let compute = if tokens > 1 { (w.macs * tokens) as f64 / (r.gmacs(w.ggml_type) * 1e9) } else { 0.0 };
        r.op_overhead_s + read.max(compute)
    }

    /// Decode-step seconds of one block on `dev`; `on_cpu[k]` puts expert tensor k in host memory.
    pub fn decode_block_s(&self, l: &Layout, b: &Block, dev: &str, on_cpu: &[bool], s: &Shape) -> f64 {
        let tokens = s.batch as u64;
        let mut t: f64 = b.dense.iter().map(|w| self.product_s(dev, w, tokens, 1.0)).sum();
        let state_read = b.state_bytes as f64 * (s.batch as f64 / s.n_seq as f64) * (s.ctx as f64 / s.n_ctx_seq as f64).min(1.0);
        t += state_read / (self.rates(dev).mem_gbps() * 1e9);
        let f = l.expert_read_fraction(s.batch);
        let mut split = false;
        for (k, e) in b.experts.iter().enumerate() {
            let host = on_cpu.get(k).copied().unwrap_or(false) && dev != CPU;
            split |= host;
            t += self.product_s(if host { CPU } else { dev }, &expert_weight(l, e), tokens, f);
        }
        if split {
            t += self.split_copy_s(l, dev, tokens);
        }
        t
    }

    /// Activations to the host and results back, paid once per block whose experts are split.
    pub fn split_copy_s(&self, l: &Layout, dev: &str, tokens: u64) -> f64 {
        let act = tokens * l.n_embd as u64 * 4;
        self.copy_s(dev, CPU, act) + self.copy_s(CPU, dev, act)
    }

    /// Decode seconds saved per step by moving one expert tensor from host memory to `dev`.
    pub fn expert_gain_s(&self, l: &Layout, e: &ExpertTensor, dev: &str, s: &Shape) -> f64 {
        let w = expert_weight(l, e);
        let f = l.expert_read_fraction(s.batch);
        self.product_s(CPU, &w, s.batch as u64, f) - self.product_s(dev, &w, s.batch as u64, f)
    }

    /// Prefill seconds for one chunk of `s.ubatch` tokens through one block.
    pub fn prefill_block_s(&self, l: &Layout, b: &Block, dev: &str, on_cpu: &[bool], s: &Shape, ctx_before: u64) -> f64 {
        let u = s.ubatch as u64;
        let r = self.rates(dev);
        let mut t: f64 = b.dense.iter().map(|w| self.product_s(dev, w, u, 1.0)).sum();
        let attn_macs = (u * (ctx_before + u / 2) * b.attn_macs_per_pos) as f64;
        t += attn_macs / (r.gmacs(GgmlType::F16) * 1e9);
        let f = l.expert_read_fraction(s.ubatch);
        let mut split = false;
        for (k, e) in b.experts.iter().enumerate() {
            let host = on_cpu.get(k).copied().unwrap_or(false) && dev != CPU;
            let w = expert_weight(l, e);
            if host && s.op_offload && s.ubatch >= 32 {
                // Large batches of host-resident experts are copied to the GPU and run there.
                t += self.copy_s(CPU, dev, (e.bytes as f64 * f) as u64) + self.product_s(dev, &w, u, f);
            } else if host {
                split = true;
                t += self.product_s(CPU, &w, u, f);
            } else {
                t += self.product_s(dev, &w, u, f);
            }
        }
        if split {
            t += self.split_copy_s(l, dev, u);
        }
        t
    }

    pub fn output_decode_s(&self, l: &Layout, dev: &str, s: &Shape) -> f64 {
        l.head.output.iter().map(|w| self.product_s(dev, w, s.batch as u64, 1.0)).sum()
    }
}

/// Storage-bound output rate <= B_eff * beta_disk / D_step: `useful_tokens` per step, `bytes_per_step`
/// read from storage at `disk_gbps`. Assumes ideal reuse and ignores compute; random reads do worse.
pub fn storage_bound_tps(useful_tokens: f64, bytes_per_step: f64, disk_gbps: f64) -> f64 {
    useful_tokens * disk_gbps * 1e9 / bytes_per_step
}

/// Whether running a host-resident expert on the CPU beats copying its weights to the GPU for one use.
pub fn cpu_beats_transfer(cpu_s: f64, transfer_bytes: u64, pcie_gbps: f64) -> bool {
    cpu_s < transfer_bytes as f64 / (pcie_gbps * 1e9)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use flux_core::hardware::{CopyPoint, HardwareInventory, KernelProbe};
    use flux_core::stats::Summary;

    fn s(v: f64) -> Summary {
        Summary { n: 1, mean: v, stddev: 0.0, min: v, p50: v, p95: v, p99: v, max: v }
    }

    /// Synthetic report: GPU0 reads q8_0 at 300 GB/s, GPU1 at 200, CPU at 40; PCIe 10 GB/s.
    pub fn report(inv: HardwareInventory) -> ProbeReport {
        let mut kernels = vec![];
        let k = 4096u64;
        let n = 4096u64;
        let bytes = GgmlType::Q8_0.row_bytes(k).unwrap() * n;
        for (dev, gbps, gmacs, threads) in [("CUDA0", 300.0, 5000.0, 0), ("CUDA1", 200.0, 3000.0, 0), (CPU, 40.0, 300.0, 12)] {
            kernels.push(KernelProbe { device: dev.into(), ggml_type: "q8_0".into(), k, n, batch: 1, threads, micros: s(bytes as f64 / (gbps * 1e3)) });
            kernels.push(KernelProbe {
                device: dev.into(),
                ggml_type: "q8_0".into(),
                k,
                n,
                batch: 512,
                threads,
                micros: s((k * n * 512) as f64 / (gmacs * 1e3)),
            });
            kernels.push(KernelProbe { device: dev.into(), ggml_type: "q8_0".into(), k: 256, n: 256, batch: 1, threads, micros: s(10.0) });
        }
        let curve = |dev: &str, dir| CopyCurve {
            device: dev.into(),
            peer: None,
            direction: dir,
            pinned: true,
            points: vec![CopyPoint { bytes: 4096, gbps: s(0.4) }, CopyPoint { bytes: 1 << 26, gbps: s(10.0) }],
        };
        ProbeReport {
            schema: 1,
            topology: "t".into(),
            backend_revision: "r".into(),
            backend_build: "b".into(),
            created: chrono::Utc::now(),
            inventory: inv,
            copies: vec![
                curve("CUDA0", CopyDirection::HostToDevice),
                curve("CUDA0", CopyDirection::DeviceToHost),
                curve("CUDA1", CopyDirection::HostToDevice),
                curve("CUDA1", CopyDirection::DeviceToHost),
            ],
            contention: vec![],
            kernels,
            cpu_bandwidth: vec![],
            storage: vec![],
            host_pages: vec![],
        }
    }

    #[test]
    fn proposal_example_c_storage_bound() {
        // 20 GB per step at 5 GB/s: 4 s per step, 0.25 tok/s; 16 sequences sharing reads: 4 tok/s.
        assert!((storage_bound_tps(1.0, 20e9, 5.0) - 0.25).abs() < 1e-12);
        assert!((storage_bound_tps(16.0, 20e9, 5.0) - 4.0).abs() < 1e-12);
        assert!((storage_bound_tps(1.0, 20e9, 0.5) - 0.025).abs() < 1e-12);
    }

    #[test]
    fn proposal_example_d_cpu_versus_moving_an_expert() {
        // 32 MB at 10 GB/s is 3.2 ms of transfer; 1.2 ms of CPU work wins for a one-off use.
        assert!(cpu_beats_transfer(1.2e-3, 32_000_000, 10.0));
        assert!(!cpu_beats_transfer(4.0e-3, 32_000_000, 10.0));
    }

    #[test]
    fn rates_follow_probes() {
        let m = CostModel::from_report(&report(crate::tests::inventory()), 12);
        let w = Weight { ggml_type: GgmlType::Q8_0, bytes: 3_000_000_000, macs: 1 };
        let gpu = m.product_s("CUDA0", &w, 1, 1.0);
        let cpu = m.product_s(CPU, &w, 1, 1.0);
        assert!((gpu - 0.01).abs() < 1e-3, "{gpu}");
        assert!((cpu - 0.075).abs() < 1e-3, "{cpu}");
        assert!(m.copy_s("CUDA0", "CUDA1", 1 << 26) > m.copy_s(CPU, "CUDA1", 1 << 26));
        assert_eq!(m.copy_s(CPU, CPU, 100), 0.0);
    }
}
