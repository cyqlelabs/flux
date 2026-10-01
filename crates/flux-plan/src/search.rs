//! Placement search. Contiguous blocks over [CPU, gpu_1, ..., gpu_k] by dynamic programming over
//! block index, device and discretized memory, with a per-block choice of routed-expert residency;
//! then greedy gain-per-byte filling of spare memory with single expert tensors, then local swaps.
//! Host memory is a constraint too: when the fastest placement overflows it, host bytes get the
//! lowest price (seconds per byte) at which the search fits. A bounded heuristic, not a proof of
//! global optimality.

use crate::cost::{CostModel, Shape, CPU};
use crate::layers::Layout;
use std::collections::{BTreeMap, HashMap};

/// Memory quantum of the dynamic program; exact bytes are re-checked after reconstruction.
const QUANTUM: u64 = 16 << 20;

pub struct SearchInput<'a> {
    pub layout: &'a Layout,
    pub cost: &'a CostModel,
    pub shape: Shape,
    /// GPUs in pipeline order; blocks may skip any of them.
    pub order: Vec<String>,
    /// Bytes per GPU for weights and state (compute buffers and reserve already deducted).
    pub capacity: HashMap<String, u64>,
    /// Keep every routed expert in host memory (the ablation without expert residency).
    pub host_experts: bool,
    /// Host bytes for weights and state; None leaves host memory unconstrained.
    pub host_capacity: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    /// Offload devices in backend order (only those holding something).
    pub devices: Vec<String>,
    pub layer_device: Vec<String>,
    pub output_device: String,
    /// `on_cpu[block][k]`: expert tensor k of a GPU block kept in host memory.
    pub on_cpu: Vec<Vec<bool>>,
    /// Predicted seconds per decode step.
    pub decode_s: f64,
    /// Weights + state bytes per device, including CPU.
    pub bytes: BTreeMap<String, u64>,
}

fn units(b: u64) -> usize {
    (b.saturating_add(QUANTUM / 2) / QUANTUM) as usize
}

#[derive(Clone, Copy)]
struct Cell {
    cost: f64,
    prev: (u8, u32),
    experts_on_cpu: bool,
}

pub fn search(inp: &SearchInput) -> Option<Assignment> {
    let fastest = search_priced(inp, 0.0)?;
    let host = |a: &Assignment| a.bytes.get(CPU).copied().unwrap_or(0);
    let Some(cap) = inp.host_capacity.filter(|&c| host(&fastest) > c) else { return Some(fastest) };
    // 1 µs per byte outweighs any decode time: if that does not fit, nothing does, and the caller
    // reports the overflow of the fastest placement.
    let (mut lo, mut hi) = (0.0, 1e-6);
    let mut best = search_priced(inp, hi).filter(|a| host(a) <= cap);
    if best.is_none() {
        return Some(fastest);
    }
    for _ in 0..16 {
        let mid = (lo + hi) / 2.0;
        match search_priced(inp, mid).filter(|a| host(a) <= cap) {
            Some(a) => {
                best = Some(a);
                hi = mid;
            }
            None => lo = mid,
        }
    }
    best
}

/// The DP and greedy fill with every host byte costing `price` seconds; `decode_s` stays unpriced.
fn search_priced(inp: &SearchInput, price: f64) -> Option<Assignment> {
    let l = inp.layout;
    let s = &inp.shape;
    let seq: Vec<&str> = std::iter::once(CPU).chain(inp.order.iter().map(String::as_str)).collect();
    let caps: Vec<usize> = seq.iter().map(|d| if *d == CPU { 0 } else { units(inp.capacity.get(*d).copied().unwrap_or(0)) }).collect();
    let act = s.batch as u64 * l.n_embd as u64 * 4;
    let n = l.blocks.len();

    // Variants of block i on device j: (gpu units, seconds, experts on host).
    let variants = |i: usize, j: usize| -> Vec<(usize, f64, bool)> {
        let b = &l.blocks[i];
        let dev = seq[j];
        if dev == CPU {
            let host = (b.dense_bytes() + b.expert_bytes() + b.state_bytes) as f64;
            return vec![(0, inp.cost.decode_block_s(l, b, CPU, &[], s) + price * host, false)];
        }
        let mut v = vec![];
        if b.experts.is_empty() || !inp.host_experts {
            v.push((units(b.dense_bytes() + b.expert_bytes() + b.state_bytes), inp.cost.decode_block_s(l, b, dev, &[], s), false));
        }
        if !b.experts.is_empty() {
            let all = vec![true; b.experts.len()];
            let host = b.expert_bytes() as f64;
            v.push((units(b.dense_bytes() + b.state_bytes), inp.cost.decode_block_s(l, b, dev, &all, s) + price * host, true));
        }
        v
    };

    // table[i][j][u]: best cost with block i on device j using u units of j.
    let mut table: Vec<Vec<Vec<Option<Cell>>>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut cur: Vec<Vec<Option<Cell>>> = caps.iter().map(|&c| vec![None; c + 1]).collect();
        for j in 0..seq.len() {
            for (du, t, host) in variants(i, j) {
                let mut relax = |u: usize, cost: f64, prev: (u8, u32)| {
                    if u <= caps[j] && cur[j][u].is_none_or(|c| cost < c.cost) {
                        cur[j][u] = Some(Cell { cost, prev, experts_on_cpu: host });
                    }
                };
                if i == 0 {
                    relax(du, t + inp.cost.copy_s(CPU, seq[j], act), (u8::MAX, 0));
                    continue;
                }
                for (jp, row) in table[i - 1].iter().enumerate().take(j + 1) {
                    for (up, c) in row.iter().enumerate() {
                        let Some(c) = c else { continue };
                        if jp == j {
                            relax(up + du, c.cost + t, (jp as u8, up as u32));
                        } else {
                            relax(du, c.cost + t + inp.cost.copy_s(seq[jp], seq[j], act), (jp as u8, up as u32));
                        }
                    }
                }
            }
        }
        table.push(cur);
    }

    // Output head on the last block's device or on a later GPU.
    let out_units = units(l.head.output_bytes());
    let logits = s.batch as u64 * l.n_vocab as u64 * 4;
    let mut best: Option<(f64, usize, usize, usize)> = None;
    for (j, row) in table[n - 1].iter().enumerate() {
        for (u, c) in row.iter().enumerate() {
            let Some(c) = c else { continue };
            for k in j..seq.len() {
                let fits = if k == j { seq[k] == CPU || u + out_units <= caps[k] } else { out_units <= caps[k] };
                if !fits {
                    continue;
                }
                let t = c.cost
                    + inp.cost.copy_s(seq[j], seq[k], act)
                    + inp.cost.output_decode_s(l, seq[k], s)
                    + inp.cost.copy_s(seq[k], CPU, logits)
                    + inp.cost.step_overhead_s;
                if best.is_none_or(|b| t < b.0) {
                    best = Some((t, j, u, k));
                }
            }
        }
    }
    let (_, mut j, mut u, out) = best?;

    let mut layer_device = vec![String::new(); n];
    let mut on_cpu = vec![vec![]; n];
    for i in (0..n).rev() {
        let c = table[i][j][u].expect("reconstruction follows filled cells");
        layer_device[i] = seq[j].to_string();
        if c.experts_on_cpu {
            on_cpu[i] = vec![true; l.blocks[i].experts.len()];
        }
        (j, u) = (c.prev.0 as usize, c.prev.1 as usize);
    }
    let mut a = Assignment { devices: vec![], layer_device, output_device: seq[out].to_string(), on_cpu, decode_s: 0.0, bytes: BTreeMap::new() };
    if !inp.host_experts {
        fill_spare(inp, &mut a, price);
    }
    finalize(inp, &mut a);
    Some(a)
}

/// Greedy: move host-resident expert tensors of GPU blocks into spare GPU memory by gain per byte,
/// then swap selections while that lowers the predicted step time.
fn fill_spare(inp: &SearchInput, a: &mut Assignment, price: f64) {
    let l = inp.layout;
    let s = &inp.shape;
    let used = device_bytes(l, a);
    for dev in &inp.order {
        let mut spare = inp.capacity.get(dev).copied().unwrap_or(0).saturating_sub(used.get(dev).copied().unwrap_or(0)) as i64;
        let mut items: Vec<(usize, usize, f64)> = vec![];
        for (i, b) in l.blocks.iter().enumerate() {
            if &a.layer_device[i] != dev {
                continue;
            }
            for (k, e) in b.experts.iter().enumerate() {
                if a.on_cpu[i].get(k) == Some(&true) {
                    let share = inp.cost.split_copy_s(l, dev, s.batch as u64) / b.experts.len() as f64;
                    items.push((i, k, (inp.cost.expert_gain_s(l, e, dev, s) + share) / e.bytes as f64 + price));
                }
            }
        }
        items.sort_by(|x, y| y.2.total_cmp(&x.2));
        for &(i, k, value) in &items {
            let bytes = l.blocks[i].experts[k].bytes as i64;
            if value > 0.0 && bytes <= spare {
                a.on_cpu[i][k] = false;
                spare -= bytes;
            }
        }
        // Local swaps: replace a resident tensor with a host one when it fits and the step gets faster.
        for _ in 0..200 {
            let mut improved = false;
            'outer: for &(ia, ka, _) in &items {
                if a.on_cpu[ia][ka] {
                    continue;
                }
                for &(ib, kb, _) in &items {
                    if !a.on_cpu[ib][kb] {
                        continue;
                    }
                    let (ba, bb) = (l.blocks[ia].experts[ka].bytes as i64, l.blocks[ib].experts[kb].bytes as i64);
                    if spare + ba < bb {
                        continue;
                    }
                    let before = block_s(inp, a, ia) + if ia != ib { block_s(inp, a, ib) } else { 0.0 };
                    a.on_cpu[ia][ka] = true;
                    a.on_cpu[ib][kb] = false;
                    let after = block_s(inp, a, ia) + if ia != ib { block_s(inp, a, ib) } else { 0.0 };
                    if after < before - 1e-9 {
                        spare += ba - bb;
                        improved = true;
                        break 'outer;
                    }
                    a.on_cpu[ia][ka] = false;
                    a.on_cpu[ib][kb] = true;
                }
            }
            if !improved {
                break;
            }
        }
    }
}

fn block_s(inp: &SearchInput, a: &Assignment, i: usize) -> f64 {
    inp.cost.decode_block_s(inp.layout, &inp.layout.blocks[i], &a.layer_device[i], &a.on_cpu[i], &inp.shape)
}

/// Exact bytes per device: block weights and state, host-resident experts, output head, input embeddings.
pub fn device_bytes(l: &Layout, a: &Assignment) -> BTreeMap<String, u64> {
    let mut m: BTreeMap<String, u64> = BTreeMap::new();
    for (i, b) in l.blocks.iter().enumerate() {
        let dev = a.layer_device[i].clone();
        *m.entry(dev.clone()).or_default() += b.dense_bytes() + b.state_bytes;
        for (k, e) in b.experts.iter().enumerate() {
            let host = a.on_cpu[i].get(k) == Some(&true);
            *m.entry(if host { CPU.to_string() } else { dev.clone() }).or_default() += e.bytes;
        }
    }
    *m.entry(a.output_device.clone()).or_default() += l.head.output_bytes();
    *m.entry(CPU.to_string()).or_default() += l.head.input_bytes;
    m
}

fn finalize(inp: &SearchInput, a: &mut Assignment) {
    let l = inp.layout;
    let s = &inp.shape;
    let act = s.batch as u64 * l.n_embd as u64 * 4;
    let mut t = inp.cost.step_overhead_s;
    let mut prev = CPU;
    for i in 0..l.blocks.len() {
        let dev = a.layer_device[i].as_str();
        t += inp.cost.copy_s(prev, dev, act) + block_s(inp, a, i);
        prev = dev;
    }
    t += inp.cost.copy_s(prev, &a.output_device, act)
        + inp.cost.output_decode_s(l, &a.output_device, s)
        + inp.cost.copy_s(&a.output_device, CPU, s.batch as u64 * l.n_vocab as u64 * 4);
    a.decode_s = t;
    a.bytes = device_bytes(l, a);
    let mut devices: Vec<String> = vec![];
    for d in a.layer_device.iter().chain(std::iter::once(&a.output_device)) {
        if d != CPU && devices.last() != Some(d) {
            devices.push(d.clone());
        }
    }
    a.devices = devices;
}

/// Predicted seconds to prefill `prompt` tokens in chunks of `shape.ubatch`.
pub fn prefill_s(inp: &SearchInput, a: &Assignment, prompt: u64) -> f64 {
    let l = inp.layout;
    let s = &inp.shape;
    let mut t = 0.0;
    let mut done = 0;
    while done < prompt {
        let chunk = (prompt - done).min(s.ubatch as u64);
        let cs = Shape { ubatch: chunk as u32, ..*s };
        let act = chunk * l.n_embd as u64 * 4;
        let mut prev = CPU;
        for (i, b) in l.blocks.iter().enumerate() {
            let dev = a.layer_device[i].as_str();
            t += inp.cost.copy_s(prev, dev, act) + inp.cost.prefill_block_s(l, b, dev, &a.on_cpu[i], &cs, done);
            prev = dev;
        }
        done += chunk;
    }
    t + inp.cost.output_decode_s(l, &a.output_device, &Shape { batch: 1, ..*s })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::layers::{Block, ExpertTensor, Head, Weight};
    use flux_core::ggml_type::GgmlType;

    const GB: u64 = 1_000_000_000;

    fn layout(n: usize, dense: u64, experts: u64) -> Layout {
        let blocks = (0..n)
            .map(|i| Block {
                index: i as u32,
                dense: vec![Weight { ggml_type: GgmlType::Q8_0, bytes: dense, macs: dense }],
                experts: if experts > 0 {
                    ["up", "gate", "down"]
                        .iter()
                        .map(|p| ExpertTensor { name: format!("blk.{i}.ffn_{p}_exps.weight"), ggml_type: GgmlType::Q8_0, bytes: experts / 3, k: 2048, n: 1408 })
                        .collect()
                } else {
                    vec![]
                },
                state_bytes: 10 << 20,
                attn_macs_per_pos: 4096,
            })
            .collect();
        Layout {
            blocks,
            head: Head { output: vec![Weight { ggml_type: GgmlType::Q8_0, bytes: GB / 2, macs: GB / 2 }], input_bytes: GB / 2 },
            n_expert: if experts > 0 { 64 } else { 0 },
            n_expert_used: 6,
            n_embd: 4096,
            n_vocab: 150_000,
        }
    }

    pub(crate) fn shape() -> Shape {
        Shape { batch: 1, ctx: 2048, n_ctx_seq: 4096, n_seq: 1, ubatch: 512, op_offload: true }
    }

    pub(crate) fn cost() -> CostModel {
        CostModel::from_report(&crate::cost::tests::report(crate::tests::inventory()), 12)
    }

    #[test]
    fn fits_on_fastest_gpu_when_possible() {
        let l = layout(28, GB / 4, 0);
        let c = cost();
        let inp = SearchInput {
            layout: &l,
            cost: &c,
            shape: shape(),
            order: vec!["CUDA0".into(), "CUDA1".into()],
            capacity: HashMap::from([("CUDA0".into(), 12 * GB), ("CUDA1".into(), 6 * GB)]),
            host_experts: false,
            host_capacity: None,
        };
        let a = search(&inp).unwrap();
        assert!(a.layer_device.iter().all(|d| d == "CUDA0"), "{:?}", a.layer_device);
        assert_eq!(a.devices, vec!["CUDA0"]);
    }

    #[test]
    fn splits_when_capacity_requires_it() {
        let l = layout(28, GB / 2, 0);
        let c = cost();
        let inp = SearchInput {
            layout: &l,
            cost: &c,
            shape: shape(),
            order: vec!["CUDA0".into(), "CUDA1".into()],
            capacity: HashMap::from([("CUDA0".into(), 10 * GB), ("CUDA1".into(), 6 * GB)]),
            host_experts: false,
            host_capacity: None,
        };
        let a = search(&inp).unwrap();
        let on0 = a.layer_device.iter().filter(|d| *d == "CUDA0").count();
        let on1 = a.layer_device.iter().filter(|d| *d == "CUDA1").count();
        assert_eq!(on0 + on1, 28, "{:?}", a.layer_device);
        assert!(a.bytes["CUDA0"] <= 10 * GB && a.bytes["CUDA1"] <= 6 * GB);
        // Contiguous: CUDA0 block first, then CUDA1.
        let first1 = a.layer_device.iter().position(|d| d == "CUDA1").unwrap();
        assert!(a.layer_device[first1..].iter().all(|d| d == "CUDA1"));
    }

    #[test]
    fn overflow_goes_to_cpu_prefix() {
        let l = layout(20, GB, 0);
        let c = cost();
        let inp = SearchInput {
            layout: &l,
            cost: &c,
            shape: shape(),
            order: vec!["CUDA0".into()],
            capacity: HashMap::from([("CUDA0".into(), 10 * GB)]),
            host_experts: false,
            host_capacity: None,
        };
        let a = search(&inp).unwrap();
        let cpu = a.layer_device.iter().take_while(|d| *d == CPU).count();
        assert!((10..=12).contains(&cpu), "{:?}", a.layer_device);
        assert!(a.bytes["CUDA0"] <= 10 * GB);
    }

    #[test]
    fn host_capacity_is_a_constraint() {
        // 24 MoE blocks (34 GB with experts) over 16 GB of GPUs: most experts stay in host memory.
        let l = layout(24, GB / 5, 6 * GB / 5);
        let c = cost();
        let mut inp = SearchInput {
            layout: &l,
            cost: &c,
            shape: shape(),
            order: vec!["CUDA0".into(), "CUDA1".into()],
            capacity: HashMap::from([("CUDA0".into(), 10 * GB), ("CUDA1".into(), 6 * GB)]),
            host_experts: false,
            host_capacity: None,
        };
        let free = search(&inp).unwrap();
        // Pricing host bytes never puts more of them in host memory.
        assert!(search_priced(&inp, 1e-6).unwrap().bytes[CPU] <= free.bytes[CPU]);
        inp.host_capacity = Some(free.bytes[CPU]);
        assert_eq!(search(&inp).unwrap(), free, "a limit the fastest plan meets keeps it");
        inp.host_capacity = Some(GB);
        assert_eq!(search(&inp).unwrap(), free, "an impossible limit leaves the fastest plan for the caller to reject");
    }

    #[test]
    fn moe_keeps_dense_parts_on_gpu_and_fills_experts_by_value() {
        // 24 blocks: 0.2 GB dense + 1.2 GB experts each; 12 GB GPU cannot hold all experts.
        let l = layout(24, GB / 5, 6 * GB / 5);
        let c = cost();
        let mut inp = SearchInput {
            layout: &l,
            cost: &c,
            shape: shape(),
            order: vec!["CUDA0".into()],
            capacity: HashMap::from([("CUDA0".into(), 12 * GB)]),
            host_experts: false,
            host_capacity: None,
        };
        let a = search(&inp).unwrap();
        assert!(a.layer_device.iter().all(|d| d == "CUDA0"), "dense parts stay on the GPU: {:?}", a.layer_device);
        let host_tensors: usize = a.on_cpu.iter().map(|v| v.iter().filter(|&&x| x).count()).sum();
        assert!(host_tensors > 0 && host_tensors < 72);
        assert!(a.bytes["CUDA0"] <= 12 * GB);
        assert!(a.bytes["CUDA0"] > 11 * GB, "spare memory is filled: {}", a.bytes["CUDA0"]);
        inp.host_experts = true;
        let ablated = search(&inp).unwrap();
        assert!(ablated.on_cpu.iter().all(|v| v.iter().all(|&h| h)), "every expert stays in host memory");
        assert!(ablated.decode_s > a.decode_s);
    }
}
