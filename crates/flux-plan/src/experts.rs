//! Per-expert residency. Whole expert tensors are the unit of `search`; with measured routing counts
//! the same GPU memory serves more selections when it holds each block's most-routed experts instead.
//! The backend runs that through a relabeled checkpoint (`flux_ingest::repack`): hot experts stay with
//! their block's GPU, the cold remainder of every split tensor is overridden to host memory.

use crate::cost::{CostModel, Shape, CPU};
use crate::layers::Layout;
use crate::search::Assignment;
use flux_core::plan::{SplitSpec, TensorOverride};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone)]
pub struct Residency {
    pub spec: SplitSpec,
    /// Blocks on a GPU whose expert tensors (whole, or the cold part when split) stay in host memory.
    pub host_blocks: Vec<usize>,
    pub hot_experts: u32,
    pub decode_s: f64,
    pub gpu_served: f64,
    pub gpu_served_by_tensors: f64,
}

/// Routing counts per layer and expert, from tracing calibration prompts.
pub type Routes = BTreeMap<u32, Vec<u64>>;

/// Chooses hot experts per GPU block by routing count per byte, within the expert bytes the base
/// assignment already keeps on each device plus `spare` (negative to shrink it). A split block also holds one all-zero expert
/// per routing slot on its GPU. Returns None when no block would be split.
pub fn choose(l: &Layout, a: &Assignment, cost: &CostModel, s: &Shape, routes: &Routes, spare: &HashMap<String, i64>) -> Option<Residency> {
    let n = l.n_expert as usize;
    if n == 0 || routes.is_empty() {
        return None;
    }
    let gpu_blocks: Vec<usize> = (0..l.blocks.len()).filter(|&i| a.layer_device[i] != CPU && !l.blocks[i].experts.is_empty()).collect();
    // Experts past the last one routed, and blocks never routed through (e.g. next-token heads), count as unused.
    let padded: Vec<Vec<u64>> = (0..l.blocks.len())
        .map(|i| {
            let mut c = routes.get(&(i as u32)).cloned().unwrap_or_default();
            c.resize(n, 0);
            c
        })
        .collect();
    let counts = |i: usize| &padded[i];
    let unit = |i: usize| l.blocks[i].expert_bytes() / n as u64;
    // Expert tensors the base keeps on the block's GPU; a missing flag means resident, as in the cost model.
    let resident = |i: usize| -> u64 { l.blocks[i].experts.iter().enumerate().filter(|(k, _)| a.on_cpu[i].get(*k) != Some(&true)).map(|(_, e)| e.bytes).sum() };
    let mut budget: HashMap<&str, i64> = HashMap::new();
    for &i in &gpu_blocks {
        *budget.entry(a.layer_device[i].as_str()).or_insert_with(|| spare.get(&a.layer_device[i]).copied().unwrap_or(0)) += resident(i) as i64;
    }
    // Greedy by selections per byte across the device's blocks; a block's first hot expert also pays its zero experts.
    let mut items: Vec<(usize, u32, f64)> = vec![];
    for &i in &gpu_blocks {
        for (e, &c) in counts(i).iter().enumerate() {
            items.push((i, e as u32, c as f64 / unit(i) as f64));
        }
    }
    items.sort_by(|x, y| y.2.total_cmp(&x.2).then(x.0.cmp(&y.0)).then(x.1.cmp(&y.1)));
    let mut hot: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for &(i, e, _) in &items {
        let left = budget.get_mut(a.layer_device[i].as_str()).unwrap();
        let first = !hot.contains_key(&i);
        let need = unit(i) as i64 * if first { 1 + l.n_expert_used as i64 } else { 1 };
        if need <= *left {
            *left -= need;
            hot.entry(i).or_default().push(e);
        }
    }
    let mut spec = SplitSpec::default();
    let (mut host_blocks, mut hot_experts) = (vec![], 0);
    let (mut served, mut served_tensors, mut total) = (0f64, 0f64, 0f64);
    let mut decode_s = a.decode_s;
    for &i in &gpu_blocks {
        let b = &l.blocks[i];
        let dev = a.layer_device[i].as_str();
        let c = counts(i);
        let sel: u64 = c.iter().sum();
        let ids = hot.remove(&i).unwrap_or_default();
        let hit: u64 = ids.iter().map(|&e| c[e as usize]).sum();
        let p = if sel == 0 { ids.len() as f64 / n as f64 } else { hit as f64 / sel as f64 };
        served += p * sel as f64;
        served_tensors += resident(i) as f64 / b.expert_bytes() as f64 * sel as f64;
        total += sel as f64;
        hot_experts += ids.len() as u32;
        // Interpolate the block between all experts resident and all in host memory by the share served.
        let t_gpu = cost.decode_block_s(l, b, dev, &[], s);
        let t_host = cost.decode_block_s(l, b, dev, &vec![true; b.experts.len()], s);
        let copy = cost.split_copy_s(l, dev, s.batch as u64);
        let t = match ids.len() {
            0 => t_host,
            k if k == n => t_gpu,
            _ => t_gpu + copy + (1.0 - p) * (t_host - copy - t_gpu),
        };
        decode_s += t - cost.decode_block_s(l, b, dev, &a.on_cpu[i], s);
        if ids.len() < n {
            host_blocks.push(i);
        }
        if !ids.is_empty() && ids.len() < n {
            let mut order = ids.clone();
            let mut cold: Vec<u32> = (0..n as u32).filter(|e| !ids.contains(e)).collect();
            cold.sort_by_key(|&e| (std::cmp::Reverse(c[e as usize]), e));
            order.extend(cold);
            spec.layers.insert(i as u32, (order, ids.len() as u32));
        }
    }
    if spec.layers.is_empty() {
        return None;
    }
    Some(Residency { spec, host_blocks, hot_experts, decode_s, gpu_served: served / total.max(1.0), gpu_served_by_tensors: served_tensors / total.max(1.0) })
}

/// Host overrides for the blocks whose experts (whole tensors, or cold parts of split ones) stay in RAM.
pub fn overrides(l: &Layout, r: &Residency) -> Vec<TensorOverride> {
    r.host_blocks
        .iter()
        .map(|&i| {
            let alts: Vec<String> = l.blocks[i].experts.iter().map(|e| e.name.replace('.', r"\.")).collect();
            TensorOverride { pattern: format!("^({})$", alts.join("|")), device: CPU.into() }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Block, ExpertTensor, Head};
    use flux_core::ggml_type::GgmlType;

    #[test]
    fn hot_experts_follow_routing_within_the_same_memory() {
        // Two GPU blocks of 4 experts at 100 bytes each; the base keeps block 0's tensor resident (no host flags).
        let block = |i: u32| Block {
            index: i,
            experts: vec![ExpertTensor { name: format!("blk.{i}.ffn_up_exps.weight"), ggml_type: GgmlType::Q8_0, bytes: 400, k: 32, n: 1 }],
            ..Default::default()
        };
        let l = Layout { blocks: vec![block(0), block(1)], head: Head::default(), n_expert: 4, n_expert_used: 1, n_embd: 32, n_vocab: 8 };
        let a = Assignment {
            devices: vec!["CUDA0".into()],
            layer_device: vec!["CUDA0".into(); 2],
            output_device: "CUDA0".into(),
            on_cpu: vec![vec![], vec![true]],
            decode_s: 1.0,
            bytes: Default::default(),
        };
        let routes = Routes::from([(0, vec![10, 0, 0, 0]), (1, vec![1, 50, 40, 0])]);
        let cost = crate::search::tests::cost();
        let s = crate::search::tests::shape();
        let r = choose(&l, &a, &cost, &s, &routes, &HashMap::from([("CUDA0".to_string(), 0)])).unwrap();
        // The 400 resident bytes go to block 1's three routed experts plus its zero expert.
        assert!(!r.spec.layers.contains_key(&0));
        assert_eq!(r.spec.layers[&1], (vec![1, 2, 0, 3], 3));
        assert_eq!(r.host_blocks, vec![0, 1]);
        assert!(r.gpu_served > r.gpu_served_by_tensors, "{} vs {}", r.gpu_served, r.gpu_served_by_tensors);
        assert_eq!(overrides(&l, &r)[1].pattern, r"^(blk\.1\.ffn_up_exps\.weight)$");
    }
}
