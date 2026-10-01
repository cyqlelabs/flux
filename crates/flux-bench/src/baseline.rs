//! The strongest llama.cpp configuration a careful user would try, found with the same tuning
//! budget Flux spent: llama.cpp's own memory fit, then for MoE models the fewest `--n-cpu-moe` blocks
//! that load with a layer split balanced to each GPU's free memory, then single-GPU and plain splits.

use crate::client::{self, Prompt};
use crate::harness::{Cell, Prompts};
use crate::servers::{start, Contestant, Kind};
use flux_core::config::FluxConfig;
use flux_core::hardware::BackendDevice;
use flux_core::model::{ModelManifest, TensorRole};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tried {
    pub contestant: Contestant,
    pub score: Option<f64>,
    pub error: Option<String>,
    pub seconds: f64,
}

fn common_args(model: &Path, cell: &Cell, threads: u32) -> Vec<String> {
    vec![
        "--model".into(),
        model.display().to_string(),
        "--ctx-size".into(),
        (cell.n_ctx_seq() * cell.concurrency).to_string(),
        "--parallel".into(),
        cell.concurrency.to_string(),
        "--flash-attn".into(),
        "on".into(),
        "--threads".into(),
        threads.to_string(),
    ]
}

/// `--n-cpu-moe k` configurations of a MoE model. `--n-cpu-moe` keeps the experts of the first k blocks
/// in RAM, so llama.cpp's default split (by free memory) overfills the GPUs holding later blocks; each
/// configuration instead fills the GPUs in order, as a careful user would set `--tensor-split`.
pub struct MoeSweep {
    common: Vec<String>,
    /// Per block: bytes that stay on its GPU, and routed-expert bytes.
    blocks: Vec<(u64, u64)>,
    head: u64,
    gpus: Vec<(String, u64)>,
}

/// Room left on each GPU for its share of the KV cache, compute buffers and the CUDA context; a split
/// that does not fit fails to load and counts as infeasible, as it would for a user trying it.
const GPU_MARGIN: u64 = 512 << 20;

impl MoeSweep {
    pub fn new(m: &ModelManifest, model: &Path, cell: &Cell, gpus: &[BackendDevice], threads: u32) -> Option<MoeSweep> {
        let facts = m.facts.as_ref()?;
        facts.moe.as_ref()?;
        let n = (facts.n_layer + facts.n_layer_nextn) as usize;
        let mut blocks = vec![(0u64, 0u64); n];
        let mut head = 0;
        for t in &m.tensors {
            match (t.layer().map(|l| l as usize).filter(|&l| l < n), t.role()) {
                (Some(l), TensorRole::FfnRoutedExpert) => blocks[l].1 += t.bytes,
                (Some(l), _) => blocks[l].0 += t.bytes,
                (None, TensorRole::Output | TensorRole::OutputNorm) => head += t.bytes,
                _ => {}
            }
        }
        (!gpus.is_empty() && blocks.iter().any(|b| b.1 > 0)).then(|| MoeSweep {
            common: common_args(model, cell, threads),
            blocks,
            head,
            gpus: gpus.iter().map(|g| (g.name.clone(), g.mem_free)).collect(),
        })
    }

    pub fn n_blocks(&self) -> u32 {
        self.blocks.len() as u32
    }

    /// Blocks per GPU in order (the last also holds the output head), or None when they cannot fit.
    fn split(&self, k: u32) -> Option<Vec<u32>> {
        let mut counts = vec![0u32; self.gpus.len()];
        let mut g = 0;
        let mut left = self.gpus[0].1.saturating_sub(GPU_MARGIN);
        for (i, &(dense, experts)) in self.blocks.iter().enumerate() {
            let need = dense + if (i as u32) < k { 0 } else { experts };
            while need > left {
                g += 1;
                left = self.gpus.get(g)?.1.saturating_sub(GPU_MARGIN);
            }
            left -= need;
            counts[g] += 1;
        }
        (self.head <= left).then_some(())?;
        counts[g] += 1;
        Some(counts)
    }

    pub fn config(&self, k: u32) -> Option<Contestant> {
        let counts = self.split(k)?;
        let mut a = self.common.clone();
        a.extend(["--fit", "off", "--n-gpu-layers", "999", "--n-cpu-moe", &k.to_string()].map(String::from));
        a.extend(["--device".to_string(), self.gpus.iter().map(|g| g.0.as_str()).collect::<Vec<_>>().join(",")]);
        if self.gpus.len() >= 2 {
            a.extend(["--tensor-split".to_string(), counts.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")]);
        }
        Some(Contestant { label: format!("llama-server: all layers, {k} MoE blocks on CPU, split {counts:?}"), kind: Kind::LlamaServer { args: a } })
    }
}

pub fn configs(model: &Path, cell: &Cell, gpus: &[BackendDevice], threads: u32) -> Vec<Contestant> {
    let common = |extra: &[&str]| -> Vec<String> {
        let mut a = common_args(model, cell, threads);
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    };
    let mut v = vec![("llama.cpp auto-fit".to_string(), common(&["--fit", "on"]))];
    if let Some(g) = gpus.first() {
        v.push((format!("auto-fit on {} only", g.name), common(&["--fit", "on", "--device", &g.name])));
        v.push((format!("all layers on {}", g.name), common(&["--fit", "off", "--n-gpu-layers", "999", "--device", &g.name])));
    }
    if gpus.len() >= 2 {
        v.push(("all layers, split by free memory".into(), common(&["--fit", "off", "--n-gpu-layers", "999"])));
        v.push(("all layers, split 2:1 (report)".into(), common(&["--fit", "off", "--n-gpu-layers", "999", "--tensor-split", "2,1"])));
    }
    v.into_iter().map(|(label, args)| Contestant { label: format!("llama-server: {label}"), kind: Kind::LlamaServer { args } }).collect()
}

/// Measures configurations until `budget_s` is spent and returns them best first. After the first
/// (llama.cpp's auto-fit), a MoE sweep bisects for the fewest CPU MoE blocks that load.
#[allow(clippy::too_many_arguments)]
pub async fn tune(
    cfg: &FluxConfig,
    configs: Vec<Contestant>,
    moe: Option<&MoeSweep>,
    cell: &Cell,
    prompts: &Prompts,
    model: &Path,
    budget_s: f64,
    logdir: &Path,
    log: &(dyn Fn(&str) + Sync),
) -> Vec<Tried> {
    let t_start = Instant::now();
    let mut tried = vec![];
    let mut configs = configs.into_iter();
    if let Some(c) = configs.next() {
        tried.push(try_config(cfg, c, cell, prompts, model, logdir, tried.len(), log).await);
    }
    if let Some(sweep) = moe {
        // All experts in RAM loads if anything does; then halve toward the fewest blocks that still load.
        let (mut lo, mut hi) = (0, sweep.n_blocks());
        let mut hi_ok = false;
        let mut k = hi;
        while t_start.elapsed().as_secs_f64() <= budget_s {
            let ok = match sweep.config(k) {
                Some(c) => {
                    let t = try_config(cfg, c, cell, prompts, model, logdir, tried.len(), log).await;
                    let ok = t.score.is_some();
                    tried.push(t);
                    ok
                }
                None => false,
            };
            if ok {
                hi = k;
                hi_ok = true;
            } else if !hi_ok {
                break;
            } else {
                lo = k;
            }
            if hi - lo <= 1 {
                break;
            }
            k = (lo + hi) / 2;
        }
    }
    for c in configs {
        if t_start.elapsed().as_secs_f64() > budget_s && !tried.is_empty() {
            tried.push(Tried { contestant: c, score: None, error: Some("not tried: tuning budget spent".into()), seconds: 0.0 });
            continue;
        }
        tried.push(try_config(cfg, c, cell, prompts, model, logdir, tried.len(), log).await);
    }
    tried.sort_by(|a, b| b.score.unwrap_or(-1.0).total_cmp(&a.score.unwrap_or(-1.0)));
    tried
}

#[allow(clippy::too_many_arguments)]
async fn try_config(
    cfg: &FluxConfig,
    c: Contestant,
    cell: &Cell,
    prompts: &Prompts,
    model: &Path,
    logdir: &Path,
    index: usize,
    log: &(dyn Fn(&str) + Sync),
) -> Tried {
    let http = reqwest::Client::new();
    log(&format!("baseline tuning: {}", c.label));
    let t0 = Instant::now();
    let file = logdir.join(format!("tune-{index}.log"));
    let (score, error) = match start(cfg, &c, model, cell.n_ctx_seq() * cell.concurrency, &file).await {
        Err(e) => (None, Some(format!("{e:#}"))),
        Ok(server) => {
            let mut rates = vec![];
            let mut err = None;
            for (i, p) in prompts.tokens.iter().take(3).enumerate() {
                let s = client::stream(&http, &server.base, c.api(), Prompt::Tokens(p), if i == 0 { 8 } else { 64 }).await;
                match (s.error, i) {
                    (Some(e), _) => {
                        err = Some(e);
                        break;
                    }
                    (None, 0) => {}
                    (None, _) => rates.extend(flux_core::stats::interactive_rate(&s.token_times_s)),
                }
            }
            server.stop().await;
            rates.sort_by(f64::total_cmp);
            (rates.get(rates.len() / 2).copied(), err)
        }
    };
    match (&score, &error) {
        (Some(s), _) => log(&format!("  {s:.2} tok/s")),
        (None, Some(e)) => log(&format!("  failed: {}", e.lines().next().unwrap_or_default())),
        _ => {}
    }
    Tried { contestant: c, score, error, seconds: t0.elapsed().as_secs_f64() }
}
