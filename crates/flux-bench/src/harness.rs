//! Paired trials: every trial runs each contestant once, in a seeded random order, cold.

use crate::client::{self, Prompt, Stream};
use crate::proc::{evict_page_cache, tree_cpu_s};
use crate::servers::{start, Contestant};
use anyhow::{Context, Result};
use flux_core::config::FluxConfig;
use flux_core::corpus::{Corpus, Role};
use flux_core::hardware::BackendDevice;
use flux_probe::sampler::Sampler;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub concurrency: u32,
}

impl Cell {
    pub fn label(&self) -> String {
        format!("p{}-o{}-c{}", self.prompt_tokens, self.output_tokens, self.concurrency)
    }

    /// Cells per sequence the plan must hold, padded like the backend pads.
    pub fn n_ctx_seq(&self) -> u32 {
        (self.prompt_tokens + self.output_tokens + 16).div_ceil(256) * 256
    }

    /// `512x1` → 512-token prompts at concurrency 1, 256 output tokens.
    pub fn parse(s: &str, output_tokens: u32) -> Result<Cell> {
        let (p, c) = s.split_once('x').context("cells look like 512x1")?;
        Ok(Cell {
            prompt_tokens: p.trim_end_matches(['k', 'K']).parse::<u32>()? * if p.ends_with(['k', 'K']) { 1024 } else { 1 },
            output_tokens,
            concurrency: c.parse()?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialRecord {
    pub cell: String,
    pub trial: usize,
    /// Position of this contestant in the trial's random order.
    pub order: usize,
    pub contestant: String,
    pub load_s: f64,
    /// Bytes read from storage during load (the model's pages were evicted first).
    pub load_read_bytes: u64,
    pub streams: Vec<Stream>,
    pub wall_s: f64,
    pub aggregate_tps: f64,
    pub device_peak_added: BTreeMap<String, u64>,
    pub device_power_peak_w: BTreeMap<String, f64>,
    pub peak_rss: u64,
    pub cpu_cores_busy: f64,
    pub error: Option<String>,
}

impl TrialRecord {
    /// The compared quantity: median single-stream decode rate at concurrency 1, aggregate tokens/s above.
    pub fn score(&self, concurrency: u32) -> Option<f64> {
        if self.error.is_some() || self.streams.iter().any(|s| s.error.is_some()) {
            return None;
        }
        if concurrency > 1 {
            return Some(self.aggregate_tps);
        }
        let mut r: Vec<f64> = self.streams.iter().filter_map(Stream::decode_rate).collect();
        r.sort_by(f64::total_cmp);
        (!r.is_empty()).then(|| r[r.len() / 2])
    }
}

/// Exact-length token prompts plus matching text for chat-level contestants.
pub struct Prompts {
    pub tokens: Vec<Vec<i32>>,
    pub texts: Vec<String>,
}

pub async fn build_prompts(
    cfg: &FluxConfig,
    tokenizer: &Contestant,
    model: &Path,
    cell: &Cell,
    count: usize,
    corpus: &Corpus,
    logdir: &Path,
) -> Result<Prompts> {
    let path = tokenizer.tokenize_path().context("the first contestant must accept token prompts")?;
    let r = start(cfg, tokenizer, model, cell.n_ctx_seq() * cell.concurrency, &logdir.join("tokenizer.log")).await?;
    let http = reqwest::Client::new();
    let n = cell.prompt_tokens as usize;
    let mut out = Prompts { tokens: vec![], texts: vec![] };
    let result = async {
        for i in 0..count {
            let text = corpus.text(Role::HeldOut, i * 3, n * 6);
            let toks = client::tokenize(&http, &r.base, path, &text).await?;
            anyhow::ensure!(toks.len() >= n, "held-out text too short for {n} tokens");
            let chars = (text.len() as f64 * n as f64 / toks.len() as f64) as usize;
            let cut = (0..=chars.min(text.len())).rev().find(|&c| text.is_char_boundary(c)).unwrap_or(0);
            out.tokens.push(toks[..n].to_vec());
            out.texts.push(text[..cut].to_string());
        }
        Ok::<(), anyhow::Error>(())
    }
    .await;
    r.stop().await;
    result.map(|_| out)
}

pub struct CellRun<'a> {
    pub cfg: &'a FluxConfig,
    pub model_files: &'a [PathBuf],
    /// Every file a contestant reads (the model, and a plan's split checkpoint): evicted before each run.
    pub cold_files: &'a [PathBuf],
    pub cell: Cell,
    pub contestants: &'a [Contestant],
    pub trials: usize,
    pub prompts: &'a Prompts,
    pub prompts_per_trial: usize,
    pub seed: u64,
    pub logdir: &'a Path,
    pub devices: &'a [BackendDevice],
}

pub async fn run_cell(run: &CellRun<'_>, log: &(dyn Fn(&str) + Sync)) -> Vec<TrialRecord> {
    let mut records = vec![];
    for t in 0..run.trials {
        let mut order: Vec<usize> = (0..run.contestants.len()).collect();
        order.shuffle(&mut rand::rngs::StdRng::seed_from_u64(run.seed + t as u64));
        for (pos, &ci) in order.iter().enumerate() {
            let c = &run.contestants[ci];
            log(&format!("{} trial {}/{}: {}", run.cell.label(), t + 1, run.trials, c.label));
            let rec = run_one(run, c, t, pos).await;
            match (&rec.error, rec.score(run.cell.concurrency)) {
                (Some(e), _) => log(&format!("  failed: {e}")),
                (None, Some(s)) => log(&format!("  {s:.2} tok/s · load {:.1} s · TTFT p50 {:.0} ms", rec.load_s, median_ttft_ms(&rec))),
                (None, None) => log("  no completed streams"),
            }
            records.push(rec);
        }
    }
    records
}

fn median_ttft_ms(r: &TrialRecord) -> f64 {
    let mut v: Vec<f64> = r.streams.iter().filter(|s| s.error.is_none()).map(|s| s.ttft_s * 1e3).collect();
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

async fn run_one(run: &CellRun<'_>, c: &Contestant, trial: usize, order: usize) -> TrialRecord {
    let mut rec = TrialRecord {
        cell: run.cell.label(),
        trial,
        order,
        contestant: c.label.clone(),
        load_s: 0.0,
        load_read_bytes: 0,
        streams: vec![],
        wall_s: 0.0,
        aggregate_tps: 0.0,
        device_peak_added: BTreeMap::new(),
        device_power_peak_w: BTreeMap::new(),
        peak_rss: 0,
        cpu_cores_busy: 0.0,
        error: None,
    };
    evict_page_cache(run.cold_files);
    let log = run.logdir.join(format!("{}-t{trial}-{}.log", run.cell.label(), c.label.replace(|ch: char| !ch.is_alphanumeric(), "_")));
    let sampler = Sampler::start(run.devices, None, Duration::from_millis(50));
    let server = match start(run.cfg, c, &run.model_files[0], run.cell.n_ctx_seq() * run.cell.concurrency, &log).await {
        Ok(s) => s,
        Err(e) => {
            sampler.finish();
            rec.error = Some(format!("{e:#}"));
            return rec;
        }
    };
    rec.load_s = server.load_s;
    rec.load_read_bytes = server.load_read_bytes;
    let http = reqwest::Client::new();
    let api = c.api();
    let pick = |i: usize| -> Prompt<'_> {
        let k = (trial * run.prompts_per_trial + i) % run.prompts.tokens.len();
        match api {
            client::Api::Chat => Prompt::Text(&run.prompts.texts[k]),
            _ => Prompt::Tokens(&run.prompts.tokens[k]),
        }
    };
    let warm = client::stream(&http, &server.base, api, pick(0), 8).await;
    if let Some(e) = warm.error {
        rec.error = Some(format!("warm-up: {e}"));
        sampler.finish();
        server.stop().await;
        return rec;
    }
    let peaks = Sampler::start(run.devices, Some(server.pid), Duration::from_millis(50));
    let cpu0 = tree_cpu_s(server.pid);
    let t0 = Instant::now();
    let n = run.prompts_per_trial.max(run.cell.concurrency as usize);
    let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(run.cell.concurrency as usize));
    let futs = (0..n).map(|i| {
        let (sem, http, base) = (sem.clone(), http.clone(), server.base.clone());
        let p = pick(i);
        let out = run.cell.output_tokens;
        async move {
            let _permit = sem.acquire().await;
            client::stream(&http, &base, api, p, out).await
        }
    });
    rec.streams = futures::future::join_all(futs).await;
    rec.wall_s = t0.elapsed().as_secs_f64();
    rec.cpu_cores_busy = (tree_cpu_s(server.pid) - cpu0) / rec.wall_s;
    let p = peaks.finish();
    let base = sampler.finish();
    // Device memory added since before the server started, so model and cache are included.
    rec.device_peak_added = p.device_used.iter().map(|(k, v)| (k.clone(), v.saturating_sub(base.device_baseline.get(k).copied().unwrap_or(0)))).collect();
    rec.device_power_peak_w = p.device_power_w;
    rec.peak_rss = p.tree_rss;
    rec.aggregate_tps = rec.streams.iter().map(|s| s.token_times_s.len()).sum::<usize>() as f64 / rec.wall_s;
    if let Some(e) = rec.streams.iter().find_map(|s| s.error.clone()) {
        rec.error = Some(e);
    }
    server.stop().await;
    rec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_parse_and_pad() {
        let c = Cell::parse("8kx4", 256).unwrap();
        assert_eq!((c.prompt_tokens, c.concurrency), (8192, 4));
        assert_eq!(c.n_ctx_seq(), 8704);
        assert_eq!(Cell::parse("512x1", 256).unwrap().label(), "p512-o256-c1");
    }
}
