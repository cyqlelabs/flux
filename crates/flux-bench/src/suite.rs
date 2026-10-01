//! One benchmark suite for one model: plans, tuned baselines, paired trials, gates and the bundle.

use crate::baseline::{self, Tried};
use crate::harness::{build_prompts, run_cell, Cell, CellRun, TrialRecord};
use crate::servers::{Contestant, Kind};
use crate::stats::{speed_gate, CellTrials, SpeedGate};
use anyhow::{Context, Result};
use flux_core::config::FluxConfig;
use flux_core::corpus::Corpus;
use flux_core::hardware::ProbeReport;
use flux_core::model::ModelManifest;
use flux_core::plan::{EngineKind, Objective, Plan, Workload};
use flux_core::stats::Summary;
use flux_plan::planner::{plan, PlanRequest};
use flux_plan::store::PlanStore;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const SCHEMA: u32 = 1;

#[derive(Debug, Clone)]
pub struct SuiteOptions {
    pub cells: Vec<Cell>,
    pub trials: usize,
    pub prompts_per_trial: usize,
    pub seed: u64,
    /// flux.toml engines to include as additional baselines when they list the architecture.
    pub externals: Vec<String>,
    pub out_dir: PathBuf,
    pub serving_p95_ms: u32,
    pub engines: Vec<EngineKind>,
    pub speculation: bool,
    /// Keep the page cache between runs (warm steady state) instead of starting every run cold.
    pub warm: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContestantSummary {
    pub label: String,
    pub successes: usize,
    pub failures: Vec<String>,
    pub score: Option<Summary>,
    pub ttft_ms: Option<Summary>,
    pub token_ms: Option<Summary>,
    pub load_s: Option<Summary>,
    pub peak_device_bytes: Vec<(String, u64)>,
    pub peak_rss: u64,
    pub load_read_bytes: u64,
    pub cpu_cores_busy: f64,
    pub peak_power_w: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellResult {
    pub cell: Cell,
    pub flux_plan: Option<String>,
    pub best_baseline: Option<String>,
    pub contestants: Vec<ContestantSummary>,
    pub baseline_tuning: Vec<Tried>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteReport {
    pub schema: u32,
    pub created: chrono::DateTime<chrono::Utc>,
    pub flux_version: String,
    pub backend_revision: String,
    pub topology: String,
    pub host: flux_core::hardware::HardwareInventory,
    pub model_files: Vec<(PathBuf, Option<String>)>,
    pub model_identity: Option<String>,
    pub architecture: String,
    pub trials: usize,
    pub seed: u64,
    #[serde(default)]
    pub warm: bool,
    pub cells: Vec<CellResult>,
    pub speed_gate: SpeedGate,
    pub plans: Vec<Plan>,
    pub records: Vec<TrialRecord>,
}

fn summarize(label: &str, recs: &[&TrialRecord], concurrency: u32) -> ContestantSummary {
    let ok: Vec<&&TrialRecord> = recs.iter().filter(|r| r.score(concurrency).is_some()).collect();
    let all = |f: &dyn Fn(&TrialRecord) -> Vec<f64>| Summary::of(&ok.iter().flat_map(|r| f(r)).collect::<Vec<_>>());
    let mut peaks: std::collections::BTreeMap<String, u64> = Default::default();
    let mut power: std::collections::BTreeMap<String, f64> = Default::default();
    for r in &ok {
        for (k, v) in &r.device_peak_added {
            let e = peaks.entry(k.clone()).or_default();
            *e = (*e).max(*v);
        }
        for (k, v) in &r.device_power_peak_w {
            let e = power.entry(k.clone()).or_default();
            *e = e.max(*v);
        }
    }
    ContestantSummary {
        label: label.into(),
        successes: ok.len(),
        failures: recs.iter().filter_map(|r| r.error.clone()).collect(),
        score: Summary::of(&ok.iter().filter_map(|r| r.score(concurrency)).collect::<Vec<_>>()),
        ttft_ms: all(&|r| r.streams.iter().map(|s| s.ttft_s * 1e3).collect()),
        token_ms: all(&|r| r.streams.iter().flat_map(|s| s.token_times_s.windows(2).map(|w| (w[1] - w[0]) * 1e3).collect::<Vec<_>>()).collect()),
        load_s: Summary::of(&ok.iter().map(|r| r.load_s).collect::<Vec<_>>()),
        peak_device_bytes: peaks.into_iter().collect(),
        peak_rss: ok.iter().map(|r| r.peak_rss).max().unwrap_or(0),
        load_read_bytes: ok.iter().map(|r| r.load_read_bytes).max().unwrap_or(0),
        cpu_cores_busy: ok.iter().map(|r| r.cpu_cores_busy).sum::<f64>() / ok.len().max(1) as f64,
        peak_power_w: power.into_iter().collect(),
    }
}

/// The plan for this workload: the saved one for the profile key, or a freshly measured one.
#[allow(clippy::too_many_arguments)]
async fn plan_for(
    cfg: &FluxConfig,
    m: &ModelManifest,
    report: &ProbeReport,
    corpus: &Corpus,
    w: Workload,
    engines: &[EngineKind],
    speculation: bool,
    log: &(dyn Fn(&str) + Sync),
) -> Result<Plan> {
    let store = PlanStore::new(&cfg.plans_dir());
    if let Some(p) = store.list().into_iter().find(|p| {
        Some(&p.key.model_identity) == m.identity.as_ref()
            && p.key.topology == report.topology
            && p.key.backend_revision == report.backend_revision
            && p.workload.n_ctx_seq >= w.n_ctx_seq
            && p.workload.concurrency == w.concurrency
            && p.workload.objective == w.objective
            && p.validation.is_some()
    }) {
        log(&format!("reusing plan {} ({})", p.id, p.placement.describe()));
        return Ok(p);
    }
    let req = PlanRequest {
        workload: w,
        engines: engines.to_vec(),
        type_k: "f16".into(),
        type_v: "f16".into(),
        n_ubatch: 512,
        allow_storage_streaming: false,
        prompt_tokens: 512,
        decode_tokens: 64,
        speculation,
        expert_residency: true,
        min_decode_tps: None,
        mlock: false,
        draft_model: None,
    };
    let p = plan(cfg, m, report, corpus, &req, log).await?;
    store.save(&p)?;
    Ok(p)
}

pub async fn run_suite(
    cfg: &FluxConfig,
    m: &ModelManifest,
    report: &ProbeReport,
    corpus: &Corpus,
    opts: &SuiteOptions,
    log: &(dyn Fn(&str) + Sync),
) -> Result<SuiteReport> {
    let facts = m.facts.as_ref().context("model has no architecture facts")?;
    std::fs::create_dir_all(&opts.out_dir)?;
    let logdir = opts.out_dir.join("logs");
    std::fs::create_dir_all(&logdir)?;
    let files = m.paths();
    let gpus: Vec<_> = report.inventory.backend_devices.iter().filter(|d| d.kind == "gpu").cloned().collect();

    let mut cells = vec![];
    let mut plans: Vec<Plan> = vec![];
    let mut records = vec![];
    let mut trials_for_gate = vec![];
    for cell in &opts.cells {
        log(&format!("cell {}", cell.label()));
        let w = Workload {
            n_ctx_seq: cell.n_ctx_seq(),
            concurrency: cell.concurrency,
            objective: if cell.concurrency == 1 { Objective::Interactive } else { Objective::Serving { max_p95_token_ms: opts.serving_p95_ms } },
        };
        if facts.n_ctx_train > 0 && cell.n_ctx_seq() > facts.n_ctx_train {
            log("  unsupported: longer than the trained context");
            cells.push(CellResult { cell: *cell, flux_plan: None, best_baseline: None, contestants: vec![], baseline_tuning: vec![] });
            trials_for_gate.push(CellTrials { cell: cell.label(), flux: vec![None; opts.trials], baseline: vec![None; opts.trials] });
            continue;
        }
        let flux_plan = match plan_for(cfg, m, report, corpus, w, &opts.engines, opts.speculation, log).await {
            Ok(p) => Some(p),
            Err(e) => {
                log(&format!("  no feasible Flux plan: {e:#}"));
                None
            }
        };
        let flux = flux_plan.as_ref().map(|p| Contestant { label: format!("flux {}", p.id), kind: Kind::Flux { plan: p.id.clone() } });
        let tune_budget = flux_plan.as_ref().and_then(|p| p.validation.as_ref()).map_or(cfg.plan.tuning_budget_s, |v| v.tuning_seconds);
        let configs = baseline::configs(&files[0], cell, &gpus, report.inventory.cpu.cores);
        let sweep = baseline::MoeSweep::new(m, &files[0], cell, &gpus, report.inventory.cpu.cores);
        let n_prompts = opts.trials * opts.prompts_per_trial.max(cell.concurrency as usize);
        let tokenizer = flux.clone().unwrap_or_else(|| configs[0].clone());
        let prompts = build_prompts(cfg, &tokenizer, &files[0], cell, n_prompts.max(3), corpus, &logdir).await?;
        let tuning = baseline::tune(cfg, configs, sweep.as_ref(), cell, &prompts, &files[0], tune_budget, &logdir, log).await;
        let mut contestants: Vec<Contestant> = flux.into_iter().collect();
        if let Some(best) = tuning.iter().find(|t| t.score.is_some()) {
            contestants.push(best.contestant.clone());
        }
        for e in &opts.externals {
            if cfg.engines.get(e).is_some_and(|x| x.architectures.contains(&facts.architecture)) {
                contestants.push(Contestant { label: e.clone(), kind: Kind::External { engine: e.clone() } });
            }
        }
        let mut cold_files = if opts.warm { vec![] } else { files.clone() };
        if !opts.warm {
            cold_files.extend(flux_plan.iter().flat_map(|p| p.model_files.iter()).filter(|f| !files.contains(f)).cloned());
        }
        let run = CellRun {
            cfg,
            model_files: &files,
            cold_files: &cold_files,
            cell: *cell,
            contestants: &contestants,
            trials: opts.trials,
            prompts: &prompts,
            prompts_per_trial: opts.prompts_per_trial,
            seed: opts.seed,
            logdir: &logdir,
            devices: &report.inventory.backend_devices,
        };
        let recs = run_cell(&run, log).await;
        let per = |label: &str| -> Vec<&TrialRecord> { recs.iter().filter(|r| r.contestant == label).collect() };
        let summaries: Vec<ContestantSummary> = contestants.iter().map(|c| summarize(&c.label, &per(&c.label), cell.concurrency)).collect();
        // The strongest baseline in this cell is the one with the best median score.
        let best_baseline = summaries
            .iter()
            .filter(|s| !s.label.starts_with("flux "))
            .max_by(|a, b| a.score.map_or(-1.0, |s| s.p50).total_cmp(&b.score.map_or(-1.0, |s| s.p50)))
            .map(|s| s.label.clone());
        let score_by_trial = |label: Option<&String>| -> Vec<Option<f64>> {
            (0..opts.trials)
                .map(|t| label.and_then(|l| recs.iter().find(|r| &r.contestant == l && r.trial == t)).and_then(|r| r.score(cell.concurrency)))
                .collect()
        };
        let flux_label = flux_plan.as_ref().map(|p| format!("flux {}", p.id));
        trials_for_gate.push(CellTrials { cell: cell.label(), flux: score_by_trial(flux_label.as_ref()), baseline: score_by_trial(best_baseline.as_ref()) });
        cells.push(CellResult {
            cell: *cell,
            flux_plan: flux_plan.as_ref().map(|p| p.id.clone()),
            best_baseline,
            contestants: summaries,
            baseline_tuning: tuning,
        });
        if let Some(p) = flux_plan {
            if !plans.iter().any(|x| x.id == p.id) {
                plans.push(p);
            }
        }
        records.extend(recs);
    }

    let gate = speed_gate(&trials_for_gate, 2000, opts.seed);
    let rep = SuiteReport {
        schema: SCHEMA,
        created: chrono::Utc::now(),
        flux_version: env!("CARGO_PKG_VERSION").into(),
        backend_revision: report.backend_revision.clone(),
        topology: report.topology.clone(),
        host: report.inventory.clone(),
        model_files: m.files.iter().map(|f| (f.path.clone(), f.sha256.clone())).collect(),
        model_identity: m.identity.clone(),
        architecture: facts.architecture.clone(),
        trials: opts.trials,
        seed: opts.seed,
        warm: opts.warm,
        cells,
        speed_gate: gate,
        plans,
        records,
    };
    flux_core::fsutil::write_json_atomic(&opts.out_dir.join("report.json"), &rep)?;
    flux_core::fsutil::write_json_atomic(&opts.out_dir.join("probe.json"), report)?;
    std::fs::write(opts.out_dir.join("summary.txt"), render(&rep))?;
    Ok(rep)
}

/// Human-readable summary; the JSON bundle next to it holds every raw trial.
pub fn render(r: &SuiteReport) -> String {
    let mut s = String::new();
    let line = |s: &mut String, t: String| {
        s.push_str(&t);
        s.push('\n');
    };
    line(
        &mut s,
        format!("Flux {} · llama.cpp {} · topology {} · {}", r.flux_version, &r.backend_revision[..9], r.topology, r.created.format("%Y-%m-%d %H:%M UTC")),
    );
    line(
        &mut s,
        format!(
            "model {} ({}) · {} paired {} trials per cell, seed {}",
            r.model_files[0].0.display(),
            r.architecture,
            r.trials,
            if r.warm { "warm" } else { "cold" },
            r.seed
        ),
    );
    for c in &r.cells {
        line(&mut s, format!("\n{}  best baseline: {}", c.cell.label(), c.best_baseline.as_deref().unwrap_or("none")));
        for k in &c.contestants {
            let sc = k.score.map_or("—".to_string(), |x| format!("{:.2} tok/s (p50 of {} trials, min {:.2})", x.p50, x.n, x.min));
            let ttft = k.ttft_ms.map_or("—".to_string(), |x| format!("{:.0} ms", x.p50));
            let tok = k.token_ms.map_or("—".to_string(), |x| format!("p50 {:.1} / p95 {:.1} / p99 {:.1} ms", x.p50, x.p95, x.p99));
            let vram: Vec<String> = k.peak_device_bytes.iter().map(|(d, b)| format!("{d} {}", flux_core::fmt_bytes(*b))).collect();
            line(&mut s, format!("  {:<52} {sc} · TTFT {ttft} · token {tok}", k.label));
            line(
                &mut s,
                format!(
                    "  {:<52} load {} · VRAM {} · RSS {} · CPU {:.1} cores · failures {}",
                    "",
                    k.load_s.map_or("—".into(), |x| format!("{:.1} s", x.p50)),
                    vram.join(", "),
                    flux_core::fmt_bytes(k.peak_rss),
                    k.cpu_cores_busy,
                    k.failures.len()
                ),
            );
        }
    }
    let g = &r.speed_gate;
    line(
        &mut s,
        format!("\nspeed gate: {} · geometric-mean speedup {:.3} (95% CI {:.3}–{:.3})", if g.pass { "PASS" } else { "FAIL" }, g.geomean, g.ci_low, g.ci_high),
    );
    for c in &g.cells {
        line(&mut s, format!("  {:<16} {:.3}x (CI {:.3}–{:.3}, {} pairs)", c.cell, c.speedup, c.ci_low, c.ci_high, c.pairs));
    }
    for reason in &g.reasons {
        line(&mut s, format!("  - {reason}"));
    }
    s
}

pub fn load(dir: &Path) -> Result<SuiteReport> {
    flux_core::fsutil::read_json(&dir.join("report.json"))
}
