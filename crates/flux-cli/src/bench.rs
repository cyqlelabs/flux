use crate::planning;
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use flux_bench::harness::{build_prompts, Cell};
use flux_bench::servers::{Contestant, Kind};
use flux_core::backend::BackendParams;
use flux_core::config::FluxConfig;
use flux_core::plan::EngineKind;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Subcommand)]
pub enum BenchCmd {
    /// Paired, randomized trials of Flux against the strongest tuned baseline.
    Run {
        model: PathBuf,
        /// Prompt tokens x concurrency per cell; output is --output tokens.
        #[arg(long, value_delimiter = ',', default_value = "512x1,8kx1,32kx1,512x4,512x16")]
        cells: Vec<String>,
        #[arg(long, default_value_t = 256)]
        output: u32,
        #[arg(long, default_value_t = 10)]
        trials: usize,
        #[arg(long, default_value_t = 3)]
        prompts: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// flux.toml engines to include as baselines (repeatable), e.g. strata.
        #[arg(long = "external")]
        externals: Vec<String>,
        #[arg(long, default_value_t = 1000)]
        serving_p95_ms: u32,
        /// Let Flux plans measure speculative decoding.
        #[arg(long)]
        speculation: bool,
        /// Warm steady-state trials: keep the page cache instead of evicting the model before each run.
        #[arg(long)]
        warm: bool,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Same-artifact quality of a plan against a reference placement (KL divergence, top-1, perplexity).
    Quality {
        plan: String,
        /// `cpu` or a single backend device such as CUDA0.
        #[arg(long, default_value = "cpu")]
        reference: String,
        #[arg(long, default_value_t = 16)]
        chunks: u32,
    },
    /// Same plan on the native engine and on llama-server: templates, special tokens, sampling and stops must match.
    Conformance { plan: String },
    /// Soak `flux serve` with overload, cancellations and injected faults.
    Soak {
        plan: String,
        #[arg(long, default_value = "30m")]
        duration: String,
        #[arg(long, default_value_t = 4)]
        clients: usize,
        #[arg(long, default_value_t = 0.2)]
        cancel: f64,
        #[arg(long)]
        kill_every: Option<String>,
        #[arg(long)]
        pressure_every: Option<String>,
        #[arg(long, default_value = "10m")]
        timeout: String,
    },
    /// Mixed arrivals (Poisson, varied lengths) plus a long-running chat, on Flux and on llama.cpp auto-fit.
    Arrivals {
        plan: String,
        #[arg(long, default_value = "10m")]
        duration: String,
        /// Mean requests per second.
        #[arg(long, default_value_t = 0.2)]
        rate: f64,
        #[arg(long, default_value_t = 12)]
        chat_turns: usize,
    },
    /// Ablations: the full plan against variants that each remove one optimization.
    Ablate {
        model: PathBuf,
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        #[arg(long, default_value_t = 300.0)]
        budget_s: f64,
    },
    /// Print the summary of a finished suite.
    Show { dir: PathBuf },
    /// Pass/fail matrix over every saved plan: speed gate, quality, conformance and soak results.
    Matrix {
        /// Additional directories holding suite bundles.
        #[arg(long = "bench-dir")]
        bench_dirs: Vec<PathBuf>,
    },
}

pub fn parse_duration(s: &str) -> Result<Duration> {
    let (num, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()));
    let n: u64 = num.parse().with_context(|| format!("bad duration {s}"))?;
    Ok(Duration::from_secs(match unit {
        "" | "s" => n,
        "m" => n * 60,
        "h" => n * 3600,
        _ => bail!("duration unit must be s, m or h: {s}"),
    }))
}

pub async fn run(cfg: FluxConfig, cmd: BenchCmd) -> Result<()> {
    match cmd {
        BenchCmd::Run { model, cells, output, trials, prompts, seed, externals, serving_p95_ms, speculation, warm, out } => {
            let m = planning::inspect_hashed(&cfg, &model)?;
            let report = planning::probe_report(&cfg, Some(&m), false, false).await?;
            let corpus = planning::corpus(&cfg).await?;
            let cells = cells.iter().map(|c| Cell::parse(c, output)).collect::<Result<Vec<_>>>()?;
            let out = out.unwrap_or_else(|| {
                cfg.data_dir.join("bench").join(format!(
                    "{}-{}",
                    chrono::Utc::now().format("%Y%m%d-%H%M%S"),
                    m.identity.as_deref().map_or("model", |i| &i[..12])
                ))
            });
            let opts = flux_bench::suite::SuiteOptions {
                cells,
                trials,
                prompts_per_trial: prompts,
                seed,
                externals,
                out_dir: out.clone(),
                serving_p95_ms,
                engines: vec![EngineKind::Native, EngineKind::LlamaServer],
                speculation,
                warm,
            };
            let r = flux_bench::suite::run_suite(&cfg, &m, &report, &corpus, &opts, &planning::log).await?;
            print!("{}", flux_bench::suite::render(&r));
            println!("bundle      {}", out.display());
        }
        BenchCmd::Quality { plan, reference, chunks } => {
            let plan = planning::find_plan(&cfg, &plan)?;
            let corpus = planning::corpus(&cfg).await?;
            let mut r: BackendParams = plan.backend_params();
            r.model = plan.model_files[0].clone();
            r.overrides.clear();
            r.split_mode = flux_core::plan::SplitMode::Layer;
            if reference == "cpu" {
                r.devices.clear();
                r.n_gpu_layers = 0;
                r.tensor_split.clear();
            } else {
                r.devices = vec![reference.clone()];
                r.n_gpu_layers = 999;
                r.tensor_split = vec![1.0];
            }
            // llama.cpp choosing its own GPU placement: the divergence users already accept.
            let accepted = vec![
                "--model".to_string(),
                plan.model_files[0].display().to_string(),
                "--fit".into(),
                "on".into(),
                "--flash-attn".into(),
                "on".into(),
                "--threads".into(),
                plan.runtime.n_threads.to_string(),
            ];
            // Reference logits can be gigabytes: they live in the cache; only the result goes to the data dir.
            let work = cfg.cache_dir.join("quality").join(&plan.id);
            let q =
                flux_bench::quality::check(&cfg, &plan, (&reference, r.tool_args()), Some(("llama.cpp auto-fit", accepted)), &corpus, chunks, &work).await?;
            println!("plan        {} vs {} reference, {} chunks of 512 held-out tokens", q.plan, q.reference, q.chunks);
            if let Some((label, d)) = &q.accepted {
                println!("accepted    {label}: ppl ratio {:?}, KL {:?}, top-1 {:?}", d.ppl_ratio, d.kl_mean, d.top1_agreement);
            }
            println!(
                "plan        ppl {:?} (reference {:?}), ratio {:?}, KL {:?}, top-1 {:?}",
                q.ppl_candidate, q.ppl_reference, q.ppl_ratio, q.kl_mean, q.top1_agreement
            );
            println!("gate        {}{}", if q.gate.pass { "PASS" } else { "FAIL" }, q.gate.reasons.iter().map(|r| format!("\n  - {r}")).collect::<String>());
            flux_core::fsutil::write_json_atomic(&cfg.data_dir.join("quality").join(format!("{}.json", plan.id)), &q)?;
        }
        BenchCmd::Soak { plan, duration, clients, cancel, kill_every, pressure_every, timeout } => {
            let p = planning::find_plan(&cfg, &plan)?;
            let corpus = planning::corpus(&cfg).await?;
            let dir = cfg.data_dir.join("soak").join(format!("{}-{}", p.id, chrono::Utc::now().format("%Y%m%d-%H%M%S")));
            std::fs::create_dir_all(&dir)?;
            let len = (p.workload.n_ctx_seq / 2).clamp(64, 1024);
            let cell = Cell { prompt_tokens: len, output_tokens: 256, concurrency: 1 };
            let tokenizer = Contestant { label: "flux".into(), kind: Kind::Flux { plan: p.id.clone() } };
            let ps = build_prompts(&cfg, &tokenizer, &p.model_files[0], &cell, 16, &corpus, &dir).await?;
            let prompts: Vec<Vec<i32>> = ps.tokens.into_iter().enumerate().map(|(i, t)| t[..(32 + i * 61).min(t.len())].to_vec()).collect();
            let opts = flux_bench::soak::SoakOptions {
                duration: parse_duration(&duration)?,
                clients,
                cancel_fraction: cancel,
                kill_every: kill_every.as_deref().map(parse_duration).transpose()?,
                pressure_every: pressure_every.as_deref().map(parse_duration).transpose()?,
                request_timeout: parse_duration(&timeout)?,
                seed: 1,
            };
            let r = flux_bench::soak::soak(&cfg, &p.id, prompts, &opts, &dir, &planning::log).await?;
            println!("{}", serde_json::to_string_pretty(&r)?);
            flux_core::fsutil::write_json_atomic(&dir.join("soak.json"), &r)?;
            println!("soak        {} · {}", if r.pass { "PASS" } else { "FAIL" }, dir.display());
        }
        BenchCmd::Conformance { plan } => {
            let p = planning::find_plan(&cfg, &plan)?;
            let corpus = planning::corpus(&cfg).await?;
            let checks = flux_bench::conformance::check(&p, &corpus, &cfg.data_dir.join("logs")).await?;
            for c in &checks {
                println!("{} {:<58} {}", if c.pass { "✓" } else { "✗" }, c.name, c.detail);
            }
            let failed = checks.iter().filter(|c| !c.pass).count();
            println!("conformance {}: {} of {} checks pass", if failed == 0 { "PASS" } else { "FAIL" }, checks.len() - failed, checks.len());
            flux_core::fsutil::write_json_atomic(&cfg.data_dir.join("conformance").join(format!("{}.json", p.id)), &checks)?;
        }
        BenchCmd::Matrix { bench_dirs } => {
            let dirs: Vec<&std::path::Path> = bench_dirs.iter().map(|d| d.as_path()).collect();
            print!("{}", flux_bench::matrix::render(&flux_bench::matrix::collect(&cfg, &dirs)));
        }
        BenchCmd::Ablate { model, ctx, budget_s } => ablate(cfg, &model, ctx, budget_s).await?,
        BenchCmd::Arrivals { plan, duration, rate, chat_turns } => {
            let p = planning::find_plan(&cfg, &plan)?;
            let corpus = planning::corpus(&cfg).await?;
            let dir = cfg.data_dir.join("arrivals").join(format!("{}-{}", p.id, chrono::Utc::now().format("%Y%m%d-%H%M%S")));
            std::fs::create_dir_all(&dir)?;
            let o = flux_bench::arrivals::ArrivalOptions { duration: parse_duration(&duration)?, rate, chat_turns, seed: 5 };
            let cell = Cell { prompt_tokens: p.workload.n_ctx_seq.saturating_sub(272), output_tokens: 256, concurrency: p.workload.concurrency };
            let mut auto = flux_bench::baseline::configs(&p.model_files[0], &cell, &[], 0).into_iter().next().context("no baseline")?;
            if let Kind::LlamaServer { args } = &mut auto.kind {
                // Same thread count as the plan; chat templates through jinja like Flux.
                let t = args.iter().position(|a| a == "--threads").unwrap();
                args[t + 1] = p.runtime.n_threads.to_string();
                args.push("--jinja".into());
            }
            let flux = Contestant { label: format!("flux {}", p.id), kind: Kind::Flux { plan: p.id.clone() } };
            let ctx_total = p.workload.n_ctx_seq * p.workload.concurrency;
            let mut reports = vec![];
            for c in [&flux, &auto] {
                planning::log(&format!("arrivals: {}", c.label));
                reports.push(flux_bench::arrivals::run(&cfg, c, &p.model_files[0], ctx_total, &corpus, &o, &dir).await?);
            }
            for r in &reports {
                println!(
                    "{:<44} {} requests, {} failed · {:.1} tok/s overall · TTFT p50 {} p95 {} ms · decode p50 {} tok/s",
                    r.contestant,
                    r.requests,
                    r.failures,
                    r.tokens_per_s,
                    r.ttft_ms.map_or("—".into(), |x| format!("{:.0}", x.p50)),
                    r.ttft_ms.map_or("—".into(), |x| format!("{:.0}", x.p95)),
                    r.decode_tps.map_or("—".into(), |x| format!("{:.1}", x.p50))
                );
                let turns: Vec<String> = r.chat_turn_ttft_ms.iter().map(|t| format!("{t:.0}")).collect();
                println!("{:<44} chat-turn TTFT ms: {}", "", turns.join(" "));
            }
            flux_core::fsutil::write_json_atomic(&dir.join("arrivals.json"), &reports)?;
        }
        BenchCmd::Show { dir } => print!("{}", flux_bench::suite::render(&flux_bench::suite::load(&dir)?)),
    }
    Ok(())
}

async fn ablate(mut cfg: FluxConfig, model: &std::path::Path, ctx: u32, budget_s: f64) -> Result<()> {
    use flux_core::plan::{Objective, Workload};
    cfg.plan.tuning_budget_s = budget_s;
    let m = planning::inspect_hashed(&cfg, model)?;
    let facts = m.facts.clone().context("no architecture facts")?;
    let report = planning::probe_report(&cfg, Some(&m), false, false).await?;
    let corpus = planning::corpus(&cfg).await?;
    let base = flux_plan::planner::PlanRequest {
        workload: Workload { n_ctx_seq: ctx, concurrency: 1, objective: Objective::Interactive },
        engines: vec![EngineKind::Native, EngineKind::LlamaServer],
        type_k: "f16".into(),
        type_v: "f16".into(),
        n_ubatch: 512,
        allow_storage_streaming: false,
        prompt_tokens: 512,
        decode_tokens: 64,
        speculation: false,
        expert_residency: true,
        min_decode_tps: None,
        mlock: false,
        draft_model: None,
    };
    let mut variants = vec![("full plan", base.clone())];
    if facts.moe.is_some() {
        variants.push(("no expert residency (all experts in RAM)", flux_plan::planner::PlanRequest { expert_residency: false, ..base.clone() }));
    }
    variants.push(("native engine only", flux_plan::planner::PlanRequest { engines: vec![EngineKind::Native], ..base.clone() }));
    variants.push(("llama-server engine only", flux_plan::planner::PlanRequest { engines: vec![EngineKind::LlamaServer], ..base.clone() }));
    variants
        .push(("KV cache q8_0 (changed-quality profile)", flux_plan::planner::PlanRequest { type_k: "q8_0".into(), type_v: "q8_0".into(), ..base.clone() }));
    if facts.n_layer_nextn > 0 {
        variants.push(("speculation (next-token heads)", flux_plan::planner::PlanRequest { speculation: true, ..base.clone() }));
    }
    let mut rows = vec![];
    for (name, req) in variants {
        planning::log(&format!("ablation: {name}"));
        let r = flux_plan::planner::plan(&cfg, &m, &report, &corpus, &req, &planning::log).await;
        let row = match r {
            Ok(p) => {
                let v = p.validation.as_ref().context("plan without validation")?;
                let c = v.candidates.iter().find(|c| c.label == v.chosen).context("chosen candidate missing")?;
                let meas = c.validation.as_ref().or(c.calibration.as_ref());
                format!(
                    "{:<44} {:>8} tok/s  TTFT {:>6} ms  {}",
                    name,
                    meas.map_or("—".into(), |x| format!("{:.2}", x.decode_tps.p50)),
                    meas.map_or("—".into(), |x| format!("{:.0}", x.ttft_ms.p50)),
                    v.chosen
                )
            }
            Err(e) => format!("{:<44} failed: {}", name, format!("{e:#}").lines().next().unwrap_or_default()),
        };
        rows.push(row);
    }
    // Planner-free reference: llama.cpp choosing everything itself.
    let cell = Cell { prompt_tokens: 512, output_tokens: 64, concurrency: 1 };
    let auto = flux_bench::baseline::configs(&m.files[0].path, &Cell { prompt_tokens: ctx.saturating_sub(272), ..cell }, &[], report.inventory.cpu.cores)
        .into_iter()
        .next()
        .context("no baseline configuration")?;
    let dir = cfg.data_dir.join("ablations");
    std::fs::create_dir_all(&dir)?;
    let tokenizer = auto.clone();
    let prompts = build_prompts(&cfg, &tokenizer, &m.files[0].path, &cell, 3, &corpus, &dir).await?;
    let tried = flux_bench::baseline::tune(&cfg, vec![auto], None, &cell, &prompts, &m.files[0].path, f64::MAX, &dir, &planning::log).await;
    rows.push(format!("{:<44} {:>8} tok/s  (no planner)", "llama.cpp auto-fit", tried[0].score.map_or("—".into(), |s| format!("{s:.2}"))));
    for r in rows {
        println!("{r}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations() {
        assert_eq!(parse_duration("90").unwrap(), Duration::from_secs(90));
        assert_eq!(parse_duration("10m").unwrap(), Duration::from_secs(600));
        assert_eq!(parse_duration("24h").unwrap(), Duration::from_secs(86400));
        assert!(parse_duration("3d").is_err());
    }
}
