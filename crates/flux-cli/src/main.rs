mod bench;
mod inspect;
mod planning;

use anyhow::Result;
use clap::{Parser, Subcommand};
use flux_core::config::FluxConfig;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "flux", version, about = "Measured LLM execution planning over llama.cpp and other engines")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Identify a model artifact: metadata, shards, hashes, compatibility.
    Inspect {
        path: PathBuf,
        /// Skip file hashing (no model identity; plans need it).
        #[arg(long)]
        no_hash: bool,
        #[arg(long)]
        json: bool,
    },
    /// Download files of a Hugging Face model at a pinned revision, verified and atomic.
    Fetch {
        repo: String,
        #[arg(long, default_value = "main")]
        revision: String,
        /// Glob of repository paths to fetch (repeatable).
        #[arg(long = "include", required = true)]
        include: Vec<String>,
        /// Destination; default `<models_dir>/<repo name>`.
        #[arg(long)]
        dest: Option<PathBuf>,
    },
    /// Convert a Hugging Face checkpoint to GGUF with the pinned converter (cached, atomic).
    Convert {
        source: PathBuf,
        /// f32, f16, bf16, q8_0 or auto.
        #[arg(long, default_value = "bf16")]
        outtype: String,
        /// Python with torch, numpy and transformers for the converter.
        #[arg(long, default_value = "python3")]
        python: PathBuf,
    },
    /// Download the WikiText-2 prompt corpus (calibration, validation and held-out sets).
    Corpus,
    /// Measure this machine: copy curves, contention, kernel shapes, CPU and storage bandwidth.
    Probe {
        /// Probe the kernel shapes and storage of this model.
        #[arg(long)]
        model: Option<PathBuf>,
        #[arg(long)]
        quick: bool,
    },
    /// Find, measure and save the fastest validated plan for a model and workload.
    Plan {
        model: PathBuf,
        /// Tokens per sequence the plan must hold (prompt + output).
        #[arg(long, default_value_t = 8192)]
        ctx: u32,
        /// Concurrent sequences the plan must hold.
        #[arg(long, default_value_t = 1)]
        concurrency: u32,
        /// Optimize aggregate tokens/s under this p95 per-token latency instead of single-stream speed.
        #[arg(long)]
        serving_p95_ms: Option<u32>,
        /// Engines to consider: native, llama-server.
        #[arg(long, value_delimiter = ',', default_value = "native,llama-server")]
        engines: Vec<String>,
        /// KV cache type; anything but f16 is a separately labelled quality profile.
        #[arg(long, default_value = "f16")]
        kv: String,
        /// Largest prompt chunk to try. Each chunk streams host-resident experts to the GPU once, so a chunk
        /// that holds the whole prompt reaches the first token sooner; smaller chunks are tried too.
        #[arg(long, default_value_t = 1024)]
        ubatch: u32,
        #[arg(long, default_value_t = 512)]
        prompt_tokens: u32,
        #[arg(long, default_value_t = 64)]
        decode_tokens: u32,
        /// Seconds for timing placement finalists (default from flux.toml); drafting and the expert cache then
        /// build on the fastest one regardless.
        #[arg(long)]
        budget_s: Option<f64>,
        #[arg(long)]
        allow_storage_streaming: bool,
        /// Also measure speculative decoding with --draft-model; models with next-token heads are measured anyway.
        #[arg(long)]
        speculation: bool,
        /// Separate draft model for --speculation.
        #[arg(long)]
        draft_model: Option<PathBuf>,
        /// Next-token (MTP) heads to graft onto the model: a GGUF of head blocks numbered after the trunk.
        /// Implies --speculation.
        #[arg(long)]
        heads: Option<PathBuf>,
        /// Prune candidates that cannot reach this decode rate even optimistically.
        #[arg(long)]
        min_tps: Option<f64>,
        /// Lock the whole model in RAM (opt-in; counted in the host budget).
        #[arg(long)]
        mlock: bool,
        /// Plan again even if a plan for this profile key exists.
        #[arg(long)]
        replan: bool,
        /// Measure hardware again instead of reusing the stored probe report.
        #[arg(long)]
        reprobe: bool,
    },
    /// List saved plans.
    Plans,
    /// Show a saved plan with its decisions and measurements.
    Show { id: String },
    /// Attribute decode-step time to devices and operations for a saved plan (native engine).
    Trace {
        plan: String,
        #[arg(long, default_value_t = 16)]
        steps: u32,
        #[arg(long, default_value_t = 512)]
        prompt_tokens: usize,
        /// Also time every operation (heavy per-node synchronization; shares are indicative only).
        #[arg(long)]
        per_op: bool,
        /// Count expert selections per MoE layer and compare per-layer with per-expert residency.
        #[arg(long)]
        routes: bool,
    },
    /// Benchmarks, quality checks and soak tests.
    Bench {
        #[command(subcommand)]
        cmd: bench::BenchCmd,
    },
    /// Serve a saved plan over an OpenAI-compatible HTTP API.
    Serve {
        /// Plan id (or prefix).
        plan: String,
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        port: Option<u16>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_env("FLUX_LOG").unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).init();
    let cfg = FluxConfig::load()?;
    match Cli::parse().cmd {
        Cmd::Inspect { path, no_hash, json } => {
            let m = flux_ingest::inspect(&path, &flux_ingest::InspectOptions { hash: !no_hash, config: &cfg })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&m)?);
            } else {
                inspect::print(&m);
            }
        }
        Cmd::Fetch { repo, revision, include, dest } => {
            let client = flux_ingest::hf::Client::new();
            let commit = client.resolve(&repo, &revision).await?;
            let files: Vec<_> =
                client.list(&repo, &commit).await?.into_iter().filter(|f| include.iter().any(|g| flux_ingest::hf::glob_match(g, &f.path))).collect();
            anyhow::ensure!(!files.is_empty(), "no files in {repo}@{commit} match {include:?}");
            let dest = dest.unwrap_or_else(|| cfg.models_dir.join(repo.rsplit('/').next().unwrap()));
            eprintln!("{repo} {revision} → {commit}, {} file(s) → {}", files.len(), dest.display());
            for f in &files {
                let shown = std::sync::atomic::AtomicU64::new(0);
                let progress = |done: u64, total: u64| {
                    let pct = done * 100 / total.max(1);
                    if pct >= shown.load(std::sync::atomic::Ordering::Relaxed) + 5 {
                        shown.store(pct, std::sync::atomic::Ordering::Relaxed);
                        eprintln!("  {} {pct}%", f.path);
                    }
                };
                let p = client.download(&repo, &commit, f, &dest, &progress).await?;
                eprintln!("  ok {}", p.display());
            }
            flux_ingest::hf::write_source_record(&dest, &repo, &revision, &commit, &files)?;
        }
        Cmd::Probe { model, quick } => {
            let m = model.map(|p| planning::inspect_hashed(&cfg, &p)).transpose()?;
            let r = planning::probe_report(&cfg, m.as_ref(), quick, true).await?;
            planning::print_probe(&r);
        }
        Cmd::Plan {
            model,
            ctx,
            concurrency,
            serving_p95_ms,
            engines,
            kv,
            ubatch,
            prompt_tokens,
            decode_tokens,
            budget_s,
            allow_storage_streaming,
            speculation,
            draft_model,
            heads,
            min_tps,
            mlock,
            replan,
            reprobe,
        } => {
            let mut cfg = cfg;
            if let Some(b) = budget_s {
                cfg.plan.tuning_budget_s = b;
            }
            let model = match &heads {
                Some(h) => planning::graft_heads(&cfg, &model, h)?,
                None => model,
            };
            let m = planning::inspect_hashed(&cfg, &model)?;
            // Drafting is measured whenever the model carries next-token heads, and kept only where it wins.
            let speculation = speculation || m.facts.as_ref().is_some_and(|f| f.n_layer_nextn > 0);
            let report = planning::probe_report(&cfg, Some(&m), false, reprobe).await?;
            if !replan {
                if let Some(p) = planning::lookup(&cfg, &m, &report, ctx, concurrency).await {
                    planning::log("a plan for this profile key exists (use --replan to measure again)");
                    planning::print_plan(&p);
                    planning::print_next(&cfg, &p);
                    return Ok(());
                }
            }
            let corpus = planning::corpus(&cfg).await?;
            let engines = engines
                .iter()
                .map(|e| match e.as_str() {
                    "native" => Ok(flux_core::plan::EngineKind::Native),
                    "llama-server" => Ok(flux_core::plan::EngineKind::LlamaServer),
                    other if cfg.engines.contains_key(other) => Ok(flux_core::plan::EngineKind::External(other.into())),
                    other => anyhow::bail!("unknown engine {other}"),
                })
                .collect::<Result<Vec<_>>>()?;
            let req = flux_plan::planner::PlanRequest {
                workload: flux_core::plan::Workload {
                    n_ctx_seq: ctx,
                    concurrency,
                    objective: match serving_p95_ms {
                        Some(ms) => flux_core::plan::Objective::Serving { max_p95_token_ms: ms },
                        None => flux_core::plan::Objective::Interactive,
                    },
                },
                engines,
                type_k: kv.clone(),
                type_v: kv,
                n_ubatch: ubatch,
                allow_storage_streaming,
                prompt_tokens,
                decode_tokens,
                speculation,
                expert_residency: true,
                min_decode_tps: min_tps,
                mlock,
                draft_model,
            };
            let plan = flux_plan::planner::plan(&cfg, &m, &report, &corpus, &req, &planning::log).await?;
            let path = planning::save(&cfg, &plan)?;
            planning::print_plan(&plan);
            println!("saved       {}", path.display());
            planning::print_next(&cfg, &plan);
        }
        Cmd::Plans => {
            for p in planning::store(&cfg).list() {
                println!(
                    "{}  {}  {:<12} ctx {:>6} × {}  {}  {}",
                    p.id,
                    p.created.format("%Y-%m-%d %H:%M"),
                    p.engine.to_string(),
                    p.workload.n_ctx_seq,
                    p.workload.concurrency,
                    p.placement.describe(),
                    p.model_files[0].file_name().unwrap_or_default().to_string_lossy()
                );
            }
        }
        Cmd::Show { id } => planning::print_plan(&planning::find_plan(&cfg, &id)?),
        Cmd::Bench { cmd } => bench::run(cfg, cmd).await?,
        Cmd::Trace { plan, steps, prompt_tokens, per_op, routes } => planning::trace(&cfg, &plan, steps, prompt_tokens, per_op, routes).await?,
        Cmd::Serve { plan, host, port } => {
            let plan = planning::find_plan(&cfg, &plan)?;
            planning::check_fits(&plan).await?;
            let host = host.unwrap_or_else(|| cfg.serve.host.clone());
            let port = port.unwrap_or(cfg.serve.port);
            let st = flux_serve::build(cfg.clone(), plan, Some(planning::replanner(cfg))).await?;
            flux_serve::run(st, &host, port).await?;
        }
        Cmd::Convert { source, outtype, python } => {
            let out = flux_ingest::convert::convert(&cfg, &source, &outtype, &python, &planning::log)?;
            let m = planning::inspect_hashed(&cfg, &out)?;
            inspect::print(&m);
        }
        Cmd::Corpus => {
            let dir = flux_core::corpus::Corpus::dir_in(&cfg.cache_dir);
            flux_ingest::corpus::fetch(&dir).await?;
            let c = flux_core::corpus::Corpus::load(&dir)?;
            println!("corpus ready at {} ({} held-out characters)", dir.display(), c.heldout_text().len());
        }
    }
    Ok(())
}
