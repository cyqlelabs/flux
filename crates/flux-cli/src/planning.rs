use anyhow::{Context, Result};
use flux_core::config::FluxConfig;
use flux_core::corpus::Corpus;
use flux_core::fmt_bytes;
use flux_core::hardware::{CopyDirection, ProbeReport};
use flux_core::model::ModelManifest;
use flux_core::plan::{ctx_bucket, Plan, ProfileKey};
use flux_plan::store::PlanStore;
use flux_probe::native::{self, ProbeOptions};
use std::path::{Path, PathBuf};

pub fn log(s: &str) {
    eprintln!("· {s}");
}

pub fn inspect_hashed(cfg: &FluxConfig, path: &Path) -> Result<ModelManifest> {
    flux_ingest::inspect(path, &flux_ingest::InspectOptions { hash: true, config: cfg })
}

/// The model with next-token heads grafted on, derived once into the cache (keyed by both identities).
pub fn graft_heads(cfg: &FluxConfig, model: &Path, heads: &Path) -> Result<PathBuf> {
    let trunk = inspect_hashed(cfg, model)?.identity.context("the model has no content identity")?;
    let mut head_file = vec![flux_ingest::hashing::model_file(heads)?];
    flux_ingest::hashing::HashIndex::open(&cfg.cache_dir).hash_all(&mut head_file)?;
    let head_id = flux_ingest::hashing::identity(&head_file).context("the heads file has no content identity")?;
    let key = flux_core::fsutil::sha256_hex(format!("{trunk}+{head_id}").as_bytes());
    let out = cfg.cache_dir.join("derived").join(format!("{}.gguf", &key[..16]));
    if !out.exists() {
        log(&format!("grafting {} onto {} -> {}", heads.display(), model.display(), out.display()));
        std::fs::create_dir_all(out.parent().unwrap())?;
        let m = flux_ingest::gguf::GgufModel::open(model)?;
        let h = flux_ingest::gguf::GgufModel::open(heads)?;
        let shown = std::cell::Cell::new(0);
        flux_ingest::repack::graft(&m, &h, &out, &|done, total| {
            let pct = done * 100 / total.max(1);
            if pct >= shown.get() + 10 {
                shown.set(pct);
                log(&format!("grafting: {pct}% of {}", fmt_bytes(total)));
            }
        })?;
    }
    Ok(out)
}

pub async fn corpus(cfg: &FluxConfig) -> Result<Corpus> {
    let dir = Corpus::dir_in(&cfg.cache_dir);
    flux_ingest::corpus::fetch(&dir).await?;
    Corpus::load(&dir)
}

/// The stored probe report for this topology, backend and model, or a fresh one.
pub async fn probe_report(cfg: &FluxConfig, model: Option<&ModelManifest>, quick: bool, fresh: bool) -> Result<ProbeReport> {
    let native::Backend { devices, pin, build } = native::backend().await?;
    let topology = tokio::task::spawn_blocking(move || flux_probe::inventory::inventory(devices)).await??.topology_fingerprint();
    if !fresh {
        if let Some(r) = native::load(cfg, &topology, &pin, &build, model) {
            log(&format!("using probe report from {}", r.created.format("%Y-%m-%d %H:%M")));
            return Ok(r);
        }
    }
    let r = native::run(&ProbeOptions { quick, model }, &log).await?;
    let p = native::save(cfg, &r, model)?;
    log(&format!("probe report saved to {}", p.display()));
    Ok(r)
}

pub fn print_probe(r: &ProbeReport) {
    let inv = &r.inventory;
    println!(
        "host        {} · {} ({}C/{}T, {}) · {} RAM",
        inv.hostname,
        inv.cpu.model,
        inv.cpu.cores,
        inv.cpu.threads,
        inv.cpu.isa.join(" "),
        fmt_bytes(inv.memory.total)
    );
    for g in &inv.gpus {
        println!(
            "gpu         {} {} · {} · PCIe gen{} x{} (max gen{} x{})",
            g.pci_bus_id,
            g.name,
            fmt_bytes(g.mem_total),
            g.pcie_gen_current.unwrap_or(0),
            g.pcie_width_current.unwrap_or(0),
            g.pcie_gen_max.unwrap_or(0),
            g.pcie_width_max.unwrap_or(0)
        );
    }
    for d in &inv.backend_devices {
        println!("backend     {} = {} ({})", d.name, d.description, d.pci_bus_id.as_deref().unwrap_or("-"));
    }
    for c in &r.copies {
        let big = c.points.last().map_or(0.0, |p| p.gbps.p50);
        let small = c.points.first().map_or(0.0, |p| p.bytes as f64 / (p.gbps.p50 * 1e3));
        let dir = match c.direction {
            CopyDirection::HostToDevice => "host→",
            CopyDirection::DeviceToHost => "→host",
            CopyDirection::DeviceToDevice => "→peer",
        };
        println!(
            "copy        {}{dir}{} {}: {big:.1} GB/s large, {small:.0} µs small",
            c.device,
            c.peer.as_deref().map(|p| format!(" {p}")).unwrap_or_default(),
            if c.pinned { "pinned" } else { "pageable" }
        );
    }
    for c in &r.contention {
        println!("contention  {}: {:.1} GB/s alone, {:.1} GB/s contended", c.scenario, c.alone_gbps, c.contended_gbps);
    }
    for p in &r.host_pages {
        if let Some(reason) = &p.unsupported {
            println!("host pages  {}: unavailable ({reason})", p.device);
        } else {
            println!(
                "host pages  {}: stream {:.1}/{:.1}, blocks {:.1}/{:.1} GB/s alone/contended ({} CPU threads, {})",
                p.device,
                p.streaming_gbps,
                p.streaming_contended_gbps,
                p.scattered_gbps,
                p.scattered_contended_gbps,
                p.cpu_threads,
                fmt_bytes(p.buffer_bytes)
            );
        }
    }
    for b in &r.cpu_bandwidth {
        println!("cpu         {:>2} threads: {:.1} GB/s weight reads", b.threads, b.gbps.p50);
    }
    for k in r.kernels.iter().filter(|k| k.batch == 1 && k.k * k.n > 65536) {
        let bytes = flux_core::ggml_type::GgmlType::from_name(&k.ggml_type).and_then(|t| t.row_bytes(k.k)).unwrap_or(0) * k.n;
        println!("kernel      {:<5} {:<8} {}x{}: {:.1} GB/s ({:.0} µs)", k.device, k.ggml_type, k.k, k.n, k.weight_gbps(bytes), k.micros.p50);
    }
    for s in &r.storage {
        println!(
            "storage     {} {} {}: {:.2} GB/s",
            if s.direct_io { "direct" } else { "buffered" },
            if s.sequential { "sequential" } else { "random" },
            fmt_bytes(s.block_bytes),
            s.gbps.p50
        );
    }
}

pub async fn lookup(cfg: &FluxConfig, m: &ModelManifest, r: &ProbeReport, req: &flux_plan::planner::PlanRequest) -> Option<Plan> {
    let build = native::backend().await.ok()?.build;
    let key = ProfileKey {
        model_identity: m.identity.clone()?,
        topology: r.topology.clone(),
        backend_revision: r.backend_revision.clone(),
        backend_build: build,
        driver: r.inventory.driver_version.clone().unwrap_or_default(),
        ctx_bucket: ctx_bucket(req.workload.n_ctx_seq.div_ceil(256) * 256),
        concurrency: req.workload.concurrency,
        policy: req.policy_key(cfg),
    };
    PlanStore::new(&cfg.plans_dir())
        .lookup(&key)
        .filter(|p| p.workload.n_ctx_seq >= req.workload.n_ctx_seq && p.workload.objective == req.workload.objective && req.engines.contains(&p.engine))
}

/// The command that serves a plan, printed once `flux plan` has one.
pub fn print_next(cfg: &FluxConfig, p: &Plan) {
    println!("next        flux serve {}  (OpenAI API at http://{}:{}/v1)", p.id, cfg.serve.host, cfg.serve.port);
}

pub fn print_plan(p: &Plan) {
    println!("plan        {} · {} · {}", p.id, p.engine, p.created.format("%Y-%m-%d %H:%M"));
    println!("model       {} ({})", p.model_files[0].display(), p.architecture);
    if let Some(s) = &p.expert_cache {
        println!(
            "experts     GPU cache of {} experts over {} layers, initially serving {:.1}% of calibration routing ({:.1}% with whole tensors); adapts while decoding",
            s.hot_experts,
            s.spec.layers.len(),
            s.gpu_served * 100.0,
            s.gpu_served_by_tensors * 100.0
        );
        for t in &s.spec.tiers {
            println!(
                "            {} serves {} of them for {} layers placed on other GPUs",
                t.device,
                t.layers.iter().map(|(_, es)| es.len()).sum::<usize>(),
                t.layers.len()
            );
        }
    }
    println!("workload    {} tokens per sequence × {} · {:?}", p.workload.n_ctx_seq, p.workload.concurrency, p.workload.objective);
    println!("placement   {}", p.placement.describe());
    for o in &p.placement.overrides {
        println!("  override  {} → {}", o.pattern, o.device);
    }
    println!(
        "runtime     {} threads ({} batch), ubatch {}, flash-attn {}, KV {}/{}",
        p.runtime.n_threads, p.runtime.n_threads_batch, p.runtime.n_ubatch, p.runtime.flash_attn, p.runtime.type_k, p.runtime.type_v
    );
    if let Some(k) = &p.runtime.kv_paging {
        println!(
            "kv pages    {} tokens per sequence in VRAM, then up to {} of RAM ({} kept free; serving recomputes both)",
            k.floor_tokens,
            fmt_bytes(k.host_budget),
            fmt_bytes(k.host_reserve)
        );
        for d in &k.devices {
            println!(
                "            {}: {} of VRAM, {} of it for caches read in full; {}",
                d.device,
                fmt_bytes(d.bytes),
                fmt_bytes(d.full_read_reserve),
                if d.host_reads { "RAM past it" } else { "no RAM pages" }
            );
        }
    }
    for b in p.budgets.iter().filter(|b| p.placement.uses_device(&b.device)) {
        let m = b.measured.unwrap_or(b.predicted);
        println!(
            "budget      {}: weights {} + state {} + compute {} = {} of {} free",
            b.device,
            fmt_bytes(m.weights),
            fmt_bytes(m.state),
            fmt_bytes(m.compute),
            fmt_bytes(m.total()),
            fmt_bytes(b.free_at_plan)
        );
    }
    println!("host        {} resident weights, {} available at plan time", fmt_bytes(p.host.resident_weights), fmt_bytes(p.host.available_at_plan));
    for d in &p.decisions {
        println!("{} {:<11} {}", if d.used { "✓" } else { "✗" }, d.resource, d.reason);
    }
    if let Some(v) = &p.validation {
        println!("tuning      {:.0} s of {:.0} s budget · chosen: {} ({})", v.tuning_seconds, v.tuning_budget_seconds, v.chosen, v.reason);
        for c in &v.candidates {
            let m = c.validation.as_ref().or(c.calibration.as_ref());
            match (m, &c.failure) {
                (Some(m), _) => println!(
                    "  {:<60} predicted {:>6.1} ms/tok · measured {:>6.2} tok/s p50 ({:.2} min) · TTFT {:>6.0} ms{}{}",
                    c.label,
                    c.predicted_token_ms,
                    m.decode_tps.p50,
                    m.decode_tps.min,
                    m.ttft_ms.p50,
                    c.depth.as_ref().map_or(String::new(), |d| format!(
                        " · {}-token prompt at {:.0} tok/s, decode {:.2} tok/s",
                        d.prompt_tokens, d.prefill_tps, d.decode_tps
                    )),
                    match c.validation.as_ref().map(|v| v.agent_decode_tps.as_ref()) {
                        Some(Some(a)) => format!(" · validated · agent conversations {:.2} tok/s", a.p50),
                        Some(None) => " · validated".to_string(),
                        None => String::new(),
                    }
                ),
                (None, Some(f)) => println!("  {:<60} {f}", c.label),
                _ => {}
            }
        }
    }
}

pub fn store(cfg: &FluxConfig) -> PlanStore {
    PlanStore::new(&cfg.plans_dir())
}

/// Saves the plan.
pub fn save(cfg: &FluxConfig, plan: &Plan) -> Result<std::path::PathBuf> {
    store(cfg).save(plan)
}

/// Fits a plan to the memory its GPUs have free now. The planner fills each GPU up to its reserve, so other programs
/// growing by a few MiB would otherwise refuse the plan: a GPU short by less than its cached experts take serves
/// with the least-routed of them dropped, in proportion per layer (the cache adapts while decoding anyway). A larger
/// shortfall refuses the plan, naming it, instead of letting the backend fail to create its context.
pub async fn fit(cfg: &FluxConfig, mut plan: Plan) -> Result<Plan> {
    let devices = native::backend().await?.devices;
    let budgets = plan.budgets.clone();
    for b in budgets.iter().filter(|b| b.required() > 0) {
        let Some(free) = devices.iter().find(|d| d.name == b.device).map(|d| d.mem_free) else { continue };
        let usable = free.saturating_sub(b.reserve);
        let short = b.required().saturating_sub(usable);
        if short == 0 {
            continue;
        }
        let dropped = match trim_cache(cfg, &mut plan, &b.device, short) {
            Ok(n) => n,
            Err(e) => anyhow::bail!(
                "{}: plan {} needs {} but only {} is free beyond the {} reserve ({} was free when it was planned), and {e:#}; free memory on {} or run `flux plan <model> --replan`",
                b.device,
                plan.id,
                fmt_bytes(b.required()),
                fmt_bytes(usable),
                fmt_bytes(b.reserve),
                fmt_bytes(b.free_at_plan),
                b.device
            ),
        };
        tracing::warn!(
            "{}: {} less free than plan {} was measured to need; serving with {dropped} fewer cached experts ({} left). `flux plan <model> --replan` sizes the cache to this memory",
            b.device,
            fmt_bytes(short),
            plan.id,
            plan.expert_cache.as_ref().map_or(0, |c| c.hot_experts)
        );
    }
    // KV pages get the RAM free now, after the weights the worker pins, and the VRAM spare now in place of the spare at planning.
    if let Some(k) = plan.runtime.kv_paging.as_mut() {
        let meminfo = std::fs::read_to_string("/proc/meminfo")?;
        let kib = |key: &str| -> Option<u64> { meminfo.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse::<u64>().ok() };
        let total = kib("MemTotal:").context("cannot read total host memory")? * 1024;
        let available = kib("MemAvailable:").context("cannot read available host memory")? * 1024;
        k.host_reserve = cfg.host_reserve_bytes(total);
        k.host_budget = available.saturating_sub(k.host_reserve + plan.host.resident_weights + plan.host.state_and_scratch);
        for d in &mut k.devices {
            let (Some(b), Some(free)) = (budgets.iter().find(|b| b.device == d.device), devices.iter().find(|x| x.name == d.device).map(|x| x.mem_free)) else {
                continue;
            };
            let spare = |free: u64| free.saturating_sub(b.reserve).saturating_sub(b.required());
            d.bytes = d.bytes.saturating_sub(spare(b.free_at_plan)) + spare(free);
        }
        tracing::info!(
            "KV pages: {} of RAM ({} kept free), VRAM {}",
            fmt_bytes(k.host_budget),
            fmt_bytes(k.host_reserve),
            k.devices.iter().map(|d| format!("{} {}", d.device, fmt_bytes(d.bytes))).collect::<Vec<_>>().join(", ")
        );
    }
    Ok(plan)
}

/// Drops at least `bytes` of the experts cached on `device`, for the layers placed there and as a tier for others,
/// the same share of each layer's, least routed first; the number dropped.
fn trim_cache(cfg: &FluxConfig, plan: &mut Plan, device: &str, bytes: u64) -> Result<u32> {
    let cache = plan.expert_cache.as_mut().context("the plan caches no experts to drop")?;
    let m = flux_ingest::inspect(&plan.model_files[0], &flux_ingest::InspectOptions { hash: false, config: cfg })?;
    let n_expert = m.facts.as_ref().and_then(|f| f.moe.as_ref()).map_or(0, |moe| moe.n_expert as u64);
    anyhow::ensure!(n_expert > 0, "the model has no routed experts");
    let mut per_expert: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for t in m.tensors.iter().filter(|t| t.role() == flux_core::model::TensorRole::FfnRoutedExpert) {
        if let Some(l) = t.layer() {
            *per_expert.entry(l).or_default() += t.bytes / n_expert;
        }
    }
    let on_device = |l: u32| plan.placement.layer_device.get(l as usize).is_some_and(|d| d == device);
    let bytes_of = |l: u32, n: usize| n as u64 * per_expert.get(&l).copied().unwrap_or(0);
    let tiers = || cache.spec.tiers.iter().filter(|t| t.device == device).flat_map(|t| &t.layers);
    let held: u64 = cache.spec.layers.iter().filter(|(l, _)| on_device(**l)).map(|(l, (_, hot))| bytes_of(*l, *hot as usize)).sum::<u64>()
        + tiers().map(|(l, es)| bytes_of(*l, es.len())).sum::<u64>();
    anyhow::ensure!(held > bytes, "its {} of cached experts there cannot cover it", fmt_bytes(held));
    let share = bytes as f64 / held as f64;
    let cut = |n: usize| ((n as f64 * share).ceil() as usize).min(n);
    let mut dropped = 0;
    for (_, (_, hot)) in cache.spec.layers.iter_mut().filter(|(l, _)| on_device(**l)) {
        let n = cut(*hot as usize);
        *hot -= n as u32;
        dropped += n as u32;
    }
    // a tier lists its experts most routed first
    for (_, es) in cache.spec.tiers.iter_mut().filter(|t| t.device == device).flat_map(|t| t.layers.iter_mut()) {
        let n = cut(es.len());
        es.truncate(es.len() - n);
        dropped += n as u32;
    }
    for t in &mut cache.spec.tiers {
        t.layers.retain(|(_, es)| !es.is_empty());
    }
    cache.spec.tiers.retain(|t| !t.layers.is_empty());
    cache.hot_experts = cache.hot_experts.saturating_sub(dropped);
    Ok(dropped)
}

pub fn find_plan(cfg: &FluxConfig, id: &str) -> Result<Plan> {
    store(cfg).by_id(id).with_context(|| format!("no saved plan matches {id}"))
}

/// Plans again for the same model, workload and engine, with fresh hardware measurements.
pub fn replanner(cfg: FluxConfig) -> flux_serve::Replanner {
    std::sync::Arc::new(move |current: Plan| {
        let cfg = cfg.clone();
        Box::pin(async move {
            let m = inspect_hashed(&cfg, &current.model_files[0])?;
            let report = probe_report(&cfg, Some(&m), false, true).await?;
            let corpus = corpus(&cfg).await?;
            let req = flux_plan::planner::PlanRequest {
                workload: current.workload.clone(),
                engines: vec![current.engine.clone()],
                type_k: current.runtime.type_k.clone(),
                type_v: current.runtime.type_v.clone(),
                n_ubatch: current.runtime.n_ubatch,
                allow_storage_streaming: false,
                prompt_tokens: 512,
                decode_tokens: 64,
                speculation: current.runtime.speculation.is_some(),
                expert_residency: true,
                min_decode_tps: None,
                mlock: current.runtime.mlock,
                draft_model: current.runtime.speculation.as_ref().and_then(|s| s.draft_model.clone()),
            };
            let p = flux_plan::planner::plan(&cfg, &m, &report, &corpus, &req, &log).await?;
            save(&cfg, &p)?;
            Ok(p)
        }) as futures::future::BoxFuture<'static, Result<Plan>>
    })
}

/// Runs the plan's placement on the native engine with per-node timing and prints where a decode step goes.
pub async fn trace(cfg: &FluxConfig, plan_id: &str, steps: u32, prompt_tokens: usize, per_op: bool, routes: bool) -> Result<()> {
    let mut plan = find_plan(cfg, plan_id)?;
    plan.engine = flux_core::plan::EngineKind::Native;
    plan.runtime.speculation = None;
    let corpus = corpus(cfg).await?;
    let logs = cfg.data_dir.join("logs");
    std::fs::create_dir_all(&logs)?;
    let w = flux_core::worker::Worker::spawn(&plan.engine, Some(&logs.join(format!("trace-{}.log", plan.id)))).await?;
    let result = async {
        w.load_with(&plan, true).await?;
        let prompt = flux_plan::run::corpus_prompt(&w, &corpus, flux_core::corpus::Role::Calibration, 3, prompt_tokens).await?;
        w.trace(prompt, steps, per_op, routes).await
    }
    .await;
    w.shutdown().await;
    let r = result?;
    if routes {
        return print_routes(cfg, &plan, &r);
    }
    let med = |k: &str| {
        let mut v: Vec<f64> = serde_json::from_value(r[k].clone()).unwrap_or_default();
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(0.0) / 1e3
    };
    let ops: Vec<(String, String, f64, i64)> = r["ops"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|o| {
                    (
                        o["device"].as_str().unwrap_or("").to_string(),
                        o["op"].as_str().unwrap_or("").to_string(),
                        o["us"].as_f64().unwrap_or(0.0),
                        o["count"].as_i64().unwrap_or(0),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let traced_total: f64 = ops.iter().map(|o| o.2).sum::<f64>().max(1e-9);
    println!("plan        {} · {}", plan.id, plan.placement.describe());
    println!(
        "decode step {:.2} ms untraced, {:.2} ms traced{}",
        med("plain_step_us"),
        med("traced_step_us"),
        if per_op { " (per-node synchronization inflates op times)" } else { " (one synchronization per device split)" }
    );
    let mut by_dev: std::collections::BTreeMap<String, f64> = Default::default();
    for o in &ops {
        *by_dev.entry(o.0.clone()).or_default() += o.2;
    }
    for (d, us) in &by_dev {
        println!("device      {d:<6} {:5.1}% of traced time ({:.2} ms)", 100.0 * us / traced_total, us / 1e3);
    }
    println!("splits      {} device changes per step", r["device_switches"]);
    if !per_op {
        let dir = cfg.data_dir.join("traces");
        flux_core::fsutil::write_json_atomic(&dir.join(format!("{}.json", plan.id)), &r)?;
        return Ok(());
    }
    let mut top = ops.clone();
    top.sort_by(|a, b| b.2.total_cmp(&a.2));
    for o in top.iter().take(12) {
        println!("  {:<6} {:<14} {:6.2} ms  {:4} nodes  {:4.1}%", o.0, o.1, o.2 / 1e3, o.3, 100.0 * o.2 / traced_total);
    }
    let dir = cfg.data_dir.join("traces");
    flux_core::fsutil::write_json_atomic(&dir.join(format!("{}.json", plan.id)), &r)?;
    Ok(())
}

/// Counts per layer from the bridge's `{layer: [count per expert]}` (serialized as pairs).
fn layer_counts(v: &serde_json::Value) -> Vec<(u32, Vec<u64>)> {
    let pairs: Vec<(u32, Vec<u64>)> = serde_json::from_value(v.clone()).unwrap_or_default();
    pairs
}

/// How skewed expert use is, and what per-expert residency would serve from the GPU compared with
/// the plan's per-layer residency in the same memory.
fn print_routes(cfg: &FluxConfig, plan: &flux_core::plan::Plan, r: &serde_json::Value) -> Result<()> {
    let decode = layer_counts(&r["decode"]);
    anyhow::ensure!(!decode.is_empty(), "no MoE routing observed: the model has no routed experts");
    let n_layers = decode.len();
    let mut need = [0usize; 3];
    for (_, c) in &decode {
        let mut c = c.clone();
        c.sort_unstable_by(|a, b| b.cmp(a));
        let total: u64 = c.iter().sum::<u64>().max(1);
        for (k, q) in [0.5, 0.8, 0.9].iter().enumerate() {
            let mut acc = 0;
            need[k] += c
                .iter()
                .take_while(|&&x| {
                    let before = acc;
                    acc += x;
                    (before as f64) < q * total as f64
                })
                .count();
        }
    }
    let n_expert = decode.iter().map(|(_, c)| c.len()).max().unwrap_or(0);
    println!("routing     {} MoE layers, {} experts each; {} decode + {} prefill tokens observed", n_layers, n_expert, r["decode_tokens"], r["prefill_tokens"]);
    for (k, q) in ["50%", "80%", "90%"].iter().enumerate() {
        println!(
            "coverage    {q} of selections need {:.1} experts per layer on average ({:.1}% of them)",
            need[k] as f64 / n_layers as f64,
            100.0 * need[k] as f64 / (n_layers * n_expert).max(1) as f64
        );
    }
    // GPU expert budget the plan already spends: layers whose routed experts are not overridden to the CPU.
    let m = inspect_hashed(cfg, &plan.model_files[0])?;
    let host_layers: std::collections::BTreeSet<u32> = m
        .tensors
        .iter()
        .filter(|t| t.role() == flux_core::model::TensorRole::FfnRoutedExpert)
        .filter(|t| {
            plan.placement.layer_device.get(t.layer().unwrap_or(0) as usize).is_some_and(|d| d == "CPU")
                || plan.placement.overrides.iter().any(|o| o.pattern.contains(&format!("blk\\.{}\\.", t.layer().unwrap_or(0))))
        })
        .filter_map(|t| t.layer())
        .collect();
    let gpu_layers = n_layers.saturating_sub(host_layers.len());
    let total_sel: u64 = decode.iter().map(|(_, c)| c.iter().sum::<u64>()).sum::<u64>().max(1);
    let layer_served: u64 = decode.iter().filter(|(l, _)| !host_layers.contains(l)).map(|(_, c)| c.iter().sum::<u64>()).sum();
    let budget_experts = gpu_layers * n_expert;
    let mut all: Vec<u64> = decode.iter().flat_map(|(_, c)| c.iter().copied()).collect();
    all.sort_unstable_by(|a, b| b.cmp(a));
    let expert_served: u64 = all.iter().take(budget_experts).sum();
    println!(
        "residency   per-layer (this plan): {} of {} expert layers on GPU serve {:.1}% of expert reads",
        gpu_layers,
        n_layers,
        100.0 * layer_served as f64 / total_sel as f64
    );
    println!(
        "residency   per-expert by frequency, same {} experts of GPU memory: {:.1}% of expert reads",
        budget_experts,
        100.0 * expert_served as f64 / total_sel as f64
    );
    let dir = cfg.data_dir.join("traces");
    flux_core::fsutil::write_json_atomic(&dir.join(format!("{}-routes.json", plan.id)), r)?;
    Ok(())
}
