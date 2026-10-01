//! Probe → prune → enumerate → verify memory with the backend → measure finalists → save.

use crate::cost::{cpu_beats_transfer, storage_bound_tps, CostModel, Shape, CPU};
use crate::experts::Routes;
use crate::layers::{layout, Layout};
use crate::params::{backend_mapping, placement};
use crate::run::{bench_sampling, corpus_prompt, run_all, summarize, RunSummary};
use crate::search::{prefill_s, search, Assignment, SearchInput};
use anyhow::{bail, ensure, Context, Result};
use flux_core::backend::BackendParams;
use flux_core::config::FluxConfig;
use flux_core::corpus::{Corpus, Role};
use flux_core::fmt_bytes;
use flux_core::ggml_type::GgmlType;
use flux_core::hardware::{BackendDevice, ProbeReport};
use flux_core::model::{Compatibility, ModelManifest, TensorRole};
use flux_core::plan::*;
use flux_core::protocol::DeviceMemory;
use flux_core::worker::{oneshot_job, Worker};
use flux_probe::sampler::Sampler;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct PlanRequest {
    pub workload: Workload,
    /// Engines allowed to run the plan, in preference order for ties.
    pub engines: Vec<EngineKind>,
    pub type_k: String,
    pub type_v: String,
    pub n_ubatch: u32,
    pub allow_storage_streaming: bool,
    /// Prompt length and generated tokens per measured stream.
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    /// Also measure speculative decoding with the model's own next-token heads (llama-server engine).
    pub speculation: bool,
    /// Place routed experts by gain per byte; false keeps them all in host memory (ablation).
    pub expert_residency: bool,
    /// Prune candidates whose optimistic decode rate cannot reach this.
    pub min_decode_tps: Option<f64>,
    /// Lock the whole model in RAM (opt-in; counted in the host budget).
    pub mlock: bool,
    /// Separate draft model for speculative decoding, measured like the model's own heads.
    pub draft_model: Option<std::path::PathBuf>,
}

#[derive(Clone)]
struct Candidate {
    label: String,
    engine: EngineKind,
    assignment: Assignment,
    placement: Placement,
    predicted_decode_s: f64,
    predicted_ttft_s: f64,
    measured: Vec<DeviceMemory>,
    storage_bound_tps: Option<f64>,
    speculation: Option<Speculation>,
    /// Prompt chunk size: smaller chunks shrink compute buffers, which can free room for weights.
    n_ubatch: u32,
    host_compute: u64,
    /// The relabeled checkpoint this candidate runs instead of the supplied artifact.
    split: Option<(PathBuf, ExpertSplit)>,
}

impl Candidate {
    fn gpus(&self) -> usize {
        self.placement.devices.len()
    }
}

struct Ctx<'a> {
    cfg: &'a FluxConfig,
    manifest: &'a ModelManifest,
    req: &'a PlanRequest,
    runtime: RuntimeParams,
    n_ctx_seq: u32,
    log: &'a (dyn Fn(&str) + Sync),
    backend_build: String,
}

impl Ctx<'_> {
    fn params(&self, model: &Path, p: &Placement, n_ubatch: u32) -> BackendParams {
        BackendParams {
            model: model.to_path_buf(),
            devices: p.devices.clone(),
            n_gpu_layers: p.n_gpu_layers,
            tensor_split: p.tensor_split.clone(),
            split_mode: p.split_mode,
            overrides: p.overrides.clone(),
            mmap: self.runtime.mmap,
            mlock: self.runtime.mlock,
            n_ctx_seq: self.n_ctx_seq,
            n_seq: self.req.workload.concurrency,
            n_batch: self.runtime.n_batch,
            n_ubatch,
            n_threads: self.runtime.n_threads,
            n_threads_batch: self.runtime.n_threads_batch,
            flash_attn: self.runtime.flash_attn,
            type_k: self.runtime.type_k.clone(),
            type_v: self.runtime.type_v.clone(),
            op_offload: self.runtime.op_offload,
            kv_unified: false,
            speculation: None,
        }
    }

    /// Backend allocation dry run: the authority on memory.
    async fn measure(&self, p: &Placement, n_ubatch: u32) -> Result<Vec<DeviceMemory>> {
        self.measure_model(&self.manifest.files[0].path, p, n_ubatch).await
    }

    async fn measure_model(&self, model: &Path, p: &Placement, n_ubatch: u32) -> Result<Vec<DeviceMemory>> {
        let v = oneshot_job(&["measure"], &serde_json::to_value(self.params(model, p, n_ubatch))?).await?;
        Ok(serde_json::from_value(v["memory"].clone())?)
    }
}

/// Tensor-parallel buffers are reported on the backend's meta device as one total; apportion them
/// to the participating GPUs by their split shares.
fn apportion_meta(measured: Vec<DeviceMemory>, p: &Placement) -> Vec<DeviceMemory> {
    let total: f32 = p.tensor_split.iter().sum::<f32>().max(1e-6);
    let (meta, mut out): (Vec<DeviceMemory>, Vec<DeviceMemory>) = measured.into_iter().partition(|m| m.device.starts_with("Meta("));
    for m in &meta {
        for (d, w) in p.devices.iter().zip(&p.tensor_split) {
            let f = (*w / total) as f64;
            let share = |b: u64| (b as f64 * f) as u64;
            match out.iter_mut().find(|x| &x.device == d) {
                Some(x) => {
                    x.model += share(m.model);
                    x.context += share(m.context);
                    x.compute += share(m.compute);
                }
                None => out.push(DeviceMemory { device: d.clone(), model: share(m.model), context: share(m.context), compute: share(m.compute) }),
            }
        }
    }
    out
}

fn total(m: &[DeviceMemory], dev: &str) -> u64 {
    m.iter().filter(|d| d.device == dev).map(|d| d.model + d.context + d.compute).sum()
}

fn ordered_subsets(gpus: &[String]) -> Vec<Vec<String>> {
    let mut out = vec![vec![]];
    let mut frontier: Vec<Vec<String>> = vec![vec![]];
    while !frontier.is_empty() {
        let mut next = vec![];
        for o in &frontier {
            for g in gpus.iter().filter(|g| !o.contains(g)) {
                let mut x = o.clone();
                x.push(g.clone());
                next.push(x);
            }
        }
        out.extend(next.iter().cloned());
        frontier = next;
    }
    out
}

fn mem_available() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("MemAvailable:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok()))
        .map_or(0, |kib| kib * 1024)
}

pub async fn plan(
    cfg: &FluxConfig,
    manifest: &ModelManifest,
    report: &ProbeReport,
    corpus: &Corpus,
    req: &PlanRequest,
    log: &(dyn Fn(&str) + Sync),
) -> Result<Plan> {
    let t_start = Instant::now();
    let facts = manifest.facts.clone().context("the artifact has no architecture facts")?;
    let identity = manifest.identity.clone().context("model identity missing: inspect with hashing enabled")?;
    match &manifest.compatibility {
        Compatibility::InspectOnly { missing, alternatives } => {
            bail!("not executable: {}; alternatives: {}", missing.join("; "), alternatives.join("; "))
        }
        Compatibility::Executable { engines } => {
            for e in &req.engines {
                ensure!(engines.contains(&e.to_string()), "engine {e} is not certified for this artifact (eligible: {})", engines.join(", "));
            }
        }
    }
    let n_ctx_seq = req.workload.n_ctx_seq.div_ceil(256) * 256;
    if facts.n_ctx_train > 0 && n_ctx_seq > facts.n_ctx_train {
        bail!("{n_ctx_seq} tokens per sequence exceeds the trained context of {}; Flux does not change context semantics", facts.n_ctx_train);
    }

    let flux_probe::native::Backend { devices, pin, build } = flux_probe::native::backend().await?;
    ensure!(pin == report.backend_revision, "the probe report was measured with backend {}, this build runs {pin}: run `flux probe`", report.backend_revision);
    let mut decisions: Vec<Decision> = vec![];

    // Device capability: every encoding must have a native kernel, or the device is excluded.
    let types: Vec<String> = manifest.encodings.keys().cloned().collect();
    let expert_types: Vec<String> = manifest.tensors.iter().filter(|t| t.role() == TensorRole::FfnRoutedExpert).map(|t| t.ggml_type.name()).collect();
    let mut gpus: Vec<&BackendDevice> = vec![];
    for g in devices.iter().filter(|d| d.kind == "gpu") {
        let v = oneshot_job(&["probe", "supports"], &json!({"device": g.name, "types": types})).await?;
        let missing: Vec<&String> = types
            .iter()
            .filter(|t| v["support"][t.as_str()]["mul_mat"] != true || (expert_types.contains(t) && v["support"][t.as_str()]["mul_mat_id"] != true))
            .collect();
        if missing.is_empty() {
            gpus.push(g);
        } else {
            decisions.push(Decision {
                resource: format!("{} ({})", g.name, g.description),
                used: false,
                reason: format!("no native kernel for {}", missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")),
            });
        }
    }

    let threads = report.best_cpu_bandwidth().map_or(report.inventory.cpu.cores, |b| b.threads);
    let runtime = RuntimeParams {
        n_batch: 2048,
        n_ubatch: req.n_ubatch,
        n_threads: threads,
        n_threads_batch: report.inventory.cpu.threads,
        flash_attn: true,
        type_k: req.type_k.clone(),
        type_v: req.type_v.clone(),
        mmap: true,
        mlock: req.mlock,
        op_offload: true,
        speculation: None,
    };
    let ctx = Ctx { cfg, manifest, req, runtime: runtime.clone(), n_ctx_seq, log, backend_build: build };
    let tk = GgmlType::from_name(&req.type_k).context("unknown cache type")?;
    let tv = GgmlType::from_name(&req.type_v).context("unknown cache type")?;
    let lay = layout(manifest, &facts, req.workload.concurrency as u64, n_ctx_seq as u64, tk, tv);
    let n_all = lay.blocks.len();

    // Compute buffers per GPU when it is the only (hence last) device, per prompt chunk size:
    // a conservative reservation. Smaller chunks trade prefill speed for room.
    let mut ubatches: Vec<u32> = [req.n_ubatch, req.n_ubatch / 2, req.n_ubatch / 4].into_iter().filter(|&u| u >= 64).collect();
    ubatches.dedup();
    let mut compute: HashMap<(String, u32), u64> = HashMap::new();
    let mut host_compute: HashMap<u32, u64> = HashMap::new();
    for &ub in &ubatches {
        for g in &gpus {
            let p = Placement {
                devices: vec![g.name.clone()],
                layer_device: vec![g.name.clone(); n_all],
                output_device: g.name.clone(),
                overrides: vec![],
                n_gpu_layers: n_all as i32 + 1,
                tensor_split: vec![1.0],
                split_mode: SplitMode::Layer,
            };
            let m = ctx.measure(&p, ub).await.with_context(|| format!("dry run on {}", g.name))?;
            compute.insert((g.name.clone(), ub), m.iter().filter(|d| d.device == g.name).map(|d| d.compute).sum());
            let h = host_compute.entry(ub).or_default();
            *h = (*h).max(m.iter().filter(|d| d.device == CPU).map(|d| d.compute).sum());
        }
    }
    let reserve = cfg.plan.device_reserve_mib << 20;
    let capacity_for = |ub: u32| -> HashMap<String, u64> {
        gpus.iter().map(|g| (g.name.clone(), g.mem_free.saturating_sub(reserve + compute.get(&(g.name.clone(), ub)).copied().unwrap_or(0)))).collect()
    };
    let capacity = capacity_for(req.n_ubatch);
    let host_avail = mem_available().saturating_sub(cfg.plan.host_reserve_mib << 20);

    let cost = CostModel::from_report(report, threads);
    let shape = Shape {
        batch: match req.workload.objective {
            Objective::Interactive => 1,
            Objective::Serving { .. } => req.workload.concurrency,
        },
        ctx: n_ctx_seq as u64 / 2,
        n_ctx_seq: n_ctx_seq as u64,
        n_seq: req.workload.concurrency as u64,
        ubatch: req.n_ubatch,
        op_offload: true,
    };
    let disk_gbps = report.storage.iter().filter(|s| s.direct_io).map(|s| s.gbps.p50).fold(0.0, f64::max);

    // Enumerate placements per GPU order, verify the best with the backend, repair overflows.
    let gpu_names: Vec<String> = gpus.iter().map(|g| g.name.clone()).collect();
    let mut cands: Vec<Candidate> = vec![];
    let mut rejected: Vec<String> = vec![];
    let mut seen = vec![];
    for (&ub, order) in ubatches.iter().flat_map(|ub| ordered_subsets(&gpu_names).into_iter().map(move |o| (ub, o))) {
        let shape = Shape { ubatch: ub, ..shape };
        let host_compute = host_compute.get(&ub).copied().unwrap_or(0);
        let mut caps = capacity_for(ub);
        for attempt in 0..3 {
            let inp = SearchInput {
                layout: &lay,
                cost: &cost,
                shape,
                order: order.clone(),
                capacity: caps.clone(),
                host_experts: !req.expert_residency,
                host_capacity: (!req.allow_storage_streaming).then(|| host_avail.saturating_sub(host_compute)),
            };
            let Some(a) = search(&inp) else {
                rejected.push(format!("order [{}]: no feasible placement", order.join(",")));
                break;
            };
            let key = (a.devices.clone(), a.layer_device.clone(), a.output_device.clone(), a.on_cpu.clone());
            if seen.contains(&key) {
                break;
            }
            let p = placement(&lay, &a);
            let mut expected = a.layer_device.clone();
            expected.push(a.output_device.clone());
            ensure!(backend_mapping(n_all, &p.devices, p.n_gpu_layers, &p.tensor_split) == expected, "placement does not reproduce: {}", p.describe());
            let host_need = a.bytes.get(CPU).copied().unwrap_or(0) + host_compute;
            let storage = (host_need > host_avail).then(|| {
                let overflow = (host_need - host_avail) as f64;
                let read = if lay.n_expert > 0 { lay.expert_read_fraction(shape.batch) } else { 1.0 };
                storage_bound_tps(shape.batch as f64, overflow * read, disk_gbps.max(0.1))
            });
            if let (Some(bound), false) = (storage, req.allow_storage_streaming) {
                rejected.push(format!(
                    "{}: needs {} of host memory, {} available; streaming weights from storage would bound decode at {bound:.3} tok/s (allow with --allow-storage-streaming)",
                    p.describe(),
                    fmt_bytes(host_need),
                    fmt_bytes(host_avail)
                ));
                break;
            }
            let measured = ctx.measure(&p, ub).await.with_context(|| format!("dry run of {}", p.describe()))?;
            let over: Vec<(String, u64)> = gpus
                .iter()
                .filter_map(|g| {
                    let need = total(&measured, &g.name);
                    let limit = g.mem_free.saturating_sub(reserve);
                    (need > limit).then(|| (g.name.clone(), need - limit))
                })
                .collect();
            if !over.is_empty() && attempt < 2 {
                for (g, by) in &over {
                    let c = caps.get_mut(g).unwrap();
                    *c = c.saturating_sub(by + (64 << 20));
                }
                log(&format!("repair {}: backend measured {:?} over budget", p.describe(), over));
                continue;
            }
            if !over.is_empty() {
                rejected.push(format!("{}: backend allocation exceeds free memory on {:?}", p.describe(), over));
                break;
            }
            seen.push(key);
            let ttft = prefill_s(&inp, &a, req.prompt_tokens as u64);
            cands.push(Candidate {
                label: if ub == req.n_ubatch { p.describe() } else { format!("{} · ubatch {ub}", p.describe()) },
                engine: EngineKind::Native,
                predicted_decode_s: a.decode_s,
                predicted_ttft_s: ttft,
                assignment: a,
                placement: p,
                measured,
                storage_bound_tps: storage,
                speculation: None,
                n_ubatch: ub,
                host_compute,
                split: None,
            });
            break;
        }
    }
    // Tensor parallelism across all eligible GPUs: estimated with per-block collectives, verified by the backend.
    if gpus.len() >= 2 && lay.n_expert == 0 {
        let mut order: Vec<&BackendDevice> = gpus.clone();
        order.sort_by_key(|g| std::cmp::Reverse(capacity[&g.name]));
        let devs: Vec<String> = order.iter().map(|g| g.name.clone()).collect();
        let total_cap: u64 = order.iter().map(|g| capacity[&g.name]).sum::<u64>().max(1);
        let split: Vec<f32> = order.iter().map(|g| (capacity[&g.name] as f64 * 100.0 / total_cap as f64).round() as f32).collect();
        for mode in [SplitMode::Row, SplitMode::Tensor] {
            let p = Placement {
                devices: devs.clone(),
                layer_device: vec![devs.join("+"); n_all],
                output_device: devs[0].clone(),
                overrides: vec![],
                n_gpu_layers: n_all as i32 + 1,
                tensor_split: split.clone(),
                split_mode: mode,
            };
            let measured = match ctx.measure(&p, req.n_ubatch).await {
                Ok(m) => apportion_meta(m, &p),
                Err(e) => {
                    rejected.push(format!("{}: backend refused: {}", p.describe(), format!("{e:#}").lines().next().unwrap_or_default()));
                    continue;
                }
            };
            if let Some(g) = gpus.iter().find(|g| total(&measured, &g.name) > g.mem_free.saturating_sub(reserve)) {
                rejected.push(format!("{}: backend allocation exceeds free memory on {}", p.describe(), g.name));
                continue;
            }
            let decode = tp_decode_s(&lay, &cost, &shape, &devs, &split, mode);
            let single = Assignment {
                devices: vec![devs[0].clone()],
                layer_device: vec![devs[0].clone(); n_all],
                output_device: devs[0].clone(),
                on_cpu: vec![vec![]; n_all],
                decode_s: decode,
                bytes: Default::default(),
            };
            let inp =
                SearchInput { layout: &lay, cost: &cost, shape, order: devs.clone(), capacity: capacity.clone(), host_experts: false, host_capacity: None };
            let ttft = prefill_s(&inp, &single, req.prompt_tokens as u64) / devs.len() as f64;
            cands.push(Candidate {
                label: p.describe(),
                engine: EngineKind::Native,
                assignment: single,
                placement: p,
                predicted_decode_s: decode,
                predicted_ttft_s: ttft,
                measured,
                storage_bound_tps: None,
                speculation: None,
                n_ubatch: req.n_ubatch,
                host_compute: host_compute.get(&req.n_ubatch).copied().unwrap_or(0),
                split: None,
            });
        }
    }
    // Objective prune: an optimistic bound (the prediction without its fixed overheads, 25% faster)
    // that still misses the requested rate cannot be rescued by measurement.
    if let Some(min) = req.min_decode_tps {
        let (keep, drop): (Vec<Candidate>, Vec<Candidate>) = cands.into_iter().partition(|c| 1.25 / c.predicted_decode_s >= min);
        for c in drop {
            rejected.push(format!("{}: optimistic bound {:.1} tok/s misses the required {min:.1} tok/s", c.label, 1.25 / c.predicted_decode_s));
        }
        cands = keep;
    }
    ensure!(!cands.is_empty(), "no feasible plan:\n  {}", rejected.join("\n  "));
    cands.sort_by(|a, b| a.predicted_decode_s.total_cmp(&b.predicted_decode_s).then(a.predicted_ttft_s.total_cmp(&b.predicted_ttft_s)));
    for c in &cands {
        log(&format!("candidate {:<48} predicted {:.2} ms/token, TTFT {:.0} ms", c.label, c.predicted_decode_s * 1e3, c.predicted_ttft_s * 1e3));
    }

    // Finalists: best predicted, the best predicted for each first device (predictions rank device
    // orders least reliably), and the best known-good single-device (or CPU) candidate.
    let mut finalists: Vec<Candidate> = cands.iter().take(cfg.plan.finalists).cloned().collect();
    for c in cands.iter().filter(|c| c.gpus() > 0) {
        if !finalists.iter().any(|f| f.placement.devices.first() == c.placement.devices.first()) {
            finalists.push(c.clone());
        }
    }
    if !finalists.iter().any(|c| c.gpus() <= 1) {
        if let Some(c) = cands.iter().find(|c| c.gpus() <= 1) {
            finalists.push(c.clone());
        }
    }
    let mut runs: Vec<Candidate> = vec![];
    for e in &req.engines {
        for (i, f) in finalists.iter().enumerate() {
            // Alternative engines are compared on the best placement only.
            if *e == EngineKind::Native || i == 0 {
                let mut c = f.clone();
                c.engine = e.clone();
                c.label = format!("{e}: {}", f.label);
                runs.push(c);
            }
        }
    }

    let hybrid = facts.recurrent.iter().any(|&r| r);
    let drafters: Vec<(String, Option<std::path::PathBuf>)> = match (&req.draft_model, facts.n_layer_nextn > 0) {
        (Some(d), _) => vec![("draft-simple".into(), Some(d.clone()))],
        (None, true) => vec![("draft-mtp".into(), None)],
        (None, false) => vec![],
    };
    // Drafting needs its own context next to the model: use the fastest placement that leaves room.
    let headroom = |c: &Candidate| -> u64 {
        gpus.iter()
            .filter(|g| c.placement.uses_device(&g.name))
            .map(|g| g.mem_free.saturating_sub(reserve + total(&c.measured, &g.name)))
            .min()
            .unwrap_or(u64::MAX)
    };
    let draft_room = 768u64 << 20;
    let draft_base = cands.iter().find(|c| headroom(c) >= draft_room).cloned();
    if req.speculation && req.engines.contains(&EngineKind::LlamaServer) && !drafters.is_empty() && draft_base.is_none() {
        rejected.push(format!("speculation: no placement leaves {} free for the draft context", fmt_bytes(draft_room)));
    }
    if let (true, true, Some(base)) = (req.speculation, req.engines.contains(&EngineKind::LlamaServer), &draft_base) {
        for (kind, draft_model) in drafters {
            for n_max in [2, 4] {
                let mut c = base.clone();
                c.engine = EngineKind::LlamaServer;
                c.label = format!("llama-server + {kind} {n_max}: {}", base.label);
                c.speculation = Some(Speculation { kind: kind.clone(), draft_model: draft_model.clone(), n_max });
                runs.push(c);
            }
        }
    }

    // Calibration, then a race of the top two on separate validation prompts.
    let budget = Duration::from_secs_f64(cfg.plan.tuning_budget_s);
    let base_plan = |c: &Candidate| -> Plan { assemble(&ctx, &facts.architecture, &identity, report, &devices, c, host_avail, vec![], None) };
    let mut results: Vec<(usize, CandidateResult, Option<RunSummary>)> = vec![];
    for (i, c) in runs.iter().enumerate() {
        if t_start.elapsed() > budget && !results.is_empty() {
            results.push((
                i,
                CandidateResult {
                    label: c.label.clone(),
                    predicted_token_ms: c.predicted_decode_s * 1e3,
                    calibration: None,
                    validation: None,
                    failure: Some("skipped: tuning budget spent".into()),
                },
                None,
            ));
            continue;
        }
        log(&format!("calibrating {}", c.label));
        if c.speculation.is_some() && hybrid {
            // Hybrid models keep recurrent state that a rejected draft must roll back exactly.
            let mut plain = c.clone();
            plain.speculation = None;
            let cert = async {
                let a = greedy_tokens(&ctx, &base_plan(&plain), corpus).await?;
                let b = greedy_tokens(&ctx, &base_plan(c), corpus).await?;
                Ok::<bool, anyhow::Error>(a == b)
            }
            .await;
            match cert {
                Ok(true) => log("  rollback certified: greedy output identical with and without drafting"),
                outcome => {
                    let why = match outcome {
                        Ok(_) => "rollback not certified: greedy output differs with drafting".to_string(),
                        Err(e) => format!("rollback certification failed: {e:#}"),
                    };
                    log(&format!("  {why}"));
                    results.push((
                        i,
                        CandidateResult {
                            label: c.label.clone(),
                            predicted_token_ms: c.predicted_decode_s * 1e3,
                            calibration: None,
                            validation: None,
                            failure: Some(why),
                        },
                        None,
                    ));
                    continue;
                }
            }
        }
        let r = measured_run(&ctx, &base_plan(c), corpus, Role::Calibration, cfg.plan.calibration_prompts, &devices).await;
        let (cal, summary, failure) = match r {
            Ok((m, s)) => (Some(m), Some(s), None),
            Err(e) => (None, None, Some(format!("{e:#}"))),
        };
        if let Some(s) = &summary {
            log(&format!("  decode {:.2} tok/s p50, TTFT {:.0} ms p50", s.decode_tps.p50, s.ttft_ms.p50));
        }
        results.push((
            i,
            CandidateResult { label: c.label.clone(), predicted_token_ms: c.predicted_decode_s * 1e3, calibration: cal, validation: None, failure },
            summary,
        ));
    }
    let score = |s: &RunSummary| match req.workload.objective {
        Objective::Interactive => s.decode_tps.p50,
        Objective::Serving { max_p95_token_ms } => {
            if s.token_ms.p95 <= max_p95_token_ms as f64 {
                s.aggregate_tps
            } else {
                -s.token_ms.p95
            }
        }
    };
    // Per-expert residency on the best measured placement that keeps experts in host memory: trace its
    // routing, write the split checkpoint, then calibrate it like any other candidate.
    let splittable = |c: &Candidate| {
        c.engine == EngineKind::Native
            && c.speculation.is_none()
            && c.placement.split_mode == SplitMode::Layer
            && c.assignment.on_cpu.iter().zip(&c.assignment.layer_device).any(|(h, d)| d != CPU && h.contains(&true))
    };
    let split_base = results
        .iter()
        .filter(|r| splittable(&runs[r.0]))
        .filter_map(|r| Some((r.0, score(r.2.as_ref()?))))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| runs[i].clone());
    if let (true, Some(base)) = (req.expert_residency && lay.n_expert > 0 && t_start.elapsed() < budget, split_base) {
        let shape = Shape { ubatch: base.n_ubatch, ..shape };
        match expert_split(&ctx, &base_plan(&base), &base, &lay, &cost, &shape, &gpus, corpus, &identity).await {
            Ok(Some(c)) => {
                log(&format!("calibrating {} (predicted {:.2} ms/token)", c.label, c.predicted_decode_s * 1e3));
                let (cal, summary, failure) = match measured_run(&ctx, &base_plan(&c), corpus, Role::Calibration, cfg.plan.calibration_prompts, &devices).await
                {
                    Ok((m, s)) => (Some(m), Some(s), None),
                    Err(e) => (None, None, Some(format!("{e:#}"))),
                };
                if let Some(s) = &summary {
                    log(&format!("  decode {:.2} tok/s p50, TTFT {:.0} ms p50", s.decode_tps.p50, s.ttft_ms.p50));
                }
                let result =
                    CandidateResult { label: c.label.clone(), predicted_token_ms: c.predicted_decode_s * 1e3, calibration: cal, validation: None, failure };
                runs.push(c);
                results.push((runs.len() - 1, result, summary));
            }
            Ok(None) => rejected.push(format!("per-expert residency on {}: no block would split", base.label)),
            Err(e) => rejected.push(format!("per-expert residency on {}: {e:#}", base.label)),
        }
    }
    // Interactive decode rates within 3% (run-to-run noise here, and the speed gate's tolerance) are
    // ties, won by the faster first token.
    let tie = match req.workload.objective {
        Objective::Interactive => 0.03,
        Objective::Serving { .. } => 0.0,
    };
    let order = |ks: &mut Vec<usize>, of: &dyn Fn(usize) -> RunSummary| {
        let top = ks.iter().map(|&k| score(&of(k))).fold(f64::MIN, f64::max);
        let near = |k: usize| score(&of(k)) >= top - tie * top.abs();
        ks.sort_by(|&a, &b| {
            near(b).cmp(&near(a)).then_with(|| match near(a) && near(b) {
                true => of(a).ttft_ms.p50.total_cmp(&of(b).ttft_ms.p50),
                false => score(&of(b)).total_cmp(&score(&of(a))),
            })
        });
    };
    let mut ranked: Vec<usize> = (0..results.len()).filter(|&k| results[k].2.is_some()).collect();
    order(&mut ranked, &|k| results[k].2.clone().unwrap());
    ensure!(
        !ranked.is_empty(),
        "every candidate failed to run:\n  {}",
        results.iter().map(|r| format!("{}: {}", r.1.label, r.1.failure.clone().unwrap_or_default())).collect::<Vec<_>>().join("\n  ")
    );
    let mut validated: Vec<(usize, RunSummary)> = vec![];
    for &k in ranked.iter().take(2) {
        log(&format!("validating {}", results[k].1.label));
        match measured_run(&ctx, &base_plan(&runs[results[k].0]), corpus, Role::Validation, cfg.plan.validation_prompts, &devices).await {
            Ok((m, s)) => {
                results[k].1.validation = Some(m);
                validated.push((k, s));
            }
            Err(e) => results[k].1.failure = Some(format!("validation: {e:#}")),
        }
    }
    let mut finals: Vec<usize> = (0..validated.len()).collect();
    order(&mut finals, &|i| validated[i].1.clone());
    let best = finals.first().map_or(ranked[0], |&i| validated[i].0);
    let chosen = &runs[results[best].0];
    let validation = ValidationRecord {
        tuning_seconds: t_start.elapsed().as_secs_f64(),
        tuning_budget_seconds: cfg.plan.tuning_budget_s,
        candidates: results.iter().map(|r| r.1.clone()).collect(),
        chosen: chosen.label.clone(),
        reason: match req.workload.objective {
            Objective::Interactive => {
                "highest median single-stream decode rate on validation prompts (rates within 3% are ties, won by the faster first token)".into()
            }
            Objective::Serving { max_p95_token_ms } => format!("highest aggregate tokens/s with p95 token latency within {max_p95_token_ms} ms"),
        },
    };
    let mut plan = assemble(&ctx, &facts.architecture, &identity, report, &devices, chosen, host_avail, decisions, Some(validation));
    explain(&mut plan, &lay, &cost, &shape, &devices, chosen, &results.iter().map(|r| (runs[r.0].clone(), r.2.clone())).collect::<Vec<_>>(), &rejected);
    plan.id = plan.compute_id();
    Ok(plan)
}

/// Decode step with every block's weights split across `devs` by `split` shares: the slowest share
/// bounds each product, and each block pays its collectives over the measured copy paths.
fn tp_decode_s(lay: &Layout, cost: &CostModel, s: &Shape, devs: &[String], split: &[f32], mode: SplitMode) -> f64 {
    let total: f32 = split.iter().sum();
    let act = s.batch as u64 * lay.n_embd as u64 * 4;
    let gather = devs[1..].iter().map(|d| cost.copy_s(d, &devs[0], act) + cost.copy_s(&devs[0], d, act)).fold(0.0, f64::max);
    let mut t = cost.step_overhead_s + cost.output_decode_s(lay, &devs[0], s);
    for b in &lay.blocks {
        let shares = devs.iter().zip(split).map(|(d, w)| cost.decode_block_s(lay, b, d, &[], s) * (*w / total) as f64);
        t += shares.fold(0.0, f64::max);
        let collectives = match mode {
            SplitMode::Row => b.dense.iter().filter(|w| w.macs > 0).count(),
            _ => 2,
        };
        t += collectives as f64 * gather;
    }
    t
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    ctx: &Ctx,
    arch: &str,
    identity: &str,
    report: &ProbeReport,
    devices: &[BackendDevice],
    c: &Candidate,
    host_avail: u64,
    decisions: Vec<Decision>,
    validation: Option<ValidationRecord>,
) -> Plan {
    let reserve = ctx.cfg.plan.device_reserve_mib << 20;
    let budgets = devices
        .iter()
        .filter(|d| d.kind == "gpu")
        .map(|d| {
            let m: Vec<&DeviceMemory> = c.measured.iter().filter(|m| m.device == d.name).collect();
            DeviceBudget {
                device: d.name.clone(),
                capacity: d.mem_total,
                free_at_plan: d.mem_free,
                reserve,
                predicted: MemSplit { weights: c.assignment.bytes.get(&d.name).copied().unwrap_or(0), state: 0, compute: 0 },
                measured: (!m.is_empty()).then(|| MemSplit {
                    weights: m.iter().map(|x| x.model).sum(),
                    state: m.iter().map(|x| x.context).sum(),
                    compute: m.iter().map(|x| x.compute).sum(),
                }),
            }
        })
        .collect();
    // The backend's dry run is the authority on what stays in host memory.
    let host_weights = match c.measured.iter().filter(|m| m.device == CPU).map(|m| m.model).sum::<u64>() {
        0 => c.assignment.bytes.get(CPU).copied().unwrap_or(0),
        measured => measured,
    };
    let mut plan = Plan {
        schema: PLAN_SCHEMA,
        id: String::new(),
        created: chrono::Utc::now(),
        key: ProfileKey {
            model_identity: identity.into(),
            topology: report.topology.clone(),
            backend_revision: report.backend_revision.clone(),
            backend_build: ctx.backend_build.clone(),
            driver: report.inventory.driver_version.clone().unwrap_or_default(),
            ctx_bucket: ctx_bucket(ctx.n_ctx_seq),
            concurrency: ctx.req.workload.concurrency,
        },
        model_files: c.split.as_ref().map_or_else(|| ctx.manifest.paths(), |(path, _)| vec![path.clone()]),
        architecture: arch.into(),
        engine: c.engine.clone(),
        workload: Workload { n_ctx_seq: ctx.n_ctx_seq, ..ctx.req.workload.clone() },
        placement: c.placement.clone(),
        runtime: RuntimeParams { speculation: c.speculation.clone(), n_ubatch: c.n_ubatch, ..ctx.runtime.clone() },
        budgets,
        host: HostBudget {
            capacity: report.inventory.memory.total,
            available_at_plan: host_avail + (ctx.cfg.plan.host_reserve_mib << 20),
            os_reserve: ctx.cfg.plan.host_reserve_mib << 20,
            resident_weights: host_weights,
            mirrored_weights: 0,
            pinned_buffers: if ctx.req.mlock { host_weights } else { 0 },
            state_and_scratch: c.host_compute,
        },
        quality: if ctx.req.type_k == "f16" && ctx.req.type_v == "f16" {
            QualityProfile::Exact
        } else {
            QualityProfile::Changed { changes: vec![format!("KV cache {}/{}", ctx.req.type_k, ctx.req.type_v)], ppl_ratio: None, top1_agreement: None }
        },
        decisions,
        validation,
        expert_split: c.split.as_ref().map(|(_, s)| s.clone()),
    };
    plan.id = plan.compute_id();
    plan
}

/// Traces routing on `base`, picks hot experts within the GPU memory it leaves, writes the relabeled
/// checkpoint (cached by its spec) and verifies the allocation with the backend, shrinking on overflow.
#[allow(clippy::too_many_arguments)]
async fn expert_split(
    ctx: &Ctx<'_>,
    base_plan: &Plan,
    base: &Candidate,
    lay: &Layout,
    cost: &CostModel,
    shape: &Shape,
    gpus: &[&BackendDevice],
    corpus: &Corpus,
    identity: &str,
) -> Result<Option<Candidate>> {
    (ctx.log)(&format!("tracing expert routing on {}", base.label));
    let routes = trace_routes(ctx, base_plan, corpus).await?;
    let reserve = ctx.cfg.plan.device_reserve_mib << 20;
    // Splitting adds a few small nodes per block; keep a margin for their buffers.
    let mut spare: HashMap<String, i64> =
        gpus.iter().map(|g| (g.name.clone(), g.mem_free as i64 - (reserve + total(&base.measured, &g.name) + (128 << 20)) as i64)).collect();
    for attempt in 0..3 {
        let Some(r) = crate::experts::choose(lay, &base.assignment, cost, shape, &routes, &spare) else { return Ok(None) };
        let dir = ctx.cfg.cache_dir.join("moe-split");
        let path = dir.join(format!("{}-{}.gguf", &flux_core::fsutil::sha256_hex(identity.as_bytes())[..16], r.spec.digest()));
        if !path.exists() {
            std::fs::create_dir_all(&dir)?;
            (ctx.log)(&format!("writing split checkpoint: {} hot experts across {} layers → {}", r.hot_experts, r.spec.layers.len(), path.display()));
            let (src, spec, out, id) = (ctx.manifest.files[0].path.clone(), r.spec.clone(), path.clone(), identity.to_string());
            tokio::task::spawn_blocking(move || flux_ingest::repack::moe_split(&flux_ingest::gguf::GgufModel::open(&src)?, &id, &spec, &out, &|_, _| {}))
                .await??;
        }
        let mut p = base.placement.clone();
        p.overrides = crate::experts::overrides(lay, &r);
        let measured = ctx.measure_model(&path, &p, base.n_ubatch).await.context("dry run of the split checkpoint")?;
        let over: Vec<(String, u64)> = gpus
            .iter()
            .filter_map(|g| {
                let (need, limit) = (total(&measured, &g.name), g.mem_free.saturating_sub(reserve));
                (need > limit).then(|| (g.name.clone(), need - limit))
            })
            .collect();
        if over.is_empty() {
            return Ok(Some(Candidate {
                label: format!("{} · {} hot experts", base.label.replace(&base.placement.describe(), &p.describe()), r.hot_experts),
                predicted_decode_s: r.decode_s,
                placement: p,
                measured,
                split: Some((
                    path,
                    ExpertSplit {
                        source: ctx.manifest.paths(),
                        spec: r.spec,
                        hot_experts: r.hot_experts,
                        gpu_served: r.gpu_served,
                        gpu_served_by_tensors: r.gpu_served_by_tensors,
                    },
                )),
                ..base.clone()
            }));
        }
        (ctx.log)(&format!("repair split (attempt {}): backend measured {over:?} over budget", attempt + 1));
        for (g, by) in &over {
            *spare.get_mut(g).unwrap() -= (by + (64 << 20)) as i64;
        }
    }
    bail!("the split checkpoint still exceeds device memory after repairs")
}

/// Routing counts per layer and expert over the calibration prompts, prefill and decode.
async fn trace_routes(ctx: &Ctx<'_>, plan: &Plan, corpus: &Corpus) -> Result<Routes> {
    let logs = ctx.cfg.data_dir.join("logs");
    std::fs::create_dir_all(&logs)?;
    let w = Worker::spawn(&EngineKind::Native, Some(&logs.join(format!("routes-{}.log", plan.id)))).await?;
    let out = async {
        w.load_with(plan, true).await?;
        let mut acc = Routes::new();
        for i in 0..ctx.cfg.plan.calibration_prompts {
            let p = corpus_prompt(&w, corpus, Role::Calibration, i * 7, ctx.req.prompt_tokens as usize).await?;
            let r = w.trace(p, ctx.req.decode_tokens, false, true).await?;
            for phase in ["prefill", "decode"] {
                // The backend lists experts up to the highest id it saw, so lengths vary.
                for (layer, c) in serde_json::from_value::<Vec<(u32, Vec<u64>)>>(r[phase].clone())? {
                    let a = acc.entry(layer).or_default();
                    if a.len() < c.len() {
                        a.resize(c.len(), 0);
                    }
                    a.iter_mut().zip(c).for_each(|(x, y)| *x += y);
                }
            }
        }
        Ok(acc)
    }
    .await;
    w.shutdown().await;
    out
}

/// Greedy tokens for two validation prompts, to compare two configurations of the same artifact.
async fn greedy_tokens(ctx: &Ctx<'_>, plan: &Plan, corpus: &Corpus) -> Result<Vec<Vec<i32>>> {
    let w = Worker::spawn(&plan.engine, Some(&ctx.cfg.data_dir.join("logs").join(format!("cert-{}.log", plan.id)))).await?;
    let out = async {
        w.load(plan).await?;
        let mut v = vec![];
        for i in 0..2 {
            let p = corpus_prompt(&w, corpus, Role::Validation, 50 + i, 256).await?;
            let r = crate::run::run_stream(&w, &format!("cert-{i}"), p, 48, bench_sampling()).await?;
            if let Some(e) = r.error {
                bail!("{e}");
            }
            v.push(r.tokens);
        }
        Ok(v)
    }
    .await;
    w.shutdown().await;
    out
}

/// One measured session: load, warm up, run prompts, record distributions and peak memory.
async fn measured_run(ctx: &Ctx<'_>, plan: &Plan, corpus: &Corpus, role: Role, n: usize, devices: &[BackendDevice]) -> Result<(Measurement, RunSummary)> {
    let logs = ctx.cfg.data_dir.join("logs");
    std::fs::create_dir_all(&logs)?;
    let w = Worker::spawn(&plan.engine, Some(&logs.join(format!("plan-{}.log", plan.id)))).await?;
    let sampler = Sampler::start(devices, w.pid().await, Duration::from_millis(50));
    let result = async {
        tokio::time::timeout(Duration::from_secs(1800), w.load(plan)).await.context("load timed out")??;
        let offset = if role == Role::Validation { 100 } else { 0 };
        let mut prompts = vec![];
        for i in 0..n {
            prompts.push(corpus_prompt(&w, corpus, role, offset + i * 7, ctx.req.prompt_tokens as usize).await?);
        }
        let warm = corpus_prompt(&w, corpus, role, 999, 32).await?;
        crate::run::run_stream(&w, "warmup", warm, 8, bench_sampling()).await?;
        let concurrency = match ctx.req.workload.objective {
            Objective::Interactive => 1,
            Objective::Serving { .. } => ctx.req.workload.concurrency as usize,
        };
        let t0 = Instant::now();
        let results = run_all(&w, "cal", prompts, ctx.req.decode_tokens, concurrency, bench_sampling()).await?;
        let wall = t0.elapsed().as_secs_f64();
        let errors: Vec<String> = results.iter().filter_map(|r| r.error.clone()).collect();
        ensure!(errors.is_empty(), "{}", errors.join("; "));
        summarize(&results, wall).context("no completed streams")
    }
    .await;
    let peaks = sampler.finish();
    w.shutdown().await;
    let s = result?;
    (ctx.log)(&format!(
        "  peak host RSS {}, devices {:?}",
        fmt_bytes(peaks.tree_rss),
        peaks.device_added().iter().map(|(k, v)| format!("{k} {}", fmt_bytes(*v))).collect::<Vec<_>>()
    ));
    Ok((
        Measurement {
            prompts: n,
            ttft_ms: s.ttft_ms,
            decode_tps: s.decode_tps,
            token_ms: s.token_ms,
            peak_device_bytes: peaks.device_added().into_iter().collect(),
            peak_host_rss: peaks.tree_rss,
        },
        s,
    ))
}

/// Why each resource is or is not used, from the measured comparison where one exists.
#[allow(clippy::too_many_arguments)]
fn explain(
    plan: &mut Plan,
    lay: &Layout,
    cost: &CostModel,
    shape: &Shape,
    devices: &[BackendDevice],
    chosen: &Candidate,
    measured: &[(Candidate, Option<RunSummary>)],
    rejected: &[String],
) {
    let rate = |c: &Candidate| measured.iter().find(|(m, _)| m.label == c.label).and_then(|(_, s)| s.as_ref().map(|s| s.decode_tps.p50));
    let chosen_rate = rate(chosen);
    let a = &chosen.assignment;
    for d in devices.iter().filter(|d| d.kind == "gpu") {
        if plan.decisions.iter().any(|x| x.resource.starts_with(&d.name)) {
            continue;
        }
        let blocks: Vec<usize> = (0..a.layer_device.len()).filter(|&i| a.layer_device[i] == d.name).collect();
        let resource = format!("{} ({})", d.name, d.description);
        if chosen.placement.uses_device(&d.name) {
            let span = match (blocks.first(), blocks.last(), chosen.placement.split_mode) {
                (_, _, SplitMode::Layer) if blocks.is_empty() => "no blocks".into(),
                (Some(f), Some(l), SplitMode::Layer) => format!("blocks {f}-{l}"),
                (_, _, mode) => {
                    let k = chosen.placement.devices.iter().position(|x| x == &d.name).unwrap_or(0);
                    format!("share {} of every block ({} split)", chosen.placement.tensor_split.get(k).copied().unwrap_or(0.0), mode.as_str())
                }
            };
            let out = if a.output_device == d.name { " + output head" } else { "" };
            let mem = plan.budgets.iter().find(|b| b.device == d.name).map(|b| b.required()).unwrap_or(0);
            plan.decisions.push(Decision {
                resource,
                used: true,
                reason: format!("{span}{out}; backend-measured {} of {} free", fmt_bytes(mem), fmt_bytes(d.mem_free)),
            });
        } else {
            let best_with = measured
                .iter()
                .filter(|(c, s)| c.placement.uses_device(&d.name) && s.is_some())
                .filter_map(|(c, s)| Some((c, s.as_ref()?.decode_tps.p50)))
                .max_by(|x, y| x.1.total_cmp(&y.1));
            let reason = match (best_with, chosen_rate) {
                (Some((c, r)), Some(cr)) => format!("unused: best plan using it ({}) measured {r:.2} tok/s vs {cr:.2} tok/s without it", c.label),
                _ => "unused: placements using it were predicted slower and not measured".into(),
            };
            plan.decisions.push(Decision { resource, used: false, reason });
        }
    }
    let cpu_blocks = a.layer_device.iter().filter(|d| *d == CPU).count();
    let host_experts: Vec<(usize, usize)> =
        a.on_cpu.iter().enumerate().flat_map(|(i, v)| v.iter().enumerate().filter(|(_, &h)| h).map(move |(k, _)| (i, k))).collect();
    let mut cpu = format!("token embeddings; {} threads (best of the measured sweep)", plan.runtime.n_threads);
    if cpu_blocks > 0 {
        cpu = format!("blocks 0-{} ({}); {cpu}", cpu_blocks - 1, fmt_bytes(lay.blocks[..cpu_blocks].iter().map(|b| b.dense_bytes() + b.expert_bytes()).sum()));
    }
    if chosen.split.is_some() {
        cpu.push_str(&format!("; cold experts of {} blocks stay in host memory", chosen.placement.overrides.len()));
    } else if !host_experts.is_empty() {
        let bytes: u64 = host_experts.iter().map(|&(i, k)| lay.blocks[i].experts[k].bytes).sum();
        let (i, k) = host_experts[0];
        let e = &lay.blocks[i].experts[k];
        let f = lay.expert_read_fraction(shape.batch);
        let dev = &a.layer_device[i];
        let pcie = cost.copy_s(CPU, dev, 64 << 20);
        let pcie_gbps = (64u64 << 20) as f64 / pcie.max(1e-9) / 1e9;
        let cpu_s = cost.decode_block_s(lay, &crate::layers::Block { experts: vec![e.clone()], ..Default::default() }, dev, &[true], shape);
        let why = if cpu_beats_transfer(cpu_s, (e.bytes as f64 * f) as u64, pcie_gbps) {
            "running them on the CPU beats copying the selected experts per token"
        } else {
            "no GPU memory left for them"
        };
        cpu.push_str(&format!("; {} routed-expert tensors ({}) stay in host memory: {why}", host_experts.len(), fmt_bytes(bytes)));
    }
    plan.decisions.push(Decision { resource: "CPU".into(), used: true, reason: cpu });
    plan.decisions.push(Decision {
        resource: "storage".into(),
        used: chosen.storage_bound_tps.is_some(),
        reason: match chosen.storage_bound_tps {
            Some(b) => format!("weights exceed host memory and stream from storage; bound {b:.3} tok/s"),
            None => format!("load only (memory-mapped); {} of weights resident in RAM", fmt_bytes(plan.host.resident_weights)),
        },
    });
    let rate_of = |pred: &dyn Fn(&Candidate) -> bool| {
        measured
            .iter()
            .filter(|(c, s)| pred(c) && s.is_some())
            .map(|(_, s)| s.as_ref().unwrap().decode_tps.p50)
            .fold(None, |m: Option<f64>, r| Some(m.map_or(r, |m| m.max(r))))
    };
    let split_rate = rate_of(&|c: &Candidate| c.split.is_some());
    let tensor_rate = rate_of(&|c: &Candidate| c.split.is_none() && c.speculation.is_none());
    if lay.n_expert > 0 {
        let split = measured.iter().find_map(|(c, _)| c.split.as_ref().map(|(_, s)| s));
        plan.decisions.push(Decision {
            resource: "routed experts".into(),
            used: chosen.split.is_some(),
            reason: match (split, split_rate, tensor_rate) {
                (Some(s), Some(r), Some(t)) => format!(
                    "per-expert residency: {} hot experts serve {:.1}% of calibration routing from GPU memory vs {:.1}% with whole tensors; measured {r:.2} vs {t:.2} tok/s{}",
                    s.hot_experts,
                    s.gpu_served * 100.0,
                    s.gpu_served_by_tensors * 100.0,
                    if chosen.split.is_some() { "" } else { "; not faster" }
                ),
                (Some(_), _, _) => "per-expert residency candidate failed to run".into(),
                _ => "whole expert tensors; per-expert residency not applicable (see candidates)".into(),
            },
        });
    }
    let spec_rate = rate_of(&|c: &Candidate| c.speculation.is_some());
    let plain_rate = rate_of(&|c: &Candidate| c.speculation.is_none());
    plan.decisions.push(Decision {
        resource: "speculation".into(),
        used: chosen.speculation.is_some(),
        reason: match (&chosen.speculation, spec_rate, plain_rate) {
            (Some(sp), Some(s), Some(p)) => format!("{} with {} drafted tokens: {s:.2} tok/s vs {p:.2} tok/s without", sp.kind, sp.n_max),
            (None, Some(s), Some(p)) => format!("measured {s:.2} tok/s with drafting vs {p:.2} tok/s without; not faster"),
            _ if measured.iter().any(|(c, _)| c.speculation.is_some()) => "drafting candidates failed or were not certified".into(),
            _ => "not measured (opt-in with --speculation; needs next-token heads or a draft model)".into(),
        },
    });
    plan.decisions.push(Decision {
        resource: "KV cache".into(),
        used: true,
        reason: format!(
            "{}/{}, {} per sequence, {} sequence(s); context semantics unchanged",
            plan.runtime.type_k, plan.runtime.type_v, plan.workload.n_ctx_seq, plan.workload.concurrency
        ),
    });
    for r in rejected {
        plan.decisions.push(Decision { resource: "candidate".into(), used: false, reason: r.clone() });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_memory_is_apportioned_by_split() {
        let p = Placement {
            devices: vec!["CUDA0".into(), "CUDA1".into()],
            layer_device: vec![],
            output_device: "CUDA0".into(),
            overrides: vec![],
            n_gpu_layers: 1,
            tensor_split: vec![2.0, 1.0],
            split_mode: SplitMode::Tensor,
        };
        let m = vec![
            DeviceMemory { device: "Meta()".into(), model: 300, context: 30, compute: 0 },
            DeviceMemory { device: "CUDA0".into(), model: 0, context: 0, compute: 10 },
        ];
        let a = apportion_meta(m, &p);
        assert_eq!(total(&a, "CUDA0"), 200 + 20 + 10);
        assert_eq!(total(&a, "CUDA1"), 100 + 10);
    }

    #[test]
    fn orders_cover_every_subset_permutation() {
        let o = ordered_subsets(&["A".into(), "B".into()]);
        assert_eq!(o.len(), 5, "{o:?}");
        assert!(o.contains(&vec![]) && o.contains(&vec!["B".into(), "A".into()]));
    }
}
