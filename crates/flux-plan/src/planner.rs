//! Probe → prune → enumerate → verify memory with the backend → measure finalists → save.

use crate::cost::{cpu_beats_transfer, storage_bound_tps, CostModel, Shape, CPU};
use crate::experts::Routes;
use crate::layers::{layout, Layout};
use crate::params::{backend_mapping, placement};
use crate::run::{agent_prompt, bench_sampling, chat_prompt, run_all, summarize, RunSummary};
use crate::search::{prefill_s, search, units, Assignment, SearchInput};
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
use std::path::Path;
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
    /// The GPU expert cache this candidate runs with.
    cache: Option<ExpertCache>,
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
            expert_cache: None,
            expert_cache_frozen: false,
        }
    }

    /// Backend allocation dry run: the authority on memory.
    async fn measure(&self, p: &Placement, n_ubatch: u32) -> Result<Vec<DeviceMemory>> {
        self.measure_model(&self.manifest.files[0].path, p, n_ubatch).await
    }

    async fn measure_model(&self, model: &Path, p: &Placement, n_ubatch: u32) -> Result<Vec<DeviceMemory>> {
        self.measure_params(&self.params(model, p, n_ubatch)).await
    }

    async fn measure_params(&self, params: &BackendParams) -> Result<Vec<DeviceMemory>> {
        let v = oneshot_job(&["measure"], &serde_json::to_value(params)?).await?;
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
                    x.staging += share(m.staging);
                }
                None => out.push(DeviceMemory { device: d.clone(), model: share(m.model), context: share(m.context), compute: share(m.compute), staging: share(m.staging) }),
            }
        }
    }
    out
}

fn total(m: &[DeviceMemory], dev: &str) -> u64 {
    m.iter().filter(|d| d.device == dev).map(|d| d.model + d.context + d.compute).sum()
}

/// GPU orders past this many are cut down to the ones the finalists are drawn from.
const MAX_ORDERS: usize = 128;

/// What the placement search can tell apart about a GPU.
struct GpuTraits {
    name: String,
    /// Description and total memory.
    model: (String, u64),
    /// Bytes for weights and state, per prompt chunk size.
    capacity: Vec<u64>,
    /// Seconds to upload 64 MiB from pinned host memory.
    upload_s: f64,
}

impl GpuTraits {
    /// Same model, room within one search quantum, and host links within 20%: an x4 slot differs from
    /// an x16 one, measurement noise does not. Kernel rates and peer links are left out because their
    /// noise would split identical cards.
    fn interchangeable(&self, o: &GpuTraits) -> bool {
        self.model == o.model
            && self.capacity.iter().zip(&o.capacity).all(|(&a, &b)| units(a).abs_diff(units(b)) <= 1)
            && (self.upload_s - o.upload_s).abs() <= 0.2 * self.upload_s.max(o.upload_s)
    }
}

/// Groups GPUs the search cannot tell apart; every member of a class is interchangeable with every
/// other. Classes and their members are listed by room, most first.
fn gpu_classes(mut gpus: Vec<GpuTraits>) -> Vec<Vec<String>> {
    gpus.sort_by(|a, b| b.capacity.first().cmp(&a.capacity.first()).then_with(|| a.name.cmp(&b.name)));
    let mut classes: Vec<Vec<GpuTraits>> = vec![];
    for g in gpus {
        match classes.iter_mut().find(|c| c.iter().all(|m| m.interchangeable(&g))) {
            Some(c) => c.push(g),
            None => classes.push(vec![g]),
        }
    }
    classes.into_iter().map(|c| c.into_iter().map(|g| g.name).collect()).collect()
}

/// GPU orders for the placement search: every ordered subset, taking the GPUs of a class in class
/// order so that mirror-image orders are searched once; past `MAX_ORDERS`, the `capped_orders`.
fn gpu_orders(classes: &[Vec<String>]) -> Vec<Vec<String>> {
    let named = |o: &[usize]| -> Vec<String> {
        let mut used = vec![0; classes.len()];
        o.iter()
            .map(|&c| {
                used[c] += 1;
                classes[c][used[c] - 1].clone()
            })
            .collect()
    };
    let mut out = vec![vec![]];
    let mut frontier: Vec<Vec<usize>> = vec![vec![]];
    while !frontier.is_empty() && out.len() <= MAX_ORDERS {
        let mut next = vec![];
        for o in &frontier {
            for (c, members) in classes.iter().enumerate() {
                if o.iter().filter(|&&x| x == c).count() < members.len() {
                    let mut x = o.clone();
                    x.push(c);
                    next.push(x);
                }
            }
        }
        out.extend(next.iter().map(|o| named(o)));
        frontier = next;
    }
    if out.len() <= MAX_ORDERS {
        out
    } else {
        capped_orders(classes)
    }
}

/// CPU only, each class's first GPU alone, and each class ahead of the others. The search may skip
/// any GPU of an order, so this loses candidate variety, not placements.
fn capped_orders(classes: &[Vec<String>]) -> Vec<Vec<String>> {
    let mut out = vec![vec![]];
    out.extend(classes.iter().map(|c| vec![c[0].clone()]));
    for i in 0..classes.len() {
        let rest = classes.iter().enumerate().filter(|&(j, _)| j != i).flat_map(|(_, c)| c);
        out.push(classes[i].iter().chain(rest).cloned().collect());
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

    let threads = report.decode_cpu_bandwidth().map_or(report.inventory.cpu.cores, |b| b.threads);
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
    let lay = layout(manifest, &facts, req.workload.concurrency as u64, n_ctx_seq as u64, tk, tv, req.mlock);
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
    let reserve = |g: &BackendDevice| cfg.plan.device_reserve(g);
    let capacity_for = |ub: u32| -> HashMap<String, u64> {
        gpus.iter().map(|g| (g.name.clone(), g.mem_free.saturating_sub(reserve(g) + compute.get(&(g.name.clone(), ub)).copied().unwrap_or(0)))).collect()
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
    let classes = gpu_classes(
        gpus.iter()
            .map(|g| GpuTraits {
                name: g.name.clone(),
                model: (g.description.clone(), g.mem_total),
                capacity: ubatches.iter().map(|&ub| capacity_for(ub)[&g.name]).collect(),
                upload_s: cost.copy_s(CPU, &g.name, 64 << 20),
            })
            .collect(),
    );
    let orders = gpu_orders(&classes);
    if !classes.is_empty() {
        log(&format!("searching {} GPU orders over {}", orders.len(), classes.iter().map(|c| c.join("=")).collect::<Vec<_>>().join(", ")));
    }
    let mut cands: Vec<Candidate> = vec![];
    let mut rejected: Vec<String> = vec![];
    let mut seen = vec![];
    for (&ub, order) in ubatches.iter().flat_map(|ub| orders.iter().cloned().map(move |o| (ub, o))) {
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
                    let limit = g.mem_free.saturating_sub(reserve(g));
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
                cache: None,
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
            if let Some(g) = gpus.iter().find(|g| total(&measured, &g.name) > g.mem_free.saturating_sub(reserve(g))) {
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
                cache: None,
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
            .map(|g| g.mem_free.saturating_sub(reserve(g) + total(&c.measured, &g.name)))
            .min()
            .unwrap_or(u64::MAX)
    };
    // llama-server drafts with a draft model or the model's heads, on the fastest predicted placement that
    // leaves room; the native engine drafts with the heads on measured placements, after calibration.
    let server_spec = req.speculation && req.engines.contains(&EngineKind::LlamaServer) && !drafters.is_empty();
    let mut draft_base: Option<Candidate> = None;
    if server_spec {
        for c in &cands {
            // A draft model needs about `DRAFT_MODEL_ROOM` next to the target's weights. The model's own heads load
            // only with drafting, so the placement is measured with them.
            let fits = match &req.draft_model {
                Some(_) => headroom(c) >= DRAFT_MODEL_ROOM,
                None => {
                    let params = BackendParams {
                        speculation: Some(Speculation { kind: "draft-mtp".into(), draft_model: None, n_max: 4, draft_vocab: None }),
                        ..ctx.params(&ctx.manifest.files[0].path, &c.placement, c.n_ubatch)
                    };
                    match ctx.measure_params(&params).await {
                        Ok(m) => gpus.iter().filter(|g| c.placement.uses_device(&g.name)).all(|g| total(&m, &g.name) + reserve(g) + SPEC_MARGIN <= g.mem_free),
                        Err(e) => {
                            rejected.push(format!("speculation (llama-server) on {}: memory measurement with the heads failed: {e:#}", c.label));
                            false
                        }
                    }
                }
            };
            if fits {
                draft_base = Some(c.clone());
                break;
            }
        }
        if draft_base.is_none() {
            rejected.push("speculation (llama-server): no placement leaves room for the draft context".into());
        }
    }
    if let (true, Some(base)) = (server_spec, &draft_base) {
        for (kind, draft_model) in &drafters {
            for n_max in [2, 4] {
                let mut c = base.clone();
                c.engine = EngineKind::LlamaServer;
                c.label = format!("llama-server + {kind} {n_max}: {}", base.label);
                c.speculation = Some(Speculation { kind: kind.clone(), draft_model: draft_model.clone(), n_max, draft_vocab: None });
                runs.push(c);
            }
        }
    }
    let native_mtp = req.speculation && req.engines.contains(&EngineKind::Native) && drafters.iter().any(|(k, _)| k == "draft-mtp");

    // Calibration, then a race of the top two on separate validation prompts.
    let budget = Duration::from_secs_f64(cfg.plan.tuning_budget_s);
    let base_plan = |c: &Candidate| -> Plan { assemble(&ctx, &facts.architecture, &identity, report, &devices, c, host_avail, vec![], None) };
    let mut results: Vec<(usize, CandidateResult, Option<RunSummary>)> = vec![];
    for (i, c) in runs.iter().enumerate() {
        if t_start.elapsed() > budget && !results.is_empty() {
            results.push((i, failed_result(c, "skipped: tuning budget spent".into()), None));
            continue;
        }
        log(&format!("calibrating {}", c.label));
        let reference = match (c.speculation.is_some() && hybrid).then(|| base_plan(&Candidate { speculation: None, ..c.clone() })) {
            Some(plain) => match greedy_tokens(&ctx, &plain, corpus).await {
                Ok(t) => Some(t),
                Err(e) => {
                    results.push((i, failed_result(c, format!("rollback certification failed: {e:#}")), None));
                    continue;
                }
            },
            None => None,
        };
        let (result, summary) = calibrate(&ctx, c, &base_plan(c), reference.as_deref(), corpus, &devices).await;
        results.push((i, result, summary));
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
    // Stages below build on the best measured run so far, ranked like the final choice.
    let best = |results: &[(usize, CandidateResult, Option<RunSummary>)], runs: &[Candidate], keep: &dyn Fn(&Candidate) -> bool| {
        let mut ks: Vec<usize> = (0..results.len()).filter(|&k| results[k].2.is_some() && keep(&runs[results[k].0])).collect();
        order(&mut ks, &|k| results[k].2.clone().unwrap());
        ks.first().map(|&k| (runs[results[k].0].clone(), results[k].2.clone().unwrap()))
    };
    // Native drafting on the fastest measured placement, one candidate per draft length.
    let mut spec_n_max = 3;
    // Whether drafting certified (rollback reproduces the plain greedy output) on the plain placement.
    let mut spec_certified = false;
    let plain_best = best(&results, &runs, &|c| c.engine == EngineKind::Native && c.speculation.is_none() && c.cache.is_none());
    // The budget caps the placement finalists above; the stages that build on the fastest one run once
    // regardless, because on MoE models they are worth more than any placement (15 -> 49 tok/s on Flash-Next).
    if let (true, Some((base, outputs))) = (native_mtp, plain_best) {
        match reference_tokens(&ctx, &base_plan(&base), hybrid, corpus).await {
            Ok(reference) => {
                let mut best: Option<(u32, f64)> = None;
                let vocab = draft_vocab(&outputs.tokens, lay.n_vocab);
                for n_max in [3, 4] {
                    let (c, result, summary) = speculate(&ctx, &base, n_max, vocab, &lay, facts.n_layer as usize, &base_plan, reference.as_deref(), &gpus, corpus, &devices).await;
                    if let Some(x) = summary.as_ref().map(&score) {
                        if best.is_none_or(|(_, b)| x > b) {
                            best = Some((n_max, x));
                        }
                    }
                    runs.push(c);
                    results.push((runs.len() - 1, result, summary));
                }
                spec_n_max = best.map_or(spec_n_max, |(n, _)| n);
                spec_certified = best.is_some();
            }
            Err(e) => rejected.push(format!("speculation on {}: greedy reference failed: {e:#}", base.label)),
        }
    }
    // Per-expert residency on the best measured placement that keeps experts in host memory: trace its
    // routing, size the GPU expert cache, then calibrate it like any other candidate.
    let cacheable = |c: &Candidate| {
        c.engine == EngineKind::Native
            && c.speculation.is_none()
            && c.placement.split_mode == SplitMode::Layer
            && c.assignment.on_cpu.iter().zip(&c.assignment.layer_device).any(|(h, d)| d != CPU && h.contains(&true))
    };
    let cache_base = best(&results, &runs, &cacheable);
    if let (true, Some((base, _))) = (req.expert_residency && lay.n_expert > 0, cache_base) {
        (ctx.log)(&format!("tracing expert routing on {}", base.label));
        match trace_routes(&ctx, &base_plan(&base), corpus).await {
            Ok(routes) => {
                // Every prompt chunk streams the host experts to the GPU once, so larger chunks process long
                // prompts faster but leave less room for cached experts: size the cache for every chunk size and
                // let the measured runs, short and long, choose.
                let sizes = std::iter::once(base.n_ubatch).chain(ubatches.iter().copied().filter(|&u| u != base.n_ubatch));
                for ub in sizes {
                    let base = match ub == base.n_ubatch {
                        true => base.clone(),
                        false => match ctx.measure(&base.placement, ub).await {
                            Ok(measured) => Candidate {
                                label: base.placement.describe(),
                                n_ubatch: ub,
                                host_compute: measured.iter().filter(|d| d.device == CPU).map(|d| d.compute).sum(),
                                measured,
                                ..base.clone()
                            },
                            Err(e) => {
                                rejected.push(format!("per-expert residency on {} with ubatch {ub}: {e:#}", base.label));
                                continue;
                            }
                        },
                    };
                    let shape = Shape { ubatch: ub, ..shape };
                    // Drafting on the cache later loads the heads, their draft context and the rollback snapshots:
                    // measure what that adds on the base placement and keep it free.
                    let mut head_room: HashMap<String, u64> = HashMap::new();
                    if native_mtp {
                        let params = BackendParams {
                            speculation: Some(Speculation { kind: "draft-mtp".into(), draft_model: None, n_max: spec_n_max, draft_vocab: None }),
                            ..ctx.params(&ctx.manifest.files[0].path, &base.placement, ub)
                        };
                        match ctx.measure_params(&params).await {
                            Ok(m) => {
                                for g in &gpus {
                                    head_room.insert(g.name.clone(), total(&m, &g.name).saturating_sub(total(&base.measured, &g.name)) + SPEC_MARGIN);
                                }
                            }
                            Err(e) => rejected.push(format!("speculation on the expert cache: memory measurement with the heads failed: {e:#}")),
                        }
                    }
                    match expert_cache(&ctx, &base, &lay, &cost, &shape, &gpus, &routes, facts.n_layer as usize, &head_room).await {
                        Ok(Some(c)) => {
                            log(&format!("calibrating {} (predicted {:.2} ms/token)", c.label, c.predicted_decode_s * 1e3));
                            let (result, summary) = calibrate(&ctx, &c, &base_plan(&c), None, corpus, &devices).await;
                            let calibrated = summary.is_some();
                            let vocab = summary.as_ref().and_then(|s| draft_vocab(&s.tokens, lay.n_vocab));
                            runs.push(c.clone());
                            results.push((runs.len() - 1, result, summary));
                            // Hot experts on the GPU also make verification cheaper: every extra token in the batch
                            // mostly reuses resident experts. Draft on the cache with the best measured draft length.
                            if native_mtp && spec_certified && calibrated {
                                match reference_tokens(&ctx, &base_plan(&c), hybrid, corpus).await {
                                    Ok(reference) => {
                                        let (s, result, summary) = speculate(&ctx, &c, spec_n_max, vocab, &lay, facts.n_layer as usize, &base_plan, reference.as_deref(), &gpus, corpus, &devices).await;
                                        runs.push(s);
                                        results.push((runs.len() - 1, result, summary));
                                    }
                                    Err(e) => rejected.push(format!("speculation on {}: greedy reference failed: {e:#}", c.label)),
                                }
                            }
                        }
                        Ok(None) => rejected.push(format!("per-expert residency on {}: no block would split", base.label)),
                        Err(e) => rejected.push(format!("per-expert residency on {}: {e:#}", base.label)),
                    }
                }
            }
            Err(e) => rejected.push(format!("per-expert residency on {}: tracing expert routing failed: {e:#}", base.label)),
        }
    }
    let mut ranked: Vec<usize> = (0..results.len()).filter(|&k| results[k].2.is_some()).collect();
    order(&mut ranked, &|k| results[k].2.clone().unwrap());
    ensure!(
        !ranked.is_empty(),
        "every candidate failed to run:\n  {}",
        results.iter().map(|r| format!("{}: {}", r.1.label, r.1.failure.clone().unwrap_or_default())).collect::<Vec<_>>().join("\n  ")
    );
    // Long prompts: one request a quarter of the planned context deep, so the choice also holds for clients that
    // send long prompts. The contenders are the two best short-prompt runs and the best of every other chunk
    // size, the setting long prompts expose; ranking them needs no assumption about the mix of requests.
    let depth_tokens = (n_ctx_seq / 4).min(n_ctx_seq.saturating_sub(2 * req.decode_tokens));
    let deep = matches!(req.workload.objective, Objective::Interactive) && depth_tokens >= 4 * req.prompt_tokens;
    if deep {
        let mut contenders: Vec<usize> = ranked.iter().copied().take(2).collect();
        for &k in &ranked {
            if contenders.len() >= MAX_CONTENDERS {
                break;
            }
            if !contenders.iter().any(|&c| runs[results[c].0].n_ubatch == runs[results[k].0].n_ubatch) {
                contenders.push(k);
            }
        }
        for k in contenders {
            log(&format!("long prompt on {}", results[k].1.label));
            let plan = base_plan(&runs[results[k].0]);
            match depth_run(&ctx, &plan, corpus, depth_tokens).await {
                Ok(d) => {
                    log(&format!(
                        "  {} prompt tokens at {:.0} tok/s ({:.1} s to the first token), decode {:.2} tok/s",
                        d.prompt_tokens,
                        d.prefill_tps,
                        d.ttft_ms / 1e3,
                        d.decode_tps
                    ));
                    results[k].1.depth = Some(d);
                }
                Err(e) => {
                    log(&format!("  failed: {e:#} (worker log: {})", cfg.data_dir.join("logs").join(format!("plan-{}.log", plan.id)).display()));
                    rejected.push(format!("{}: long prompt failed: {e:#}", results[k].1.label));
                }
            }
        }
        let measured: Vec<usize> = ranked.iter().copied().filter(|&k| results[k].1.depth.is_some()).collect();
        if !measured.is_empty() {
            ranked = measured;
            robust(&mut ranked, &|k| results[k].2.clone().unwrap(), &|k| results[k].1.depth.clone().unwrap());
        }
    }
    let deep = deep && results[ranked[0]].1.depth.is_some();
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
    if deep {
        robust(&mut finals, &|i| validated[i].1.clone(), &|i| results[validated[i].0].1.depth.clone().unwrap());
    } else {
        order(&mut finals, &|i| validated[i].1.clone());
    }
    let best = finals.first().map_or(ranked[0], |&i| validated[i].0);
    let chosen = &runs[results[best].0];
    let validation = ValidationRecord {
        tuning_seconds: t_start.elapsed().as_secs_f64(),
        tuning_budget_seconds: cfg.plan.tuning_budget_s,
        candidates: results
            .iter()
            .map(|(i, r, _)| {
                let c = &runs[*i];
                CandidateResult {
                    expert_cache: c.cache.as_ref().map(|x| x.spec.clone()),
                    placement: Some(c.placement.clone()),
                    n_ubatch: Some(c.n_ubatch),
                    speculation: c.speculation.clone(),
                    ..r.clone()
                }
            })
            .collect(),
        chosen: chosen.label.clone(),
        reason: match req.workload.objective {
            Objective::Interactive if deep => format!(
                "best worst case over decode rate and first-token time on short prompts ({} tokens) and on a long one ({depth_tokens} tokens), each relative to the best candidate there (within 3% are ties, won by the higher geometric mean)",
                req.prompt_tokens
            ),
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
    let budgets = devices
        .iter()
        .filter(|d| d.kind == "gpu")
        .map(|d| {
            let m: Vec<&DeviceMemory> = c.measured.iter().filter(|m| m.device == d.name).collect();
            DeviceBudget {
                device: d.name.clone(),
                capacity: d.mem_total,
                free_at_plan: d.mem_free,
                reserve: ctx.cfg.plan.device_reserve(d),
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
        model_files: ctx.manifest.paths(),
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
        expert_cache: c.cache.clone(),
    };
    plan.id = plan.compute_id();
    plan
}

/// Per-expert residency on `base`: chooses each GPU block's most-routed experts (by `routes`) within the
/// memory the base gives its experts (plus `extra_room` kept free), keeps those blocks' expert tensors in host
/// memory behind a GPU cache that starts with the chosen experts and adapts while decoding, and verifies the
/// allocation with the backend, shrinking on overflow.
#[allow(clippy::too_many_arguments)]
async fn expert_cache(
    ctx: &Ctx<'_>,
    base: &Candidate,
    lay: &Layout,
    cost: &CostModel,
    shape: &Shape,
    gpus: &[&BackendDevice],
    routes: &Routes,
    n_trunk: usize,
    extra_room: &HashMap<String, u64>,
) -> Result<Option<Candidate>> {
    // The cached path adds a few small nodes per block; keep a margin for their buffers, and `extra_room` for
    // what the plan will also have to hold (next-token heads and their draft context).
    let mut spare: HashMap<String, i64> = gpus
        .iter()
        .map(|g| {
            // The pools of the blocks on this device will hold its prefill staging (see `add_cache`).
            let holds = base.assignment.layer_device.iter().any(|d| d == &g.name);
            let staging: u64 = if holds { base.measured.iter().filter(|m| m.device == g.name).map(|m| m.staging).sum() } else { 0 };
            let keep = ctx.cfg.plan.device_reserve(g) + total(&base.measured, &g.name).saturating_sub(staging) + (128 << 20) + extra_room.get(&g.name).copied().unwrap_or(0);
            (g.name.clone(), g.mem_free as i64 - keep as i64)
        })
        .collect();
    for attempt in 0..3 {
        let Some(r) = crate::experts::choose(lay, n_trunk, &base.assignment, cost, shape, routes, &spare) else { return Ok(None) };
        let mut p = base.placement.clone();
        p.overrides = crate::experts::overrides(lay, &r);
        let mut measured = ctx.measure_model(&ctx.manifest.files[0].path, &p, base.n_ubatch).await.context("dry run with the experts in host memory")?;
        add_cache(&mut measured, crate::experts::cache_bytes(lay, &base.assignment, &r.spec));
        let over: Vec<(String, u64)> = gpus
            .iter()
            .filter_map(|g| {
                let (need, limit) = (total(&measured, &g.name), g.mem_free.saturating_sub(ctx.cfg.plan.device_reserve(g)));
                (need > limit).then(|| (g.name.clone(), need - limit))
            })
            .collect();
        if over.is_empty() {
            // The assignment must describe this placement: only the cached blocks keep experts in host memory,
            // so anything rebuilding the placement from it (drafting forces head experts onto their GPU)
            // reproduces the cache's overrides.
            let mut assignment = base.assignment.clone();
            for (i, b) in lay.blocks.iter().enumerate() {
                assignment.on_cpu[i] = vec![r.host_blocks.contains(&i); b.experts.len()];
            }
            return Ok(Some(Candidate {
                label: format!("{} · {} cached experts", base.label.replace(&base.placement.describe(), &p.describe()), r.hot_experts),
                predicted_decode_s: r.decode_s,
                assignment,
                placement: p,
                measured,
                cache: Some(ExpertCache {
                    spec: r.spec,
                    hot_experts: r.hot_experts,
                    gpu_served: r.gpu_served,
                    gpu_served_by_tensors: r.gpu_served_by_tensors,
                    frozen: false,
                }),
                ..base.clone()
            }));
        }
        (ctx.log)(&format!("repair expert cache (attempt {}): measured {over:?} over budget", attempt + 1));
        for (g, by) in &over {
            *spare.get_mut(g).unwrap() -= (by + (64 << 20)) as i64;
        }
    }
    bail!("the expert cache still exceeds device memory after repairs")
}

/// Adds an expert cache's pools to a dry run's memory (a dry run allocates no cache). The backend lends a
/// device's prefill staging from the pools of the blocks placed there when those alone can hold it, so the
/// staging then stops counting as compute.
fn add_cache(measured: &mut Vec<DeviceMemory>, cache: Vec<(String, crate::experts::CacheBytes)>) {
    for (dev, bytes) in cache {
        match measured.iter_mut().find(|m| m.device == dev) {
            Some(m) => {
                m.model += bytes.own + bytes.tier;
                if bytes.own >= m.staging {
                    m.compute = m.compute.saturating_sub(m.staging);
                    m.staging = 0;
                }
            }
            None => measured.push(DeviceMemory { device: dev, model: bytes.own + bytes.tier, ..Default::default() }),
        }
    }
}

/// The prefix of token ids to draft from: every id the model emitted in `outputs`, rounded up to a multiple of
/// 16384. None (the whole vocabulary) when that would keep over three quarters of it or nothing was emitted.
fn draft_vocab(outputs: &[i32], n_vocab: u32) -> Option<u32> {
    let n = (*outputs.iter().max()? as u32 + 1).div_ceil(16384) * 16384;
    (n as u64 * 4 <= n_vocab as u64 * 3).then_some(n)
}

/// Generated tokens traced per calibration prompt for routing counts.
const ROUTE_TOKENS: u32 = 256;

/// Routing counts per layer and expert over tokens generated for the calibration prompts: as many coding-agent
/// conversations as prose chat prompts. A cache traced on prose alone served 31% of agent decoding against 74%
/// of prose; the even mix serves about 55% of each, and the cache adapts to the real traffic from there.
async fn trace_routes(ctx: &Ctx<'_>, plan: &Plan, corpus: &Corpus) -> Result<Routes> {
    let logs = ctx.cfg.data_dir.join("logs");
    std::fs::create_dir_all(&logs)?;
    let w = Worker::spawn(&EngineKind::Native, Some(&logs.join(format!("routes-{}.log", plan.id)))).await?;
    let out = async {
        w.load_with(plan, true).await?;
        let mut acc = Routes::new();
        let n = ctx.cfg.plan.calibration_prompts;
        // Residency serves decoding: prompts reach the GPU's experts in bulk, so only generated tokens count.
        for i in 0..2 * n {
            let p = match i % 2 {
                0 => chat_prompt(&w, corpus, Role::Calibration, i / 2 * 7, ctx.req.prompt_tokens as usize).await?,
                _ => agent_prompt(&w, i / 2, n).await?,
            };
            let r = w.trace(p, ROUTE_TOKENS, false, true).await?;
            // The backend lists experts up to the highest id it saw, so lengths vary.
            for (layer, c) in serde_json::from_value::<Vec<(u32, Vec<u64>)>>(r["decode"].clone())? {
                let a = acc.entry(layer).or_default();
                if a.len() < c.len() {
                    a.resize(c.len(), 0);
                }
                a.iter_mut().zip(c).for_each(|(x, y)| *x += y);
            }
        }
        Ok(acc)
    }
    .await;
    w.shutdown().await;
    out
}

/// Drafting's runtime growth beyond the buffers measured with the heads (batches, sampler scratch).
const SPEC_MARGIN: u64 = 64 << 20;

/// Room a separate draft model needs next to the target's weights on each of the plan's GPUs.
const DRAFT_MODEL_ROOM: u64 = 768 << 20;

/// Greedy output of a plan without drafting, the reference drafting must reproduce (hybrid models only).
async fn reference_tokens(ctx: &Ctx<'_>, plan: &Plan, hybrid: bool, corpus: &Corpus) -> Result<Option<Vec<Greedy>>> {
    Ok(if hybrid { Some(greedy_tokens(ctx, plan, corpus).await?) } else { None })
}

/// Drafting with the model's own heads on `base`. The heads, their draft context and the rollback
/// snapshots load only with speculation, so the plan's memory is measured with them; where it does not
/// fit, whole expert tensors of trunk blocks on that device move to host memory (split candidates keep
/// the room `expert_cache` reserved instead).
#[allow(clippy::too_many_arguments)]
async fn speculate(
    ctx: &Ctx<'_>,
    base: &Candidate,
    n_max: u32,
    draft_vocab: Option<u32>,
    lay: &Layout,
    n_trunk: usize,
    plan_of: &(dyn Fn(&Candidate) -> Plan + Sync),
    reference: Option<&[Greedy]>,
    gpus: &[&BackendDevice],
    corpus: &Corpus,
    devices: &[BackendDevice],
) -> (Candidate, CandidateResult, Option<RunSummary>) {
    let mut c = Candidate { speculation: Some(Speculation { kind: "draft-mtp".into(), draft_model: None, n_max, draft_vocab }), ..base.clone() };
    // The heads run every draft step and follow every verification: their experts stay on their GPU.
    // (Output without drafting never runs them, so this does not change the certification reference.)
    let heads: Vec<usize> = (n_trunk..lay.blocks.len()).filter(|&i| c.assignment.layer_device[i] != CPU && c.assignment.on_cpu[i].contains(&true)).collect();
    for &i in &heads {
        c.assignment.on_cpu[i] = vec![false; lay.blocks[i].experts.len()];
    }
    if !heads.is_empty() {
        c.placement = crate::params::placement(lay, &c.assignment);
    }
    let mut repaired = false;
    for attempt in 0..3 {
        c.label = match (attempt, &c.cache) {
            (0, _) => format!("native + draft-mtp {n_max}: {}", base.label),
            (_, Some(x)) => format!("native + draft-mtp {n_max}: {} · ubatch {} · {} cached experts", c.placement.describe(), c.n_ubatch, x.hot_experts),
            (_, None) => format!("native + draft-mtp {n_max}: {} · ubatch {}", c.placement.describe(), c.n_ubatch),
        };
        let params = BackendParams { speculation: c.speculation.clone(), expert_cache: c.cache.as_ref().map(|x| x.spec.clone()), ..ctx.params(&ctx.manifest.files[0].path, &c.placement, c.n_ubatch) };
        let mut m = match ctx.measure_params(&params).await {
            Ok(m) => m,
            Err(e) => return (c.clone(), failed_result(&c, format!("memory measurement with the heads failed: {e:#}")), None),
        };
        add_cache(&mut m, c.cache.as_ref().map(|x| crate::experts::cache_bytes(lay, &c.assignment, &x.spec)).unwrap_or_default());
        let over: Vec<(String, u64)> = gpus
            .iter()
            .filter_map(|g| {
                let need = total(&m, &g.name) + ctx.cfg.plan.device_reserve(g) + SPEC_MARGIN;
                (need > g.mem_free).then(|| (g.name.clone(), need - g.mem_free))
            })
            .collect();
        if over.is_empty() {
            c.measured = m;
            break;
        }
        if attempt == 2 {
            let why = format!("does not fit with the heads loaded ({})", over.iter().map(|(d, b)| format!("{d} over by {}", fmt_bytes(*b))).collect::<Vec<_>>().join(", "));
            (ctx.log)(&format!("skipping {}: {why}", c.label));
            return (c.clone(), failed_result(&c, why), None);
        }
        // With a cache, give up its least-routed experts on that device; otherwise move whole expert tensors.
        if let Some(cache) = c.cache.as_mut() {
            let unit = |l: u32| lay.blocks[l as usize].expert_bytes() / lay.n_expert as u64;
            for (dev, by) in &over {
                let mut freed = 0u64;
                // Tiers hold the least-routed cached experts (last in each list): give those up first.
                for t in cache.spec.tiers.iter_mut().filter(|t| t.device == *dev) {
                    while freed <= by + (64 << 20) && !t.layers.is_empty() {
                        for (l, es) in t.layers.iter_mut() {
                            if freed <= by + (64 << 20) && es.pop().is_some() {
                                cache.hot_experts -= 1;
                                freed += unit(*l) * if es.is_empty() { 2 } else { 1 };
                            }
                        }
                        t.layers.retain(|(_, es)| !es.is_empty());
                    }
                }
                cache.spec.tiers.retain(|t| !t.layers.is_empty());
                let layers: Vec<u32> = cache.spec.layers.keys().copied().filter(|&l| c.assignment.layer_device[l as usize] == *dev).collect();
                while freed <= by + (64 << 20) && layers.iter().any(|l| cache.spec.layers[l].1 > 0) {
                    for l in &layers {
                        let hot = &mut cache.spec.layers.get_mut(l).unwrap().1;
                        if *hot > 0 && freed <= by + (64 << 20) {
                            *hot -= 1;
                            cache.hot_experts -= 1;
                            freed += unit(*l);
                        }
                    }
                }
            }
            repaired = true;
            (ctx.log)(&format!("repair for drafting: measured {over:?} over budget, shrinking the expert cache"));
            continue;
        }
        for (dev, by) in &over {
            let mut freed = 0u64;
            for i in (0..n_trunk).filter(|&i| c.assignment.layer_device[i] == *dev) {
                if freed > by + (64 << 20) {
                    break;
                }
                let experts = &lay.blocks[i].experts;
                c.assignment.on_cpu[i].resize(experts.len(), false);
                for (k, e) in experts.iter().enumerate() {
                    if !c.assignment.on_cpu[i][k] {
                        c.assignment.on_cpu[i][k] = true;
                        freed += e.bytes;
                    }
                }
            }
        }
        c.placement = crate::params::placement(lay, &c.assignment);
        repaired = true;
        (ctx.log)(&format!("repair for drafting: measured {over:?} over budget, moving experts to host memory"));
    }
    // Host and GPU kernels round differently: certify against the repaired placement without drafting.
    let fresh;
    let reference = match (reference, repaired) {
        (Some(_), true) => match greedy_tokens(ctx, &plan_of(&Candidate { speculation: None, ..c.clone() }), corpus).await {
            Ok(t) => {
                fresh = t;
                Some(&fresh[..])
            }
            Err(e) => return (c.clone(), failed_result(&c, format!("rollback certification failed: {e:#}")), None),
        },
        (r, _) => r,
    };
    (ctx.log)(&format!("calibrating {}", c.label));
    let (result, summary) = calibrate(ctx, &c, &plan_of(&c), reference, corpus, devices).await;
    (c, result, summary)
}

fn failed_result(c: &Candidate, why: String) -> CandidateResult {
    CandidateResult { label: c.label.clone(), predicted_token_ms: c.predicted_decode_s * 1e3, calibration: None, validation: None, failure: Some(why), expert_cache: None, placement: None, n_ubatch: None, speculation: None, depth: None }
}

/// Calibrates one candidate. With `reference` (greedy output of the same plan without drafting, for
/// hybrid models) drafting is first certified to roll recurrent state back exactly.
async fn calibrate(ctx: &Ctx<'_>, c: &Candidate, plan: &Plan, reference: Option<&[Greedy]>, corpus: &Corpus, devices: &[BackendDevice]) -> (CandidateResult, Option<RunSummary>) {
    let failed = |why: String| failed_result(c, why);
    if let Some(a) = reference {
        // Verification batches round differently from single-token steps, so greedy output may flip at a
        // near-tie: where the drafted run first departs, it must pick the plain run's runner-up. A wrong
        // rollback leaves stale state, whose choice there is arbitrary.
        let cert = async {
            let b = greedy_tokens(ctx, plan, corpus).await?;
            Ok::<bool, anyhow::Error>(a.iter().zip(&b).all(|((x, alts), (y, _))| match x.iter().zip(y).position(|(p, q)| p != q) {
                None => true,
                Some(d) => alts[d] == Some(y[d]),
            }))
        }
        .await;
        match cert {
            Ok(true) => (ctx.log)("  rollback certified: greedy output matches without drafting up to near-ties"),
            Ok(false) => {
                let why = "rollback not certified: greedy output departs from the plain run at a non-tie".to_string();
                (ctx.log)(&format!("  {why}"));
                return (failed(why), None);
            }
            Err(e) => {
                let why = format!("rollback certification failed: {e:#}");
                (ctx.log)(&format!("  {why}"));
                return (failed(why), None);
            }
        }
    }
    match measured_run(ctx, plan, corpus, Role::Calibration, ctx.cfg.plan.calibration_prompts, devices).await {
        Ok((m, s)) => {
            (ctx.log)(&format!("  decode {:.2} tok/s p50, TTFT {:.0} ms p50", s.decode_tps.p50, s.ttft_ms.p50));
            (CandidateResult { label: c.label.clone(), predicted_token_ms: c.predicted_decode_s * 1e3, calibration: Some(m), validation: None, failure: None, expert_cache: None, placement: None, n_ubatch: None, speculation: None, depth: None }, Some(s))
        }
        Err(e) => (failed(format!("{e:#}")), None),
    }
}

/// Greedy tokens of one certification prompt, with each row's runner-up (two validation prompts compare
/// two configurations of the same artifact).
type Greedy = (Vec<i32>, Vec<Option<i32>>);

async fn greedy_tokens(ctx: &Ctx<'_>, plan: &Plan, corpus: &Corpus) -> Result<Vec<Greedy>> {
    let w = Worker::spawn(&plan.engine, Some(&ctx.cfg.data_dir.join("logs").join(format!("cert-{}.log", plan.id)))).await?;
    // The same residency in both runs: an adapting cache would move experts between CPU and GPU kernels.
    let mut plan = plan.clone();
    if let Some(c) = plan.expert_cache.as_mut() {
        c.frozen = true;
    }
    let out = async {
        w.load(&plan).await?;
        let mut v = vec![];
        for i in 0..2 {
            let p = chat_prompt(&w, corpus, Role::Validation, 50 + i, 256).await?;
            let r = crate::run::run_stream(&w, &format!("cert-{i}"), p, 48, flux_core::protocol::Sampling { runner_up: Some(true), ..bench_sampling() }).await?;
            if let Some(e) = r.error {
                bail!("{e}");
            }
            v.push((r.tokens, r.alts));
        }
        Ok(v)
    }
    .await;
    w.shutdown().await;
    out
}

/// Contenders measured at depth, besides the two best short-prompt runs.
const MAX_CONTENDERS: usize = 4;

/// Orders candidates by their worst condition, each condition's rate taken relative to the best candidate there:
/// decode and first-token time on short prompts, prefill and decode on the long one. No condition is assumed
/// more common than another. Within 3% of the best worst case, the higher geometric mean wins.
fn robust(ks: &mut [usize], short: &dyn Fn(usize) -> RunSummary, deep: &dyn Fn(usize) -> DepthMeasurement) {
    let rates = |k: usize| {
        let (s, d) = (short(k), deep(k));
        [s.decode_tps.p50, 1e3 / s.ttft_ms.p50.max(1e-9), d.prefill_tps, d.decode_tps]
    };
    let all: Vec<[f64; 4]> = ks.iter().map(|&k| rates(k)).collect();
    let top: Vec<f64> = (0..4).map(|c| all.iter().map(|r| r[c]).fold(0.0, f64::max).max(1e-12)).collect();
    let score: HashMap<usize, (f64, f64)> = ks
        .iter()
        .zip(&all)
        .map(|(&k, r)| {
            let rel: Vec<f64> = r.iter().zip(&top).map(|(v, t)| v / t).collect();
            let worst = rel.iter().copied().fold(f64::MAX, f64::min);
            let mean = rel.iter().map(|x| x.max(1e-12).ln()).sum::<f64>() / rel.len() as f64;
            (k, (worst, mean))
        })
        .collect();
    let best_worst = score.values().map(|s| s.0).fold(0.0, f64::max);
    let near = |k: usize| score[&k].0 >= best_worst - 0.03;
    ks.sort_by(|&a, &b| {
        near(b).cmp(&near(a)).then_with(|| match near(a) && near(b) {
            true => score[&b].1.total_cmp(&score[&a].1),
            false => score[&b].0.total_cmp(&score[&a].0),
        })
    });
}

/// One long request on a candidate: prefill rate and decode rate at that depth.
async fn depth_run(ctx: &Ctx<'_>, plan: &Plan, corpus: &Corpus, n_tokens: u32) -> Result<DepthMeasurement> {
    let logs = ctx.cfg.data_dir.join("logs");
    std::fs::create_dir_all(&logs)?;
    let w = Worker::spawn(&plan.engine, Some(&logs.join(format!("plan-{}.log", plan.id)))).await?;
    let result = async {
        tokio::time::timeout(Duration::from_secs(1800), w.load(plan)).await.context("load timed out")??;
        let warm = chat_prompt(&w, corpus, Role::Calibration, 999, 32).await?;
        crate::run::run_stream(&w, "warmup", warm, 8, bench_sampling()).await?;
        let p = chat_prompt(&w, corpus, Role::Calibration, 500, n_tokens as usize).await?;
        let r = crate::run::run_stream(&w, "depth", p, ctx.req.decode_tokens, bench_sampling()).await?;
        if let Some(e) = r.error {
            bail!("{e}");
        }
        Ok(DepthMeasurement {
            prompt_tokens: r.n_prompt as u32,
            ttft_ms: r.ttft_s * 1e3,
            prefill_tps: r.n_prompt as f64 / r.ttft_s.max(1e-9),
            decode_tps: r.decode_rate().context("too few tokens to time decoding")?,
        })
    }
    .await;
    w.shutdown().await;
    result
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
            prompts.push(chat_prompt(&w, corpus, role, offset + i * 7, ctx.req.prompt_tokens as usize).await?);
        }
        let warm = chat_prompt(&w, corpus, role, 999, 32).await?;
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
        let s = summarize(&results, wall).context("no completed streams")?;
        let agent = if role == Role::Validation {
            let mut prompts = vec![];
            for i in 0..n {
                prompts.push(agent_prompt(&w, i, n).await?);
            }
            let t0 = Instant::now();
            let results = run_all(&w, "agent", prompts, ctx.req.decode_tokens, concurrency, bench_sampling()).await?;
            summarize(&results, t0.elapsed().as_secs_f64()).map(|a| a.decode_tps)
        } else {
            None
        };
        anyhow::Ok((s, agent))
    }
    .await;
    let peaks = sampler.finish();
    w.shutdown().await;
    let (s, agent_decode_tps) = result?;
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
            agent_decode_tps,
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
    let mut cpu = format!("token embeddings; {} threads (fewest per physical core near the best measured bandwidth)", plan.runtime.n_threads);
    if cpu_blocks > 0 {
        cpu = format!("blocks 0-{} ({}); {cpu}", cpu_blocks - 1, fmt_bytes(lay.blocks[..cpu_blocks].iter().map(|b| b.dense_bytes() + b.expert_bytes()).sum()));
    }
    if chosen.cache.is_some() {
        cpu.push_str(&format!("; the experts of {} blocks stay in host memory behind the GPU cache", chosen.placement.overrides.len()));
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
    // Without drafting on either side; the speculation decision reports what drafting adds.
    let cache_rate = rate_of(&|c: &Candidate| c.cache.is_some() && c.speculation.is_none());
    let tensor_rate = rate_of(&|c: &Candidate| c.cache.is_none() && c.speculation.is_none());
    if lay.n_expert > 0 {
        let cache = chosen.cache.as_ref().or_else(|| measured.iter().find_map(|(c, _)| c.cache.as_ref()));
        plan.decisions.push(Decision {
            resource: "routed experts".into(),
            used: chosen.cache.is_some(),
            reason: match (cache, cache_rate, tensor_rate) {
                (Some(s), Some(r), Some(t)) => format!(
                    "per-expert GPU cache: {} experts initially serve {:.1}% of calibration routing from GPU memory vs {:.1}% with whole tensors, then adapt while decoding; measured {r:.2} vs {t:.2} tok/s{}",
                    s.hot_experts,
                    s.gpu_served * 100.0,
                    s.gpu_served_by_tensors * 100.0,
                    if chosen.cache.is_some() { "" } else { "; not faster" }
                ),
                (Some(_), _, _) => "per-expert cache candidate failed to run".into(),
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
            DeviceMemory { device: "Meta()".into(), model: 300, context: 30, ..Default::default() },
            DeviceMemory { device: "CUDA0".into(), compute: 10, ..Default::default() },
        ];
        let a = apportion_meta(m, &p);
        assert_eq!(total(&a, "CUDA0"), 200 + 20 + 10);
        assert_eq!(total(&a, "CUDA1"), 100 + 10);
    }

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn singletons(n: usize) -> Vec<Vec<String>> {
        (0..n).map(|i| vec![format!("G{i}")]).collect()
    }

    #[test]
    fn orders_cover_every_subset_permutation() {
        let o = gpu_orders(&singletons(2));
        assert_eq!(o.len(), 5, "{o:?}");
        assert!(o.contains(&vec![]) && o.contains(&names(&["G1", "G0"])));
        assert_eq!(gpu_orders(&singletons(4)).len(), 65);
    }

    #[test]
    fn interchangeable_gpus_are_taken_in_class_order() {
        let o = gpu_orders(&[names(&["A1", "A2", "A3"])]);
        assert_eq!(o, vec![vec![], names(&["A1"]), names(&["A1", "A2"]), names(&["A1", "A2", "A3"])]);
        let eight: Vec<String> = (0..8).map(|i| format!("CUDA{i}")).collect();
        assert_eq!(gpu_orders(&[eight]).len(), 9);
    }

    #[test]
    fn mixed_classes_search_mirror_images_once() {
        let o = gpu_orders(&[names(&["A1", "A2"]), names(&["B"])]);
        assert_eq!(o.len(), 9, "{o:?}");
        assert!(o.contains(&names(&["A1", "B", "A2"])) && o.contains(&names(&["B", "A1", "A2"])));
        let pos = |x: &Vec<String>, g: &str| x.iter().position(|n| n == g);
        assert!(o.iter().all(|x| pos(x, "A2").is_none_or(|p2| pos(x, "A1").is_some_and(|p1| p1 < p2))), "{o:?}");
    }

    #[test]
    fn many_distinct_gpus_are_capped() {
        let classes = singletons(5);
        let o = gpu_orders(&classes);
        assert_eq!(o.len(), 1 + 5 + 5, "{o:?}");
        assert!(o.contains(&vec![]));
        assert!(classes.iter().all(|c| o.contains(c)));
        let full: Vec<&Vec<String>> = o.iter().filter(|x| x.len() == 5).collect();
        assert_eq!(full.len(), 5);
        assert_eq!(*full[2], names(&["G2", "G0", "G1", "G3", "G4"]));
        let eight = gpu_orders(&singletons(8));
        assert_eq!(eight.len(), 1 + 8 + 8);
    }

    #[test]
    fn capped_orders_keep_classes_together() {
        let classes: Vec<Vec<String>> = ["A", "B", "C", "D"].iter().map(|c| vec![format!("{c}1"), format!("{c}2")]).collect();
        let o = gpu_orders(&classes);
        assert_eq!(o.len(), 1 + 4 + 4, "{o:?}");
        assert!(o.contains(&names(&["B1"])) && !o.contains(&names(&["B2"])));
        assert!(o.contains(&names(&["B1", "B2", "A1", "A2", "C1", "C2", "D1", "D2"])));
    }

    const GIB: u64 = 1 << 30;

    fn traits(name: &str, model: &str, capacity: u64, upload_s: f64) -> GpuTraits {
        GpuTraits { name: name.into(), model: (model.into(), 24 * GIB), capacity: vec![capacity, capacity + GIB], upload_s }
    }

    #[test]
    fn identical_gpus_form_one_class_with_most_room_first() {
        // 8 MiB more room rounds to one more search quantum; 8% slower uploads are probe noise.
        let c = gpu_classes(vec![traits("CUDA0", "RTX 3090", 20 * GIB, 5.0e-3), traits("CUDA1", "RTX 3090", 20 * GIB + (8 << 20), 5.4e-3)]);
        assert_eq!(c, vec![names(&["CUDA1", "CUDA0"])]);
    }

    #[test]
    fn gpus_differing_in_room_model_or_link_stay_apart() {
        let classes = |other: GpuTraits| gpu_classes(vec![traits("CUDA0", "RTX 3090", 20 * GIB, 5.0e-3), other]);
        // A desktop display holding 40 MiB, an x4 slot, a different card.
        assert_eq!(classes(traits("CUDA1", "RTX 3090", 20 * GIB - (40 << 20), 5.0e-3)), vec![names(&["CUDA0"]), names(&["CUDA1"])]);
        assert_eq!(classes(traits("CUDA1", "RTX 3090", 20 * GIB, 15.0e-3)).len(), 2);
        assert_eq!(classes(traits("CUDA1", "RTX 4090", 20 * GIB, 5.0e-3)).len(), 2);
    }

    #[test]
    fn classes_hold_only_mutually_interchangeable_gpus() {
        // Each neighbour is within a quantum of the next, but the ends are two apart.
        let q = 16u64 << 20;
        let c = gpu_classes(vec![
            traits("CUDA0", "RTX 3090", 20 * GIB, 5.0e-3),
            traits("CUDA1", "RTX 3090", 20 * GIB + q, 5.0e-3),
            traits("CUDA2", "RTX 3090", 20 * GIB + 2 * q, 5.0e-3),
        ]);
        assert_eq!(c, vec![names(&["CUDA2", "CUDA1"]), names(&["CUDA0"])]);
    }
}
