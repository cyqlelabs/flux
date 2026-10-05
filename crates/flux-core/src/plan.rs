//! The immutable execution plan: what runs where, why, and how it measured.

use crate::stats::Summary;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const PLAN_SCHEMA: u32 = 1;

/// Everything a saved plan's validity depends on. A change to any field requires reprofiling.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProfileKey {
    pub model_identity: String,
    pub topology: String,
    pub backend_revision: String,
    pub backend_build: String,
    pub driver: String,
    /// Context per sequence rounded up to a power of two.
    pub ctx_bucket: u32,
    pub concurrency: u32,
    /// Complete workload, runtime policy, and planner revision fingerprint.
    #[serde(default)]
    pub policy: String,
}

impl ProfileKey {
    pub fn digest(&self) -> String {
        let canonical = serde_json::to_string(self).expect("profile key serializes");
        crate::fsutil::sha256_hex(canonical.as_bytes())[..20].to_string()
    }
}

pub fn ctx_bucket(n_ctx: u32) -> u32 {
    n_ctx.max(1).next_power_of_two()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Minimize single-stream decode latency, R = (N-1)/(t_last-t_first).
    Interactive,
    /// Maximize aggregate emitted tokens/s subject to a per-request latency limit.
    Serving { max_p95_token_ms: u32 },
}

/// What the plan must be able to hold at once; admission reserves this worst case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Workload {
    pub n_ctx_seq: u32,
    pub concurrency: u32,
    pub objective: Objective,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    /// flux-worker driving the pinned llama.cpp through flux-native.
    Native,
    /// The pinned llama-server binary with its own scheduler.
    LlamaServer,
    /// Another OpenAI-compatible engine from the registry, by name.
    External(String),
}

impl std::fmt::Display for EngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineKind::Native => f.write_str("native"),
            EngineKind::LlamaServer => f.write_str("llama-server"),
            EngineKind::External(n) => f.write_str(n),
        }
    }
}

/// How the backend divides work between GPUs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitMode {
    /// Contiguous blocks per device (pipeline order, no overlap for a single stream).
    #[default]
    Layer,
    /// Rows of each weight matrix split across devices, results gathered per product.
    Row,
    /// Tensor parallelism through the backend's meta device (architecture-dependent).
    Tensor,
}

impl SplitMode {
    pub fn as_str(self) -> &'static str {
        match self {
            SplitMode::Layer => "layer",
            SplitMode::Row => "row",
            SplitMode::Tensor => "tensor",
        }
    }
}

/// Pins one tensor-name regex to a device, e.g. routed experts of layer 7 to `CPU`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorOverride {
    pub pattern: String,
    pub device: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    /// Offload devices in pipeline order (ggml names, e.g. `CUDA0`); empty means CPU only.
    pub devices: Vec<String>,
    /// Device of every block including appended NextN blocks (`n_layer_all` entries),
    /// `CPU` or one of `devices`; contiguous by construction.
    pub layer_device: Vec<String>,
    pub output_device: String,
    pub overrides: Vec<TensorOverride>,
    /// Backend parameters that reproduce `layer_device` exactly.
    pub n_gpu_layers: i32,
    pub tensor_split: Vec<f32>,
    #[serde(default)]
    pub split_mode: SplitMode,
}

impl Placement {
    pub fn uses_device(&self, dev: &str) -> bool {
        self.devices.iter().any(|d| d == dev)
            || self.layer_device.iter().any(|d| d == dev)
            || self.output_device == dev
            || self.overrides.iter().any(|o| o.device == dev)
    }

    /// Short human summary such as `CPU 0-3 | CUDA0 4-27 | out CUDA0 | 6 overrides`.
    pub fn describe(&self) -> String {
        if self.split_mode != SplitMode::Layer {
            let shares: Vec<String> = self.devices.iter().zip(&self.tensor_split).map(|(d, w)| format!("{d}:{w}")).collect();
            return format!("{}-split {}", self.split_mode.as_str(), shares.join(" "));
        }
        let mut spans: Vec<(String, usize, usize)> = vec![];
        for (i, d) in self.layer_device.iter().enumerate() {
            match spans.last_mut() {
                Some((dev, _, end)) if dev == d => *end = i,
                _ => spans.push((d.clone(), i, i)),
            }
        }
        let mut s: Vec<String> = spans.iter().map(|(d, a, b)| format!("{d} {a}-{b}")).collect();
        s.push(format!("out {}", self.output_device));
        if !self.overrides.is_empty() {
            s.push(format!("{} overrides", self.overrides.len()));
        }
        s.join(" | ")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Speculation {
    /// `draft-mtp` (model's own next-token heads) or `draft` (separate model).
    pub kind: String,
    pub draft_model: Option<PathBuf>,
    pub n_max: u32,
    /// The next-token heads draft only from token ids below this, which cuts the cost of their output head.
    /// None drafts from the whole vocabulary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_vocab: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeParams {
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub flash_attn: bool,
    pub type_k: String,
    pub type_v: String,
    pub mmap: bool,
    pub mlock: bool,
    pub op_offload: bool,
    pub speculation: Option<Speculation>,
}

/// Bytes on one device split by purpose.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemSplit {
    pub weights: u64,
    pub state: u64,
    pub compute: u64,
}

impl MemSplit {
    pub fn total(&self) -> u64 {
        self.weights + self.state + self.compute
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceBudget {
    pub device: String,
    pub capacity: u64,
    pub free_at_plan: u64,
    /// Held back for the driver, other processes and fragmentation.
    pub reserve: u64,
    pub predicted: MemSplit,
    /// From the backend's own allocation dry run; the authority when present.
    pub measured: Option<MemSplit>,
}

impl DeviceBudget {
    pub fn usable(&self) -> u64 {
        self.free_at_plan.saturating_sub(self.reserve)
    }

    pub fn required(&self) -> u64 {
        self.measured.unwrap_or(self.predicted).total()
    }
}

/// Host RAM accounting without double-counting mapped pages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostBudget {
    pub capacity: u64,
    pub available_at_plan: u64,
    pub os_reserve: u64,
    /// Weights executed from host memory (resident CPU layers and experts).
    pub resident_weights: u64,
    /// Host copies kept for weights that also live on a GPU (non-mmap loads, mirrors).
    pub mirrored_weights: u64,
    pub pinned_buffers: u64,
    pub state_and_scratch: u64,
}

impl HostBudget {
    pub fn required(&self) -> u64 {
        self.os_reserve + self.resident_weights + self.mirrored_weights + self.pinned_buffers + self.state_and_scratch
    }

    pub fn fits(&self) -> bool {
        self.required() <= self.capacity && self.required() - self.os_reserve <= self.available_at_plan
    }
}

/// Why a resource is or is not part of the plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    pub resource: String,
    pub used: bool,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub prompts: usize,
    pub ttft_ms: Summary,
    pub decode_tps: Summary,
    pub token_ms: Summary,
    pub peak_device_bytes: Vec<(String, u64)>,
    pub peak_host_rss: u64,
    /// Decode rate on the built-in coding-agent conversations (validation runs): agent clients route experts
    /// unlike the prose prompts above, so serving judges requests that carry tools against this rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_decode_tps: Option<Summary>,
}

/// One long request on a candidate: a prompt a quarter of the planned context, then decoding at that depth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepthMeasurement {
    pub prompt_tokens: u32,
    pub ttft_ms: f64,
    /// Prompt tokens per second of time to first token.
    pub prefill_tps: f64,
    pub decode_tps: f64,
}

impl DepthMeasurement {
    /// Expected decode rate at `depth` cached tokens, from the short-prompt rate and this one: time per token
    /// grows linearly with depth (attention reads every cached position), clamped at twice the measured depth.
    pub fn decode_tps_at(&self, short_tps: f64, depth: u32) -> f64 {
        let (t0, t1) = (1.0 / short_tps.max(1e-9), 1.0 / self.decode_tps.max(1e-9));
        let x = (depth as f64 / self.prompt_tokens.max(1) as f64).min(2.0);
        1.0 / (t0 + (t1 - t0) * x).max(1e-9)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateResult {
    pub label: String,
    pub predicted_token_ms: f64,
    pub calibration: Option<Measurement>,
    pub validation: Option<Measurement>,
    pub failure: Option<String>,
    /// What the candidate ran, so it can be served or measured again: its expert cache, placement,
    /// micro-batch and drafting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_cache: Option<SplitSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<Placement>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ubatch: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculation: Option<Speculation>,
    /// The long-prompt measurement, for candidates that reached the final comparison.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<DepthMeasurement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationRecord {
    pub tuning_seconds: f64,
    pub tuning_budget_seconds: f64,
    pub candidates: Vec<CandidateResult>,
    pub chosen: String,
    pub reason: String,
}

/// Whether the plan changes model semantics relative to the supplied artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QualityProfile {
    /// Same artifact, same context semantics; only placement and scheduling differ.
    Exact,
    /// Opt-in profile with a recorded quality measurement against the exact reference.
    Changed { changes: Vec<String>, ppl_ratio: Option<f64>, top1_agreement: Option<f64> },
}

/// Per layer: expert ids in their new order (hot first) and how many of them are hot.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SplitSpec {
    #[serde(with = "layer_list")]
    pub layers: BTreeMap<u32, (Vec<u32>, u32)>,
    /// Experts other GPUs serve for layers placed elsewhere, beyond those layers' own hot experts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<Tier>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tier {
    pub device: String,
    /// A list of `[layer, [experts]]`.
    pub layers: Vec<(u32, Vec<u32>)>,
}

/// A list of `[layer, [order], hot]`: integer map keys do not survive the protocol's tagged enums.
mod layer_list {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    type Layers = BTreeMap<u32, (Vec<u32>, u32)>;

    pub fn serialize<S: Serializer>(m: &Layers, s: S) -> Result<S::Ok, S::Error> {
        m.iter().map(|(l, (order, hot))| (l, order, hot)).collect::<Vec<_>>().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Layers, D::Error> {
        Ok(Vec::<(u32, Vec<u32>, u32)>::deserialize(d)?.into_iter().map(|(l, order, hot)| (l, (order, hot))).collect())
    }
}

/// Per-expert residency: the expert tensors of `spec`'s layers stay in host memory while a GPU cache next
/// to each layer holds its most-routed experts. `spec` gives each layer's initial cached experts (the first
/// `hot` of its order); the backend adapts them to the traffic while decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertCache {
    pub spec: SplitSpec,
    pub hot_experts: u32,
    /// Share of routed selections on calibration prompts served from GPU memory, with the initial cache and
    /// with whole expert tensors in the same memory.
    pub gpu_served: f64,
    pub gpu_served_by_tensors: f64,
    /// Keeps residency fixed: certification and quality gates need reproducible runs.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub frozen: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub schema: u32,
    pub id: String,
    pub created: chrono::DateTime<chrono::Utc>,
    pub key: ProfileKey,
    pub model_files: Vec<PathBuf>,
    pub architecture: String,
    pub engine: EngineKind,
    pub workload: Workload,
    pub placement: Placement,
    pub runtime: RuntimeParams,
    pub budgets: Vec<DeviceBudget>,
    pub host: HostBudget,
    pub quality: QualityProfile,
    pub decisions: Vec<Decision>,
    pub validation: Option<ValidationRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_cache: Option<ExpertCache>,
}

impl Plan {
    pub fn backend_params(&self) -> crate::backend::BackendParams {
        let (p, r) = (&self.placement, &self.runtime);
        crate::backend::BackendParams {
            model: self.model_files[0].clone(),
            devices: p.devices.clone(),
            n_gpu_layers: p.n_gpu_layers,
            tensor_split: p.tensor_split.clone(),
            split_mode: p.split_mode,
            overrides: p.overrides.clone(),
            mmap: r.mmap,
            mlock: r.mlock,
            n_ctx_seq: self.workload.n_ctx_seq,
            n_seq: self.workload.concurrency,
            n_batch: r.n_batch,
            n_ubatch: r.n_ubatch,
            n_threads: r.n_threads,
            n_threads_batch: r.n_threads_batch,
            flash_attn: r.flash_attn,
            type_k: r.type_k.clone(),
            type_v: r.type_v.clone(),
            op_offload: r.op_offload,
            kv_unified: false,
            speculation: r.speculation.clone(),
            expert_cache: self.expert_cache.as_ref().map(|c| c.spec.clone()),
            expert_cache_frozen: self.expert_cache.as_ref().is_some_and(|c| c.frozen),
        }
    }

    /// Content id over everything except `id` and `created`.
    pub fn compute_id(&self) -> String {
        let mut v = serde_json::to_value(self).expect("plan serializes");
        if let Some(o) = v.as_object_mut() {
            o.remove("id");
            o.remove("created");
        }
        crate::fsutil::sha256_hex(v.to_string().as_bytes())[..16].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_spans() {
        let p = Placement {
            devices: vec!["CUDA0".into()],
            layer_device: ["CPU", "CPU", "CUDA0", "CUDA0", "CUDA0"].iter().map(|s| s.to_string()).collect(),
            output_device: "CUDA0".into(),
            overrides: vec![],
            n_gpu_layers: 4,
            tensor_split: vec![],
            split_mode: SplitMode::Layer,
        };
        assert_eq!(p.describe(), "CPU 0-1 | CUDA0 2-4 | out CUDA0");
    }

    #[test]
    fn host_budget_proposal_example_b() {
        let gib = 1u64 << 30;
        let mut h = HostBudget {
            capacity: 64 * gib,
            available_at_plan: 60 * gib,
            os_reserve: 8 * gib,
            resident_weights: 43 * gib,
            mirrored_weights: 0,
            pinned_buffers: 3 * gib,
            state_and_scratch: 4 * gib,
        };
        assert_eq!(h.required(), 58 * gib);
        assert!(h.fits());
        h.mirrored_weights = 12 * gib;
        assert_eq!(h.required(), 70 * gib);
        assert!(!h.fits());
    }
}
