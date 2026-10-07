//! One set of llama.cpp load parameters, rendered both for the flux-native bridge (JSON)
//! and for the pinned llama-server binary (CLI flags), so every engine runs the same placement.

use crate::plan::{Speculation, SplitMode, SplitSpec, TensorOverride};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendParams {
    pub model: PathBuf,
    /// Offload devices in order; empty = CPU only.
    pub devices: Vec<String>,
    pub n_gpu_layers: i32,
    pub tensor_split: Vec<f32>,
    #[serde(default)]
    pub split_mode: SplitMode,
    pub overrides: Vec<TensorOverride>,
    pub mmap: bool,
    pub mlock: bool,
    pub n_ctx_seq: u32,
    pub n_seq: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_threads: u32,
    pub n_threads_batch: u32,
    pub flash_attn: bool,
    pub type_k: String,
    pub type_v: String,
    pub op_offload: bool,
    pub kv_unified: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_paging: Option<crate::plan::KvPaging>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub qsa_pooled: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub qsa_blocks: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub qsa_indexed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculation: Option<Speculation>,
    /// Layers whose experts a GPU cache serves (native engine), with their initial cached experts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_cache: Option<SplitSpec>,
    /// Keeps the cache's residency fixed, for reproducible runs (rollback certification).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub expert_cache_frozen: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_cache_policy: Option<crate::plan::CachePolicy>,
}

impl BackendParams {
    /// Placement and runtime flags shared by every llama.cpp tool (server, perplexity, bench);
    /// `--fit off` keeps the tool from re-fitting them.
    pub fn tool_args(&self) -> Vec<String> {
        let mut a: Vec<String> = vec![
            "--model".into(),
            self.model.display().to_string(),
            "--device".into(),
            if self.devices.is_empty() { "none".into() } else { self.devices.join(",") },
            "--split-mode".into(),
            self.split_mode.as_str().into(),
            "--n-gpu-layers".into(),
            self.n_gpu_layers.to_string(),
        ];
        if !self.tensor_split.is_empty() {
            a.push("--tensor-split".into());
            a.push(self.tensor_split.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
        }
        if !self.overrides.is_empty() {
            a.push("--override-tensor".into());
            a.push(self.overrides.iter().map(|o| format!("{}={}", o.pattern, o.device)).collect::<Vec<_>>().join(","));
        }
        let load = match (self.mmap, self.mlock) {
            (true, true) => "mmap+mlock",
            (true, false) => "mmap",
            (false, true) => "mlock",
            (false, false) => "none",
        };
        a.extend(
            [
                "--load-mode",
                load,
                "--batch-size",
                &self.n_batch.to_string(),
                "--ubatch-size",
                &self.n_ubatch.to_string(),
                "--threads",
                &self.n_threads.to_string(),
                "--threads-batch",
                &self.n_threads_batch.to_string(),
                "--flash-attn",
                if self.flash_attn { "on" } else { "off" },
                "--cache-type-k",
                &self.type_k,
                "--cache-type-v",
                &self.type_v,
                "--fit",
                "off",
            ]
            .map(String::from),
        );
        if !self.op_offload {
            a.push("--no-op-offload".into());
        }
        a
    }

    /// llama-server flags reproducing these parameters.
    pub fn llama_server_args(&self) -> Vec<String> {
        let mut a = self.tool_args();
        a.extend(["--ctx-size".into(), (self.n_ctx_seq * self.n_seq).to_string(), "--parallel".into(), self.n_seq.to_string()]);
        a.push(if self.kv_unified { "--kv-unified" } else { "--no-kv-unified" }.into());
        if let Some(sp) = &self.speculation {
            a.extend(["--spec-type".into(), sp.kind.clone(), "--spec-draft-n-max".into(), sp.n_max.to_string(), "--spec-draft-p-min".into(), "0".into()]);
            if let Some(d) = &sp.draft_model {
                a.extend(["--spec-draft-model".into(), d.display().to_string()]);
            }
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_args_carry_placement() {
        let p = BackendParams {
            model: "/m.gguf".into(),
            devices: vec!["CUDA0".into(), "CUDA1".into()],
            n_gpu_layers: 25,
            tensor_split: vec![10.0, 15.0],
            split_mode: SplitMode::Layer,
            overrides: vec![TensorOverride { pattern: r"^blk\.3\.ffn_up_exps\.weight$".into(), device: "CPU".into() }],
            mmap: true,
            mlock: false,
            n_ctx_seq: 4096,
            n_seq: 2,
            n_batch: 2048,
            n_ubatch: 512,
            n_threads: 12,
            n_threads_batch: 24,
            flash_attn: true,
            type_k: "f16".into(),
            type_v: "f16".into(),
            op_offload: true,
            kv_unified: false,
            kv_paging: None,
            qsa_pooled: false,
            qsa_blocks: false,
            qsa_indexed: false,
            speculation: Some(Speculation { kind: "draft-mtp".into(), draft_model: None, n_max: 3, draft_vocab: None }),
            expert_cache: None,
            expert_cache_frozen: false,
            expert_cache_policy: None,
        };
        let a = p.llama_server_args().join(" ");
        assert!(a.ends_with("--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0"));
        assert!(a.contains("--device CUDA0,CUDA1"));
        assert!(a.contains("--tensor-split 10,15"));
        assert!(a.contains(r"--override-tensor ^blk\.3\.ffn_up_exps\.weight$=CPU"));
        assert!(a.contains("--ctx-size 8192 --parallel 2"));
        assert!(a.contains("--fit off"));
    }
}
