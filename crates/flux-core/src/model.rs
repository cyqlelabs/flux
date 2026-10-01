//! What Flux knows about a model artifact before executing it.

use crate::ggml_type::GgmlType;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const MANIFEST_SCHEMA: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Gguf,
    Safetensors,
    Exl3,
    Gptq,
    Awq,
    Fp8,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    pub path: PathBuf,
    pub size: u64,
    pub mtime: i64,
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HfSource {
    pub repo: Option<String>,
    pub revision: Option<String>,
    pub commit: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: GgmlType,
    pub dims: Vec<u64>,
    pub bytes: u64,
    pub shard: u32,
    pub offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    TokenEmbedding,
    Output,
    OutputNorm,
    Attention,
    Norm,
    FfnDense,
    FfnRouter,
    FfnSharedExpert,
    FfnRoutedExpert,
    Recurrent,
    Other,
}

impl TensorInfo {
    /// Block index for `blk.N.*` tensors.
    pub fn layer(&self) -> Option<u32> {
        self.name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
    }

    /// Name without the `blk.N.` prefix, e.g. `ffn_up_exps.weight`.
    pub fn local_name(&self) -> &str {
        match self.layer() {
            Some(_) => self.name.splitn(3, '.').nth(2).unwrap_or(&self.name),
            None => &self.name,
        }
    }

    pub fn role(&self) -> TensorRole {
        let local = self.local_name();
        let stem = local.split('.').next().unwrap_or(local);
        if self.layer().is_none() {
            return match stem {
                "token_embd" | "per_layer_token_embd" => TensorRole::TokenEmbedding,
                "output" => TensorRole::Output,
                "output_norm" => TensorRole::OutputNorm,
                _ => TensorRole::Other,
            };
        }
        if stem.ends_with("_exps") || stem.ends_with("_exps_b") {
            TensorRole::FfnRoutedExpert
        } else if stem.ends_with("_shexp") || stem == "ffn_gate_inp_shexp" {
            TensorRole::FfnSharedExpert
        } else if stem == "ffn_gate_inp" || stem == "exp_probs_b" {
            TensorRole::FfnRouter
        } else if stem.ends_with("_norm") || stem.contains("norm") {
            TensorRole::Norm
        } else if stem.starts_with("attn") || stem.starts_with("indexer") {
            TensorRole::Attention
        } else if stem.starts_with("ffn") {
            TensorRole::FfnDense
        } else if stem.starts_with("ssm") || stem.starts_with("time_mix") || stem.starts_with("channel_mix") || stem.starts_with("shortconv") {
            TensorRole::Recurrent
        } else {
            TensorRole::Other
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeFacts {
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_expert_shared: u32,
    pub n_dense_lead: u32,
}

/// Execution-relevant hyperparameters read from the checkpoint, not from a family name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchFacts {
    pub architecture: String,
    /// Blocks in the main stack, excluding appended next-token-prediction blocks.
    pub n_layer: u32,
    pub n_layer_nextn: u32,
    pub n_ctx_train: u32,
    pub n_embd: u32,
    pub n_vocab: u32,
    pub n_head: Vec<u32>,
    /// Per layer; 0 means the layer keeps no attention cache.
    pub n_head_kv: Vec<u32>,
    pub key_length: u32,
    pub value_length: u32,
    /// Compressed latent attention: only the latent+positional key row is cached; values are views of it.
    pub mla: bool,
    pub sliding_window: Option<u32>,
    pub recurrent: Vec<bool>,
    /// f32 elements of recurrent state per recurrent layer and sequence (conv + ssm state).
    pub recurrent_state_elems: u64,
    pub moe: Option<MoeFacts>,
    /// Features present in the checkpoint that the analytic state model does not cover;
    /// the backend's measured allocation decides for these.
    pub unmodeled: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateBytes {
    pub kv: u64,
    pub recurrent: u64,
}

impl ArchFacts {
    /// Conventional cache payload per layer for `n_seq` sequences of `n_ctx_seq` cells:
    /// KV = B * S * sum_l [Hkv_l * (Dk*Pk + Dv*Pv)], plus fixed-size recurrent state per sequence.
    /// Excludes block rounding, scales outside the encoding, and allocator metadata.
    pub fn state_bytes_per_layer(&self, n_seq: u64, n_ctx_seq: u64, type_k: GgmlType, type_v: GgmlType) -> Vec<StateBytes> {
        (0..self.n_layer as usize)
            .map(|il| {
                if self.recurrent.get(il).copied().unwrap_or(false) {
                    return StateBytes { kv: 0, recurrent: n_seq * self.recurrent_state_elems * 4 };
                }
                let hkv = self.n_head_kv.get(il).copied().unwrap_or(0) as u64;
                if hkv == 0 {
                    return StateBytes { kv: 0, recurrent: 0 };
                }
                let cells = n_seq * n_ctx_seq;
                let k = cells as f64 * encoded_bytes(type_k, hkv * self.key_length as u64);
                let v = if self.mla { 0.0 } else { cells as f64 * encoded_bytes(type_v, hkv * self.value_length as u64) };
                StateBytes { kv: (k + v).ceil() as u64, recurrent: 0 }
            })
            .collect()
    }

    pub fn state_bytes_total(&self, n_seq: u64, n_ctx_seq: u64, type_k: GgmlType, type_v: GgmlType) -> u64 {
        self.state_bytes_per_layer(n_seq, n_ctx_seq, type_k, type_v).iter().map(|s| s.kv + s.recurrent).sum()
    }
}

fn encoded_bytes(t: GgmlType, n: u64) -> f64 {
    t.row_bytes(n).map(|b| b as f64).unwrap_or_else(|| n as f64 * t.bytes_per_element().unwrap_or(2.0))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerIdentity {
    pub model: String,
    pub pre: Option<String>,
    pub n_tokens: u32,
    /// sha256 over the token list, types, merges and special-token ids.
    pub sha256: String,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub eot: Option<u32>,
    pub add_bos: Option<bool>,
}

/// Whether Flux may execute the artifact, and with which engines.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Compatibility {
    Executable { engines: Vec<String> },
    InspectOnly { missing: Vec<String>, alternatives: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub schema: u32,
    pub format: ModelFormat,
    pub files: Vec<ModelFile>,
    pub source: Option<HfSource>,
    pub name: Option<String>,
    pub facts: Option<ArchFacts>,
    pub tensors: Vec<TensorInfo>,
    /// Bytes per tensor encoding.
    pub encodings: BTreeMap<String, u64>,
    pub tokenizer: Option<TokenizerIdentity>,
    pub chat_template_sha256: Option<String>,
    pub compatibility: Compatibility,
    /// sha256 over the ordered file hashes; stable across paths and mtimes.
    pub identity: Option<String>,
}

impl ModelManifest {
    pub fn total_tensor_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes).sum()
    }

    pub fn paths(&self) -> Vec<PathBuf> {
        self.files.iter().map(|f| f.path.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen25_7b() -> ArchFacts {
        ArchFacts {
            architecture: "qwen2".into(),
            n_layer: 28,
            n_layer_nextn: 0,
            n_ctx_train: 32768,
            n_embd: 3584,
            n_vocab: 152064,
            n_head: vec![28; 28],
            n_head_kv: vec![4; 28],
            key_length: 128,
            value_length: 128,
            mla: false,
            sliding_window: None,
            recurrent: vec![false; 28],
            recurrent_state_elems: 0,
            moe: None,
            unmodeled: vec![],
        }
    }

    #[test]
    fn proposal_kv_examples() {
        let f = qwen25_7b();
        // 8K tokens, batch 1, FP16: 469,762,048 bytes = 448 MiB.
        assert_eq!(f.state_bytes_total(1, 8192, GgmlType::F16, GgmlType::F16), 469_762_048);
        // 32K tokens: 1.75 GiB per sequence; four sequences need 7 GiB before block overhead.
        assert_eq!(f.state_bytes_total(1, 32768, GgmlType::F16, GgmlType::F16), 7 * (1 << 30) / 4);
        assert_eq!(f.state_bytes_total(4, 32768, GgmlType::F16, GgmlType::F16), 7 * (1 << 30));
    }

    #[test]
    fn q8_cache_is_block_encoded() {
        let f = qwen25_7b();
        let q8 = f.state_bytes_total(1, 8192, GgmlType::Q8_0, GgmlType::Q8_0);
        assert_eq!(q8, 469_762_048 / 64 * 34);
    }

    #[test]
    fn roles_and_layers() {
        let t = |name: &str| TensorInfo { name: name.into(), ggml_type: GgmlType::F16, dims: vec![], bytes: 0, shard: 0, offset: 0 };
        assert_eq!(t("blk.3.ffn_up_exps.weight").role(), TensorRole::FfnRoutedExpert);
        assert_eq!(t("blk.3.ffn_up_exps.weight").layer(), Some(3));
        assert_eq!(t("blk.3.ffn_up_exps.weight").local_name(), "ffn_up_exps.weight");
        assert_eq!(t("blk.3.ffn_down_shexp.weight").role(), TensorRole::FfnSharedExpert);
        assert_eq!(t("blk.3.ffn_gate_inp.weight").role(), TensorRole::FfnRouter);
        assert_eq!(t("blk.3.attn_kv_a_mqa.weight").role(), TensorRole::Attention);
        assert_eq!(t("blk.3.attn_norm.weight").role(), TensorRole::Norm);
        assert_eq!(t("blk.3.ffn_down.weight").role(), TensorRole::FfnDense);
        assert_eq!(t("token_embd.weight").role(), TensorRole::TokenEmbedding);
        assert_eq!(t("output.weight").role(), TensorRole::Output);
    }
}
