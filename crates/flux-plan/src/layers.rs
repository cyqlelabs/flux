//! Per-block accounting from the manifest's exact tensor sizes.

use flux_core::ggml_type::GgmlType;
use flux_core::model::{ArchFacts, ModelManifest, TensorInfo, TensorRole};

/// One routed-expert weight of one block (e.g. `blk.7.ffn_up_exps.weight`): the unit of expert residency.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertTensor {
    pub name: String,
    pub ggml_type: GgmlType,
    /// Bytes of all experts in the tensor.
    pub bytes: u64,
    /// Rows x cols of one expert's matrix.
    pub k: u64,
    pub n: u64,
}

/// One weight read every step, with the multiply-accumulates it costs per token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Weight {
    pub ggml_type: GgmlType,
    pub bytes: u64,
    pub macs: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Block {
    pub index: u32,
    /// Weights that always live on the block's device: attention, norms, router, shared experts, dense FFN.
    pub dense: Vec<Weight>,
    pub experts: Vec<ExpertTensor>,
    /// Cache bytes for the planned sequences (KV or recurrent state).
    pub state_bytes: u64,
    /// Attention multiply-accumulates per token per cached position (0 for layers without KV).
    pub attn_macs_per_pos: u64,
}

impl Block {
    pub fn dense_bytes(&self) -> u64 {
        self.dense.iter().map(|w| w.bytes).sum()
    }

    pub fn expert_bytes(&self) -> u64 {
        self.experts.iter().map(|e| e.bytes).sum()
    }
}

/// Weights outside the blocks: output head (placed with the last device) and input embeddings (host).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Head {
    pub output: Vec<Weight>,
    /// Resident host bytes of the input embedding tables (see `layout`).
    pub input_bytes: u64,
}

impl Head {
    pub fn output_bytes(&self) -> u64 {
        self.output.iter().map(|w| w.bytes).sum()
    }
}

pub struct Layout {
    pub blocks: Vec<Block>,
    pub head: Head,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_embd: u32,
    pub n_vocab: u32,
}

fn weight(t: &TensorInfo) -> Weight {
    let macs = if t.dims.len() >= 2 { t.dims[0] * t.dims[1] } else { 0 };
    Weight { ggml_type: t.ggml_type, bytes: t.bytes, macs }
}

/// Host bytes an input embedding table keeps resident when it is memory-mapped: lookups touch only the rows of
/// the tokens seen, so a large table (e.g. per-layer n-gram embeddings) never becomes resident as a whole.
const INPUT_RESIDENT_CAP: u64 = 1 << 30;

/// Splits the manifest into blocks with state sized for `n_seq` sequences of `n_ctx_seq` cells. With `mlock`,
/// input embedding tables count in full; otherwise up to `INPUT_RESIDENT_CAP` each.
pub fn layout(m: &ModelManifest, facts: &ArchFacts, n_seq: u64, n_ctx_seq: u64, type_k: GgmlType, type_v: GgmlType, mlock: bool) -> Layout {
    let n_all = (facts.n_layer + facts.n_layer_nextn) as usize;
    let mut blocks: Vec<Block> = (0..n_all as u32).map(|index| Block { index, ..Default::default() }).collect();
    let state = facts.state_bytes_per_layer(n_seq, n_ctx_seq, type_k, type_v);
    for (i, (b, s)) in blocks.iter_mut().zip(&state).enumerate() {
        b.state_bytes = s.kv + s.recurrent;
        if s.kv > 0 {
            b.attn_macs_per_pos = facts.n_head.get(i).copied().unwrap_or(0) as u64 * (facts.key_length + facts.value_length) as u64;
        }
    }
    let mut head = Head::default();
    for t in &m.tensors {
        match (t.layer(), t.role()) {
            (Some(l), TensorRole::FfnRoutedExpert) if (l as usize) < n_all && t.dims.len() == 3 => {
                let b = &mut blocks[l as usize];
                b.experts.push(ExpertTensor { name: t.name.clone(), ggml_type: t.ggml_type, bytes: t.bytes, k: t.dims[0], n: t.dims[1] });
            }
            (Some(l), _) if (l as usize) < n_all => blocks[l as usize].dense.push(weight(t)),
            (None, TensorRole::Output | TensorRole::OutputNorm) => head.output.push(weight(t)),
            (None, TensorRole::TokenEmbedding) => head.input_bytes += if mlock { t.bytes } else { t.bytes.min(INPUT_RESIDENT_CAP) },
            _ => {}
        }
    }
    // Tied embeddings: the head reads token_embd when there is no separate output matrix.
    if head.output.iter().all(|w| w.macs == 0) {
        if let Some(t) = m.tensors.iter().find(|t| t.name == "token_embd.weight") {
            head.output.push(weight(t));
        }
    }
    let moe = facts.moe.as_ref();
    Layout {
        blocks,
        head,
        n_expert: moe.map_or(0, |m| m.n_expert),
        n_expert_used: moe.map_or(0, |m| m.n_expert_used),
        n_embd: facts.n_embd,
        n_vocab: facts.n_vocab,
    }
}

impl Layout {
    /// Fraction of a block's routed-expert bytes read in one step of `batch` tokens:
    /// the expected share of distinct experts selected, 1 - (1 - k/E)^batch.
    pub fn expert_read_fraction(&self, batch: u32) -> f64 {
        if self.n_expert == 0 {
            return 0.0;
        }
        let p = self.n_expert_used as f64 / self.n_expert as f64;
        1.0 - (1.0 - p).powi(batch as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flux_core::model::{Compatibility, ModelFormat, MoeFacts};

    fn t(name: &str, dims: &[u64], bytes: u64) -> TensorInfo {
        TensorInfo { name: name.into(), ggml_type: GgmlType::Q8_0, dims: dims.to_vec(), bytes, shard: 0, offset: 0 }
    }

    #[test]
    fn splits_dense_and_expert_weights() {
        let facts = ArchFacts {
            architecture: "x".into(),
            n_layer: 2,
            n_layer_nextn: 0,
            n_ctx_train: 4096,
            n_embd: 64,
            n_vocab: 100,
            n_head: vec![4; 2],
            n_head_kv: vec![2; 2],
            key_length: 16,
            value_length: 16,
            mla: false,
            sliding_window: None,
            recurrent: vec![false; 2],
            recurrent_state_elems: 0,
            moe: Some(MoeFacts { n_expert: 8, n_expert_used: 2, n_expert_shared: 0, n_dense_lead: 1 }),
            unmodeled: vec![],
        };
        let m = ModelManifest {
            schema: 1,
            format: ModelFormat::Gguf,
            files: vec![],
            source: None,
            name: None,
            facts: Some(facts.clone()),
            tensors: vec![
                t("token_embd.weight", &[64, 100], 1000),
                t("blk.0.ffn_up.weight", &[64, 128], 500),
                t("blk.1.attn_q.weight", &[64, 64], 200),
                t("blk.1.ffn_up_exps.weight", &[64, 32, 8], 800),
                t("output.weight", &[64, 100], 900),
            ],
            encodings: Default::default(),
            tokenizer: None,
            chat_template_sha256: None,
            compatibility: Compatibility::Executable { engines: vec![] },
            identity: None,
        };
        let l = layout(&m, &facts, 1, 1024, GgmlType::F16, GgmlType::F16, false);
        assert_eq!(l.blocks[0].dense_bytes(), 500);
        assert_eq!(l.blocks[1].expert_bytes(), 800);
        assert_eq!(l.blocks[1].experts[0].n, 32);
        assert_eq!(l.head.input_bytes, 1000);
        assert_eq!(l.head.output.iter().map(|w| w.macs).sum::<u64>(), 6400);
        assert_eq!(l.blocks[1].attn_macs_per_pos, 4 * 32);
        assert_eq!(l.blocks[0].state_bytes, 1024 * 2 * (16 * 2 * 2));
        assert!((l.expert_read_fraction(1) - 0.25).abs() < 1e-9);
        assert!(l.expert_read_fraction(16) > 0.98);
    }
}
