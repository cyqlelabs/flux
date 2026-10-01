//! Execution facts from checkpoint metadata. Families are not graphs: every number comes from the artifact.

use crate::gguf::{GgufModel, Value};
use anyhow::{Context, Result};
use flux_core::model::{ArchFacts, MoeFacts};

/// Scalar or per-layer array key, expanded to one value per layer.
fn per_layer(m: &GgufModel, key: &str, n: usize) -> Option<Vec<u32>> {
    match m.get(key)? {
        Value::Array(a) => Some(a.iter().map(|v| v.as_u64().unwrap_or(0) as u32).chain(std::iter::repeat(0)).take(n).collect()),
        v => v.as_u64().map(|x| vec![x as u32; n]),
    }
}

pub fn facts_from_gguf(m: &GgufModel) -> Result<ArchFacts> {
    let arch = m.get("general.architecture").and_then(Value::as_str).context("general.architecture missing")?.to_string();
    let key = |k: &str| format!("{arch}.{k}");
    let u = |k: &str| m.get(&key(k)).and_then(Value::as_u64);

    let n_layer_all = u("block_count").context("block_count missing")? as u32;
    let n_layer_nextn = u("nextn_predict_layers").unwrap_or(0) as u32;
    let n_layer = n_layer_all - n_layer_nextn.min(n_layer_all);
    let n_embd = u("embedding_length").unwrap_or(0) as u32;
    let n = n_layer as usize;

    let n_head = per_layer(m, &key("attention.head_count"), n).unwrap_or_else(|| vec![0; n]);
    let mut n_head_kv = per_layer(m, &key("attention.head_count_kv"), n).unwrap_or_else(|| n_head.clone());
    let head0 = n_head.iter().copied().find(|&h| h > 0).unwrap_or(1);
    let key_length = u("attention.key_length").map_or(n_embd / head0, |v| v as u32);
    let value_length = u("attention.value_length").map_or(n_embd / head0, |v| v as u32);
    let mla = u("attention.key_length_mla").unwrap_or(0) > 0 && u("attention.value_length_mla").unwrap_or(0) > 0;

    let n_vocab = u("vocab_size").or_else(|| m.get("tokenizer.ggml.tokens").and_then(Value::as_array).map(|a| a.len() as u64)).unwrap_or(0) as u32;

    let mut unmodeled = vec![];
    let sliding_window = u("attention.sliding_window").map(|v| v as u32).filter(|&v| v > 0);
    if sliding_window.is_some() {
        unmodeled.push("sliding-window cache size (analytic model assumes full-length cache)".into());
    }

    let mut recurrent = vec![false; n];
    let mut recurrent_state_elems = 0;
    if let Some(d_conv) = u("ssm.conv_kernel") {
        let d_state = u("ssm.state_size").unwrap_or(0);
        let d_inner = u("ssm.inner_size").unwrap_or(0);
        let n_group = u("ssm.group_count").unwrap_or(0);
        recurrent_state_elems = d_conv.saturating_sub(1) * (d_inner + 2 * n_group * d_state) + d_state * d_inner;
        recurrent = match u("full_attention_interval") {
            Some(k) if k > 0 => (0..n as u64).map(|i| !(i + 1).is_multiple_of(k)).collect(),
            _ if matches!(m.get(&key("attention.head_count_kv")), Some(Value::Array(_))) => n_head_kv.iter().map(|&h| h == 0).collect(),
            _ => vec![true; n],
        };
        for (i, r) in recurrent.iter().enumerate() {
            if *r {
                n_head_kv[i] = 0;
            }
        }
    }
    for (k, what) in [
        ("rwkv.head_size", "RWKV state"),
        ("attention.indexer.head_count", "sparse-attention indexer cache"),
        ("attention.compress_ratios", "compressed attention layers"),
        ("embedding_length_per_layer_input", "per-layer input embeddings"),
    ] {
        if m.get(&key(k)).is_some() || m.get(k).is_some() {
            unmodeled.push(what.into());
        }
    }

    let moe = u("expert_count").filter(|&e| e > 0).map(|e| MoeFacts {
        n_expert: e as u32,
        n_expert_used: u("expert_used_count").unwrap_or(0) as u32,
        // Some checkpoints give only the shared expert's width; that still means one shared expert.
        n_expert_shared: u("expert_shared_count").unwrap_or(u64::from(u("expert_shared_feed_forward_length").unwrap_or(0) > 0)) as u32,
        n_dense_lead: u("leading_dense_block_count").unwrap_or(0) as u32,
    });

    Ok(ArchFacts {
        architecture: arch.clone(),
        n_layer,
        n_layer_nextn,
        n_ctx_train: u("context_length").unwrap_or(0) as u32,
        n_embd,
        n_vocab,
        n_head,
        n_head_kv,
        key_length,
        value_length,
        mla,
        sliding_window,
        recurrent,
        recurrent_state_elems,
        moe,
        unmodeled,
    })
}

/// Facts from a Hugging Face `config.json`, for planning arithmetic on artifacts Flux will not execute itself.
pub fn facts_from_hf_config(cfg: &serde_json::Value) -> Option<ArchFacts> {
    let c = cfg.get("text_config").unwrap_or(cfg);
    let u = |k: &str| c.get(k).and_then(serde_json::Value::as_u64);
    let n_layer = u("num_hidden_layers")? as u32;
    let n_embd = u("hidden_size").unwrap_or(0) as u32;
    let heads = u("num_attention_heads").unwrap_or(1) as u32;
    let kv_heads = u("num_key_value_heads").map_or(heads, |v| v as u32);
    let head_dim = u("head_dim").map_or(n_embd / heads.max(1), |v| v as u32);
    let kv_lora = u("kv_lora_rank");
    let (key_length, value_length, n_head_kv, mla) = match kv_lora {
        Some(r) => ((r + u("qk_rope_head_dim").unwrap_or(0)) as u32, r as u32, 1, true),
        None => (head_dim, head_dim, kv_heads, false),
    };
    let n_expert = u("num_experts").or_else(|| u("n_routed_experts")).or_else(|| u("num_local_experts"));
    let n = n_layer as usize;
    Some(ArchFacts {
        architecture: c.get("model_type").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
        n_layer,
        n_layer_nextn: u("num_nextn_predict_layers").unwrap_or(0) as u32,
        n_ctx_train: u("max_position_embeddings").unwrap_or(0) as u32,
        n_embd,
        n_vocab: u("vocab_size").unwrap_or(0) as u32,
        n_head: vec![heads; n],
        n_head_kv: vec![n_head_kv; n],
        key_length,
        value_length,
        mla,
        sliding_window: u("sliding_window").map(|v| v as u32),
        recurrent: vec![false; n],
        recurrent_state_elems: 0,
        moe: n_expert.filter(|&e| e > 0).map(|e| MoeFacts {
            n_expert: e as u32,
            n_expert_used: u("num_experts_per_tok").unwrap_or(0) as u32,
            n_expert_shared: u("n_shared_experts").unwrap_or(0) as u32,
            n_dense_lead: u("first_k_dense_replace").unwrap_or(0) as u32,
        }),
        unmodeled: vec!["derived from config.json, not from an executable artifact".into()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::Builder;
    use flux_core::ggml_type::GgmlType;

    #[test]
    fn qwen2_style_gqa() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("q.gguf");
        Builder::new()
            .str("general.architecture", "qwen2")
            .u32("qwen2.block_count", 28)
            .u32("qwen2.embedding_length", 3584)
            .u32("qwen2.attention.head_count", 28)
            .u32("qwen2.attention.head_count_kv", 4)
            .tensor("w", &[32])
            .write(&p);
        let f = facts_from_gguf(&GgufModel::open(&p).unwrap()).unwrap();
        assert_eq!(f.key_length, 128);
        assert_eq!(f.state_bytes_total(1, 8192, GgmlType::F16, GgmlType::F16), 469_762_048);
    }

    #[test]
    fn hybrid_recurrent_layers_hold_no_kv() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("h.gguf");
        Builder::new()
            .str("general.architecture", "qwen35")
            .u32("qwen35.block_count", 9)
            .u32("qwen35.nextn_predict_layers", 1)
            .u32("qwen35.attention.head_count", 24)
            .u32("qwen35.attention.head_count_kv", 4)
            .u32("qwen35.attention.key_length", 256)
            .u32("qwen35.attention.value_length", 256)
            .u32("qwen35.ssm.conv_kernel", 4)
            .u32("qwen35.ssm.state_size", 128)
            .u32("qwen35.ssm.inner_size", 6144)
            .u32("qwen35.ssm.group_count", 16)
            .u32("qwen35.full_attention_interval", 4)
            .tensor("w", &[32])
            .write(&p);
        let f = facts_from_gguf(&GgufModel::open(&p).unwrap()).unwrap();
        assert_eq!(f.n_layer, 8);
        assert_eq!(f.recurrent, vec![true, true, true, false, true, true, true, false]);
        assert_eq!(f.recurrent_state_elems, 3 * (6144 + 2 * 16 * 128) + 128 * 6144);
        let per = f.state_bytes_per_layer(1, 1024, GgmlType::F16, GgmlType::F16);
        assert_eq!(per[3].kv, 1024 * 4 * (256 * 2 + 256 * 2));
        assert_eq!(per[0].kv, 0);
        assert_eq!(per[0].recurrent, f.recurrent_state_elems * 4);
    }

    #[test]
    fn mla_from_hf_config() {
        let cfg: serde_json::Value = serde_json::json!({
            "model_type": "deepseek_v2", "num_hidden_layers": 27, "hidden_size": 2048,
            "num_attention_heads": 16, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
            "n_routed_experts": 64, "num_experts_per_tok": 6, "n_shared_experts": 2, "first_k_dense_replace": 1
        });
        let f = facts_from_hf_config(&cfg).unwrap();
        assert!(f.mla);
        // Latent (512) + positional (64) components only, not 16 expanded heads.
        assert_eq!(f.state_bytes_total(1, 1, GgmlType::F16, GgmlType::F16), 27 * 576 * 2);
        assert_eq!(f.moe.unwrap().n_dense_lead, 1);
    }
}
