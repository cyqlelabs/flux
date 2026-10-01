//! Tokenizer and chat-template identity. Two artifacts with equal identities tokenize and prompt identically.

use crate::gguf::{GgufModel, Value};
use flux_core::model::TokenizerIdentity;
use sha2::{Digest, Sha256};

const IDENTITY_KEYS: &[&str] = &[
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_sep_token",
];

pub fn tokenizer_identity(m: &GgufModel) -> Option<TokenizerIdentity> {
    let model = m.get("tokenizer.ggml.model")?.as_str()?.to_string();
    let mut h = Sha256::new();
    for k in IDENTITY_KEYS {
        h.update(k.as_bytes());
        match m.get(k) {
            Some(v) => v.hash_into(&mut h),
            None => h.update([0xff]),
        }
    }
    let id = |k: &str| m.get(k).and_then(Value::as_u64).map(|v| v as u32);
    Some(TokenizerIdentity {
        model,
        pre: m.get("tokenizer.ggml.pre").and_then(Value::as_str).map(str::to_string),
        n_tokens: m.get("tokenizer.ggml.tokens").and_then(Value::as_array).map_or(0, |a| a.len() as u32),
        sha256: hex::encode(h.finalize()),
        bos: id("tokenizer.ggml.bos_token_id"),
        eos: id("tokenizer.ggml.eos_token_id"),
        eot: id("tokenizer.ggml.eot_token_id"),
        add_bos: m.get("tokenizer.ggml.add_bos_token").and_then(Value::as_bool),
    })
}

pub fn chat_template_sha256(m: &GgufModel) -> Option<String> {
    m.get("tokenizer.chat_template").and_then(Value::as_str).map(|t| flux_core::fsutil::sha256_hex(t.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::Builder;

    fn identity(tokens: &[&str], eos: u32) -> TokenizerIdentity {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.gguf");
        Builder::new()
            .str("general.architecture", "llama")
            .str("tokenizer.ggml.model", "gpt2")
            .strs("tokenizer.ggml.tokens", tokens)
            .u32("tokenizer.ggml.eos_token_id", eos)
            .tensor("w", &[32])
            .write(&p);
        tokenizer_identity(&GgufModel::open(&p).unwrap()).unwrap()
    }

    #[test]
    fn special_token_change_changes_identity() {
        let a = identity(&["a", "b"], 1);
        assert_eq!(a, identity(&["a", "b"], 1));
        assert_ne!(a.sha256, identity(&["a", "b"], 0).sha256);
        assert_ne!(a.sha256, identity(&["a", "c"], 1).sha256);
        assert_eq!(a.n_tokens, 2);
    }
}
