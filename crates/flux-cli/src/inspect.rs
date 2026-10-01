use flux_core::fmt_bytes;
use flux_core::ggml_type::GgmlType;
use flux_core::model::{Compatibility, ModelManifest};

pub fn print(m: &ModelManifest) {
    println!("format        {:?}", m.format);
    for f in &m.files {
        let hash = f.sha256.as_deref().map_or("not hashed".to_string(), |h| h[..16].to_string());
        println!("file          {} ({}, sha256 {hash})", f.path.display(), fmt_bytes(f.size));
    }
    if let Some(s) = &m.source {
        println!("source        {} @ {}", s.repo.as_deref().unwrap_or("?"), &s.commit[..12.min(s.commit.len())]);
    }
    if let Some(id) = &m.identity {
        println!("identity      {id}");
    }
    if let Some(n) = &m.name {
        println!("name          {n}");
    }
    if let Some(f) = &m.facts {
        println!("architecture  {}", f.architecture);
        let nextn = if f.n_layer_nextn > 0 { format!(" + {} next-token block(s)", f.n_layer_nextn) } else { String::new() };
        println!("blocks        {}{nextn}, embedding {}, vocab {}, trained context {}", f.n_layer, f.n_embd, f.n_vocab, f.n_ctx_train);
        let attn = f.n_head_kv.iter().filter(|&&h| h > 0).count();
        let kv = f.n_head_kv.iter().copied().max().unwrap_or(0);
        println!(
            "attention     {attn} layers with cache, {} query / {kv} KV heads, key {} value {}{}",
            f.n_head.first().copied().unwrap_or(0),
            f.key_length,
            f.value_length,
            if f.mla { ", compressed latent (MLA)" } else { "" }
        );
        let rec = f.recurrent.iter().filter(|&&r| r).count();
        if rec > 0 {
            println!("recurrent     {rec} layers, {} per sequence", fmt_bytes(rec as u64 * f.recurrent_state_elems * 4));
        }
        if let Some(moe) = &f.moe {
            println!(
                "experts       {} routed ({} active), {} shared, {} leading dense block(s)",
                moe.n_expert, moe.n_expert_used, moe.n_expert_shared, moe.n_dense_lead
            );
        }
        for ctx in [8192u64, 32768] {
            let b = f.state_bytes_total(1, ctx, GgmlType::F16, GgmlType::F16);
            println!("state @ {:>5}  {} per sequence (f16 cache; allocator overhead extra)", ctx, fmt_bytes(b));
        }
        for u in &f.unmodeled {
            println!("note          {u}: measured by the backend, not by the formula");
        }
    }
    let total = m.total_tensor_bytes();
    if total > 0 {
        let enc: Vec<String> = m.encodings.iter().map(|(k, v)| format!("{k} {:.1}%", 100.0 * *v as f64 / total as f64)).collect();
        println!("weights       {} in {} tensors: {}", fmt_bytes(total), m.tensors.len(), enc.join(", "));
    }
    if let Some(t) = &m.tokenizer {
        println!("tokenizer     {} / {} ({} tokens), identity {}", t.model, t.pre.as_deref().unwrap_or("-"), t.n_tokens, &t.sha256[..16]);
    }
    if let Some(c) = &m.chat_template_sha256 {
        println!("chat template {}", &c[..16]);
    }
    match &m.compatibility {
        Compatibility::Executable { engines } => println!("executable    yes: {}", engines.join(", ")),
        Compatibility::InspectOnly { missing, alternatives } => {
            println!("executable    no");
            missing.iter().for_each(|x| println!("  missing     {x}"));
            alternatives.iter().for_each(|x| println!("  alternative {x}"));
        }
    }
}
