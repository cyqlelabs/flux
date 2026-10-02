//! Next-token heads grafted onto a model: a GGUF holding only next-token (MTP) blocks numbered after the
//! trunk is appended as the model's `nextn_predict_layers`. Every tensor's bytes are copied as they are.

use crate::gguf::{GgufModel, Value};
use anyhow::{ensure, Context, Result};
use flux_core::ggml_type::GgmlType;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

/// One output tensor, copied whole from a source shard.
struct OutTensor {
    name: String,
    ggml_type: GgmlType,
    dims: Vec<u64>,
    shard: usize,
    offset: u64,
    bytes: u64,
}

/// Writes `model` with `heads` appended to `out` (atomically) and returns its size.
pub fn graft(model: &GgufModel, heads: &GgufModel, out: &Path, progress: &dyn Fn(u64, u64)) -> Result<u64> {
    let first = &model.shards[0];
    let arch = model.get("general.architecture").and_then(Value::as_str).context("no general.architecture")?;
    let mut kv: Vec<(String, Value, Option<u32>)> =
        first.key_order.iter().filter(|k| !k.starts_with("split.")).map(|k| (k.clone(), first.kv[k].clone(), first.array_types.get(k).copied())).collect();
    graft_keys(&mut kv, arch, model, heads)?;
    let shards = model.shards.len();
    let tensors: Vec<OutTensor> = model
        .tensors
        .iter()
        .map(|t| (t, t.shard as usize))
        .chain(heads.tensors.iter().map(|t| (t, shards + t.shard as usize)))
        .map(|(t, shard)| OutTensor { name: t.name.clone(), ggml_type: t.ggml_type, dims: t.dims.clone(), shard, offset: t.offset, bytes: t.bytes })
        .collect();
    let sources = model
        .shards
        .iter()
        .chain(&heads.shards)
        .map(|s| File::open(&s.path).with_context(|| format!("opening {}", s.path.display())))
        .collect::<Result<Vec<_>>>()?;
    let tmp = flux_core::fsutil::tmp_path(out);
    match write(&tmp, &kv, &tensors, &sources, first.alignment, progress) {
        Ok(n) => {
            std::fs::rename(&tmp, out)?;
            Ok(n)
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// The trunk's metadata grown by the head blocks: `block_count` and `nextn_predict_layers`, and every
/// per-layer array extended by the heads file's own entries, or else by the last trunk layer's (the
/// loader sets the attention kind of next-token layers itself).
fn graft_keys(kv: &mut Vec<(String, Value, Option<u32>)>, arch: &str, model: &GgufModel, heads: &GgufModel) -> Result<()> {
    ensure!(heads.get("general.architecture").and_then(Value::as_str) == Some(arch), "the heads are not {arch} blocks");
    let nextn = format!("{arch}.nextn_predict_layers");
    ensure!(model.get(&nextn).is_none(), "the model already has next-token heads");
    let n_layer = model.get(&format!("{arch}.block_count")).and_then(Value::as_u64).context("no block_count")?;
    let n_heads = heads.get(&nextn).and_then(Value::as_u64).context("the heads file names no nextn_predict_layers")?;
    let numbered = |l: u32| (n_layer..n_layer + n_heads).contains(&(l as u64));
    ensure!(heads.tensors.iter().all(|t| t.layer().is_some_and(numbered)), "head tensors must be numbered {n_layer}..{}", n_layer + n_heads - 1);
    let prefix = format!("{arch}.");
    for (k, v, _) in kv.iter_mut() {
        if *k == format!("{arch}.block_count") {
            *v = Value::U32((n_layer + n_heads) as u32);
        } else if let (true, Value::Array(a)) = (k.starts_with(&prefix), &mut *v) {
            if a.len() as u64 == n_layer {
                let own = heads.get(k).and_then(Value::as_array).filter(|h| h.len() as u64 == n_heads);
                let extra = own.map_or_else(|| vec![a[a.len() - 1].clone(); n_heads as usize], <[Value]>::to_vec);
                a.extend(extra);
            }
        }
    }
    kv.push((nextn, Value::U32(n_heads as u32), None));
    Ok(())
}

fn write(
    path: &Path,
    kv: &[(String, Value, Option<u32>)],
    tensors: &[OutTensor],
    sources: &[File],
    alignment: u64,
    progress: &dyn Fn(u64, u64),
) -> Result<u64> {
    let pad = |n: u64| n.div_ceil(alignment) * alignment;
    let mut header = b"GGUF".to_vec();
    header.extend(3u32.to_le_bytes());
    header.extend((tensors.len() as u64).to_le_bytes());
    header.extend((kv.len() as u64).to_le_bytes());
    for (k, v, elem) in kv {
        Value::String(k.clone()).write_payload(&mut header, None);
        header.extend(v.type_id().to_le_bytes());
        v.write_payload(&mut header, *elem);
    }
    let mut offset = 0u64;
    for t in tensors {
        Value::String(t.name.clone()).write_payload(&mut header, None);
        header.extend((t.dims.len() as u32).to_le_bytes());
        t.dims.iter().for_each(|d| header.extend(d.to_le_bytes()));
        header.extend(t.ggml_type.0.to_le_bytes());
        header.extend(offset.to_le_bytes());
        ensure!(
            t.ggml_type.row_bytes(t.dims[0]).map(|r| r * t.dims[1..].iter().product::<u64>()) == Some(t.bytes),
            "{}: data size does not match its shape",
            t.name
        );
        offset = pad(offset + t.bytes);
    }
    let total = pad(header.len() as u64) + offset;
    let mut w = BufWriter::with_capacity(16 << 20, File::create(path).with_context(|| format!("creating {}", path.display()))?);
    w.write_all(&header)?;
    let mut done = header.len() as u64;
    let zeros = vec![0u8; 1 << 20];
    let fill = |w: &mut BufWriter<File>, mut n: u64, done: &mut u64| -> Result<()> {
        while n > 0 {
            let k = n.min(zeros.len() as u64);
            w.write_all(&zeros[..k as usize])?;
            n -= k;
            *done += k;
        }
        Ok(())
    };
    fill(&mut w, pad(done) - done, &mut done)?;
    let mut buf = vec![0u8; 8 << 20];
    for t in tensors {
        let mut at = t.offset;
        while at < t.offset + t.bytes {
            let k = (t.offset + t.bytes - at).min(buf.len() as u64) as usize;
            sources[t.shard].read_exact_at(&mut buf[..k], at)?;
            w.write_all(&buf[..k])?;
            at += k as u64;
            done += k as u64;
        }
        fill(&mut w, pad(done) - done, &mut done)?;
        progress(done, total);
    }
    let f = w.into_inner().map_err(|e| e.into_error())?;
    f.sync_all()?;
    ensure!(done == total, "wrote {done} bytes, expected {total}");
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tests::Builder;

    #[test]
    fn graft_appends_head_blocks_and_grows_per_layer_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let (trunk, heads, out) = (dir.path().join("trunk.gguf"), dir.path().join("heads.gguf"), dir.path().join("grafted.gguf"));
        Builder::new()
            .str("general.architecture", "qwen4exp")
            .u32("qwen4exp.block_count", 2)
            .u32s("qwen4exp.attention.compress_ratios", &[0, 4])
            .tensor("blk.0.attn_norm.weight", &[32])
            .tensor("blk.1.attn_norm.weight", &[32])
            .write(&trunk);
        Builder::new().str("general.architecture", "qwen4exp").u32("qwen4exp.nextn_predict_layers", 1).tensor("blk.2.nextn.enorm.weight", &[32]).write(&heads);
        graft(&GgufModel::open(&trunk).unwrap(), &GgufModel::open(&heads).unwrap(), &out, &|_, _| {}).unwrap();

        let g = GgufModel::open(&out).unwrap();
        assert_eq!(g.get("qwen4exp.block_count").and_then(Value::as_u64), Some(3));
        assert_eq!(g.get("qwen4exp.nextn_predict_layers").and_then(Value::as_u64), Some(1));
        let ratios: Vec<u64> = g.get("qwen4exp.attention.compress_ratios").and_then(Value::as_array).unwrap().iter().filter_map(Value::as_u64).collect();
        assert_eq!(ratios, vec![0, 4, 4]);
        assert!(g.tensors.iter().any(|t| t.name == "blk.2.nextn.enorm.weight"));
        assert_eq!(g.tensors.len(), 3);
    }
}
