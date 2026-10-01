//! Lossless MoE expert split. Each split layer's experts are relabeled so the hot ones come first
//! (the router's rows are permuted the same way, so routing is unchanged), then every routed-expert
//! tensor is cut into `<name>_hot` (hot experts) and `<name>` (the rest), each followed by one
//! all-zero expert per routing slot. Two f32 maps give each relabeled expert {its index in the part,
//! or the part's first zero expert; 1 if it belongs to the other part}: a slot routed elsewhere uses
//! the zero expert for its slot, so no token uses an expert twice. Quantized bytes are copied, never
//! re-encoded.

use crate::gguf::{GgufModel, Value};
use anyhow::{bail, ensure, Context, Result};
use flux_core::ggml_type::GgmlType;
pub use flux_core::plan::SplitSpec;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

pub const SPLIT_VERSION: u32 = 2;
const EXPERT_TENSORS: [&str; 4] = ["ffn_up_exps", "ffn_gate_exps", "ffn_down_exps", "ffn_gate_up_exps"];

enum Piece {
    File { shard: usize, offset: u64, len: u64 },
    Zeros(u64),
    Bytes(Vec<u8>),
}

struct OutTensor {
    name: String,
    ggml_type: GgmlType,
    dims: Vec<u64>,
    pieces: Vec<Piece>,
}

impl OutTensor {
    fn bytes(&self) -> u64 {
        self.pieces
            .iter()
            .map(|p| match p {
                Piece::File { len, .. } => *len,
                Piece::Zeros(n) => *n,
                Piece::Bytes(b) => b.len() as u64,
            })
            .sum()
    }
}

/// Writes the split checkpoint to `out` (atomically) and returns its size.
pub fn moe_split(model: &GgufModel, source_identity: &str, spec: &SplitSpec, out: &Path, progress: &dyn Fn(u64, u64)) -> Result<u64> {
    let first = &model.shards[0];
    let arch = model.get("general.architecture").and_then(Value::as_str).context("no general.architecture")?;
    let slots = model.get(&format!("{arch}.expert_used_count")).and_then(Value::as_u64).context("no expert_used_count")?;
    let mut kv: Vec<(String, Value, Option<u32>)> =
        first.key_order.iter().filter(|k| !k.starts_with("split.")).map(|k| (k.clone(), first.kv[k].clone(), first.array_types.get(k).copied())).collect();
    kv.push(("flux.moe_split.version".into(), Value::U32(SPLIT_VERSION), None));
    kv.push(("flux.moe_split.source".into(), Value::String(source_identity.into()), None));
    kv.push(("flux.moe_split.spec".into(), Value::String(spec.digest()), None));

    let mut tensors: Vec<OutTensor> = vec![];
    for t in &model.tensors {
        let local = t.local_name().to_string();
        let split = t.layer().and_then(|l| spec.layers.get(&l).map(|s| (l, s)));
        let Some((layer, (order, hot))) = split else {
            tensors.push(OutTensor {
                name: t.name.clone(),
                ggml_type: t.ggml_type,
                dims: t.dims.clone(),
                pieces: vec![Piece::File { shard: t.shard as usize, offset: t.offset, len: t.bytes }],
            });
            continue;
        };
        let e = order.len() as u64;
        let hot = *hot as u64;
        let stem = local.split('.').next().unwrap_or_default();
        let gather = |ids: &[u32], unit: u64| -> Vec<Piece> {
            ids.iter().map(|&i| Piece::File { shard: t.shard as usize, offset: t.offset + i as u64 * unit, len: unit }).collect()
        };
        if EXPERT_TENSORS.contains(&stem) && local.ends_with(".weight") {
            ensure!(t.dims.len() == 3 && t.dims[2] == e, "{} does not hold {e} experts", t.name);
            let unit = t.bytes / e;
            let mut hot_pieces = gather(&order[..hot as usize], unit);
            hot_pieces.push(Piece::Zeros(unit * slots));
            let mut cold_pieces = gather(&order[hot as usize..], unit);
            cold_pieces.push(Piece::Zeros(unit * slots));
            tensors.push(OutTensor {
                name: format!("blk.{layer}.{stem}_hot.weight"),
                ggml_type: t.ggml_type,
                dims: vec![t.dims[0], t.dims[1], hot + slots],
                pieces: hot_pieces,
            });
            tensors.push(OutTensor { name: t.name.clone(), ggml_type: t.ggml_type, dims: vec![t.dims[0], t.dims[1], e - hot + slots], pieces: cold_pieces });
        } else if stem.contains("_exps") {
            bail!("{} carries per-expert parameters the split does not support", t.name);
        } else if stem == "ffn_gate_inp" || stem == "exp_probs_b" {
            // Router rows (or bias entries) follow the experts' new order.
            ensure!(t.dims.last() == Some(&e), "{} is not indexed by {e} experts", t.name);
            tensors.push(OutTensor { name: t.name.clone(), ggml_type: t.ggml_type, dims: t.dims.clone(), pieces: gather(order, t.bytes / e) });
        } else {
            tensors.push(OutTensor {
                name: t.name.clone(),
                ggml_type: t.ggml_type,
                dims: t.dims.clone(),
                pieces: vec![Piece::File { shard: t.shard as usize, offset: t.offset, len: t.bytes }],
            });
        }
    }
    for (layer, (order, hot)) in &spec.layers {
        let e = order.len() as u32;
        let to_bytes = |f: &dyn Fn(u32) -> (u32, bool)| -> Vec<u8> {
            (0..e).map(f).flat_map(|(index, other)| [index as f32, other as u32 as f32]).flat_map(f32::to_le_bytes).collect()
        };
        let map_hot = to_bytes(&|i| if i < *hot { (i, false) } else { (*hot, true) });
        let map_cold = to_bytes(&|i| if i >= *hot { (i - hot, false) } else { (e - hot, true) });
        for (name, data) in [("ffn_moe_map_hot", map_hot), ("ffn_moe_map_cold", map_cold)] {
            tensors.push(OutTensor {
                name: format!("blk.{layer}.{name}.weight"),
                ggml_type: GgmlType::F32,
                dims: vec![2, e as u64],
                pieces: vec![Piece::Bytes(data)],
            });
        }
    }

    let sources = model.shards.iter().map(|s| File::open(&s.path).with_context(|| format!("opening {}", s.path.display()))).collect::<Result<Vec<_>>>()?;
    let alignment = first.alignment;
    let tmp = flux_core::fsutil::tmp_path(out);
    let written = write(&tmp, &kv, &tensors, &sources, alignment, progress);
    match written {
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
            t.ggml_type.row_bytes(t.dims[0]).map(|r| r * t.dims[1..].iter().product::<u64>()) == Some(t.bytes()),
            "{}: data size does not match its shape",
            t.name
        );
        offset = pad(offset + t.bytes());
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
        for p in &t.pieces {
            match p {
                Piece::File { shard, offset, len } => {
                    let mut at = *offset;
                    let end = offset + len;
                    while at < end {
                        let k = (end - at).min(buf.len() as u64) as usize;
                        sources[*shard].read_exact_at(&mut buf[..k], at)?;
                        w.write_all(&buf[..k])?;
                        at += k as u64;
                        done += k as u64;
                    }
                }
                Piece::Zeros(n) => fill(&mut w, *n, &mut done)?,
                Piece::Bytes(b) => {
                    w.write_all(b)?;
                    done += b.len() as u64;
                }
            }
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
    fn split_relabels_experts_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("moe.gguf");
        // 4 experts of 32x2 f16 each; router 32x4.
        Builder::new()
            .str("general.architecture", "qwen35moe")
            .u32("qwen35moe.expert_used_count", 2)
            .tensor("blk.0.ffn_up_exps.weight", &[32, 2, 4])
            .tensor("blk.0.ffn_gate_inp.weight", &[32, 4])
            .tensor("output.weight", &[32, 2])
            .write(&src);
        // Mark each expert's bytes so the relabeling is visible: expert i filled with byte i+1.
        let m = GgufModel::open(&src).unwrap();
        let f = std::fs::OpenOptions::new().write(true).open(&src).unwrap();
        for name in ["blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_inp.weight"] {
            let t = m.tensors.iter().find(|t| t.name == name).unwrap();
            let unit = t.bytes / 4;
            for i in 0..4u64 {
                f.write_all_at(&vec![i as u8 + 1; unit as usize], t.offset + i * unit).unwrap();
            }
        }
        let m = GgufModel::open(&src).unwrap();
        let spec = SplitSpec { layers: std::collections::BTreeMap::from([(0, (vec![2, 0, 3, 1], 1))]) };
        let out = dir.path().join("split.gguf");
        moe_split(&m, "id", &spec, &out, &|_, _| {}).unwrap();

        let s = GgufModel::open(&out).unwrap();
        assert_eq!(s.get("flux.moe_split.version").and_then(Value::as_u64), Some(2));
        let get = |name: &str| s.tensors.iter().find(|t| t.name == name).unwrap().clone();
        let read = |t: &flux_core::model::TensorInfo| {
            let mut b = vec![0u8; t.bytes as usize];
            File::open(&out).unwrap().read_exact_at(&mut b, t.offset).unwrap();
            b
        };
        let hot = get("blk.0.ffn_up_exps_hot.weight");
        assert_eq!(hot.dims, vec![32, 2, 3]);
        let hb = read(&hot);
        let unit = hb.len() / 3;
        assert!(hb[..unit].iter().all(|&b| b == 3) && hb[unit..].iter().all(|&b| b == 0), "hot = expert 2 then a zero expert per slot");
        let cold = read(&get("blk.0.ffn_up_exps.weight"));
        let firsts: Vec<u8> = cold.chunks(unit).map(|c| c[0]).collect();
        assert_eq!(firsts, vec![1, 4, 2, 0, 0], "cold = experts 0,3,1 then a zero expert per slot");
        let router = read(&get("blk.0.ffn_gate_inp.weight"));
        let rows: Vec<u8> = router.chunks(router.len() / 4).map(|c| c[0]).collect();
        assert_eq!(rows, vec![3, 1, 4, 2], "router rows follow the new order");
        let map = |name: &str| -> Vec<f32> { read(&get(name)).chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
        assert_eq!(map("blk.0.ffn_moe_map_hot.weight"), vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(map("blk.0.ffn_moe_map_cold.weight"), vec![3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0]);
        assert!(s.tensors.iter().any(|t| t.name == "output.weight"));
    }
}
