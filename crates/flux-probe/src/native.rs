//! Native probes run inside flux-worker processes: copy curves, contention, kernel shapes taken
//! from the model being planned, CPU bandwidth by thread count, and storage reads of the model file.

use crate::storage::{probe_file, StorageOpts};
use anyhow::{Context, Result};
use flux_core::config::FluxConfig;
use flux_core::fsutil::{read_json, write_json_atomic};
use flux_core::ggml_type::GgmlType;
use flux_core::hardware::{BackendDevice, ContentionProbe, CopyCurve, CopyDirection, CopyPoint, CpuBandwidth, KernelProbe, ProbeReport};
use flux_core::model::ModelManifest;
use flux_core::stats::Summary;
use flux_core::worker::oneshot_job;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const SCHEMA: u32 = 1;

pub struct ProbeOptions<'a> {
    pub quick: bool,
    /// Kernel shapes and storage reads come from this model when given.
    pub model: Option<&'a ModelManifest>,
}

/// A weight shape to time: encoding, rows x cols, and where real bytes of that shape live.
#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub ggml_type: GgmlType,
    pub k: u64,
    pub n: u64,
    pub source: Option<(PathBuf, u64)>,
}

/// Shapes covering most of the model's bytes, at least one per encoding that holds >1% of them.
pub fn model_shapes(m: &ModelManifest, max: usize) -> Vec<Shape> {
    let total = m.total_tensor_bytes().max(1);
    type Source = Option<(PathBuf, u64)>;
    let mut groups: BTreeMap<(GgmlType, u64, u64), (u64, Source)> = BTreeMap::new();
    for t in &m.tensors {
        if t.dims.len() < 2 || t.dims[1] < 64 {
            continue;
        }
        let src = m.files.get(t.shard as usize).map(|f| (f.path.clone(), t.offset));
        let e = groups.entry((t.ggml_type, t.dims[0], t.dims[1])).or_insert((0, src));
        e.0 += t.bytes;
    }
    let mut v: Vec<_> = groups.into_iter().collect();
    v.sort_by_key(|(_, (b, _))| std::cmp::Reverse(*b));
    let mut out: Vec<Shape> = vec![];
    let mut covered = 0u64;
    for ((t, k, n), (bytes, src)) in &v {
        let new_type = !out.iter().any(|s| s.ggml_type == *t);
        let significant = *bytes * 100 >= total;
        if (covered * 10 < total * 9 && out.len() < max) || (new_type && significant) {
            // Vocabulary-sized matrices (embeddings, per-layer inputs) need not fit a small GPU to time a
            // kernel: cap the weights plus the f32 output of the largest probed batch.
            let per_row = t.row_bytes(*k).unwrap_or(1) + MAX_PROBE_BATCH * 4;
            let n = (*n).min((MAX_PROBE_BYTES / per_row).max(64));
            out.push(Shape { ggml_type: *t, k: *k, n, source: src.clone() });
            covered += bytes;
        }
    }
    out
}

/// Largest weights-plus-output a kernel probe allocates, and the largest batch it times (`prefill`).
const MAX_PROBE_BYTES: u64 = 256 << 20;
const MAX_PROBE_BATCH: u64 = 512;

/// Same encoding and row length, widened to at least `min_bytes` so the weights cannot stay in the
/// CPU's last-level cache: decode streams the whole model through DRAM, and so must the probe.
fn dram_sized(s: &Shape, min_bytes: u64) -> Shape {
    let bytes = s.ggml_type.row_bytes(s.k).unwrap_or(1) * s.n;
    let n = s.n * min_bytes.div_ceil(bytes.max(1)).max(1);
    let need = s.ggml_type.row_bytes(s.k).unwrap_or(1) * n;
    let source = s.source.as_ref().and_then(|(p, off)| {
        let size = std::fs::metadata(p).ok()?.len();
        (size >= need).then(|| (p.clone(), (*off).min(size - need) / 4096 * 4096))
    });
    Shape { ggml_type: s.ggml_type, k: s.k, n, source }
}

/// Larger than the last-level cache of any desktop CPU.
const CPU_PROBE_BYTES: u64 = 512 << 20;

fn generic_shapes() -> Vec<Shape> {
    ["q4_K", "q8_0", "f16"].iter().map(|t| Shape { ggml_type: GgmlType::from_name(t).unwrap(), k: 4096, n: 4096, source: None }).collect()
}

fn summary(v: &Value) -> Summary {
    let xs: Vec<f64> = serde_json::from_value(v.clone()).unwrap_or_default();
    Summary::of(&xs).unwrap_or(Summary { n: 0, mean: 0.0, stddev: 0.0, min: 0.0, p50: 0.0, p95: 0.0, p99: 0.0, max: 0.0 })
}

async fn copy_curve(device: &str, peer: Option<&str>, direction: CopyDirection, pinned: bool, sizes: &[u64], iters: u32) -> Result<CopyCurve> {
    let dir = match direction {
        CopyDirection::HostToDevice => "h2d",
        CopyDirection::DeviceToHost => "d2h",
        CopyDirection::DeviceToDevice => "d2d",
    };
    let v =
        oneshot_job(&["probe", "copy"], &json!({"device": device, "peer": peer, "direction": dir, "pinned": pinned, "sizes": sizes, "iters": iters})).await?;
    let points = v["results"]
        .as_array()
        .context("copy probe result")?
        .iter()
        .map(|r| {
            let bytes = r["bytes"].as_u64().unwrap_or(0);
            let gbps: Vec<f64> =
                serde_json::from_value::<Vec<f64>>(r["micros"].clone()).unwrap_or_default().iter().map(|us| bytes as f64 / (us * 1e3)).collect();
            CopyPoint { bytes, gbps: Summary::of(&gbps).unwrap_or_else(|| summary(&Value::Null)) }
        })
        .collect();
    Ok(CopyCurve { device: device.into(), peer: peer.map(str::to_string), direction, pinned: v["pinned"].as_bool().unwrap_or(pinned), points })
}

async fn kernel(device: &str, s: &Shape, batches: &[u64], threads: u32, iters: u32) -> Result<Vec<KernelProbe>> {
    let mut req = json!({"device": device, "type": s.ggml_type.name(), "k": s.k, "n": s.n, "batches": batches, "threads": threads, "iters": iters});
    if let Some((path, offset)) = &s.source {
        req["source"] = json!({"path": path, "offset": offset});
    }
    let v = oneshot_job(&["probe", "matmul"], &req).await?;
    Ok(v["results"]
        .as_array()
        .context("matmul probe result")?
        .iter()
        .map(|r| KernelProbe {
            device: device.into(),
            ggml_type: s.ggml_type.name(),
            k: s.k,
            n: s.n,
            batch: r["batch"].as_u64().unwrap_or(1),
            threads,
            micros: summary(&r["micros"]),
        })
        .collect())
}

/// What the pinned backend reports: its devices, revision and build identity (pin + patches).
pub struct Backend {
    pub devices: Vec<BackendDevice>,
    pub pin: String,
    pub build: String,
}

pub async fn backend() -> Result<Backend> {
    let info = oneshot_job(&["info"], &json!({})).await?;
    let devices: Vec<BackendDevice> = info["devices"]
        .as_array()
        .context("backend info devices")?
        .iter()
        .map(|d| BackendDevice {
            name: d["name"].as_str().unwrap_or_default().into(),
            description: d["description"].as_str().unwrap_or_default().into(),
            kind: d["kind"].as_str().unwrap_or_default().into(),
            mem_total: d["mem_total"].as_u64().unwrap_or(0),
            mem_free: d["mem_free"].as_u64().unwrap_or(0),
            pci_bus_id: d["pci_bus_id"].as_str().map(str::to_string),
        })
        .collect();
    Ok(Backend { devices, pin: info["pin"].as_str().unwrap_or_default().to_string(), build: info["build"].as_str().unwrap_or_default().to_string() })
}

/// Runs every probe and returns the report; `log` receives one line per step.
pub async fn run(opts: &ProbeOptions<'_>, log: &(dyn Fn(&str) + Sync)) -> Result<ProbeReport> {
    let Backend { devices, pin, .. } = backend().await?;
    let inventory = tokio::task::spawn_blocking(move || crate::inventory::inventory(devices)).await??;
    let gpus: Vec<&BackendDevice> = inventory.backend_devices.iter().filter(|d| d.kind == "gpu").collect();
    let iters = if opts.quick { 4 } else { 10 };
    let sizes: Vec<u64> = if opts.quick { vec![4 << 10, 1 << 20, 32 << 20] } else { vec![4 << 10, 64 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20] };

    let mut copies = vec![];
    for g in &gpus {
        log(&format!("copy curves {} ({})", g.name, g.description));
        copies.push(copy_curve(&g.name, None, CopyDirection::HostToDevice, true, &sizes, iters).await?);
        copies.push(copy_curve(&g.name, None, CopyDirection::DeviceToHost, true, &sizes, iters).await?);
        copies.push(copy_curve(&g.name, None, CopyDirection::HostToDevice, false, &sizes, iters).await?);
        for p in gpus.iter().filter(|p| p.name != g.name) {
            copies.push(copy_curve(&g.name, Some(&p.name), CopyDirection::DeviceToDevice, false, &sizes, iters).await?);
        }
    }

    let cores = inventory.cpu.cores.max(1);
    let mut contention = vec![];
    let seconds = if opts.quick { 0.6 } else { 1.5 };
    if gpus.len() >= 2 {
        log("contention: two GPUs reading host memory at once");
        let v = oneshot_job(&["probe", "contention"], &json!({"scenario": "h2d_pair", "devices": [gpus[0].name, gpus[1].name], "seconds": seconds})).await?;
        contention.extend(contention_results(&v));
    }
    for g in &gpus {
        log(&format!("contention: {} transfers while CPU threads stream weights", g.name));
        let v = oneshot_job(&["probe", "contention"], &json!({"scenario": "h2d_cpu", "device": g.name, "threads": cores, "seconds": seconds})).await?;
        contention.extend(contention_results(&v));
    }

    let shapes = match opts.model {
        Some(m) => model_shapes(m, if opts.quick { 3 } else { 6 }),
        None => generic_shapes(),
    };
    let main = shapes.first().cloned().unwrap_or_else(|| generic_shapes()[0].clone());
    let tiny = Shape { ggml_type: main.ggml_type, k: 256, n: 256, source: None };

    // CPU thread sweep on the dominant shape: memory bandwidth saturates before all threads help.
    let threads = inventory.cpu.threads.max(1);
    let mut sweep: Vec<u32> = vec![cores / 2, cores, cores + cores / 2, threads];
    sweep.retain(|&t| t > 0 && t <= threads);
    sweep.dedup();
    let mut cpu_bandwidth = vec![];
    let big = dram_sized(&main, CPU_PROBE_BYTES);
    let main_bytes = big.ggml_type.row_bytes(big.k).unwrap_or(0) * big.n;
    for &t in &sweep {
        log(&format!("cpu {t} threads: {} {}x{}", big.ggml_type, big.k, big.n));
        let k = kernel("CPU", &big, &[1], t, iters).await?;
        let gbps: Vec<f64> = k.iter().map(|p| p.weight_gbps(main_bytes)).collect();
        cpu_bandwidth.push(CpuBandwidth { threads: t, gbps: Summary::of(&gbps).unwrap() });
    }
    let best_threads = cpu_bandwidth.iter().max_by(|a, b| a.gbps.p50.total_cmp(&b.gbps.p50)).map_or(cores, |b| b.threads);

    let mut kernels = vec![];
    let prefill = if opts.quick { 256 } else { 512 };
    for dev in gpus.iter().map(|g| (g.name.as_str(), 0u32, g.mem_free)).chain(std::iter::once(("CPU", best_threads, u64::MAX))) {
        for s in shapes.iter().chain(std::iter::once(&tiny)) {
            let small = s.k * s.n <= 256 * 256;
            // Decode streams weights from memory: on the CPU that means wider than the cache.
            // Prompt-sized products are compute-bound, so they keep the model's own shape.
            let mut runs: Vec<(Shape, Vec<u64>)> = match (dev.0 == "CPU", small) {
                (_, true) => vec![(s.clone(), vec![1])],
                (true, false) => vec![(dram_sized(s, CPU_PROBE_BYTES), vec![1]), (s.clone(), vec![prefill])],
                (false, false) => vec![(s.clone(), vec![1, prefill])],
            };
            runs.retain(|(r, batches)| {
                let output = r.n * batches.iter().max().copied().unwrap_or(1) * 4;
                let fits = r.ggml_type.row_bytes(r.k).unwrap_or(0) * r.n + output <= dev.2 / 4;
                if !fits {
                    log(&format!("skip {} {} {}x{}: larger than a quarter of free memory", dev.0, r.ggml_type, r.k, r.n));
                }
                fits
            });
            for (r, batches) in runs {
                log(&format!("kernel {} {} {}x{} batch {:?}", dev.0, r.ggml_type, r.k, r.n, batches));
                kernels.extend(kernel(dev.0, &r, &batches, dev.1, iters).await?);
            }
        }
    }

    let mut storage = vec![];
    if let Some(m) = opts.model {
        let path = m.files[0].path.clone();
        log(&format!("storage reads of {}", path.display()));
        let o = if opts.quick { StorageOpts { bytes: 256 << 20, runs: 2, random_reads: 64, ..Default::default() } } else { StorageOpts::default() };
        storage = tokio::task::spawn_blocking(move || probe_file(&path, &o)).await??;
    }

    Ok(ProbeReport {
        schema: SCHEMA,
        topology: inventory.topology_fingerprint(),
        backend_revision: pin,
        created: chrono::Utc::now(),
        inventory,
        copies,
        contention,
        kernels,
        cpu_bandwidth,
        storage,
    })
}

fn contention_results(v: &Value) -> Vec<ContentionProbe> {
    v["results"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|r| ContentionProbe {
                    scenario: r["label"].as_str().unwrap_or_default().into(),
                    alone_gbps: r["alone_gbps"].as_f64().unwrap_or(0.0),
                    contended_gbps: r["contended_gbps"].as_f64().unwrap_or(0.0),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Reports are stored per topology and model shapes: `<probes>/<topology>-<model identity or generic>.json`.
pub fn report_path(cfg: &FluxConfig, topology: &str, model: Option<&ModelManifest>) -> PathBuf {
    let tag = model.and_then(|m| m.identity.as_deref()).map_or("generic".to_string(), |id| id[..12].to_string());
    cfg.probes_dir().join(format!("{topology}-{tag}.json"))
}

pub fn save(cfg: &FluxConfig, r: &ProbeReport, model: Option<&ModelManifest>) -> Result<PathBuf> {
    let p = report_path(cfg, &r.topology, model);
    write_json_atomic(&p, r)?;
    Ok(p)
}

/// A stored report for this topology and backend, if one exists.
pub fn load(cfg: &FluxConfig, topology: &str, backend_revision: &str, model: Option<&ModelManifest>) -> Option<ProbeReport> {
    read_json::<ProbeReport>(&report_path(cfg, topology, model)).ok().filter(|r| r.schema == SCHEMA && r.backend_revision == backend_revision)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flux_core::model::{Compatibility, ModelFile, ModelFormat, TensorInfo};

    #[test]
    fn shapes_cover_bytes_and_every_significant_encoding() {
        let t = |name: &str, ty: &str, dims: &[u64], bytes: u64| TensorInfo {
            name: name.into(),
            ggml_type: GgmlType::from_name(ty).unwrap(),
            dims: dims.to_vec(),
            bytes,
            shard: 0,
            offset: 4096,
        };
        let m = ModelManifest {
            schema: 1,
            format: ModelFormat::Gguf,
            files: vec![ModelFile { path: "/m.gguf".into(), size: 0, mtime: 0, sha256: None }],
            source: None,
            name: None,
            facts: None,
            tensors: vec![
                t("blk.0.ffn_up.weight", "q4_K", &[4096, 14336], 900),
                t("blk.1.ffn_up.weight", "q4_K", &[4096, 14336], 900),
                t("blk.0.ffn_down.weight", "q6_K", &[14336, 4096], 150),
                t("blk.0.attn_norm.weight", "f32", &[4096], 1),
                t("output.weight", "q8_0", &[4096, 150000], 40),
            ],
            encodings: Default::default(),
            tokenizer: None,
            chat_template_sha256: None,
            compatibility: Compatibility::Executable { engines: vec![] },
            identity: None,
        };
        let s = model_shapes(&m, 6);
        assert_eq!(s[0].ggml_type.name(), "q4_K");
        assert_eq!(s[0].source, Some(("/m.gguf".into(), 4096)));
        assert!(s.iter().any(|x| x.ggml_type.name() == "q6_K"));
        assert!(s.iter().any(|x| x.ggml_type.name() == "q8_0"), "{s:?}");
        assert!(!s.iter().any(|x| x.ggml_type.name() == "f32"));
    }
}
