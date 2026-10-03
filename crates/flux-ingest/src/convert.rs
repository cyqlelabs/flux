//! Offline conversion of a supported Hugging Face checkpoint to GGUF with the pinned converter.
//! Results are cached by source hashes, converter revision and output encoding, written atomically,
//! and admitted only when the destination has room for them.

use crate::{hashing, safetensors, BACKEND_PIN, CONVERTIBLE_HF_ARCHITECTURES};
use anyhow::{bail, ensure, Context, Result};
use flux_core::config::FluxConfig;
use flux_core::model::ModelFormat;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub source_dir: PathBuf,
    pub source_identity: String,
    pub hf_architecture: String,
    pub converter_revision: String,
    pub outtype: String,
    pub seconds: f64,
}

/// Bytes the output will take: float sources keep their size at bf16/f16, q8_0 is ~53%.
fn estimate(source_bytes: u64, outtype: &str) -> u64 {
    match outtype {
        "q8_0" => source_bytes * 34 / 64,
        "f32" => source_bytes * 2,
        _ => source_bytes,
    }
}

fn free_bytes(dir: &Path) -> Result<u64> {
    let c = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes())?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    ensure!(unsafe { libc::statvfs(c.as_ptr(), &mut s) } == 0, "statvfs failed for {}", dir.display());
    Ok(s.f_bavail * s.f_frsize)
}

/// Converts `source` (a Hugging Face directory) and returns the cached GGUF path.
pub fn convert(cfg: &FluxConfig, source: &Path, outtype: &str, python: &Path, log: &dyn Fn(&str)) -> Result<PathBuf> {
    let d = safetensors::inspect_dir(source)?;
    if d.format != ModelFormat::Safetensors {
        bail!(
            "{} holds {:?} weights; converting a low-bit source would dequantize and requantize it, which is not a recovery of the original weights. Use an engine certified for that format instead",
            source.display(),
            d.format
        );
    }
    let class = d.hf_architecture.clone().context("config.json has no architectures entry")?;
    ensure!(CONVERTIBLE_HF_ARCHITECTURES.contains(&class.as_str()), "the pinned converter does not support {class}");
    ensure!(["f32", "f16", "bf16", "q8_0", "auto"].contains(&outtype), "outtype must be f32, f16, bf16, q8_0 or auto");

    let mut files = d.files.iter().map(|p| hashing::model_file(p)).collect::<Result<Vec<_>>>()?;
    hashing::HashIndex::open(&cfg.cache_dir).hash_all(&mut files)?;
    let identity = hashing::identity(&files).context("source hashes missing")?;
    let script = cfg.llama_dir.join("convert_hf_to_gguf.py");
    let script_sha = hashing::sha256_file(&script, &|_| {})?;
    let key = flux_core::fsutil::sha256_hex(format!("{identity}|{BACKEND_PIN}|{script_sha}|{outtype}").as_bytes());
    let dir = cfg.cache_dir.join("converted").join(&key[..16]);
    let name = source.file_name().and_then(|n| n.to_str()).unwrap_or("model");
    let out = dir.join(format!("{name}-{outtype}.gguf"));
    if out.exists() {
        log(&format!("cached conversion {}", out.display()));
        return Ok(out);
    }

    std::fs::create_dir_all(&dir)?;
    let source_bytes: u64 = d.dtype_bytes.values().sum();
    let need = estimate(source_bytes, outtype) + (1 << 30);
    let free = free_bytes(&dir)?;
    ensure!(free >= need, "conversion needs about {} free in {}, only {} available", flux_core::fmt_bytes(need), dir.display(), flux_core::fmt_bytes(free));

    let part = flux_core::fsutil::tmp_path(&out);
    log(&format!("converting {class} ({}) with the pinned converter → {}", flux_core::fmt_bytes(source_bytes), out.display()));
    let t0 = Instant::now();
    let status = std::process::Command::new(python)
        .arg(&script)
        .arg(source)
        .arg("--outfile")
        .arg(&part)
        .arg("--outtype")
        .arg(outtype)
        .status()
        .with_context(|| format!("running {} {}", python.display(), script.display()))?;
    if !status.success() {
        let _ = std::fs::remove_file(&part);
        bail!("converter exited with {status}");
    }
    std::fs::rename(&part, &out)?;
    let prov = Provenance {
        source_dir: source.to_path_buf(),
        source_identity: identity,
        hf_architecture: class,
        converter_revision: BACKEND_PIN.into(),
        outtype: outtype.into(),
        seconds: t0.elapsed().as_secs_f64(),
    };
    flux_core::fsutil::write_json_atomic(&dir.join("provenance.json"), &prov)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_low_bit_sources_and_sizes_outputs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"architectures":["Qwen2ForCausalLM"],"quantization_config":{"quant_method":"awq"}}"#).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), [2u8, 0, 0, 0, 0, 0, 0, 0, b'{', b'}']).unwrap();
        let err = convert(&FluxConfig::default(), dir.path(), "bf16", Path::new("python3"), &|_| {}).unwrap_err().to_string();
        assert!(err.contains("not a recovery"), "{err}");
        assert_eq!(estimate(64, "q8_0"), 34);
    }
}
