//! Hugging Face directory inspection. Safetensors is storage, not an executable graph:
//! this reads headers and `config.json` for identification and arithmetic only.

use anyhow::{bail, ensure, Context, Result};
use flux_core::model::ModelFormat;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

pub struct HfDir {
    pub config: serde_json::Value,
    pub format: ModelFormat,
    pub files: Vec<PathBuf>,
    /// Payload bytes per safetensors dtype.
    pub dtype_bytes: BTreeMap<String, u64>,
    pub n_tensors: usize,
    pub hf_architecture: Option<String>,
}

pub fn format_of(config: &serde_json::Value) -> ModelFormat {
    let q = config.get("quantization_config").or_else(|| config.get("text_config").and_then(|t| t.get("quantization_config")));
    let method = q.and_then(|q| q.get("quant_method")).and_then(|m| m.as_str()).unwrap_or("").to_ascii_lowercase();
    match method.as_str() {
        "" => ModelFormat::Safetensors,
        "exl3" => ModelFormat::Exl3,
        "gptq" => ModelFormat::Gptq,
        "awq" => ModelFormat::Awq,
        "fp8" | "compressed-tensors" | "fbgemm_fp8" => ModelFormat::Fp8,
        _ => ModelFormat::Unknown,
    }
}

pub fn inspect_dir(dir: &Path) -> Result<HfDir> {
    let config_path = dir.join("config.json");
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path).with_context(|| format!("reading {}", config_path.display()))?)?;
    let mut files: Vec<PathBuf> =
        std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "safetensors")).collect();
    files.sort();
    if files.is_empty() {
        bail!("{} has config.json but no .safetensors files", dir.display());
    }
    if let Ok(idx) = std::fs::read(dir.join("model.safetensors.index.json")) {
        let idx: serde_json::Value = serde_json::from_slice(&idx)?;
        if let Some(map) = idx.get("weight_map").and_then(|m| m.as_object()) {
            let missing: Vec<&str> = map.values().filter_map(|v| v.as_str()).filter(|f| !dir.join(f).exists()).collect();
            ensure!(missing.is_empty(), "index references missing shard(s): {}", missing.join(", "));
        }
    }

    let mut dtype_bytes = BTreeMap::new();
    let mut n_tensors = 0;
    for f in &files {
        let (header, data_len) = read_header(f)?;
        for (name, t) in header.as_object().into_iter().flatten() {
            if name == "__metadata__" {
                continue;
            }
            let dtype = t.get("dtype").and_then(|d| d.as_str()).unwrap_or("?").to_string();
            let off = t.get("data_offsets").and_then(|o| o.as_array()).context("tensor without data_offsets")?;
            let (a, b) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
            ensure!(a <= b && b <= data_len, "{name} in {} points outside the file", f.display());
            *dtype_bytes.entry(dtype).or_insert(0) += b - a;
            n_tensors += 1;
        }
    }
    let hf_architecture = config.get("architectures").and_then(|a| a.get(0)).and_then(|a| a.as_str()).map(str::to_string);
    Ok(HfDir { format: format_of(&config), config, files, dtype_bytes, n_tensors, hf_architecture })
}

fn read_header(path: &Path) -> Result<(serde_json::Value, u64)> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len();
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len);
    ensure!(n <= 100 << 20 && n + 8 <= size, "{} has an invalid header length", path.display());
    let mut buf = vec![0u8; n as usize];
    f.read_exact(&mut buf)?;
    Ok((serde_json::from_slice(&buf)?, size - 8 - n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn detects_quantization_formats_and_sizes() {
        assert_eq!(format_of(&serde_json::json!({"quantization_config": {"quant_method": "exl3"}})), ModelFormat::Exl3);
        assert_eq!(format_of(&serde_json::json!({"quantization_config": {"quant_method": "awq"}})), ModelFormat::Awq);
        assert_eq!(format_of(&serde_json::json!({})), ModelFormat::Safetensors);

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"architectures":["Qwen2ForCausalLM"],"num_hidden_layers":2}"#).unwrap();
        let header = br#"{"a":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}"#;
        let mut f = File::create(dir.path().join("model.safetensors")).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header).unwrap();
        f.write_all(&[0u8; 8]).unwrap();
        let d = inspect_dir(dir.path()).unwrap();
        assert_eq!(d.dtype_bytes["BF16"], 8);
        assert_eq!(d.hf_architecture.as_deref(), Some("Qwen2ForCausalLM"));
    }
}
