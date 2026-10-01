//! `flux.toml`: where Flux keeps state and which engines it may launch.
//! Looked up at `$FLUX_CONFIG`, then `~/.config/flux/flux.toml`; every field has a default.

use crate::fsutil::expand_home;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FluxConfig {
    /// Plans, probe reports, journals: small files.
    pub data_dir: PathBuf,
    /// Hash index, prepared artifacts, benchmark corpora: large files.
    pub cache_dir: PathBuf,
    /// Destination of `flux fetch`.
    pub models_dir: PathBuf,
    /// Pinned llama.cpp checkout with its `build/bin`.
    pub llama_dir: PathBuf,
    pub plan: PlanConfig,
    pub serve: ServeConfig,
    /// Additional OpenAI-compatible engines, selectable by name.
    pub engines: BTreeMap<String, ExternalEngine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PlanConfig {
    pub tuning_budget_s: f64,
    /// Device memory held back per GPU beyond the backend's own accounting.
    pub device_reserve_mib: u64,
    /// Host memory left for the OS and other processes.
    pub host_reserve_mib: u64,
    pub finalists: usize,
    pub calibration_prompts: usize,
    pub validation_prompts: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub queue_depth: usize,
    pub max_body_bytes: usize,
    /// Stop admitting new requests when MemAvailable falls below this.
    pub min_available_mib: u64,
    pub journal_ttl_s: u64,
    /// Tokens the server still expects to generate. When decode drift is established and the time it
    /// would save over this horizon exceeds the cost of retuning, the server replans by itself. 0 disables.
    pub retune_horizon_tokens: u64,
}

/// An engine that keeps its own scheduler and speaks the OpenAI HTTP API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalEngine {
    pub command: Vec<String>,
    /// Extra arguments; `{port}`, `{model}` and `{ctx}` are substituted.
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub cwd: Option<PathBuf>,
    /// `general.architecture` (GGUF) or `model_type` (Hugging Face) values it is certified for.
    pub architectures: Vec<String>,
    #[serde(default = "default_formats")]
    pub formats: Vec<crate::model::ModelFormat>,
    #[serde(default = "default_health")]
    pub health_path: String,
    #[serde(default = "default_startup")]
    pub startup_timeout_s: u64,
}

fn default_formats() -> Vec<crate::model::ModelFormat> {
    vec![crate::model::ModelFormat::Gguf]
}

fn default_health() -> String {
    "/health".into()
}

fn default_startup() -> u64 {
    600
}

impl Default for FluxConfig {
    fn default() -> Self {
        let data = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| expand_home("~/.local/share")).join("flux");
        FluxConfig {
            cache_dir: data.join("cache"),
            models_dir: data.join("models"),
            data_dir: data,
            llama_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../third_party/llama.cpp"),
            plan: PlanConfig::default(),
            serve: ServeConfig::default(),
            engines: BTreeMap::new(),
        }
    }
}

impl Default for PlanConfig {
    fn default() -> Self {
        PlanConfig { tuning_budget_s: 600.0, device_reserve_mib: 256, host_reserve_mib: 6144, finalists: 3, calibration_prompts: 3, validation_prompts: 4 }
    }
}

impl Default for ServeConfig {
    fn default() -> Self {
        ServeConfig {
            host: "127.0.0.1".into(),
            port: 8090,
            queue_depth: 64,
            max_body_bytes: 8 << 20,
            min_available_mib: 2048,
            journal_ttl_s: 600,
            retune_horizon_tokens: 0,
        }
    }
}

impl FluxConfig {
    pub fn load() -> Result<FluxConfig> {
        let path = std::env::var_os("FLUX_CONFIG").map(PathBuf::from).unwrap_or_else(|| expand_home("~/.config/flux/flux.toml"));
        if !path.exists() {
            return Ok(FluxConfig::default());
        }
        Self::load_from(&path)
    }

    pub fn load_from(path: &Path) -> Result<FluxConfig> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut cfg: FluxConfig = toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        for p in [&mut cfg.data_dir, &mut cfg.cache_dir, &mut cfg.models_dir, &mut cfg.llama_dir] {
            *p = expand_home(&p.to_string_lossy());
        }
        Ok(cfg)
    }

    /// A pinned llama.cpp tool: from the build tree, or from an installed package layout.
    pub fn llama_bin(&self, name: &str) -> PathBuf {
        let built = self.llama_dir.join("build/bin").join(name);
        if built.exists() {
            built
        } else {
            self.llama_dir.join("bin").join(name)
        }
    }

    pub fn plans_dir(&self) -> PathBuf {
        self.data_dir.join("plans")
    }

    pub fn probes_dir(&self) -> PathBuf {
        self.data_dir.join("probes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_file_keeps_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("flux.toml");
        std::fs::write(
            &p,
            "cache_dir = \"/big/cache\"\n[plan]\ntuning_budget_s = 60\n[engines.strata]\ncommand = [\"strata\"]\narchitectures = [\"qwen3next\"]\n",
        )
        .unwrap();
        let c = FluxConfig::load_from(&p).unwrap();
        assert_eq!(c.cache_dir, PathBuf::from("/big/cache"));
        assert_eq!(c.plan.tuning_budget_s, 60.0);
        assert_eq!(c.plan.finalists, 3);
        assert_eq!(c.serve.host, "127.0.0.1");
        assert_eq!(c.engines["strata"].health_path, "/health");
    }
}
