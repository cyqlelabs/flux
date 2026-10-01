//! flux-ingest: resolve what an artifact is before anything tries to execute it.

pub mod arch;
pub mod capability;
pub mod convert;
pub mod corpus;
pub mod gguf;
pub mod hashing;
pub mod hf;
pub mod repack;
pub mod safetensors;
pub mod tokenizer;

include!(concat!(env!("OUT_DIR"), "/backend_archs.rs"));

use anyhow::{bail, Context, Result};
use flux_core::config::FluxConfig;
use flux_core::model::{HfSource, ModelFormat, ModelManifest, MANIFEST_SCHEMA};
use std::collections::BTreeMap;
use std::path::Path;

pub struct InspectOptions<'a> {
    /// Hash every file (cached); required for a model identity and therefore for plans.
    pub hash: bool,
    pub config: &'a FluxConfig,
}

pub fn inspect(path: &Path, opts: &InspectOptions) -> Result<ModelManifest> {
    if path.is_dir() {
        return inspect_hf_dir(path, opts);
    }
    let mut magic = [0u8; 4];
    std::io::Read::read_exact(&mut std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?, &mut magic)?;
    if &magic != b"GGUF" {
        bail!("{} is neither a GGUF file nor a Hugging Face model directory", path.display());
    }
    inspect_gguf(path, opts)
}

fn inspect_gguf(path: &Path, opts: &InspectOptions) -> Result<ModelManifest> {
    let model = gguf::GgufModel::open(path)?;
    let facts = arch::facts_from_gguf(&model)?;
    let mut files = model.shards.iter().map(|s| hashing::model_file(&s.path)).collect::<Result<Vec<_>>>()?;
    let source = source_of(&files[0].path);
    if opts.hash {
        let mut idx = hashing::HashIndex::open(&opts.config.cache_dir);
        idx.hash_all(&mut files)?;
        verify_against_source(&files)?;
    }
    let mut encodings = BTreeMap::new();
    for t in &model.tensors {
        *encodings.entry(t.ggml_type.name()).or_insert(0) += t.bytes;
    }
    let compatibility =
        capability::assess(&capability::Subject { format: ModelFormat::Gguf, architecture: Some(&facts.architecture), hf_class: None }, opts.config);
    Ok(ModelManifest {
        schema: MANIFEST_SCHEMA,
        format: ModelFormat::Gguf,
        identity: hashing::identity(&files),
        files,
        source,
        name: model.get("general.name").and_then(gguf::Value::as_str).map(str::to_string),
        tokenizer: tokenizer::tokenizer_identity(&model),
        chat_template_sha256: tokenizer::chat_template_sha256(&model),
        facts: Some(facts),
        tensors: model.tensors,
        encodings,
        compatibility,
    })
}

fn inspect_hf_dir(dir: &Path, opts: &InspectOptions) -> Result<ModelManifest> {
    let d = safetensors::inspect_dir(dir)?;
    let facts = arch::facts_from_hf_config(&d.config);
    let mut files = d.files.iter().map(|p| hashing::model_file(p)).collect::<Result<Vec<_>>>()?;
    if opts.hash {
        hashing::HashIndex::open(&opts.config.cache_dir).hash_all(&mut files)?;
    }
    let compatibility = capability::assess(
        &capability::Subject { format: d.format, architecture: facts.as_ref().map(|f| f.architecture.as_str()), hf_class: d.hf_architecture.as_deref() },
        opts.config,
    );
    Ok(ModelManifest {
        schema: MANIFEST_SCHEMA,
        format: d.format,
        identity: hashing::identity(&files),
        source: source_of(&files[0].path),
        files,
        name: d.hf_architecture.clone(),
        facts,
        tensors: vec![],
        encodings: d.dtype_bytes,
        tokenizer: None,
        chat_template_sha256: None,
        compatibility,
    })
}

/// Where the file came from: `flux fetch` writes `flux-source.json`; `hf download --local-dir`
/// leaves `.cache/huggingface/download/<file>.metadata` (commit, etag, timestamp).
fn source_of(file: &Path) -> Option<HfSource> {
    let dir = file.parent()?;
    if let Ok(s) = flux_core::fsutil::read_json::<hf::SourceRecord>(&dir.join(hf::SOURCE_FILE)) {
        return Some(s.source);
    }
    let meta = std::fs::read_to_string(hf_cli_metadata(file)?).ok()?;
    let commit = meta.lines().next()?.trim().to_string();
    (commit.len() == 40).then_some(HfSource { repo: None, revision: None, commit })
}

fn hf_cli_metadata(file: &Path) -> Option<std::path::PathBuf> {
    let name = file.file_name()?.to_str()?;
    Some(file.parent()?.join(".cache/huggingface/download").join(format!("{name}.metadata")))
}

/// Expected sha256 per file name recorded at download time.
fn expected_hashes(file: &Path) -> BTreeMap<String, String> {
    let dir = file.parent().unwrap_or(Path::new("."));
    if let Ok(s) = flux_core::fsutil::read_json::<hf::SourceRecord>(&dir.join(hf::SOURCE_FILE)) {
        return s.sha256;
    }
    let mut out = BTreeMap::new();
    if let Some(name) = file.file_name().and_then(|n| n.to_str()) {
        let etag = hf_cli_metadata(file).and_then(|m| std::fs::read_to_string(m).ok()).and_then(|m| m.lines().nth(1).map(str::to_string));
        if let Some(e) = etag.map(|e| e.trim().trim_matches('"').to_string()).filter(|e| e.len() == 64) {
            out.insert(name.to_string(), e);
        }
    }
    out
}

fn verify_against_source(files: &[flux_core::model::ModelFile]) -> Result<()> {
    for f in files {
        let name = f.path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if let (Some(want), Some(got)) = (expected_hashes(&f.path).get(name), &f.sha256) {
            if want != got {
                bail!("{} sha256 {got} does not match the source's {want}; the download is corrupt", f.path.display());
            }
        }
    }
    Ok(())
}
