//! Hugging Face revision resolution and verified, atomic downloads.

use anyhow::{bail, ensure, Context, Result};
use flux_core::model::HfSource;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

pub const SOURCE_FILE: &str = "flux-source.json";
const API: &str = "https://huggingface.co";

/// Written next to fetched files; the manifest's source and expected hashes come from here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRecord {
    pub source: HfSource,
    pub sha256: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TreeEntry {
    path: String,
    #[serde(rename = "type")]
    kind: String,
    size: Option<u64>,
    lfs: Option<Lfs>,
}

#[derive(Debug, Clone, Deserialize)]
struct Lfs {
    oid: String,
}

pub struct RemoteFile {
    pub path: String,
    pub size: u64,
    pub sha256: Option<String>,
}

pub struct Client {
    http: reqwest::Client,
    token: Option<String>,
}

impl Client {
    pub fn new() -> Client {
        let token = std::env::var("HF_TOKEN").ok().or_else(|| {
            let home = std::env::var("HF_HOME").map(PathBuf::from).unwrap_or_else(|_| flux_core::fsutil::expand_home("~/.cache/huggingface"));
            std::fs::read_to_string(home.join("token")).ok().map(|t| t.trim().to_string())
        });
        Client { http: reqwest::Client::builder().user_agent("flux/0.1").build().expect("http client"), token }
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        let r = self.http.get(url);
        match &self.token {
            Some(t) => r.bearer_auth(t),
            None => r,
        }
    }

    /// Branch, tag or commit → immutable commit id.
    pub async fn resolve(&self, repo: &str, revision: &str) -> Result<String> {
        let url = format!("{API}/api/models/{repo}/revision/{revision}");
        let v: serde_json::Value = self.get(&url).send().await?.error_for_status().with_context(|| format!("resolving {repo}@{revision}"))?.json().await?;
        v.get("sha").and_then(|s| s.as_str()).map(str::to_string).context("revision response has no sha")
    }

    pub async fn list(&self, repo: &str, commit: &str) -> Result<Vec<RemoteFile>> {
        let url = format!("{API}/api/models/{repo}/tree/{commit}?recursive=true");
        let entries: Vec<TreeEntry> = self.get(&url).send().await?.error_for_status()?.json().await?;
        Ok(entries
            .into_iter()
            .filter(|e| e.kind == "file")
            .map(|e| RemoteFile { path: e.path, size: e.size.unwrap_or(0), sha256: e.lfs.map(|l| l.oid) })
            .collect())
    }

    /// Downloads into `dest/<path>` through a `.part` file, resuming partial data,
    /// verifying size and LFS sha256 before the final rename.
    pub async fn download(&self, repo: &str, commit: &str, f: &RemoteFile, dest: &Path, progress: &(dyn Fn(u64, u64) + Sync)) -> Result<PathBuf> {
        let target = dest.join(&f.path);
        if target.exists() && std::fs::metadata(&target)?.len() == f.size {
            return Ok(target);
        }
        std::fs::create_dir_all(target.parent().unwrap())?;
        let part = target.with_extension(format!("{}.part", target.extension().and_then(|e| e.to_str()).unwrap_or("")));
        let mut have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if have > f.size {
            std::fs::remove_file(&part)?;
            have = 0;
        }
        let url = format!("{API}/{repo}/resolve/{commit}/{}", f.path);
        let mut req = self.get(&url);
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let resp = req.send().await?.error_for_status().with_context(|| format!("downloading {}", f.path))?;
        if have > 0 && resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            have = 0;
        }
        let mut out = tokio::fs::OpenOptions::new().create(true).write(true).truncate(have == 0).append(have > 0).open(&part).await?;
        let mut stream = resp.bytes_stream();
        let mut done = have;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            out.write_all(&chunk).await?;
            done += chunk.len() as u64;
            progress(done, f.size);
        }
        out.sync_all().await?;
        drop(out);
        ensure!(done == f.size, "{}: received {done} of {} bytes", f.path, f.size);
        if let Some(want) = &f.sha256 {
            let p = part.clone();
            let got = tokio::task::spawn_blocking(move || crate::hashing::sha256_file(&p, &|_| {})).await??;
            if &got != want {
                std::fs::remove_file(&part)?;
                bail!("{}: sha256 {got} does not match LFS {want}; partial file removed", f.path);
            }
        }
        std::fs::rename(&part, &target)?;
        Ok(target)
    }
}

impl Default for Client {
    fn default() -> Self {
        Client::new()
    }
}

/// Glob with `*` wildcards, matched against the full repository path.
pub fn glob_match(pattern: &str, path: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == path;
    }
    let mut rest = path;
    for (i, p) in parts.iter().enumerate() {
        if i == 0 {
            let Some(r) = rest.strip_prefix(p) else { return false };
            rest = r;
        } else if i == parts.len() - 1 {
            return rest.ends_with(p);
        } else {
            let Some(pos) = rest.find(p) else { return false };
            rest = &rest[pos + p.len()..];
        }
    }
    true
}

pub fn write_source_record(dest: &Path, repo: &str, revision: &str, commit: &str, files: &[RemoteFile]) -> Result<()> {
    let path = dest.join(SOURCE_FILE);
    let mut rec: SourceRecord = flux_core::fsutil::read_json(&path).unwrap_or(SourceRecord {
        source: HfSource { repo: Some(repo.into()), revision: Some(revision.into()), commit: commit.into() },
        sha256: BTreeMap::new(),
    });
    ensure!(rec.source.commit == commit, "{} already holds files from commit {}", dest.display(), rec.source.commit);
    for f in files {
        if let Some(h) = &f.sha256 {
            rec.sha256.insert(Path::new(&f.path).file_name().unwrap().to_string_lossy().into(), h.clone());
        }
    }
    flux_core::fsutil::write_json_atomic(&path, &rec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob_match("*.gguf", "a/b.gguf"));
        assert!(glob_match("GLM-*/*-00001-of-*.gguf", "GLM-IQ3/x-00001-of-00002.gguf"));
        assert!(!glob_match("*q4_k_m*", "model-q8_0.gguf"));
        assert!(glob_match("config.json", "config.json"));
    }
}
