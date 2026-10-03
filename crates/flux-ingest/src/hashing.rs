//! File hashes for the model manifest, cached by (path, size, mtime) so large artifacts are read once.

use anyhow::{Context, Result};
use flux_core::fsutil::{read_json, write_json_atomic};
use flux_core::model::ModelFile;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    size: u64,
    mtime: i64,
    sha256: String,
}

pub struct HashIndex {
    path: PathBuf,
    entries: BTreeMap<String, Entry>,
}

impl HashIndex {
    pub fn open(cache_dir: &Path) -> HashIndex {
        let path = cache_dir.join("hash-index.json");
        let entries = read_json(&path).unwrap_or_default();
        HashIndex { path, entries }
    }

    fn save(&self) -> Result<()> {
        write_json_atomic(&self.path, &self.entries)
    }

    pub fn lookup(&self, f: &ModelFile) -> Option<String> {
        let e = self.entries.get(&f.path.to_string_lossy().to_string())?;
        (e.size == f.size && e.mtime == f.mtime).then(|| e.sha256.clone())
    }

    /// Records a hash obtained elsewhere, e.g. verified during download.
    pub fn record(&mut self, f: &ModelFile, sha256: &str) -> Result<()> {
        self.entries.insert(f.path.to_string_lossy().to_string(), Entry { size: f.size, mtime: f.mtime, sha256: sha256.to_string() });
        self.save()
    }

    /// Fills `sha256` for every file, hashing uncached files in parallel and logging each one's progress.
    pub fn hash_all(&mut self, files: &mut [ModelFile]) -> Result<()> {
        let todo: Vec<usize> = (0..files.len()).filter(|&i| self.lookup(&files[i]).is_none()).collect();
        let shared = &*files;
        let results: Vec<(usize, Result<String>)> = std::thread::scope(|s| {
            let handles: Vec<_> = todo.iter().map(|&i| (i, s.spawn(move || hash_logged(&shared[i])))).collect();
            handles.into_iter().map(|(i, h)| (i, h.join().expect("hash thread panicked"))).collect()
        });
        for (i, r) in results {
            let sha = r?;
            self.entries.insert(files[i].path.to_string_lossy().to_string(), Entry { size: files[i].size, mtime: files[i].mtime, sha256: sha });
        }
        if !todo.is_empty() {
            self.save()?;
        }
        for f in files.iter_mut() {
            f.sha256 = self.lookup(f);
        }
        Ok(())
    }
}

/// Hashing a multi-GB model takes minutes, so report every 10%.
fn hash_logged(f: &ModelFile) -> Result<String> {
    let name = f.path.file_name().unwrap_or_default().to_string_lossy();
    tracing::info!("hashing {name} ({}); later runs reuse the hash", flux_core::fmt_bytes(f.size));
    let shown = Cell::new(0);
    sha256_file(&f.path, &|done| {
        let pct = done * 100 / f.size.max(1);
        if pct >= shown.get() + 10 {
            shown.set(pct);
            tracing::info!("hashing {name}: {pct}%");
        }
    })
}

/// `progress` receives the bytes hashed so far.
pub fn sha256_file(path: &Path, progress: &dyn Fn(u64)) -> Result<String> {
    let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut done = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        done += n as u64;
        progress(done);
    }
    Ok(hex::encode(h.finalize()))
}

pub fn model_file(path: &Path) -> Result<ModelFile> {
    let path = path.canonicalize().with_context(|| format!("resolving {}", path.display()))?;
    let meta = std::fs::metadata(&path)?;
    let mtime = meta.modified()?.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    Ok(ModelFile { path, size: meta.len(), mtime, sha256: None })
}

/// Content identity of an ordered file set.
pub fn identity(files: &[ModelFile]) -> Option<String> {
    let mut h = Sha256::new();
    for f in files {
        h.update(f.sha256.as_ref()?.as_bytes());
    }
    Some(hex::encode(h.finalize())[..32].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_hash_is_reused_until_file_changes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.bin");
        std::fs::write(&p, b"abc").unwrap();
        let mut idx = HashIndex::open(dir.path());
        let mut files = vec![model_file(&p).unwrap()];
        idx.hash_all(&mut files).unwrap();
        assert_eq!(files[0].sha256.as_deref(), Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
        let idx2 = HashIndex::open(dir.path());
        assert!(idx2.lookup(&files[0]).is_some());
        std::fs::write(&p, b"abcd").unwrap();
        assert!(idx2.lookup(&model_file(&p).unwrap()).is_none());
    }
}
