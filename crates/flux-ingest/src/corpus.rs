//! Fetches the WikiText-2 (raw) prompt corpus that llama.cpp's own perplexity scripts use.

use anyhow::{Context, Result};
use flux_core::corpus::{TEST_FILE, VALID_FILE};
use std::io::Read;
use std::path::Path;

const URL: &str = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip";

pub async fn fetch(dir: &Path) -> Result<()> {
    if dir.join(VALID_FILE).exists() && dir.join(TEST_FILE).exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    let bytes = reqwest::get(URL).await?.error_for_status().context("downloading WikiText-2")?.bytes().await?;
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).context("reading WikiText-2 archive")?;
    for name in [VALID_FILE, TEST_FILE] {
        let entry = (0..zip.len())
            .find(|&i| zip.by_index(i).map(|f| f.name().ends_with(name)).unwrap_or(false))
            .with_context(|| format!("{name} missing from archive"))?;
        let mut text = String::new();
        zip.by_index(entry)?.read_to_string(&mut text)?;
        flux_core::fsutil::write_atomic(&dir.join(name), text.as_bytes())?;
    }
    Ok(())
}
