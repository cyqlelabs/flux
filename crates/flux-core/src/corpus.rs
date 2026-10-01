//! Prompt sources with disjoint roles: the planner tunes on `Calibration`, races finalists on
//! `Validation`, and benchmarks and quality checks use `HeldOut`, which tuning never sees.
//! Built from WikiText-2 (raw): validation split → calibration + validation halves, test split → held-out.

use anyhow::{ensure, Context, Result};
use std::path::{Path, PathBuf};

pub const VALID_FILE: &str = "wiki.valid.raw";
pub const TEST_FILE: &str = "wiki.test.raw";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Calibration,
    Validation,
    HeldOut,
}

pub struct Corpus {
    pub dir: PathBuf,
    calibration: Vec<String>,
    validation: Vec<String>,
    heldout: Vec<String>,
}

/// Splits WikiText into articles at top-level ` = Title = ` headings.
fn articles(text: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![];
    for line in text.lines() {
        let t = line.trim();
        let top = t.starts_with("= ") && t.ends_with(" =") && !t.starts_with("= =");
        if top || out.is_empty() {
            out.push(String::new());
        }
        let cur = out.last_mut().unwrap();
        cur.push_str(line);
        cur.push('\n');
    }
    out.retain(|a| a.trim().len() > 200);
    out
}

impl Corpus {
    pub fn dir_in(cache_dir: &Path) -> PathBuf {
        cache_dir.join("corpora/wikitext-2-raw")
    }

    pub fn load(dir: &Path) -> Result<Corpus> {
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).with_context(|| format!("reading {} (run `flux corpus`)", dir.join(f).display()));
        let valid = articles(&read(VALID_FILE)?);
        let test = articles(&read(TEST_FILE)?);
        ensure!(valid.len() >= 4 && !test.is_empty(), "corpus at {} is too small", dir.display());
        let half = valid.len() / 2;
        Ok(Corpus { dir: dir.to_path_buf(), calibration: valid[..half].to_vec(), validation: valid[half..].to_vec(), heldout: test })
    }

    fn set(&self, role: Role) -> &[String] {
        match role {
            Role::Calibration => &self.calibration,
            Role::Validation => &self.validation,
            Role::HeldOut => &self.heldout,
        }
    }

    /// Text of at least `min_chars`, made of consecutive articles starting at article `index`
    /// (wrapping around). The caller tokenizes and truncates to an exact token count.
    pub fn text(&self, role: Role, index: usize, min_chars: usize) -> String {
        let set = self.set(role);
        let mut out = String::new();
        let mut i = index;
        while out.len() < min_chars {
            out.push_str(&set[i % set.len()]);
            i += 1;
        }
        out
    }

    /// The whole held-out split, for perplexity.
    pub fn heldout_text(&self) -> String {
        self.heldout.concat()
    }

    pub fn heldout_path(&self) -> PathBuf {
        self.dir.join(TEST_FILE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_are_disjoint_and_text_is_long_enough() {
        let dir = tempfile::tempdir().unwrap();
        let body = |n: usize, tag: &str| (0..n).map(|i| format!(" = {tag} {i} = \n{}\n", "word ".repeat(80))).collect::<String>();
        std::fs::write(dir.path().join(VALID_FILE), body(6, "V")).unwrap();
        std::fs::write(dir.path().join(TEST_FILE), body(3, "T")).unwrap();
        let c = Corpus::load(dir.path()).unwrap();
        let cal = c.text(Role::Calibration, 0, 1000);
        let val = c.text(Role::Validation, 0, 1000);
        assert!(cal.contains("V 0") && !cal.contains("V 3"));
        assert!(val.contains("V 3") && !val.contains("V 0"));
        assert!(c.text(Role::HeldOut, 0, 2000).len() >= 2000);
        assert!(!c.heldout_text().contains(" = V"));
    }
}
