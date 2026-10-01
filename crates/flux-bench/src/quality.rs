//! Same-artifact quality: llama-perplexity saves reference logits on held-out text with a
//! reference placement, then measures the plan's placement against them.

use crate::stats::{quality_gate, Divergence, QualityGate};
use anyhow::{bail, Context, Result};
use flux_core::config::FluxConfig;
use flux_core::corpus::Corpus;
use flux_core::plan::{Plan, QualityProfile};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub plan: String,
    pub reference: String,
    pub chunks: u32,
    pub ppl_reference: Option<f64>,
    pub ppl_candidate: Option<f64>,
    pub ppl_ratio: Option<f64>,
    pub kl_mean: Option<f64>,
    pub top1_agreement: Option<f64>,
    /// Divergence of an accepted configuration from the same reference, when one was measured.
    pub accepted: Option<(String, Divergence)>,
    pub gate: QualityGate,
}

/// First number after `label` on its line.
fn number_after(out: &str, label: &str) -> Option<f64> {
    let line = out.lines().find(|l| l.trim_start().starts_with(label))?;
    line[line.find(':')? + 1..].split_whitespace().next()?.trim_end_matches('%').parse().ok()
}

async fn perplexity(cfg: &FluxConfig, args: &[String], log: &Path) -> Result<String> {
    let out = Command::new(cfg.llama_bin("llama-perplexity")).args(args).output().await.context("starting llama-perplexity")?;
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    std::fs::write(log, &text)?;
    if !out.status.success() {
        bail!("llama-perplexity failed ({}); see {}", out.status, log.display());
    }
    Ok(text)
}

/// Scores `chunks` of 512 held-out tokens. `reference` (llama.cpp tool flags) saves the trusted logits;
/// `accepted`, when given, is measured against them first to calibrate placement-only tolerances.
pub async fn check(
    cfg: &FluxConfig,
    plan: &Plan,
    reference: (&str, Vec<String>),
    accepted: Option<(&str, Vec<String>)>,
    corpus: &Corpus,
    chunks: u32,
    workdir: &Path,
) -> Result<QualityReport> {
    std::fs::create_dir_all(workdir)?;
    let base = workdir.join(format!("logits-{}.bin", &plan.key.model_identity[..12]));
    let with_text = |mut a: Vec<String>, extra: &[&str]| -> Vec<String> {
        a.extend(["--file".into(), corpus.heldout_path().display().to_string(), "--ctx-size".into(), "512".into(), "--chunks".into(), chunks.to_string()]);
        a.extend(["--kl-divergence-base".into(), base.display().to_string()]);
        a.extend(extra.iter().map(|s| s.to_string()));
        a
    };
    perplexity(cfg, &with_text(reference.1, &[]), &workdir.join("reference.log")).await?;
    let divergence = |out: &str| Divergence {
        ppl_ratio: number_after(out, "Mean PPL(Q)/PPL(base)"),
        kl_mean: number_after(out, "Mean    KLD"),
        top1_agreement: number_after(out, "Same top p").map(|p| p / 100.0),
    };
    let accepted = match accepted {
        Some((label, args)) => {
            let out = perplexity(cfg, &with_text(args, &["--kl-divergence"]), &workdir.join("accepted.log")).await?;
            Some((label.to_string(), divergence(&out)))
        }
        None => None,
    };
    let out = perplexity(cfg, &with_text(plan.backend_params().tool_args(), &["--kl-divergence"]), &workdir.join("candidate.log")).await?;
    let _ = std::fs::remove_file(&base);
    let d = divergence(&out);
    let exact = matches!(plan.quality, QualityProfile::Exact);
    Ok(QualityReport {
        plan: plan.id.clone(),
        reference: reference.0.into(),
        chunks,
        ppl_reference: number_after(&out, "Mean PPL(base)"),
        ppl_candidate: number_after(&out, "Mean PPL(Q)"),
        ppl_ratio: d.ppl_ratio,
        kl_mean: d.kl_mean,
        top1_agreement: d.top1_agreement,
        gate: quality_gate(exact, d.ppl_ratio, d.kl_mean, d.top1_agreement, None, accepted.as_ref().map(|a| a.1)),
        accepted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_perplexity_summary() {
        let out = "Mean PPL(Q)                   :  10.123456 ±   0.1\nMean PPL(Q)/PPL(base)         :   1.000200 ±   0.0001\nMean    KLD:   0.000123 ±   0.00001\nSame top p: 99.512 ±  0.1 %\n";
        assert_eq!(number_after(out, "Mean PPL(Q)/PPL(base)"), Some(1.0002));
        assert_eq!(number_after(out, "Mean PPL(Q)"), Some(10.123456));
        assert_eq!(number_after(out, "Mean    KLD"), Some(0.000123));
        assert_eq!(number_after(out, "Same top p"), Some(99.512));
    }
}
