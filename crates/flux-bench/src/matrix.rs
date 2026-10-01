//! Flexibility gate: one row per architecture, encodings, engine and hardware topology, with every
//! result Flux has recorded for it. Missing evidence is shown as missing, never as a pass.

use crate::quality::QualityReport;
use crate::soak::SoakReport;
use crate::suite::SuiteReport;
use flux_core::config::FluxConfig;
use flux_core::fsutil::read_json;
use flux_core::plan::Plan;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub architecture: String,
    pub model: String,
    pub engine: String,
    pub topology: String,
    pub plan: String,
    pub placement: String,
    pub validated_tps: Option<f64>,
    pub speed_gate: Option<bool>,
    pub quality: Option<bool>,
    pub conformance: Option<bool>,
    pub soak: Option<bool>,
}

fn json_files(dir: &Path) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(dir).map(|rd| rd.flatten().map(|e| e.path()).collect()).unwrap_or_default()
}

pub fn collect(cfg: &FluxConfig, extra_bench_dirs: &[&Path]) -> Vec<Row> {
    let plans = flux_plan::store::PlanStore::new(&cfg.plans_dir()).list();
    let mut suites: Vec<SuiteReport> = vec![];
    for root in std::iter::once(cfg.data_dir.join("bench").as_path()).chain(extra_bench_dirs.iter().copied()) {
        suites.extend(json_files(root).iter().filter_map(|d| read_json::<SuiteReport>(&d.join("report.json")).ok()));
    }
    let soaks: Vec<(String, SoakReport)> = json_files(&cfg.data_dir.join("soak"))
        .iter()
        .filter_map(|d| Some((d.file_name()?.to_str()?.split('-').next()?.to_string(), read_json(&d.join("soak.json")).ok()?)))
        .collect();
    plans
        .iter()
        .map(|p: &Plan| {
            let v = p.validation.as_ref();
            let validated_tps = v.and_then(|v| v.candidates.iter().find(|c| c.label == v.chosen)).and_then(|c| c.validation.as_ref()).map(|m| m.decode_tps.p50);
            let speed_gate = suites.iter().filter(|s| s.plans.iter().any(|x| x.id == p.id)).map(|s| s.speed_gate.pass).reduce(|a, b| a && b);
            let quality = read_json::<QualityReport>(&cfg.data_dir.join("quality").join(format!("{}.json", p.id))).ok().map(|q| q.gate.pass);
            let conformance = read_json::<Vec<crate::conformance::Check>>(&cfg.data_dir.join("conformance").join(format!("{}.json", p.id)))
                .ok()
                .map(|c| c.iter().all(|x| x.pass));
            let soak = soaks.iter().filter(|(id, _)| id == &p.id).map(|(_, r)| r.pass).reduce(|a, b| a && b);
            Row {
                architecture: p.architecture.clone(),
                model: p.source_files()[0].file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default(),
                engine: p.engine.to_string(),
                topology: p.key.topology.clone(),
                plan: p.id.clone(),
                placement: p.placement.describe(),
                validated_tps,
                speed_gate,
                quality,
                conformance,
                soak,
            }
        })
        .collect()
}

pub fn render(rows: &[Row]) -> String {
    let mark = |b: Option<bool>| match b {
        Some(true) => "pass",
        Some(false) => "FAIL",
        None => "—",
    };
    let mut s = format!(
        "{:<10} {:<46} {:<12} {:<17} {:>8} {:<6} {:<7} {:<11} {:<5}\n",
        "arch", "model", "engine", "plan", "tok/s", "speed", "quality", "conformance", "soak"
    );
    for r in rows {
        s.push_str(&format!(
            "{:<10} {:<46} {:<12} {:<17} {:>8} {:<6} {:<7} {:<11} {:<5}\n",
            r.architecture,
            r.model.chars().take(46).collect::<String>(),
            r.engine,
            r.plan,
            r.validated_tps.map_or("—".into(), |t| format!("{t:.1}")),
            mark(r.speed_gate),
            mark(r.quality),
            mark(r.conformance),
            mark(r.soak)
        ));
    }
    s
}
