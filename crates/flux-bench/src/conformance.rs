//! Behavioural conformance against the reference engine: the same plan on the native worker and on
//! llama-server must render chat templates, tokenize special tokens, sample and stop identically.
//! A numerically correct engine can still answer wrongly through prompting or end-of-sequence handling.

use anyhow::Result;
use flux_core::corpus::{Corpus, Role};
use flux_core::plan::{EngineKind, Plan};
use flux_core::protocol::{Event, FinishReason, Sampling};
use flux_core::worker::Worker;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;

/// Leading greedy tokens that must agree: enough to show both engines saw the same prompt.
const FIRST_TOKENS: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Check {
    pub name: String,
    pub pass: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
struct Observed {
    templates: Vec<String>,
    tokens: Vec<Vec<i32>>,
    greedy: Vec<Vec<i32>>,
    /// The same greedy runs again in the same process: how reproducible the engine is by itself.
    greedy_repeat: Vec<Vec<i32>>,
    sampled: Vec<i32>,
    stopped: (String, Option<FinishReason>),
}

fn conversations() -> Vec<(&'static str, Value, Option<Value>)> {
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]);
    vec![
        ("single user turn", json!([{"role": "user", "content": "Hello"}]), None),
        (
            "system and multi-turn",
            json!([{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Name a prime."}, {"role": "assistant", "content": "7"}, {"role": "user", "content": "Another?"}]),
            None,
        ),
        ("tool definitions", json!([{"role": "user", "content": "Weather in Lima?"}]), Some(tools)),
        ("non-ASCII text", json!([{"role": "user", "content": "¿Qué tal? 日本語 🚀 naïve café"}]), None),
    ]
}

async fn collect(w: &Worker, req: &str, prompt: Vec<i32>, sampling: Sampling, stop: Vec<String>, n: u32) -> Result<(Vec<i32>, String, Option<FinishReason>)> {
    let mut rx = w.start(req, prompt, sampling, stop, n, vec![]).await?;
    w.credit(req, n).await?;
    let (mut toks, mut text, mut fin) = (vec![], String::new(), None);
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Token { token, text: t, .. } => {
                toks.push(token);
                text.push_str(&t);
            }
            Event::Error { message, .. } => anyhow::bail!("{message}"),
            Event::Finished { reason, tail, .. } => {
                text.push_str(&tail);
                fin = Some(reason);
                break;
            }
            _ => {}
        }
    }
    Ok((toks, text, fin))
}

async fn observe(plan: &Plan, engine: EngineKind, corpus: &Corpus, log: &Path) -> Result<Observed> {
    let mut p = plan.clone();
    p.engine = engine.clone();
    p.runtime.speculation = None;
    let w = Worker::spawn(&engine, Some(log)).await?;
    let out = async {
        w.load(&p).await?;
        let mut o = Observed::default();
        for (_, messages, tools) in conversations() {
            let t = w.apply_template(messages, tools).await?;
            o.tokens.push(w.tokenize(&t.prompt, true).await?);
            o.templates.push(t.prompt);
        }
        let greedy = Sampling { temperature: Some(0.0), ..Default::default() };
        for i in 0..2 {
            let prompt = flux_plan::run::corpus_prompt(&w, corpus, Role::HeldOut, 20 + i, 128).await?;
            o.greedy.push(collect(&w, &format!("g{i}"), prompt.clone(), greedy.clone(), vec![], 32).await?.0);
            o.greedy_repeat.push(collect(&w, &format!("r{i}"), prompt, greedy.clone(), vec![], 32).await?.0);
        }
        let prompt = flux_plan::run::corpus_prompt(&w, corpus, Role::HeldOut, 30, 128).await?;
        let seeded = Sampling { temperature: Some(0.8), top_k: Some(40), top_p: Some(0.95), seed: Some(42), ..Default::default() };
        o.sampled = collect(&w, "s", prompt.clone(), seeded, vec![], 32).await?.0;
        let (_, text, fin) = collect(&w, "stop", o.tokens[0].clone(), greedy, vec![".".into(), "\n".into()], 64).await?;
        o.stopped = (text, fin);
        Ok::<Observed, anyhow::Error>(o)
    }
    .await;
    w.shutdown().await;
    out
}

/// Runs the plan's placement on both engines and compares what each one renders, tokenizes and emits.
pub async fn check(plan: &Plan, corpus: &Corpus, logdir: &Path) -> Result<Vec<Check>> {
    std::fs::create_dir_all(logdir)?;
    let native = observe(plan, EngineKind::Native, corpus, &logdir.join("conformance-native.log")).await?;
    let reference = observe(plan, EngineKind::LlamaServer, corpus, &logdir.join("conformance-llama-server.log")).await?;
    let mut checks = vec![];
    for (i, (name, _, _)) in conversations().iter().enumerate() {
        let same = native.templates[i] == reference.templates[i];
        checks.push(Check {
            name: format!("chat template: {name}"),
            pass: same,
            detail: if same {
                format!("{} characters identical", native.templates[i].len())
            } else {
                first_difference(&native.templates[i], &reference.templates[i])
            },
        });
        let same = native.tokens[i] == reference.tokens[i];
        checks.push(Check {
            name: format!("tokenization with special tokens: {name}"),
            pass: same,
            detail: format!("{} vs {} tokens", native.tokens[i].len(), reference.tokens[i].len()),
        });
    }
    let agree = |a: &[i32], b: &[i32]| a.iter().zip(b).take_while(|(x, y)| x == y).count();
    for (i, (a, b)) in native.greedy.iter().zip(&reference.greedy).enumerate() {
        let cross = agree(a, b);
        let own = agree(a, &native.greedy_repeat[i]).min(agree(b, &reference.greedy_repeat[i]));
        let n = a.len().min(b.len());
        // Disagreement on the first tokens means the engines saw different prompts; later divergence is
        // numerical (different kernels or cache layouts) and is judged by the quality gate instead.
        let pass = cross >= FIRST_TOKENS.min(n);
        let detail = match (cross == n, own < n) {
            (true, _) => format!("{n} of {n} tokens identical"),
            (false, true) => format!("first {cross} identical; the engines also diverge from their own repeat at token {own} (run-to-run numerics)"),
            (false, false) => format!("first {cross} identical; both engines are deterministic, so the later divergence is a numerical path difference"),
        };
        checks.push(Check { name: format!("greedy continuation {i}"), pass, detail });
    }
    let k = native.sampled.iter().zip(&reference.sampled).take_while(|(x, y)| x == y).count();
    checks.push(Check {
        name: "seeded sampling (temperature 0.8, top-k 40, top-p 0.95)".into(),
        pass: native.sampled == reference.sampled,
        detail: format!("first {k} of {} tokens identical", native.sampled.len().min(reference.sampled.len())),
    });
    checks.push(Check {
        name: "stop strings".into(),
        pass: native.stopped == reference.stopped,
        detail: format!("native {:?} / reference {:?}", native.stopped, reference.stopped),
    });
    Ok(checks)
}

fn first_difference(a: &str, b: &str) -> String {
    let i = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    let ctx = |s: &str| s.chars().skip(i.saturating_sub(20)).take(50).collect::<String>();
    format!("differs at character {i}: {:?} vs {:?}", ctx(a), ctx(b))
}
