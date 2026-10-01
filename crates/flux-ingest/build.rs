//! Extracts the architectures the pinned llama.cpp can execute from its source,
//! so the compatibility check moves with the pin instead of a hand-kept list.

use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let arch_src = root.join("third_party/llama.cpp/src/llama-arch.cpp");
    let pin_file = root.join("backend.pin");
    println!("cargo:rerun-if-changed={}", arch_src.display());
    println!("cargo:rerun-if-changed={}", pin_file.display());

    let src = std::fs::read_to_string(&arch_src).expect("third_party/llama.cpp missing: run `git submodule update --init`");
    let table = src.split("LLM_ARCH_NAMES").nth(1).and_then(|s| s.split("};").next()).expect("LLM_ARCH_NAMES table");
    let mut names: Vec<String> = table
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            let rest = l.strip_prefix("{ LLM_ARCH_")?;
            let q = rest.find('"')?;
            let end = rest[q + 1..].find('"')?;
            Some(rest[q + 1..q + 1 + end].to_string())
        })
        .filter(|n| n != "clip" && n != "(unknown)")
        .collect();
    names.sort();
    assert!(names.len() > 50, "parsed only {} architectures", names.len());

    // Hugging Face architecture classes the pinned converter can turn into GGUF.
    let conv_dir = root.join("third_party/llama.cpp/conversion");
    println!("cargo:rerun-if-changed={}", conv_dir.display());
    let mut convertible: Vec<String> = vec![];
    for entry in std::fs::read_dir(&conv_dir).expect("conversion/ directory") {
        let src = std::fs::read_to_string(entry.unwrap().path()).unwrap_or_default();
        for call in src.split("register(").skip(1) {
            let args = call.split(')').next().unwrap_or("");
            convertible.extend(args.split(',').filter_map(|a| Some(a.trim().strip_prefix('"')?.strip_suffix('"')?.to_string())));
        }
    }
    convertible.sort();
    convertible.dedup();

    let pin = std::fs::read_to_string(&pin_file).expect("backend.pin").trim().to_string();
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("backend_archs.rs");
    std::fs::write(
        out,
        format!(
            "pub const BACKEND_PIN: &str = {pin:?};\npub const BACKEND_ARCHITECTURES: &[&str] = &{names:?};\npub const CONVERTIBLE_HF_ARCHITECTURES: &[&str] = &{convertible:?};\n"
        ),
    )
    .unwrap();
}
