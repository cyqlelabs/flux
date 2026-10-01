//! Compiles the C++ bridge against the pinned llama.cpp and links its shared libraries.
//! FLUX_LLAMA_DIR may point at another checkout, but only one whose HEAD equals `backend.pin`.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    let pin = std::fs::read_to_string(root.join("backend.pin")).expect("backend.pin").trim().to_string();
    let llama = std::env::var_os("FLUX_LLAMA_DIR").map(PathBuf::from).unwrap_or_else(|| root.join("third_party/llama.cpp"));
    let llama = llama.canonicalize().expect("llama.cpp checkout not found; run `git submodule update --init`");
    println!("cargo:rerun-if-env-changed=FLUX_LLAMA_DIR");
    println!("cargo:rerun-if-changed=native/flux_native.cpp");
    println!("cargo:rerun-if-changed=native/flux_native.h");
    println!("cargo:rerun-if-changed={}", root.join("backend.pin").display());

    let head = Command::new("git").arg("-C").arg(&llama).args(["rev-parse", "HEAD"]).output().expect("git");
    let head = String::from_utf8_lossy(&head.stdout).trim().to_string();
    assert_eq!(head, pin, "{} is at {head}, backend.pin requires {pin}", llama.display());

    let lib_dir = llama.join("build/bin");
    assert!(lib_dir.join("libllama.so").exists(), "{} has no libllama.so; run scripts/build-backend.sh", lib_dir.display());

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("native/flux_native.cpp")
        .include("native")
        .include(llama.join("include"))
        .include(llama.join("ggml/include"))
        .include(llama.join("common"))
        .include(llama.join("src"))
        .include(llama.join("vendor"))
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-function")
        .opt_level(2)
        .compile("flux_native");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    for lib in ["llama-common", "llama", "ggml", "ggml-base"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // This crate's tests get an rpath directly; dependents read `lib_dir` and add their own.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    println!("cargo:lib_dir={}", lib_dir.display());
    println!("cargo:rustc-env=FLUX_BACKEND_PIN={pin}");

    // Build identity: the pin plus the Flux patches applied on top of it.
    let patches = root.join("patches/llama.cpp");
    println!("cargo:rerun-if-changed={}", patches.display());
    let mut files: Vec<PathBuf> = std::fs::read_dir(&patches).map(|rd| rd.flatten().map(|e| e.path()).collect()).unwrap_or_default();
    files.sort();
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    for f in &files {
        h.update(std::fs::read(f).unwrap_or_default());
    }
    let digest: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let build = if files.is_empty() { pin[..12].to_string() } else { format!("{}+{}", &pin[..12], &digest[..12]) };
    println!("cargo:rustc-env=FLUX_BACKEND_BUILD={build}");
}
