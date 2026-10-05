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

    // Identify both the inputs and the linked artifacts, including uncommitted backend edits.
    let patches = root.join("patches/llama.cpp");
    println!("cargo:rerun-if-changed={}", patches.display());
    let mut files: Vec<PathBuf> = std::fs::read_dir(&patches).map(|rd| rd.flatten().map(|e| e.path()).collect()).unwrap_or_default();
    files.sort();
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(std::fs::read(PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("libflux_native.a")).expect("compiled bridge"));
    h.update(pin.as_bytes());
    let diff = Command::new("git").arg("-C").arg(&llama).args(["diff", "HEAD", "--binary"]).output().expect("backend diff");
    assert!(diff.status.success(), "cannot identify backend working tree");
    h.update(&diff.stdout);
    for dir in [root.join("crates/flux-native/native"), root.join("crates/flux-native/src"), root.join("crates/flux-worker/src")] {
        println!("cargo:rerun-if-changed={}", dir.display());
        let mut inputs: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_file()).collect();
        inputs.sort();
        for p in inputs {
            h.update(p.file_name().unwrap().as_encoded_bytes());
            h.update(std::fs::read(p).unwrap());
        }
    }
    for p in [llama.join("build/CMakeCache.txt"), root.join("scripts/build-backend.sh")] {
        println!("cargo:rerun-if-changed={}", p.display());
        h.update(std::fs::read(p).expect("backend build input"));
    }
    for dir in ["src", "ggml/src", "common", "include", "ggml/include"] {
        println!("cargo:rerun-if-changed={}", llama.join(dir).display());
    }
    let mut libs: Vec<_> = std::fs::read_dir(&lib_dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|e| e == "so")).collect();
    let server = lib_dir.join("llama-server");
    println!("cargo:rustc-env=FLUX_BACKEND_SERVER={}", server.display());
    println!("cargo:rerun-if-changed={}", server.display());
    if server.is_file() {
        libs.push(server);
    }
    libs.sort();
    let mut manifest = String::new();
    for p in libs {
        println!("cargo:rerun-if-changed={}", p.display());
        let bytes = std::fs::read(&p).expect("backend library");
        let digest = format!("{:x}", sha2::Sha256::digest(bytes));
        manifest.push_str(&format!("{} {digest}\n", p.file_name().unwrap().to_string_lossy()));
    }
    h.update(manifest.as_bytes());
    let manifest_path = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("backend-libraries.txt");
    std::fs::write(&manifest_path, manifest).unwrap();
    println!("cargo:rustc-env=FLUX_BACKEND_LIBRARIES={}", manifest_path.display());
    for f in &files {
        h.update(std::fs::read(f).unwrap_or_default());
    }
    let digest: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let build = format!("{}+{}", &pin[..12], &digest[..24]);
    println!("cargo:rustc-env=FLUX_BACKEND_BUILD={build}");
}
