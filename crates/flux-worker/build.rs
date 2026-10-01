fn main() {
    // Find the pinned backend libraries at run time without LD_LIBRARY_PATH. Release packages set
    // FLUX_RPATH='$ORIGIN/../lib'; DT_RPATH (not RUNPATH) so the backend's own dependencies resolve too.
    println!("cargo:rerun-if-env-changed=FLUX_RPATH");
    let dir = std::env::var("FLUX_RPATH").unwrap_or_else(|_| std::env::var("DEP_FLUX_NATIVE_LIB_DIR").expect("flux-native exports lib_dir"));
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags,-rpath,{dir}");
}
