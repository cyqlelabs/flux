//! Runs the maintained regression suite after remediation.
fn main() -> std::process::ExitCode {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let status = std::process::Command::new("cargo")
        .current_dir(root)
        .args(["test", "--release", "--workspace", "--offline", "--", "--test-threads=2"])
        .status()
        .expect("run Cargo regression suite");
    if status.success() {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}
