pub mod backend;
pub mod config;
pub mod corpus;
pub mod fsutil;
pub mod ggml_type;
pub mod hardware;
pub mod model;
pub mod plan;
pub mod protocol;
pub mod stats;
pub mod worker;

pub const GIB: u64 = 1 << 30;
pub const MIB: u64 = 1 << 20;

/// `12.3 GiB`, `512.0 MiB`, `640 B`.
pub fn fmt_bytes(b: u64) -> String {
    match b {
        b if b >= GIB => format!("{:.2} GiB", b as f64 / GIB as f64),
        b if b >= MIB => format!("{:.1} MiB", b as f64 / MIB as f64),
        b => format!("{b} B"),
    }
}
