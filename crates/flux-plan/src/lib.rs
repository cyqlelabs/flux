//! flux-plan: feasibility filter, measured cost model, placement search, validation and plan store.

pub mod cost;
pub mod experts;
pub mod layers;
pub mod params;
pub mod planner;
pub mod run;
pub mod search;
pub mod store;

#[cfg(test)]
pub(crate) mod tests {
    use flux_core::hardware::{CpuInfo, HardwareInventory, MemoryInfo};

    pub fn inventory() -> HardwareInventory {
        HardwareInventory {
            hostname: "test".into(),
            kernel: "6".into(),
            cpu: CpuInfo { model: "cpu".into(), vendor: "x".into(), sockets: 1, cores: 12, threads: 24, isa: vec!["avx2".into()], max_mhz: None },
            memory: MemoryInfo {
                total: 64 << 30,
                available: 56 << 30,
                swap_total: 0,
                swap_free: 0,
                hugepage_size: 2 << 20,
                hugepages_free: 0,
                thp: "madvise".into(),
            },
            numa: vec![],
            gpus: vec![],
            driver_version: None,
            cuda_driver_version: None,
            storage: vec![],
            backend_devices: vec![],
        }
    }
}
