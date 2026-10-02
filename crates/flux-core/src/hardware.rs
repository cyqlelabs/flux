//! Host inventory and measured hardware distributions.

use crate::stats::Summary;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub vendor: String,
    pub sockets: u32,
    pub cores: u32,
    pub threads: u32,
    /// SIMD and related features detected on this CPU, never inferred from the model name.
    pub isa: Vec<String>,
    pub max_mhz: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total: u64,
    pub available: u64,
    pub swap_total: u64,
    pub swap_free: u64,
    pub hugepage_size: u64,
    pub hugepages_free: u64,
    pub thp: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumaNode {
    pub id: u32,
    pub cpus: Vec<u32>,
    pub mem_total: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub nvml_index: u32,
    pub name: String,
    pub uuid: String,
    /// Lower-case `0000:06:00.0` form.
    pub pci_bus_id: String,
    pub mem_total: u64,
    pub mem_free: u64,
    pub compute_capability: (i32, i32),
    pub pcie_gen_current: Option<u32>,
    pub pcie_gen_max: Option<u32>,
    pub pcie_width_current: Option<u32>,
    pub pcie_width_max: Option<u32>,
    pub power_limit_w: Option<f64>,
    pub sm_clock_max_mhz: Option<u32>,
    pub mem_clock_max_mhz: Option<u32>,
    pub numa_node: Option<i32>,
    /// Other processes holding device memory (desktop, other services) at probe time.
    pub foreign_processes: Vec<(u32, u64)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageInfo {
    pub mount: String,
    pub device: String,
    pub model: Option<String>,
    pub transport: String,
    pub rotational: bool,
    pub size: u64,
}

/// A ggml backend device as the pinned backend enumerates it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendDevice {
    pub name: String,
    pub description: String,
    /// `cpu`, `gpu`, `igpu` or `accel`.
    pub kind: String,
    pub mem_total: u64,
    pub mem_free: u64,
    pub pci_bus_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareInventory {
    pub hostname: String,
    pub kernel: String,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub numa: Vec<NumaNode>,
    pub gpus: Vec<GpuInfo>,
    pub driver_version: Option<String>,
    pub cuda_driver_version: Option<i32>,
    pub storage: Vec<StorageInfo>,
    pub backend_devices: Vec<BackendDevice>,
}

impl HardwareInventory {
    /// Stable description of the topology: what is installed and how it is connected, not what is free now.
    pub fn topology_fingerprint(&self) -> String {
        let mut parts = vec![
            format!("cpu={}x{}c{}t", self.cpu.model, self.cpu.cores, self.cpu.threads),
            format!("isa={}", self.cpu.isa.join(",")),
            format!("ram={}", self.memory.total >> 30),
            format!("numa={}", self.numa.len()),
        ];
        for g in &self.gpus {
            parts.push(format!(
                "gpu={}@{}:{}MiB:pcie{}x{}",
                g.name,
                g.pci_bus_id,
                g.mem_total >> 20,
                g.pcie_gen_max.unwrap_or(0),
                g.pcie_width_max.unwrap_or(0)
            ));
        }
        // The devices the backend can use: masking GPUs (e.g. CUDA_VISIBLE_DEVICES) is another topology.
        for d in &self.backend_devices {
            parts.push(format!("dev={}@{}", d.name, d.pci_bus_id.as_deref().unwrap_or("")));
        }
        crate::fsutil::sha256_hex(parts.join("|").as_bytes())[..16].to_string()
    }

    pub fn gpu_by_bus(&self, bus: &str) -> Option<&GpuInfo> {
        self.gpus.iter().find(|g| g.pci_bus_id.eq_ignore_ascii_case(bus))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CopyDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

/// Measured bandwidth for one payload size, in GB/s (10^9 bytes per second).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CopyPoint {
    pub bytes: u64,
    pub gbps: Summary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CopyCurve {
    pub device: String,
    pub peer: Option<String>,
    pub direction: CopyDirection,
    pub pinned: bool,
    pub points: Vec<CopyPoint>,
}

impl CopyCurve {
    /// Interpolated median seconds to move `bytes`.
    pub fn seconds_for(&self, bytes: u64) -> f64 {
        let pts = &self.points;
        let gbps = match pts.iter().position(|p| p.bytes >= bytes) {
            Some(0) => pts[0].gbps.p50,
            Some(i) => {
                let (a, b) = (&pts[i - 1], &pts[i]);
                let t = (bytes - a.bytes) as f64 / (b.bytes - a.bytes) as f64;
                a.gbps.p50 + t * (b.gbps.p50 - a.gbps.p50)
            }
            None => pts.last().map_or(1.0, |p| p.gbps.p50),
        };
        bytes as f64 / (gbps.max(1e-6) * 1e9)
    }
}

/// Bandwidth of one activity alone and while another runs, to expose shared-link contention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentionProbe {
    pub scenario: String,
    pub alone_gbps: f64,
    pub contended_gbps: f64,
}

/// Measured time of one weight-matrix product shape on one device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelProbe {
    pub device: String,
    pub ggml_type: String,
    /// Weight is `k x n`; activations are `k x batch`.
    pub k: u64,
    pub n: u64,
    pub batch: u64,
    pub threads: u32,
    pub micros: Summary,
}

impl KernelProbe {
    /// Effective weight-read bandwidth implied by the median time, GB/s.
    pub fn weight_gbps(&self, weight_bytes: u64) -> f64 {
        weight_bytes as f64 / (self.micros.p50 * 1e3)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CpuBandwidth {
    pub threads: u32,
    pub gbps: Summary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StorageProbe {
    pub path: String,
    pub direct_io: bool,
    pub block_bytes: u64,
    pub sequential: bool,
    pub gbps: Summary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbeReport {
    pub schema: u32,
    pub topology: String,
    pub backend_revision: String,
    pub created: chrono::DateTime<chrono::Utc>,
    pub inventory: HardwareInventory,
    pub copies: Vec<CopyCurve>,
    pub contention: Vec<ContentionProbe>,
    pub kernels: Vec<KernelProbe>,
    pub cpu_bandwidth: Vec<CpuBandwidth>,
    pub storage: Vec<StorageProbe>,
}

impl ProbeReport {
    pub fn copy_curve(&self, device: &str, direction: CopyDirection, pinned: bool) -> Option<&CopyCurve> {
        self.copies.iter().find(|c| c.device == device && c.direction == direction && c.pinned == pinned && c.peer.is_none())
    }

    /// Decode CPU bandwidth and the thread count chosen for it (see `decode_threads`).
    pub fn decode_cpu_bandwidth(&self) -> Option<&CpuBandwidth> {
        decode_threads(&self.cpu_bandwidth, self.inventory.cpu.cores)
    }
}

/// The fewest threads, at most one per physical core, within 10% of the best bandwidth among them. Decode
/// synchronizes every thread at each CPU operation, so threads past bandwidth saturation only add barrier
/// cost, and SMT siblings stall the whole step when other processes want the cores.
pub fn decode_threads(sweep: &[CpuBandwidth], cores: u32) -> Option<&CpuBandwidth> {
    let within: Vec<&CpuBandwidth> = sweep.iter().filter(|b| b.threads <= cores).collect();
    let best = within.iter().map(|b| b.gbps.p50).fold(0.0, f64::max);
    within.into_iter().filter(|b| b.gbps.p50 >= 0.9 * best).min_by_key(|b| b.threads)
}
