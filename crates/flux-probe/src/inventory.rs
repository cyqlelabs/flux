use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::Result;
use flux_core::hardware::{BackendDevice, CpuInfo, GpuInfo, HardwareInventory, MemoryInfo, NumaNode, StorageInfo};
use nvml_wrapper::{enum_wrappers::device::Clock, enums::device::UsedGpuMemory, Nvml};

pub fn inventory(backend_devices: Vec<BackendDevice>) -> Result<HardwareInventory> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
    let mut cpu = parse_cpuinfo(&cpuinfo);
    cpu.isa = runtime_isa(&cpuinfo);
    cpu.max_mhz = read_trimmed("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").and_then(|s| s.parse::<f64>().ok()).map(|khz| khz / 1000.0);
    let mut memory = parse_meminfo(&fs::read_to_string("/proc/meminfo")?);
    memory.thp = read_trimmed("/sys/kernel/mm/transparent_hugepage/enabled")
        .and_then(|s| s.split_once('[').and_then(|(_, s)| s.split_once(']')).map(|(s, _)| s.to_owned()))
        .unwrap_or_default();
    let numa = numa_nodes(&cpuinfo, memory.total)?;
    let (gpus, driver_version, cuda_driver_version) = match Nvml::init() {
        Ok(nvml) => {
            let gpus = (0..nvml.device_count()?).map(|index| gpu_info(&nvml, index)).collect::<Result<Vec<_>>>()?;
            (gpus, nvml.sys_driver_version().ok(), nvml.sys_cuda_driver_version().ok())
        }
        Err(_) => (Vec::new(), None, None),
    };
    Ok(HardwareInventory {
        hostname: fs::read_to_string("/proc/sys/kernel/hostname")?.trim().to_owned(),
        kernel: fs::read_to_string("/proc/sys/kernel/osrelease")?.trim().to_owned(),
        cpu,
        memory,
        numa,
        gpus,
        driver_version,
        cuda_driver_version,
        storage: storage_inventory(&fs::read_to_string("/proc/mounts")?),
        backend_devices,
    })
}

pub fn storage_for<'a>(inv: &'a HardwareInventory, path: &Path) -> Option<&'a StorageInfo> {
    select_mount(&inv.storage, &fs::canonicalize(path).ok()?)
}

fn select_mount<'a>(storage: &'a [StorageInfo], path: &Path) -> Option<&'a StorageInfo> {
    storage.iter().filter(|s| path.starts_with(Path::new(&s.mount))).max_by_key(|s| Path::new(&s.mount).components().count())
}

fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_owned())
}

fn parse_cpuinfo(text: &str) -> CpuInfo {
    let records: Vec<BTreeMap<&str, &str>> = text
        .split("\n\n")
        .map(|record| record.lines().filter_map(|line| line.split_once(':').map(|(k, v)| (k.trim(), v.trim()))).collect())
        .filter(|record: &BTreeMap<_, _>| record.contains_key("processor"))
        .collect();
    let first = records.first();
    let field = |name| first.and_then(|r| r.get(name)).copied().unwrap_or("");
    let sockets: BTreeSet<_> = records.iter().filter_map(|r| r.get("physical id")).collect();
    let cores: BTreeSet<_> = records.iter().filter_map(|r| r.get("core id").map(|core| (r.get("physical id").copied().unwrap_or("0"), *core))).collect();
    CpuInfo {
        model: field("model name").to_owned(),
        vendor: field("vendor_id").to_owned(),
        sockets: sockets.len().max(1) as u32,
        cores: if cores.is_empty() { records.len() } else { cores.len() } as u32,
        threads: records.len() as u32,
        isa: Vec::new(),
        max_mhz: None,
    }
}

#[cfg(target_arch = "x86_64")]
fn runtime_isa(_cpuinfo: &str) -> Vec<String> {
    let mut isa = Vec::new();
    macro_rules! detect {
        ($($feature:tt),* $(,)?) => { $(
            if std::arch::is_x86_feature_detected!($feature) { isa.push($feature.to_owned()); }
        )* };
    }
    detect!("sse4.2", "avx", "avx2", "fma", "f16c", "bmi2", "avx512f", "avx512bw", "avx512vl", "avx512vnni", "avx512bf16", "avxvnni", "sha");
    isa
}

#[cfg(not(target_arch = "x86_64"))]
fn runtime_isa(cpuinfo: &str) -> Vec<String> {
    cpuinfo
        .lines()
        .filter_map(|line| line.split_once(':'))
        .find(|(key, _)| matches!(key.trim(), "flags" | "Features"))
        .map(|(_, flags)| flags.split_whitespace().map(str::to_owned).collect())
        .unwrap_or_default()
}

fn meminfo_values(text: &str) -> BTreeMap<&str, u64> {
    text.lines()
        .filter_map(|line| {
            let (key, value) = line.rsplit_once(':')?;
            let mut words = value.split_whitespace();
            let value = words.next()?.parse::<u64>().ok()?;
            let scale = if words.next() == Some("kB") { 1024 } else { 1 };
            Some((key.split_whitespace().last()?, value.saturating_mul(scale)))
        })
        .collect()
}

fn parse_meminfo(text: &str) -> MemoryInfo {
    let values = meminfo_values(text);
    let get = |key| values.get(key).copied().unwrap_or(0);
    MemoryInfo {
        total: get("MemTotal"),
        available: get("MemAvailable"),
        swap_total: get("SwapTotal"),
        swap_free: get("SwapFree"),
        hugepage_size: get("Hugepagesize"),
        hugepages_free: get("HugePages_Free").saturating_mul(get("Hugepagesize")),
        thp: String::new(),
    }
}

fn parse_cpulist(text: &str) -> Result<Vec<u32>> {
    let mut cpus = BTreeSet::new();
    for part in text.trim().split(',').filter(|s| !s.is_empty()) {
        if let Some((lo, hi)) = part.split_once('-') {
            let lo = lo.trim().parse::<u32>()?;
            let hi = hi.trim().parse::<u32>()?;
            anyhow::ensure!(lo <= hi, "reversed CPU range: {part}");
            cpus.extend(lo..=hi);
        } else {
            cpus.insert(part.trim().parse()?);
        }
    }
    Ok(cpus.into_iter().collect())
}

fn logical_cpu_ids(cpuinfo: &str) -> Vec<u32> {
    cpuinfo
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == "processor").then(|| value.trim().parse().ok()).flatten()
        })
        .collect()
}

fn numa_nodes(cpuinfo: &str, total: u64) -> Result<Vec<NumaNode>> {
    let mut nodes = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let Some(id) = name.to_str().and_then(|s| s.strip_prefix("node")).and_then(|s| s.parse().ok()) else {
                continue;
            };
            let path = entry.path();
            nodes.push(NumaNode {
                id,
                cpus: parse_cpulist(&fs::read_to_string(path.join("cpulist"))?)?,
                mem_total: meminfo_values(&fs::read_to_string(path.join("meminfo"))?).get("MemTotal").copied().unwrap_or(0),
            });
        }
    }
    if nodes.is_empty() {
        nodes.push(NumaNode { id: 0, cpus: logical_cpu_ids(cpuinfo), mem_total: total });
    }
    nodes.sort_by_key(|n| n.id);
    Ok(nodes)
}

fn normalize_pci_bus_id(bus: &str) -> Result<String> {
    let parts: Vec<_> = bus.trim().split([':', '.']).collect();
    anyhow::ensure!(parts.len() == 4, "invalid PCI bus id: {bus}");
    let fields = parts.iter().map(|s| u32::from_str_radix(s, 16)).collect::<std::result::Result<Vec<_>, _>>()?;
    anyhow::ensure!(fields[0] <= 0xffff && fields[1] <= 0xff && fields[2] <= 0x1f && fields[3] <= 7, "invalid PCI bus id: {bus}");
    Ok(format!("{:04x}:{:02x}:{:02x}.{:x}", fields[0], fields[1], fields[2], fields[3]))
}

fn gpu_info(nvml: &Nvml, index: u32) -> Result<GpuInfo> {
    let device = nvml.device_by_index(index)?;
    let pci_bus_id = normalize_pci_bus_id(&device.pci_info()?.bus_id)?;
    let memory = device.memory_info()?;
    let capability = device.cuda_compute_capability()?;
    let mut processes = BTreeMap::<u32, u64>::new();
    let compute = device.running_compute_processes().or_else(|_| device.running_compute_processes_v2()).unwrap_or_default();
    let graphics = device.running_graphics_processes().or_else(|_| device.running_graphics_processes_v2()).unwrap_or_default();
    for process in compute.into_iter().chain(graphics) {
        let bytes = match process.used_gpu_memory {
            UsedGpuMemory::Used(bytes) => bytes,
            UsedGpuMemory::Unavailable => 0,
        };
        processes.entry(process.pid).and_modify(|used| *used = (*used).max(bytes)).or_insert(bytes);
    }
    Ok(GpuInfo {
        nvml_index: index,
        name: device.name()?,
        uuid: device.uuid()?,
        mem_total: memory.total,
        mem_free: memory.free,
        compute_capability: (capability.major, capability.minor),
        pcie_gen_current: device.current_pcie_link_gen().ok(),
        pcie_gen_max: device.max_pcie_link_gen().ok(),
        pcie_width_current: device.current_pcie_link_width().ok(),
        pcie_width_max: device.max_pcie_link_width().ok(),
        power_limit_w: device.power_management_limit().ok().map(|mw| mw as f64 / 1000.0),
        sm_clock_max_mhz: device.max_clock_info(Clock::SM).ok(),
        mem_clock_max_mhz: device.max_clock_info(Clock::Memory).ok(),
        numa_node: read_trimmed(format!("/sys/bus/pci/devices/{pci_bus_id}/numa_node")).and_then(|s| s.parse::<i32>().ok()).filter(|&id| id >= 0),
        pci_bus_id,
        foreign_processes: processes.into_iter().collect(),
    })
}

fn unescape_mount(text: &str) -> String {
    let mut bytes = Vec::new();
    let input = text.as_bytes();
    let mut i = 0;
    while i < input.len() {
        if input[i] == b'\\' && i + 3 < input.len() && input[i + 1..i + 4].iter().all(|b| (b'0'..=b'7').contains(b)) {
            let value = (input[i + 1] - b'0') as u16 * 64 + (input[i + 2] - b'0') as u16 * 8 + (input[i + 3] - b'0') as u16;
            if let Ok(value) = u8::try_from(value) {
                bytes.push(value);
                i += 4;
                continue;
            }
        }
        bytes.push(input[i]);
        i += 1;
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn storage_inventory(mounts: &str) -> Vec<StorageInfo> {
    let mut seen = BTreeSet::new();
    let mut storage = Vec::new();
    for line in mounts.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() < 3 {
            continue;
        }
        let kind = fields[2];
        if matches!(kind, "tmpfs" | "proc" | "sysfs" | "overlay" | "squashfs" | "efivarfs" | "devtmpfs")
            || kind.starts_with("cgroup")
            || kind.starts_with("fuse.")
        {
            continue;
        }
        let source = unescape_mount(fields[0]);
        let Ok(device) = fs::canonicalize(&source) else { continue };
        let Some(part) = device.file_name().and_then(|s| s.to_str()) else { continue };
        let sys_part = Path::new("/sys/class/block").join(part);
        let Ok(sys_device) = fs::canonicalize(&sys_part) else { continue };
        if !seen.insert(device) {
            continue;
        }
        let disk_path = if sys_part.join("partition").exists() { sys_device.parent().unwrap_or(&sys_device) } else { &sys_device };
        let Some(disk) = disk_path.file_name().and_then(|s| s.to_str()) else { continue };
        let base = Path::new("/sys/block").join(disk);
        let device_path = fs::canonicalize(base.join("device")).ok();
        let transport = if disk.starts_with("nvme") {
            "nvme"
        } else if device_path.as_ref().is_some_and(|p| p.components().any(|c| c.as_os_str().to_string_lossy().starts_with("usb"))) {
            "usb"
        } else if disk.starts_with("sd") {
            "sata"
        } else {
            "virtual"
        };
        storage.push(StorageInfo {
            mount: unescape_mount(fields[1]),
            device: disk.to_owned(),
            model: read_trimmed(base.join("device/model")),
            transport: transport.to_owned(),
            rotational: read_trimmed(base.join("queue/rotational")).is_some_and(|s| s == "1"),
            size: read_trimmed(base.join("size")).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0).saturating_mul(512),
        });
    }
    storage
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpulist_ranges() {
        assert_eq!(parse_cpulist("0-5,12-17\n").unwrap(), (0..=5).chain(12..=17).collect::<Vec<_>>());
        assert_eq!(parse_cpulist("3,1-3,8").unwrap(), vec![1, 2, 3, 8]);
        assert!(parse_cpulist("").unwrap().is_empty());
        assert!(parse_cpulist("5-2").is_err());
        assert!(parse_cpulist("bad").is_err());
    }

    #[test]
    fn meminfo_units() {
        let memory =
            parse_meminfo("MemTotal: 8192 kB\nMemAvailable: 4096 kB\nSwapTotal: 2048 kB\nSwapFree: 1024 kB\nHugepagesize: 2048 kB\nHugePages_Free: 3\n");
        assert_eq!((memory.total, memory.available), (8192 * 1024, 4096 * 1024));
        assert_eq!((memory.swap_total, memory.swap_free), (2048 * 1024, 1024 * 1024));
        assert_eq!(memory.hugepage_size, 2048 * 1024);
        assert_eq!(memory.hugepages_free, 3 * 2048 * 1024);
        assert_eq!(meminfo_values("Node 2 MemTotal: 512 kB\n")["MemTotal"], 512 * 1024);
        assert_eq!(parse_meminfo("").available, 0);
    }

    #[test]
    fn pci_bus_normalization() {
        assert_eq!(normalize_pci_bus_id("00000000:06:00.0").unwrap(), "0000:06:00.0");
        assert_eq!(normalize_pci_bus_id("000000AB:AF:1F.7").unwrap(), "00ab:af:1f.7");
        assert!(normalize_pci_bus_id("invalid").is_err());
    }

    #[test]
    fn mount_prefix_selection() {
        let storage = ["/", "/mnt/models", "/mnt/models/nested"].map(|mount| StorageInfo {
            mount: mount.to_owned(),
            device: String::new(),
            model: None,
            transport: String::new(),
            rotational: false,
            size: 0,
        });
        assert_eq!(select_mount(&storage, Path::new("/mnt/models/nested/model.gguf")).unwrap().mount, "/mnt/models/nested");
        assert_eq!(select_mount(&storage, Path::new("/mnt/models-other/model.gguf")).unwrap().mount, "/");
        assert!(select_mount(&storage[1..], Path::new("/other")).is_none());
    }

    #[test]
    fn storage_path_canonicalization() {
        let dir = tempfile::tempdir().unwrap();
        let mount = dir.path().join("models");
        fs::create_dir(&mount).unwrap();
        let file = mount.join("weights");
        fs::write(&file, []).unwrap();
        let link = dir.path().join("alias");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        let inv = HardwareInventory {
            hostname: String::new(),
            kernel: String::new(),
            cpu: parse_cpuinfo(""),
            memory: parse_meminfo(""),
            numa: Vec::new(),
            gpus: Vec::new(),
            driver_version: None,
            cuda_driver_version: None,
            backend_devices: Vec::new(),
            storage: vec![StorageInfo {
                mount: fs::canonicalize(&mount).unwrap().to_string_lossy().into_owned(),
                device: "disk".to_owned(),
                model: None,
                transport: String::new(),
                rotational: false,
                size: 0,
            }],
        };
        assert_eq!(storage_for(&inv, &link).unwrap().device, "disk");
        assert!(storage_for(&inv, &dir.path().join("missing")).is_none());
    }

    #[test]
    fn cpu_topology() {
        let cpu = parse_cpuinfo("processor: 0\nmodel name: Test\nvendor_id: Vendor\nphysical id: 0\ncore id: 0\n\nprocessor: 1\nphysical id: 0\ncore id: 0\n\nprocessor: 2\nphysical id: 1\ncore id: 0\n");
        assert_eq!((cpu.sockets, cpu.cores, cpu.threads), (2, 2, 3));
        assert_eq!(cpu.model, "Test");
    }

    #[test]
    fn sparse_cpu_ids() {
        assert_eq!(logical_cpu_ids("processor: 0\n\nprocessor: 4\n\nprocessor: 12\n"), vec![0, 4, 12]);
    }

    #[test]
    fn escaped_mount_paths() {
        assert_eq!(unescape_mount("/mnt/model\\040files\\134name"), "/mnt/model files\\name");
    }
}
