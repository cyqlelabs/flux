//! Background sampling of device memory, power and a process tree's resident memory during a run.

use flux_core::hardware::BackendDevice;
use nvml_wrapper::Nvml;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Debug, Clone, Default)]
pub struct Peaks {
    /// Peak used bytes per backend device name (whole device, all processes).
    pub device_used: BTreeMap<String, u64>,
    /// Used bytes per device at the first sample, before the measured work allocated anything.
    pub device_baseline: BTreeMap<String, u64>,
    /// Peak power per backend device name, watts.
    pub device_power_w: BTreeMap<String, f64>,
    /// Peak summed RSS of the root process and its descendants.
    pub tree_rss: u64,
    /// Physical bytes read from storage by the tree (from /proc/<pid>/io read_bytes) since start.
    pub tree_read_bytes: u64,
}

impl Peaks {
    /// Peak bytes added on each device since sampling began.
    pub fn device_added(&self) -> BTreeMap<String, u64> {
        self.device_used.iter().map(|(k, v)| (k.clone(), v.saturating_sub(self.device_baseline.get(k).copied().unwrap_or(0)))).collect()
    }
}

pub struct Sampler {
    stop: Arc<AtomicBool>,
    peaks: Arc<Mutex<Peaks>>,
    handle: Option<JoinHandle<()>>,
}

impl Sampler {
    /// Samples every `period`; `devices` maps NVML devices to backend names by PCI bus id.
    pub fn start(devices: &[BackendDevice], root_pid: Option<u32>, period: Duration) -> Sampler {
        let stop = Arc::new(AtomicBool::new(false));
        let peaks = Arc::new(Mutex::new(Peaks::default()));
        let by_bus: Vec<(String, String)> = devices.iter().filter_map(|d| Some((d.pci_bus_id.clone()?, d.name.clone()))).collect();
        let (s, p) = (stop.clone(), peaks.clone());
        let handle = std::thread::spawn(move || {
            let nvml = Nvml::init().ok();
            let read0 = root_pid.map(tree_read_bytes).unwrap_or(0);
            while !s.load(Ordering::Relaxed) {
                let mut pk = p.lock().unwrap();
                if let Some(nvml) = &nvml {
                    for (bus, name) in &by_bus {
                        let nvml_bus = format!("0000{}", bus.to_ascii_uppercase());
                        if let Ok(d) = nvml.device_by_pci_bus_id(nvml_bus.as_str()) {
                            if let Ok(m) = d.memory_info() {
                                pk.device_baseline.entry(name.clone()).or_insert(m.used);
                                let e = pk.device_used.entry(name.clone()).or_default();
                                *e = (*e).max(m.used);
                            }
                            if let Ok(mw) = d.power_usage() {
                                let e = pk.device_power_w.entry(name.clone()).or_default();
                                *e = e.max(mw as f64 / 1000.0);
                            }
                        }
                    }
                }
                if let Some(pid) = root_pid {
                    pk.tree_rss = pk.tree_rss.max(tree_rss(pid));
                    pk.tree_read_bytes = tree_read_bytes(pid).saturating_sub(read0);
                }
                drop(pk);
                std::thread::sleep(period);
            }
        });
        Sampler { stop, peaks, handle: Some(handle) }
    }

    pub fn finish(mut self) -> Peaks {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.peaks.lock().unwrap().clone()
    }
}

fn tree(pid: u32) -> Vec<u32> {
    let mut out = vec![pid];
    let mut i = 0;
    while i < out.len() {
        let p = out[i];
        if let Ok(tasks) = std::fs::read_dir(format!("/proc/{p}/task")) {
            for t in tasks.flatten() {
                if let Ok(c) = std::fs::read_to_string(t.path().join("children")) {
                    out.extend(c.split_whitespace().filter_map(|x| x.parse::<u32>().ok()));
                }
            }
        }
        i += 1;
    }
    out
}

fn status_kib(pid: u32, key: &str) -> u64 {
    std::fs::read_to_string(format!("/proc/{pid}/status"))
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse().ok()))
        .unwrap_or(0)
}

pub fn tree_rss(pid: u32) -> u64 {
    tree(pid).into_iter().map(|p| status_kib(p, "VmRSS:") * 1024).sum()
}

fn tree_read_bytes(pid: u32) -> u64 {
    tree(pid)
        .into_iter()
        .filter_map(|p| {
            let io = std::fs::read_to_string(format!("/proc/{p}/io")).ok()?;
            io.lines().find(|l| l.starts_with("read_bytes:"))?.split_whitespace().nth(1)?.parse::<u64>().ok()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_own_process_tree() {
        let s = Sampler::start(&[], Some(std::process::id()), Duration::from_millis(5));
        // black_box: an unused buffer may be optimized away in release builds.
        let buf = std::hint::black_box(vec![1u8; 32 << 20]);
        std::thread::sleep(Duration::from_millis(30));
        let p = s.finish();
        drop(std::hint::black_box(buf));
        assert!(p.tree_rss > 32 << 20, "{}", p.tree_rss);
    }
}
