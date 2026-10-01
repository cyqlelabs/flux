//! Process-tree accounting and page-cache control for cold runs.

use std::os::fd::AsRawFd;
use std::path::Path;

fn tree(pid: u32) -> Vec<u32> {
    let mut out = vec![pid];
    let mut i = 0;
    while i < out.len() {
        if let Ok(tasks) = std::fs::read_dir(format!("/proc/{}/task", out[i])) {
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

/// Bytes the tree has read from storage (excludes page-cache hits).
pub fn tree_io_read(pid: u32) -> u64 {
    tree(pid)
        .into_iter()
        .filter_map(|p| {
            let io = std::fs::read_to_string(format!("/proc/{p}/io")).ok()?;
            io.lines().find(|l| l.starts_with("read_bytes:"))?.split_whitespace().nth(1)?.parse::<u64>().ok()
        })
        .sum()
}

/// CPU seconds (user + system) consumed by the tree so far.
pub fn tree_cpu_s(pid: u32) -> f64 {
    let tick = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    tree(pid)
        .into_iter()
        .filter_map(|p| {
            let stat = std::fs::read_to_string(format!("/proc/{p}/stat")).ok()?;
            let rest = stat.rsplit_once(") ")?.1;
            let f: Vec<&str> = rest.split_whitespace().collect();
            Some((f.get(11)?.parse::<f64>().ok()? + f.get(12)?.parse::<f64>().ok()?) / tick)
        })
        .sum()
}

/// Drops clean cached pages of the files so the next load reads from storage. Pages still mapped
/// by a running process stay cached; no privileges needed, nothing is written.
pub fn evict_page_cache(files: &[impl AsRef<Path>]) {
    for f in files {
        if let Ok(file) = std::fs::File::open(f) {
            unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
        }
    }
}

/// Fraction of the file's pages currently in the page cache.
pub fn cached_fraction(file: &Path) -> f64 {
    let Ok(f) = std::fs::File::open(file) else { return 0.0 };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0) as usize;
    if len == 0 {
        return 0.0;
    }
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    unsafe {
        let addr = libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0);
        if addr == libc::MAP_FAILED {
            return 0.0;
        }
        let pages = len.div_ceil(page);
        let mut vec = vec![0u8; pages];
        let ok = libc::mincore(addr, len, vec.as_mut_ptr()) == 0;
        libc::munmap(addr, len);
        if !ok {
            return 0.0;
        }
        vec.iter().filter(|&&b| b & 1 == 1).count() as f64 / pages as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eviction_drops_cached_pages() {
        // Not /tmp: tmpfs pages are the storage itself and cannot be evicted.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../target/flux-evict-{}", std::process::id()));
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&dir).unwrap();
            f.write_all(&vec![7u8; 8 << 20]).unwrap();
            // Dirty pages cannot be dropped until written back.
            f.sync_all().unwrap();
        }
        let _ = std::fs::read(&dir).unwrap();
        evict_page_cache(&[&dir]);
        assert!(cached_fraction(&dir) < 0.5);
        std::fs::remove_file(&dir).unwrap();
        assert!(tree_cpu_s(std::process::id()) > 0.0);
    }
}
