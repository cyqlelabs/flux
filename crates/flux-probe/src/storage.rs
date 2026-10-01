use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use flux_core::{hardware::StorageProbe, stats::Summary};

const ALIGNMENT: u64 = 4096;
const SEQUENTIAL_BLOCK: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct StorageOpts {
    pub bytes: u64,
    pub runs: usize,
    pub random_block: u64,
    pub random_reads: usize,
}

impl Default for StorageOpts {
    fn default() -> Self {
        Self { bytes: 1024 * 1024 * 1024, runs: 3, random_block: 1024 * 1024, random_reads: 256 }
    }
}

/// Measures reads only; cache eviction discards clean pages in each buffered read range.
pub fn probe_file(path: &Path, opts: &StorageOpts) -> Result<Vec<StorageProbe>> {
    anyhow::ensure!(opts.bytes > 0 && opts.runs > 0, "bytes and runs must be positive");
    anyhow::ensure!(opts.random_block > 0 && opts.random_block.is_multiple_of(ALIGNMENT), "random_block must be a positive multiple of {ALIGNMENT}");
    anyhow::ensure!(opts.random_reads > 0, "random_reads must be positive");
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let metadata = file.metadata()?;
    anyhow::ensure!(metadata.is_file() && metadata.len() > 0, "probe requires a nonempty regular file");
    let file_len = metadata.len();
    anyhow::ensure!(file_len <= i64::MAX as u64, "file exceeds the supported offset range");
    let bytes = opts.bytes.min(file_len);
    let mut rng = Xorshift(0x4d59_5df4_d0f3_3173);
    let mut buffer = vec![0; SEQUENTIAL_BLOCK];
    let mut samples = Vec::with_capacity(opts.runs);
    for _ in 0..opts.runs {
        let start = rng.next() % (file_len - bytes + 1);
        evict_cache(&file, start, bytes)?;
        samples.push(sequential_read(&file, &mut buffer, start, bytes)?);
    }
    let mut probes = vec![probe(path, false, SEQUENTIAL_BLOCK as u64, true, &samples)];
    let direct = match OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path) {
        Ok(file) => file,
        Err(error) if direct_unsupported(&error) => return Ok(probes),
        Err(error) => return Err(error).context("open file with O_DIRECT"),
    };
    let direct_bytes = bytes / ALIGNMENT * ALIGNMENT;
    if direct_bytes > 0 {
        let mut buffer = AlignedBuffer::new(SEQUENTIAL_BLOCK)?;
        samples.clear();
        for _ in 0..opts.runs {
            let start = rng.offset(file_len, direct_bytes);
            match sequential_read(&direct, buffer.as_mut_slice(), start, direct_bytes) {
                Ok(gbps) => samples.push(gbps),
                Err(error) if direct_unsupported(&error) => return Ok(probes),
                Err(error) => return Err(error).context("sequential O_DIRECT read"),
            }
        }
        probes.push(probe(path, true, SEQUENTIAL_BLOCK as u64, true, &samples));
    }
    if opts.random_block <= file_len {
        let block_size = usize::try_from(opts.random_block).context("random_block is too large")?;
        let mut buffer = AlignedBuffer::new(block_size)?;
        samples.clear();
        for _ in 0..opts.runs {
            let offsets: Vec<_> = (0..opts.random_reads).map(|_| rng.offset(file_len, opts.random_block)).collect();
            let begin = Instant::now();
            for offset in offsets {
                match direct.read_exact_at(buffer.as_mut_slice(), offset) {
                    Ok(()) => {}
                    Err(error) if direct_unsupported(&error) => return Ok(probes),
                    Err(error) => return Err(error).context("random O_DIRECT read"),
                }
            }
            samples.push(bandwidth(opts.random_block as f64 * opts.random_reads as f64, begin));
        }
        probes.push(probe(path, true, opts.random_block, false, &samples));
    }
    Ok(probes)
}

fn probe(path: &Path, direct_io: bool, block_bytes: u64, sequential: bool, samples: &[f64]) -> StorageProbe {
    StorageProbe { path: path.to_string_lossy().into_owned(), direct_io, block_bytes, sequential, gbps: Summary::of(samples).expect("runs is positive") }
}

fn sequential_read(file: &File, buffer: &mut [u8], start: u64, bytes: u64) -> io::Result<f64> {
    let begin = Instant::now();
    let mut done = 0;
    while done < bytes {
        let len = (bytes - done).min(buffer.len() as u64) as usize;
        file.read_exact_at(&mut buffer[..len], start + done)?;
        done += len as u64;
    }
    Ok(bandwidth(bytes as f64, begin))
}

fn bandwidth(bytes: f64, begin: Instant) -> f64 {
    bytes / begin.elapsed().as_secs_f64().max(1e-9) / 1e9
}

fn evict_cache(file: &File, offset: u64, bytes: u64) -> io::Result<()> {
    // The descriptor is live and the validated range fits off_t. DONTNEED never writes file data.
    let status = unsafe { libc::posix_fadvise(file.as_raw_fd(), offset as libc::off_t, bytes as libc::off_t, libc::POSIX_FADV_DONTNEED) };
    if status == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(status))
    }
}

fn direct_unsupported(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(libc::EINVAL | libc::EOPNOTSUPP | libc::ENOSYS))
}

struct AlignedBuffer {
    bytes: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuffer {
    fn new(len: usize) -> Result<Self> {
        let capacity = len.checked_add(ALIGNMENT as usize - 1).context("buffer size overflow")?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(capacity)?;
        bytes.resize(capacity, 0);
        let start = bytes.as_ptr().align_offset(ALIGNMENT as usize);
        Ok(Self { bytes, start, len })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes[self.start..self.start + self.len]
    }
}

struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn offset(&mut self, file_len: u64, bytes: u64) -> u64 {
        let slots = (file_len - bytes) / ALIGNMENT + 1;
        (self.next() % slots) * ALIGNMENT
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn reads_model_file() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let data = vec![0x5a; 1024 * 1024];
        for _ in 0..64 {
            file.write_all(&data).unwrap();
        }
        file.as_file().sync_all().unwrap();
        let opts = StorageOpts { bytes: 12 * 1024 * 1024, runs: 2, random_block: 64 * 1024, random_reads: 8 };
        let probes = probe_file(file.path(), &opts).unwrap();
        assert!((1..=3).contains(&probes.len()));
        assert!(!probes[0].direct_io && probes[0].sequential);
        for probe in &probes {
            assert_eq!(probe.gbps.n, opts.runs);
            assert!(probe.gbps.min > 0.0 && probe.gbps.max.is_finite());
            assert_eq!(probe.path, file.path().to_string_lossy());
        }
        assert_eq!(file.as_file().metadata().unwrap().len(), 64 * 1024 * 1024);
        let mut contents = vec![0; data.len()];
        file.as_file().read_exact_at(&mut contents, 0).unwrap();
        assert_eq!(contents, data);
        file.as_file().read_exact_at(&mut contents, 63 * 1024 * 1024).unwrap();
        assert_eq!(contents, data);
    }

    #[test]
    fn aligned_buffers_and_offsets() {
        let mut buffer = AlignedBuffer::new(SEQUENTIAL_BLOCK).unwrap();
        assert_eq!(buffer.as_mut_slice().as_ptr() as usize % ALIGNMENT as usize, 0);
        let mut rng = Xorshift(123);
        let offsets: Vec<_> = (0..32).map(|_| rng.offset(100_001, 8192)).collect();
        assert!(offsets.iter().all(|offset| offset.is_multiple_of(ALIGNMENT) && offset + 8192 <= 100_001));
        assert!(offsets.iter().any(|&offset| offset != 0));
    }

    #[test]
    fn invalid_options_and_empty_file() {
        let file = tempfile::NamedTempFile::new().unwrap();
        assert!(probe_file(file.path(), &StorageOpts::default()).is_err());
        assert!(probe_file(file.path(), &StorageOpts { runs: 0, ..StorageOpts::default() }).is_err());
        assert!(probe_file(file.path(), &StorageOpts { random_block: 1, ..StorageOpts::default() }).is_err());
    }

    #[test]
    fn short_unaligned_file() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&[1; 123]).unwrap();
        let probes = probe_file(file.path(), &StorageOpts { runs: 1, ..StorageOpts::default() }).unwrap();
        assert_eq!(probes.len(), 1);
        assert!(probes[0].gbps.min > 0.0);
    }
}
