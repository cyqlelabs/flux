//! GGUF container reader and validator. Reads metadata and tensor descriptors only;
//! tensor payloads are checked for bounds and overlap, never loaded.

use anyhow::{bail, ensure, Context, Result};
use flux_core::ggml_type::GgmlType;
use flux_core::model::TensorInfo;
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
/// Guards against corrupt lengths turning into huge allocations.
const MAX_STRING: u64 = 1 << 26;
const MAX_ARRAY: u64 = 1 << 28;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    /// GGUF type id of this value.
    pub fn type_id(&self) -> u32 {
        match self {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::String(_) => 8,
            Value::Array(_) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }

    /// Little-endian payload as GGUF stores it (without the type id). `elem` is the array element type.
    pub fn write_payload(&self, out: &mut Vec<u8>, elem: Option<u32>) {
        match self {
            Value::U8(v) => out.extend(v.to_le_bytes()),
            Value::I8(v) => out.extend(v.to_le_bytes()),
            Value::U16(v) => out.extend(v.to_le_bytes()),
            Value::I16(v) => out.extend(v.to_le_bytes()),
            Value::U32(v) => out.extend(v.to_le_bytes()),
            Value::I32(v) => out.extend(v.to_le_bytes()),
            Value::F32(v) => out.extend(v.to_le_bytes()),
            Value::Bool(v) => out.push(*v as u8),
            Value::String(s) => {
                out.extend((s.len() as u64).to_le_bytes());
                out.extend(s.as_bytes());
            }
            Value::Array(a) => {
                out.extend(elem.or_else(|| a.first().map(Value::type_id)).unwrap_or(0).to_le_bytes());
                out.extend((a.len() as u64).to_le_bytes());
                for v in a {
                    v.write_payload(out, None);
                }
            }
            Value::U64(v) => out.extend(v.to_le_bytes()),
            Value::I64(v) => out.extend(v.to_le_bytes()),
            Value::F64(v) => out.extend(v.to_le_bytes()),
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            Value::Bool(v) => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v as f64),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// Canonical bytes for hashing metadata identity.
    pub fn hash_into(&self, h: &mut impl sha2::Digest) {
        match self {
            Value::String(s) => {
                h.update((s.len() as u64).to_le_bytes());
                h.update(s.as_bytes());
            }
            Value::Array(a) => {
                h.update((a.len() as u64).to_le_bytes());
                a.iter().for_each(|v| v.hash_into(h));
            }
            Value::F32(v) => h.update(v.to_le_bytes()),
            Value::F64(v) => h.update(v.to_le_bytes()),
            v => h.update(v.as_u64().unwrap_or(u64::MAX).to_le_bytes()),
        }
    }
}

/// One `.gguf` file: metadata, tensor descriptors and where the payload region starts.
#[derive(Debug, Clone)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub size: u64,
    pub alignment: u64,
    pub data_offset: u64,
    pub kv: BTreeMap<String, Value>,
    /// Element type of every array-valued key, so empty arrays can be written back exactly.
    pub array_types: BTreeMap<String, u32>,
    /// Keys in file order.
    pub key_order: Vec<String>,
    pub tensors: Vec<RawTensor>,
}

#[derive(Debug, Clone)]
pub struct RawTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

struct Reader<R> {
    r: R,
}

impl<R: Read> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b).context("unexpected end of GGUF header")?;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        ensure!(n <= MAX_STRING, "string length {n} exceeds limit");
        let mut b = vec![0u8; n as usize];
        self.r.read_exact(&mut b).context("unexpected end of GGUF string")?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
    fn array_body(&mut self, ety: u32, depth: u32) -> Result<Value> {
        ensure!(depth < 4, "nested arrays too deep");
        let n = self.u64()?;
        ensure!(n <= MAX_ARRAY, "array length {n} exceeds limit");
        let mut v = Vec::with_capacity(n.min(1 << 20) as usize);
        for _ in 0..n {
            v.push(self.value(ety, depth + 1)?);
        }
        Ok(Value::Array(v))
    }

    fn value(&mut self, ty: u32, depth: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(u8::from_le_bytes(self.bytes()?)),
            1 => Value::I8(i8::from_le_bytes(self.bytes()?)),
            2 => Value::U16(u16::from_le_bytes(self.bytes()?)),
            3 => Value::I16(i16::from_le_bytes(self.bytes()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.bytes()?)),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => Value::Bool(u8::from_le_bytes(self.bytes()?) != 0),
            8 => Value::String(self.string()?),
            9 => {
                let ety = self.u32()?;
                self.array_body(ety, depth)?
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            t => bail!("unknown GGUF value type {t}"),
        })
    }
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let size = file.metadata()?.len();
        let mut buf = BufReader::with_capacity(1 << 20, file);
        let mut r = Reader { r: &mut buf };

        let magic: [u8; 4] = r.bytes()?;
        ensure!(&magic == MAGIC, "{} is not a GGUF file", path.display());
        let version = r.u32()?;
        ensure!(version == 2 || version == 3, "unsupported GGUF version {version} (big-endian or v1 files are not supported)");
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;
        ensure!(n_tensors <= 1 << 20 && n_kv <= 1 << 20, "implausible header counts: {n_tensors} tensors, {n_kv} keys");

        let mut kv = BTreeMap::new();
        let mut array_types = BTreeMap::new();
        let mut key_order = vec![];
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            let val = if ty == 9 {
                let ety = r.u32()?;
                array_types.insert(key.clone(), ety);
                r.array_body(ety, 0)
            } else {
                r.value(ty, 0)
            }
            .with_context(|| format!("reading metadata key {key}"))?;
            ensure!(kv.insert(key.clone(), val).is_none(), "duplicate metadata key {key}");
            key_order.push(key);
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()?;
            ensure!((1..=4).contains(&n_dims), "tensor {name} has {n_dims} dimensions");
            let dims = (0..n_dims).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
            let ggml_type = GgmlType(r.u32()?);
            let offset = r.u64()?;
            tensors.push(RawTensor { name, dims, ggml_type, offset });
        }

        let alignment = kv.get("general.alignment").and_then(Value::as_u64).unwrap_or(DEFAULT_ALIGNMENT);
        ensure!(alignment.is_power_of_two(), "general.alignment {alignment} is not a power of two");
        let header_end = buf.stream_position()?;
        let data_offset = header_end.div_ceil(alignment) * alignment;
        buf.seek(SeekFrom::Start(0))?;

        Ok(GgufFile { path: path.to_path_buf(), version, size, alignment, data_offset, kv, array_types, key_order, tensors })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(Value::as_str)
    }

    /// Checks every descriptor against the pinned encodings and the file bounds; returns sized tensors.
    pub fn validate(&self, shard: u32) -> Result<Vec<TensorInfo>> {
        let mut out = Vec::with_capacity(self.tensors.len());
        let mut errors = vec![];
        for t in &self.tensors {
            match self.size_tensor(t, shard) {
                Ok(info) => out.push(info),
                Err(e) => errors.push(format!("{}: {e}", t.name)),
            }
        }
        if !errors.is_empty() {
            bail!("{} invalid tensor(s) in {}:\n  {}", errors.len(), self.path.display(), errors.join("\n  "));
        }
        let mut spans: Vec<(u64, u64, &str)> = out.iter().map(|t| (t.offset, t.offset + t.bytes, t.name.as_str())).collect();
        spans.sort();
        for w in spans.windows(2) {
            ensure!(w[0].1 <= w[1].0, "tensors {} and {} overlap in {}", w[0].2, w[1].2, self.path.display());
        }
        Ok(out)
    }

    fn size_tensor(&self, t: &RawTensor, shard: u32) -> Result<TensorInfo> {
        ensure!(t.ggml_type.is_known(), "encoding {} is not supported by the pinned ggml", t.ggml_type.name());
        ensure!(t.dims.iter().all(|&d| d > 0), "zero-sized dimension {:?}", t.dims);
        let row = t
            .ggml_type
            .row_bytes(t.dims[0])
            .with_context(|| format!("row of {} elements is not whole {} blocks of {}", t.dims[0], t.ggml_type, t.ggml_type.block_size().unwrap_or(0)))?;
        let bytes = t.dims[1..].iter().try_fold(row, |acc, &d| acc.checked_mul(d)).context("tensor size overflows")?;
        ensure!(t.offset.is_multiple_of(self.alignment), "offset {} not aligned to {}", t.offset, self.alignment);
        let end = self.data_offset.checked_add(t.offset).and_then(|o| o.checked_add(bytes)).context("offset overflows")?;
        ensure!(end <= self.size, "data ends at {end} beyond file size {} (truncated download?)", self.size);
        Ok(TensorInfo { name: t.name.clone(), ggml_type: t.ggml_type, dims: t.dims.clone(), bytes, shard, offset: self.data_offset + t.offset })
    }
}

/// `name-00001-of-00003.gguf` → all shard paths, in order. A single file returns itself.
pub fn shard_paths(first: &Path) -> Result<Vec<PathBuf>> {
    let name = first.file_name().and_then(|n| n.to_str()).context("invalid model path")?;
    let Some(stem) = name.strip_suffix(".gguf") else { return Ok(vec![first.to_path_buf()]) };
    let parts: Vec<&str> = stem.rsplitn(4, '-').collect();
    if parts.len() < 4 || parts[1] != "of" {
        return Ok(vec![first.to_path_buf()]);
    }
    let (Ok(count), Ok(_)) = (parts[0].parse::<u32>(), parts[2].parse::<u32>()) else { return Ok(vec![first.to_path_buf()]) };
    let width = parts[0].len();
    Ok((1..=count).map(|i| first.with_file_name(format!("{}-{i:0width$}-of-{}.gguf", parts[3], parts[0]))).collect())
}

/// A complete, validated GGUF artifact (one or more shards).
#[derive(Debug, Clone)]
pub struct GgufModel {
    pub shards: Vec<GgufFile>,
    pub tensors: Vec<TensorInfo>,
}

impl GgufModel {
    pub fn open(path: &Path) -> Result<GgufModel> {
        let paths = shard_paths(path)?;
        let missing: Vec<_> = paths.iter().filter(|p| !p.exists()).map(|p| p.display().to_string()).collect();
        ensure!(missing.is_empty(), "missing shard(s): {}", missing.join(", "));
        let shards = paths.iter().map(|p| GgufFile::open(p)).collect::<Result<Vec<_>>>()?;

        let count = shards[0].get("split.count").and_then(Value::as_u64).unwrap_or(1);
        ensure!(count as usize == shards.len(), "split.count is {count} but {} shard file(s) found", shards.len());
        let mut tensors = vec![];
        for (i, s) in shards.iter().enumerate() {
            if shards.len() > 1 {
                let no = s.get("split.no").and_then(Value::as_u64);
                ensure!(no == Some(i as u64), "{} declares split.no {:?}, expected {i}", s.path.display(), no);
            }
            tensors.extend(s.validate(i as u32)?);
        }
        if let Some(expected) = shards[0].get("split.tensors.count").and_then(Value::as_u64) {
            ensure!(expected as usize == tensors.len(), "split.tensors.count is {expected} but shards hold {} tensors", tensors.len());
        }
        let mut seen = HashSet::new();
        for t in &tensors {
            ensure!(seen.insert(t.name.as_str()), "tensor {} appears twice", t.name);
        }
        Ok(GgufModel { shards, tensors })
    }

    /// Metadata of the first shard, which carries the model-level keys.
    pub fn kv(&self) -> &BTreeMap<String, Value> {
        &self.shards[0].kv
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.shards[0].get(key)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// Minimal GGUF v3 writer for tests.
    pub struct Builder {
        kv: Vec<(String, u32, Vec<u8>)>,
        tensors: Vec<(String, Vec<u64>, u32, u64)>,
        data: Vec<u8>,
    }

    fn s(v: &str) -> Vec<u8> {
        let mut b = (v.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(v.as_bytes());
        b
    }

    impl Builder {
        pub fn new() -> Self {
            Builder { kv: vec![], tensors: vec![], data: vec![] }
        }
        pub fn str(mut self, k: &str, v: &str) -> Self {
            self.kv.push((k.into(), 8, s(v)));
            self
        }
        pub fn u32(mut self, k: &str, v: u32) -> Self {
            self.kv.push((k.into(), 4, v.to_le_bytes().to_vec()));
            self
        }
        pub fn u16(mut self, k: &str, v: u16) -> Self {
            self.kv.push((k.into(), 2, v.to_le_bytes().to_vec()));
            self
        }
        pub fn i32(mut self, k: &str, v: i32) -> Self {
            self.kv.push((k.into(), 5, v.to_le_bytes().to_vec()));
            self
        }
        pub fn strs(mut self, k: &str, vals: &[&str]) -> Self {
            let mut b = 8u32.to_le_bytes().to_vec();
            b.extend((vals.len() as u64).to_le_bytes());
            vals.iter().for_each(|v| b.extend(s(v)));
            self.kv.push((k.into(), 9, b));
            self
        }
        /// Adds an f16 tensor with zeroed payload.
        pub fn tensor(mut self, name: &str, dims: &[u64]) -> Self {
            let bytes = dims.iter().product::<u64>() * 2;
            let off = self.data.len() as u64;
            self.data.resize((off + bytes).div_ceil(32) as usize * 32, 0);
            self.tensors.push((name.into(), dims.to_vec(), 1, off));
            self
        }
        pub fn write(self, path: &Path) {
            let mut h = b"GGUF".to_vec();
            h.extend(3u32.to_le_bytes());
            h.extend((self.tensors.len() as u64).to_le_bytes());
            h.extend((self.kv.len() as u64).to_le_bytes());
            for (k, ty, v) in &self.kv {
                h.extend(s(k));
                h.extend(ty.to_le_bytes());
                h.extend(v);
            }
            for (name, dims, ty, off) in &self.tensors {
                h.extend(s(name));
                h.extend((dims.len() as u32).to_le_bytes());
                dims.iter().for_each(|d| h.extend(d.to_le_bytes()));
                h.extend(ty.to_le_bytes());
                h.extend(off.to_le_bytes());
            }
            h.resize(h.len().div_ceil(32) * 32, 0);
            h.extend(&self.data);
            File::create(path).unwrap().write_all(&h).unwrap();
        }
    }

    #[test]
    fn reads_metadata_and_sizes() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.gguf");
        Builder::new().str("general.architecture", "llama").u32("llama.block_count", 2).tensor("blk.0.attn_q.weight", &[64, 64]).write(&p);
        let m = GgufModel::open(&p).unwrap();
        assert_eq!(m.get("general.architecture").unwrap().as_str(), Some("llama"));
        assert_eq!(m.tensors[0].bytes, 64 * 64 * 2);
        assert_eq!(m.tensors[0].offset % 32, 0);
    }

    #[test]
    fn rejects_truncated_payload() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.gguf");
        Builder::new().str("general.architecture", "llama").tensor("w", &[64, 64]).write(&p);
        let f = std::fs::OpenOptions::new().write(true).open(&p).unwrap();
        f.set_len(f.metadata().unwrap().len() - 100).unwrap();
        let err = GgufModel::open(&p).unwrap_err().to_string();
        assert!(err.contains("beyond file size"), "{err}");
    }

    #[test]
    fn shards_must_be_complete() {
        let dir = tempfile::tempdir().unwrap();
        let p1 = dir.path().join("m-00001-of-00002.gguf");
        Builder::new()
            .str("general.architecture", "llama")
            .u16("split.no", 0)
            .u16("split.count", 2)
            .i32("split.tensors.count", 2)
            .tensor("a", &[32])
            .write(&p1);
        assert!(GgufModel::open(&p1).unwrap_err().to_string().contains("missing shard"));
        let p2 = dir.path().join("m-00002-of-00002.gguf");
        Builder::new().u16("split.no", 1).u16("split.count", 2).tensor("b", &[32]).write(&p2);
        let m = GgufModel::open(&p1).unwrap();
        assert_eq!(m.tensors.len(), 2);
        assert_eq!(m.tensors[1].shard, 1);
    }

    #[test]
    fn shard_names() {
        let v = shard_paths(Path::new("/m/qwen-q4_k_m-00001-of-00002.gguf")).unwrap();
        assert_eq!(v[1], PathBuf::from("/m/qwen-q4_k_m-00002-of-00002.gguf"));
        assert_eq!(shard_paths(Path::new("/m/model.gguf")).unwrap().len(), 1);
    }
}
