//! Safe wrapper over the flux-native C ABI. Only flux-worker links this crate, so native
//! failures stay inside a worker process.

use anyhow::{bail, Result};
use serde_json::Value;
use std::ffi::{c_char, CStr, CString};
use std::ptr::NonNull;

/// Revision of the backend this crate was compiled against.
pub const BACKEND_PIN: &str = env!("FLUX_BACKEND_PIN");
/// The pin plus a digest of the Flux patches applied to it.
pub const BACKEND_BUILD: &str = env!("FLUX_BACKEND_BUILD");

mod ffi {
    use std::ffi::c_char;

    #[repr(C)]
    pub struct Engine {
        _p: [u8; 0],
    }
    #[repr(C)]
    pub struct Sampler {
        _p: [u8; 0],
    }

    extern "C" {
        pub fn fx_free(p: *mut c_char);
        pub fn fx_backend_info() -> *mut c_char;
        pub fn fx_measure(params: *const c_char) -> *mut c_char;
        pub fn fx_probe_matmul(req: *const c_char) -> *mut c_char;
        pub fn fx_probe_copy(req: *const c_char) -> *mut c_char;
        pub fn fx_probe_contention(req: *const c_char) -> *mut c_char;
        pub fn fx_supports(req: *const c_char) -> *mut c_char;

        pub fn fx_engine_load(params: *const c_char, error: *mut *mut c_char) -> *mut Engine;
        pub fn fx_engine_free(e: *mut Engine);
        pub fn fx_engine_info(e: *mut Engine) -> *mut c_char;
        pub fn fx_tokenize(e: *mut Engine, text: *const c_char, len: i32, add_special: bool, out: *mut i32, cap: i32) -> i32;
        pub fn fx_token_piece(e: *mut Engine, token: i32, special: bool, buf: *mut c_char, cap: i32) -> i32;
        pub fn fx_is_eog(e: *mut Engine, token: i32) -> bool;
        pub fn fx_apply_template(e: *mut Engine, req: *const c_char) -> *mut c_char;
        pub fn fx_decode(e: *mut Engine, n: i32, tokens: *const i32, pos: *const i32, seq: *const i32, logits: *const i8) -> i32;
        pub fn fx_trace(e: *mut Engine, req: *const c_char) -> *mut c_char;
        pub fn fx_route_stats(e: *mut Engine, req: *const c_char) -> *mut c_char;
        pub fn fx_seq_clear(e: *mut Engine, seq: i32);

        pub fn fx_sampler_new(e: *mut Engine, sampling: *const c_char) -> *mut Sampler;
        pub fn fx_sampler_free(s: *mut Sampler);
        pub fn fx_sampler_accept_prompt(s: *mut Sampler, token: i32);
        pub fn fx_sampler_sample(s: *mut Sampler, e: *mut Engine, idx: i32) -> i32;
    }
}

/// Takes ownership of a malloc'd JSON string from the bridge and parses it, surfacing `"error"`.
fn take_json(p: *mut c_char) -> Result<Value> {
    if p.is_null() {
        bail!("native call returned no result");
    }
    let text = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { ffi::fx_free(p) };
    let v: Value = serde_json::from_str(&text)?;
    if let Some(e) = v.get("error").and_then(Value::as_str) {
        bail!("{e}");
    }
    Ok(v)
}

fn call(f: unsafe extern "C" fn(*const c_char) -> *mut c_char, input: &Value) -> Result<Value> {
    let c = CString::new(input.to_string())?;
    take_json(unsafe { f(c.as_ptr()) })
}

pub fn backend_info() -> Result<Value> {
    take_json(unsafe { ffi::fx_backend_info() })
}

/// Allocation dry run: what the backend would place on each device for these load parameters.
pub fn measure(params: &Value) -> Result<Value> {
    call(ffi::fx_measure, params)
}

pub fn probe_matmul(req: &Value) -> Result<Value> {
    call(ffi::fx_probe_matmul, req)
}

pub fn probe_copy(req: &Value) -> Result<Value> {
    call(ffi::fx_probe_copy, req)
}

pub fn probe_contention(req: &Value) -> Result<Value> {
    call(ffi::fx_probe_contention, req)
}

pub fn supports(req: &Value) -> Result<Value> {
    call(ffi::fx_supports, req)
}

/// A loaded model and context. Not thread-safe: use from one thread at a time.
pub struct Engine {
    ptr: NonNull<ffi::Engine>,
}

// The backend context may move between threads as long as only one uses it at a time.
unsafe impl Send for Engine {}

impl Engine {
    pub fn load(params: &Value) -> Result<Engine> {
        let c = CString::new(params.to_string())?;
        let mut err: *mut c_char = std::ptr::null_mut();
        let p = unsafe { ffi::fx_engine_load(c.as_ptr(), &mut err) };
        match NonNull::new(p) {
            Some(ptr) => Ok(Engine { ptr }),
            None => {
                let msg = if err.is_null() {
                    "model load failed".to_string()
                } else {
                    let m = unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned();
                    unsafe { ffi::fx_free(err) };
                    m
                };
                bail!("{msg}")
            }
        }
    }

    pub fn info(&self) -> Result<Value> {
        take_json(unsafe { ffi::fx_engine_info(self.ptr.as_ptr()) })
    }

    pub fn tokenize(&self, text: &str, add_special: bool) -> Vec<i32> {
        let mut out = vec![0i32; text.len() + 8];
        loop {
            let n = unsafe { ffi::fx_tokenize(self.ptr.as_ptr(), text.as_ptr().cast(), text.len() as i32, add_special, out.as_mut_ptr(), out.len() as i32) };
            if n >= 0 {
                out.truncate(n as usize);
                return out;
            }
            out.resize((-n) as usize, 0);
        }
    }

    /// Raw bytes of a token's text; may end inside a UTF-8 sequence.
    pub fn token_bytes(&self, token: i32, special: bool) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        loop {
            let n = unsafe { ffi::fx_token_piece(self.ptr.as_ptr(), token, special, buf.as_mut_ptr().cast(), buf.len() as i32) };
            if n >= 0 {
                buf.truncate(n as usize);
                return buf;
            }
            buf.resize((-n) as usize, 0);
        }
    }

    pub fn is_eog(&self, token: i32) -> bool {
        unsafe { ffi::fx_is_eog(self.ptr.as_ptr(), token) }
    }

    pub fn apply_template(&self, req: &Value) -> Result<Value> {
        let c = CString::new(req.to_string())?;
        take_json(unsafe { ffi::fx_apply_template(self.ptr.as_ptr(), c.as_ptr()) })
    }

    /// One backend step over the given tokens; `logits[i] != 0` requests output for row i.
    pub fn decode(&mut self, tokens: &[i32], pos: &[i32], seq: &[i32], logits: &[i8]) -> Result<()> {
        assert!(tokens.len() == pos.len() && pos.len() == seq.len() && seq.len() == logits.len());
        let rc = unsafe { ffi::fx_decode(self.ptr.as_ptr(), tokens.len() as i32, tokens.as_ptr(), pos.as_ptr(), seq.as_ptr(), logits.as_ptr()) };
        match rc {
            0 => Ok(()),
            1 => bail!("no KV slot available for this batch (context full)"),
            2 => bail!("decode aborted"),
            -100 => bail!("batch larger than n_batch"),
            rc => bail!("llama_decode failed with status {rc}"),
        }
    }

    /// Expert selection counts per MoE layer; requires the engine to be loaded with `"trace": true`.
    pub fn route_stats(&mut self, req: &Value) -> Result<Value> {
        let c = CString::new(req.to_string())?;
        take_json(unsafe { ffi::fx_route_stats(self.ptr.as_ptr(), c.as_ptr()) })
    }

    /// Per-device, per-op decode time; requires the engine to be loaded with `"trace": true`.
    pub fn trace(&mut self, req: &Value) -> Result<Value> {
        let c = CString::new(req.to_string())?;
        take_json(unsafe { ffi::fx_trace(self.ptr.as_ptr(), c.as_ptr()) })
    }

    pub fn seq_clear(&mut self, seq: i32) {
        unsafe { ffi::fx_seq_clear(self.ptr.as_ptr(), seq) }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ffi::fx_engine_free(self.ptr.as_ptr()) }
    }
}

/// llama-server's sampling chain for one sequence.
pub struct Sampler {
    ptr: NonNull<ffi::Sampler>,
}

unsafe impl Send for Sampler {}

impl Sampler {
    pub fn new(engine: &Engine, sampling: &Value) -> Result<Sampler> {
        let c = CString::new(sampling.to_string())?;
        let p = unsafe { ffi::fx_sampler_new(engine.ptr.as_ptr(), c.as_ptr()) };
        NonNull::new(p).map(|ptr| Sampler { ptr }).ok_or_else(|| anyhow::anyhow!("invalid sampling parameters"))
    }

    pub fn accept_prompt(&mut self, token: i32) {
        unsafe { ffi::fx_sampler_accept_prompt(self.ptr.as_ptr(), token) }
    }

    pub fn sample(&mut self, engine: &mut Engine, row: i32) -> i32 {
        unsafe { ffi::fx_sampler_sample(self.ptr.as_ptr(), engine.ptr.as_ptr(), row) }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { ffi::fx_sampler_free(self.ptr.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_reports_pinned_build_and_cpu() {
        let info = backend_info().unwrap();
        let devices = info["devices"].as_array().unwrap();
        assert!(devices.iter().any(|d| d["kind"] == "cpu"));
        assert!(info["max_devices"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn supports_reports_known_types() {
        let v = supports(&serde_json::json!({"device": "CPU", "types": ["q4_K", "f16"]})).unwrap();
        assert_eq!(v["support"]["q4_K"]["mul_mat"], true);
    }

    #[test]
    fn cpu_matmul_probe_times_runs() {
        let v = probe_matmul(&serde_json::json!({"device": "CPU", "type": "q8_0", "k": 512, "n": 256, "batches": [1], "iters": 3, "threads": 2})).unwrap();
        assert_eq!(v["results"][0]["micros"].as_array().unwrap().len(), 3);
    }
}
