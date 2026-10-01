//! Assignment → backend placement parameters that reproduce it exactly.
//! The pinned llama.cpp assigns block i (and the output as pseudo-block L) to the CPU when
//! i < L + 1 - n_gpu_layers, else to the device whose cumulative tensor_split share first exceeds
//! (i - start) / active. Integer layer counts as split weights make that mapping exact.

use crate::cost::CPU;
use crate::layers::Layout;
use crate::search::Assignment;
use flux_core::plan::{Placement, TensorOverride};

pub fn placement(l: &Layout, a: &Assignment) -> Placement {
    let n = a.layer_device.len();
    let cpu_prefix = a.layer_device.iter().take_while(|d| *d == CPU).count();
    let gpu_tail = a.output_device != CPU;
    let (n_gpu_layers, tensor_split) = if a.devices.is_empty() || !gpu_tail {
        (0, vec![])
    } else {
        let mut split = vec![0f32; a.devices.len()];
        for d in a.layer_device[cpu_prefix..].iter().chain(std::iter::once(&a.output_device)) {
            let k = a.devices.iter().position(|x| x == d).expect("device listed");
            split[k] += 1.0;
        }
        ((n + 1 - cpu_prefix) as i32, split)
    };
    let mut overrides = vec![];
    for (i, b) in l.blocks.iter().enumerate() {
        let names: Vec<&str> = b.experts.iter().zip(&a.on_cpu[i]).filter(|(_, &host)| host && a.layer_device[i] != CPU).map(|(e, _)| e.name.as_str()).collect();
        if !names.is_empty() {
            let alts: Vec<String> = names.iter().map(|n| regex_escape(n)).collect();
            overrides.push(TensorOverride { pattern: format!("^({})$", alts.join("|")), device: CPU.into() });
        }
    }
    Placement {
        devices: a.devices.clone(),
        layer_device: a.layer_device.clone(),
        output_device: a.output_device.clone(),
        overrides,
        n_gpu_layers,
        tensor_split,
        split_mode: flux_core::plan::SplitMode::Layer,
    }
}

fn regex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        if r".+*?()|[]{}^$\".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// The backend's own mapping (codex-verified against llama-model.cpp at the pin), used to check `placement`.
pub fn backend_mapping(n_layer_all: usize, devices: &[String], n_gpu_layers: i32, split: &[f32]) -> Vec<String> {
    let l = n_layer_all;
    let g = if n_gpu_layers >= 0 { n_gpu_layers as usize } else { l + 1 };
    let start = (l + 1).saturating_sub(g);
    let active = if devices.is_empty() { 0 } else { g.min(l + 1) };
    let mut cum = vec![];
    let mut acc = 0f32;
    for &w in split {
        acc += w;
        cum.push(acc);
    }
    let total = acc;
    (0..=l)
        .map(|i| {
            if i < start || i - start >= active {
                return CPU.to_string();
            }
            let frac = (i - start) as f32 / active as f32;
            let k = cum.iter().position(|&c| c / total > frac).unwrap_or(devices.len() - 1);
            devices[k].clone()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{Block, ExpertTensor, Head};
    use flux_core::ggml_type::GgmlType;

    fn layout(n: usize) -> Layout {
        Layout {
            blocks: (0..n)
                .map(|i| Block {
                    index: i as u32,
                    experts: vec![
                        ExpertTensor { name: format!("blk.{i}.ffn_up_exps.weight"), ggml_type: GgmlType::Q8_0, bytes: 1, k: 1, n: 1 },
                        ExpertTensor { name: format!("blk.{i}.ffn_down_exps.weight"), ggml_type: GgmlType::Q8_0, bytes: 1, k: 1, n: 1 },
                    ],
                    ..Default::default()
                })
                .collect(),
            head: Head::default(),
            n_expert: 8,
            n_expert_used: 2,
            n_embd: 1,
            n_vocab: 1,
        }
    }

    fn assignment(devs: &[&str], output: &str, on_cpu: Vec<Vec<bool>>) -> Assignment {
        let mut d: Vec<String> = vec![];
        for x in devs.iter().chain(std::iter::once(&output)) {
            if *x != CPU && d.last().map(String::as_str) != Some(*x) {
                d.push(x.to_string());
            }
        }
        Assignment {
            devices: d,
            layer_device: devs.iter().map(|s| s.to_string()).collect(),
            output_device: output.into(),
            on_cpu,
            decode_s: 0.0,
            bytes: Default::default(),
        }
    }

    #[test]
    fn reproduces_backend_mapping() {
        let l = layout(10);
        let cases = [
            (vec!["CPU", "CPU", "CUDA0", "CUDA0", "CUDA0", "CUDA0", "CUDA1", "CUDA1", "CUDA1", "CUDA1"], "CUDA1"),
            (vec!["CUDA1"; 10], "CUDA1"),
            (vec!["CPU"; 10], "CUDA0"),
            (vec!["CPU"; 10], "CPU"),
            (vec!["CUDA0", "CUDA0", "CUDA0", "CUDA0", "CUDA0", "CUDA0", "CUDA0", "CUDA0", "CUDA0", "CUDA0"], "CUDA1"),
        ];
        for (devs, out) in cases {
            let a = assignment(&devs, out, vec![vec![]; 10]);
            let p = placement(&l, &a);
            let got = backend_mapping(10, &p.devices, p.n_gpu_layers, &p.tensor_split);
            let mut want: Vec<String> = devs.iter().map(|s| s.to_string()).collect();
            want.push(out.to_string());
            assert_eq!(got, want, "devices {:?} ngl {} split {:?}", p.devices, p.n_gpu_layers, p.tensor_split);
        }
    }

    #[test]
    fn host_experts_become_anchored_overrides() {
        let l = layout(2);
        let a = assignment(&["CUDA0", "CUDA0"], "CUDA0", vec![vec![true, false], vec![true, true]]);
        let p = placement(&l, &a);
        assert_eq!(p.overrides[0].pattern, r"^(blk\.0\.ffn_up_exps\.weight)$");
        assert_eq!(p.overrides[1].pattern, r"^(blk\.1\.ffn_up_exps\.weight|blk\.1\.ffn_down_exps\.weight)$");
        assert!(p.overrides.iter().all(|o| o.device == "CPU"));
    }
}
