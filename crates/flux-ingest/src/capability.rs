//! Compatibility key: architecture + tensor encodings + backend revision (+ device capability at plan time).
//! Fails early with the missing capability and an eligible alternative; never guesses a tensor mapping.

use crate::{BACKEND_ARCHITECTURES, BACKEND_PIN, CONVERTIBLE_HF_ARCHITECTURES};
use flux_core::config::FluxConfig;
use flux_core::model::{Compatibility, ModelFormat};

pub struct Subject<'a> {
    pub format: ModelFormat,
    /// GGUF `general.architecture` or HF `model_type`.
    pub architecture: Option<&'a str>,
    /// HF `architectures[0]`, e.g. `Qwen2ForCausalLM`.
    pub hf_class: Option<&'a str>,
}

pub fn assess(s: &Subject, cfg: &FluxConfig) -> Compatibility {
    let pin = &BACKEND_PIN[..9];
    let external: Vec<String> = cfg
        .engines
        .iter()
        .filter(|(_, e)| e.formats.contains(&s.format) && s.architecture.is_some_and(|a| e.architectures.iter().any(|x| x == a)))
        .map(|(n, _)| n.clone())
        .collect();

    let mut missing = vec![];
    let mut alternatives = vec![];
    match s.format {
        ModelFormat::Gguf => {
            match s.architecture {
                Some(a) if BACKEND_ARCHITECTURES.contains(&a) => {}
                Some(a) => missing.push(format!("architecture `{a}` is not implemented by llama.cpp@{pin}")),
                None => missing.push("general.architecture is absent".into()),
            }
            if missing.is_empty() {
                let mut engines = vec!["native".to_string(), "llama-server".to_string()];
                engines.extend(external);
                return Compatibility::Executable { engines };
            }
            if !external.is_empty() {
                return Compatibility::Executable { engines: external };
            }
        }
        ModelFormat::Safetensors | ModelFormat::Exl3 | ModelFormat::Gptq | ModelFormat::Awq | ModelFormat::Fp8 => {
            if !external.is_empty() {
                return Compatibility::Executable { engines: external };
            }
            missing.push(format!("no registered engine is certified for {:?} artifacts of `{}`", s.format, s.architecture.unwrap_or("?")));
            match s.format {
                ModelFormat::Exl3 => alternatives.push("register an ExLlamaV3 server (e.g. TabbyAPI) under [engines] with formats = [\"exl3\"]".into()),
                ModelFormat::Gptq | ModelFormat::Awq | ModelFormat::Fp8 => {
                    alternatives.push("register a backend certified for this quantization metadata and GPU (e.g. vLLM) under [engines]".into())
                }
                _ => {}
            }
            if s.hf_class.is_some_and(|c| CONVERTIBLE_HF_ARCHITECTURES.contains(&c)) && s.format == ModelFormat::Safetensors {
                alternatives.push(format!(
                    "convert offline with third_party/llama.cpp/convert_hf_to_gguf.py (supports {}), then `flux inspect` the GGUF",
                    s.hf_class.unwrap_or_default()
                ));
            }
        }
        ModelFormat::Unknown => missing.push("unrecognized container or quantization method; metadata inspection only".into()),
    }
    if alternatives.is_empty() {
        alternatives.push("none known for this backend revision".into());
    }
    Compatibility::InspectOnly { missing, alternatives }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(format: ModelFormat, arch: &'static str, class: &'static str) -> Subject<'static> {
        Subject { format, architecture: Some(arch), hf_class: Some(class) }
    }

    #[test]
    fn gguf_known_arch_is_executable() {
        let c = assess(&subject(ModelFormat::Gguf, "qwen2", ""), &FluxConfig::default());
        assert!(matches!(c, Compatibility::Executable { ref engines } if engines[0] == "native"));
    }

    #[test]
    fn unknown_arch_is_inspect_only() {
        let c = assess(&subject(ModelFormat::Gguf, "made-up-arch", ""), &FluxConfig::default());
        let Compatibility::InspectOnly { missing, .. } = c else { panic!() };
        assert!(missing[0].contains("made-up-arch"));
    }

    #[test]
    fn safetensors_offers_conversion() {
        let c = assess(&subject(ModelFormat::Safetensors, "qwen2", "Qwen2ForCausalLM"), &FluxConfig::default());
        let Compatibility::InspectOnly { alternatives, .. } = c else { panic!() };
        assert!(alternatives.iter().any(|a| a.contains("convert_hf_to_gguf")), "{alternatives:?}");
    }
}
