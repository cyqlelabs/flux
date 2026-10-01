//! Tensor encodings of the pinned ggml revision (see `backend.pin`).
//! Values were printed from `ggml_blck_size` / `ggml_type_size` of that build.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GgmlType(pub u32);

struct Traits {
    id: u32,
    name: &'static str,
    block: u64,
    size: u64,
}

const TYPES: &[Traits] = &[
    Traits { id: 0, name: "f32", block: 1, size: 4 },
    Traits { id: 1, name: "f16", block: 1, size: 2 },
    Traits { id: 2, name: "q4_0", block: 32, size: 18 },
    Traits { id: 3, name: "q4_1", block: 32, size: 20 },
    Traits { id: 6, name: "q5_0", block: 32, size: 22 },
    Traits { id: 7, name: "q5_1", block: 32, size: 24 },
    Traits { id: 8, name: "q8_0", block: 32, size: 34 },
    Traits { id: 9, name: "q8_1", block: 32, size: 36 },
    Traits { id: 10, name: "q2_K", block: 256, size: 84 },
    Traits { id: 11, name: "q3_K", block: 256, size: 110 },
    Traits { id: 12, name: "q4_K", block: 256, size: 144 },
    Traits { id: 13, name: "q5_K", block: 256, size: 176 },
    Traits { id: 14, name: "q6_K", block: 256, size: 210 },
    Traits { id: 15, name: "q8_K", block: 256, size: 292 },
    Traits { id: 16, name: "iq2_xxs", block: 256, size: 66 },
    Traits { id: 17, name: "iq2_xs", block: 256, size: 74 },
    Traits { id: 18, name: "iq3_xxs", block: 256, size: 98 },
    Traits { id: 19, name: "iq1_s", block: 256, size: 50 },
    Traits { id: 20, name: "iq4_nl", block: 32, size: 18 },
    Traits { id: 21, name: "iq3_s", block: 256, size: 110 },
    Traits { id: 22, name: "iq2_s", block: 256, size: 82 },
    Traits { id: 23, name: "iq4_xs", block: 256, size: 136 },
    Traits { id: 24, name: "i8", block: 1, size: 1 },
    Traits { id: 25, name: "i16", block: 1, size: 2 },
    Traits { id: 26, name: "i32", block: 1, size: 4 },
    Traits { id: 27, name: "i64", block: 1, size: 8 },
    Traits { id: 28, name: "f64", block: 1, size: 8 },
    Traits { id: 29, name: "iq1_m", block: 256, size: 56 },
    Traits { id: 30, name: "bf16", block: 1, size: 2 },
    Traits { id: 34, name: "tq1_0", block: 256, size: 54 },
    Traits { id: 35, name: "tq2_0", block: 256, size: 66 },
    Traits { id: 39, name: "mxfp4", block: 32, size: 17 },
    Traits { id: 40, name: "nvfp4", block: 64, size: 36 },
    Traits { id: 41, name: "q1_0", block: 128, size: 18 },
    Traits { id: 42, name: "q2_0", block: 64, size: 18 },
];

impl GgmlType {
    pub const F32: GgmlType = GgmlType(0);
    pub const F16: GgmlType = GgmlType(1);
    pub const Q4_0: GgmlType = GgmlType(2);
    pub const Q8_0: GgmlType = GgmlType(8);

    fn traits(self) -> Option<&'static Traits> {
        TYPES.iter().find(|t| t.id == self.0)
    }

    /// Whether the pinned ggml knows this encoding.
    pub fn is_known(self) -> bool {
        self.traits().is_some()
    }

    pub fn name(self) -> String {
        self.traits().map_or_else(|| format!("unknown({})", self.0), |t| t.name.to_string())
    }

    pub fn from_name(name: &str) -> Option<GgmlType> {
        TYPES.iter().find(|t| t.name.eq_ignore_ascii_case(name)).map(|t| GgmlType(t.id))
    }

    pub fn block_size(self) -> Option<u64> {
        self.traits().map(|t| t.block)
    }

    /// Bytes of one row of `n` elements, or `None` for an unknown type or a row that is not whole blocks.
    pub fn row_bytes(self, n: u64) -> Option<u64> {
        let t = self.traits()?;
        n.is_multiple_of(t.block).then(|| n / t.block * t.size)
    }

    /// Bytes per element, fractional for block encodings.
    pub fn bytes_per_element(self) -> Option<f64> {
        self.traits().map(|t| t.size as f64 / t.block as f64)
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_bytes_match_block_layout() {
        assert_eq!(GgmlType::F16.row_bytes(128), Some(256));
        assert_eq!(GgmlType::Q8_0.row_bytes(4096), Some(4096 / 32 * 34));
        assert_eq!(GgmlType::Q8_0.row_bytes(100), None);
        assert_eq!(GgmlType(99).row_bytes(32), None);
        assert_eq!(GgmlType::from_name("Q4_K"), Some(GgmlType(12)));
    }
}
