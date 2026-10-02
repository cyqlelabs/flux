#!/usr/bin/env python3
"""Convert Qwen3.8-Flash-Next Strata MTP runtime weights to a tensor-only GGUF.

Usage:
  python scripts/strata-mtp-to-gguf.py --rt /path/to/mtp/rt --out mtp.gguf --verify

The output holds only the MTP block (blk.48.*); it needs the trunk's metadata,
token embeddings and output head. Graft it with:
  flux plan <trunk.gguf> --heads mtp.gguf
Norm weights already include Gemma's +1; no dequantization or requantization
is performed. The reduced draft vocabulary is intentionally unused.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/llama.cpp/gguf-py"))
import gguf  # noqa: E402


H, FF, NE, HC, LR = 2560, 640, 512, 4, 320
Q = gguf.GGMLQuantizationType
KINDS = {"q8_0": Q.Q8_0, "bf16": Q.BF16, "f32": Q.F32}
# Source name -> (llama.cpp name, rows, columns, runtime kind).
WEIGHTS = {
    "fc_embedding.weight": ("nextn.e_proj", H, H, "q8_0"),
    "fc_hidden.weight": ("nextn.h_proj", H, H, "q8_0"),
    "pre_fc_norm_embedding.weight": ("nextn.enorm", 1, H, "f32"),
    "pre_fc_norm_hidden.weight": ("nextn.hnorm", 1, HC * H, "f32"),
    "hyper_connection_mixer.hc_norm.weight": ("nextn.hc_head_norm", 1, HC * H, "f32"),
    "hyper_connection_mixer.input_mix_weight_down.weight": ("nextn.hc_head_down", LR, HC * H, "bf16"),
    "hyper_connection_mixer.input_mix_weight_up.weight": ("nextn.hc_head_up", HC * H, LR, "bf16"),
    "self_attn.q_proj.weight": ("attn_q", 48 * 256, H, "q8_0"),
    "self_attn.k_proj.weight": ("attn_k", 2 * 256, H, "q8_0"),
    "self_attn.v_proj.weight": ("attn_v", 2 * 256, H, "q8_0"),
    "self_attn.o_proj.weight": ("attn_output", H, 24 * 256, "q8_0"),
    "self_attn.q_norm.weight": ("attn_q_norm", 1, 256, "f32"),
    "self_attn.k_norm.weight": ("attn_k_norm", 1, 256, "f32"),
    "self_attn.indexer.index_qk_proj.weight": ("indexer.qk_proj", 5 * 128, H, "q8_0"),
    "self_attn.indexer.q_layernorm.weight": ("indexer.q_norm", 1, 128, "f32"),
    "self_attn.indexer.k_layernorm.weight": ("indexer.k_norm", 1, 128, "f32"),
    "mlp.gate.weight": ("ffn_gate_inp", NE, H, "bf16"),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp", FF, H, "q8_0"),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp", FF, H, "q8_0"),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp", H, FF, "q8_0"),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp", 1, H, "bf16"),
}
for source, dest in (("attn", "attn"), ("mlp", "ffn")):
    for suffix, target, rows, cols, kind in (
        ("hc_norm", "norm", 1, HC * H, "f32"),
        ("input_mix_weight_down", "down", LR, HC * H, "bf16"),
        ("input_mix_weight_up", "up", HC * H, LR, "bf16"),
        ("block_inject_weight", "inject", HC, HC * H, "bf16"),
    ):
        WEIGHTS[f"{source}_hyper_connection.{suffix}.weight"] = (f"hc_{dest}_{target}", rows, cols, kind)


@dataclass
class DenseTensor:
    name: str
    qtype: Q
    data: np.ndarray


def read_dense(rt: Path) -> list[DenseTensor]:
    raw = np.memmap(rt / "dense.bin", mode="r", dtype=np.uint8)
    seen = set()
    tensors = []
    end = 0
    for line in (rt / "dense.txt").read_text().splitlines():
        if not line.strip():
            continue
        name, kind, *numbers = line.split()
        rows, cols, offset, size = map(int, numbers)
        if name not in WEIGHTS or name in seen:
            raise ValueError(f"unknown or duplicate tensor: {name}")
        dest, want_rows, want_cols, want_kind = WEIGHTS[name]
        if (rows, cols, kind) != (want_rows, want_cols, want_kind):
            raise ValueError(f"{name}: unexpected shape/type {(rows, cols, kind)}")
        block, width = gguf.GGML_QUANT_SIZES[KINDS[kind]]
        row_bytes = cols // block * width
        if cols % block or size != rows * row_bytes or offset != end or offset + size > raw.size:
            raise ValueError(f"{name}: invalid size/offset")
        data = raw[offset:offset + size].reshape(rows, row_bytes)
        if dest == "indexer.qk_proj":
            # Same Q-then-K row split as conversion/qwen4exp.py.
            tensors.append(DenseTensor("blk.48.indexer.q_proj.weight", KINDS[kind], data[:512]))
            tensors.append(DenseTensor("blk.48.indexer.k_proj.weight", KINDS[kind], data[512:]))
        else:
            if rows == 1:
                data = data.reshape(-1)
            tensors.append(DenseTensor(f"blk.48.{dest}.weight", KINDS[kind], data))
        seen.add(name)
        end = offset + size
    if seen != WEIGHTS.keys() or end != raw.size:
        raise ValueError(f"incomplete manifest or trailing dense bytes; missing {WEIGHTS.keys() - seen}")
    return tensors


def restore_experts(raw: np.ndarray, which: str, h: int = H, ff: int = FF) -> np.ndarray:
    """Invert mtp_rt.blob_of: [GU codes, D codes, GU scales, D scales]."""
    if h % 64 or ff % 64 or which not in ("gate", "up", "down"):
        raise ValueError("invalid Q2_0 expert geometry or projection")
    gu_blocks, dn_blocks = 2 * ff * (h // 64), h * (ff // 64)
    if raw.ndim != 2 or raw.shape[1] != (gu_blocks + dn_blocks) * 18:
        raise ValueError("invalid expert blob size")
    rows, blocks = (h, ff // 64) if which == "down" else (ff, h // 64)
    out = np.empty((raw.shape[0], rows, blocks, 18), dtype=np.uint8)
    for e, blob in enumerate(raw):
        if which == "down":
            codes = blob[gu_blocks * 16:(gu_blocks + dn_blocks) * 16].reshape(rows, blocks, 16)
            scales = blob[(gu_blocks + dn_blocks) * 16 + gu_blocks * 2:].reshape(rows, blocks, 2)
        else:
            parity = 0 if which == "gate" else 1
            codes = blob[:gu_blocks * 16].reshape(2 * ff, blocks, 16)[parity::2]
            scales = blob[(gu_blocks + dn_blocks) * 16:][:gu_blocks * 2].reshape(2 * ff, blocks, 2)[parity::2]
        out[e, :, :, :2] = scales
        out[e, :, :, 2:] = codes
    return out.reshape(raw.shape[0], rows, blocks * 18)


def verify_experts(raw: np.ndarray, tensors: dict, h: int = H, ff: int = FF) -> None:
    """Rebuild each original runtime blob from the three GGUF tensors."""
    for e, original in enumerate(raw):
        gu = np.empty((2 * ff, h // 64, 18), dtype=np.uint8)
        gu[0::2] = tensors["gate"][e].reshape(ff, h // 64, 18)
        gu[1::2] = tensors["up"][e].reshape(ff, h // 64, 18)
        down = tensors["down"][e].reshape(h, ff // 64, 18)
        restored = b"".join(part.tobytes() for part in (gu[:, :, 2:], down[:, :, 2:], gu[:, :, :2], down[:, :, :2]))
        if restored != original.tobytes():
            raise ValueError(f"expert {e}: byte round-trip failed")


def self_test() -> None:
    h, ff, ne = 128, 64, 3
    raw = np.random.default_rng(42).integers(0, 256, (ne, 3 * h * ff // 64 * 18), dtype=np.uint8)
    tensors = {key: restore_experts(raw, key, h, ff) for key in ("gate", "up", "down")}
    verify_experts(raw, tensors, h, ff)
    # Known row/plane offsets independently guard against matching inverse mistakes.
    assert tensors["gate"][0, 0, :2].tolist() == raw[0, 3 * h * ff // 64 * 16:][:2].tolist()
    assert tensors["up"][0, 0, 2:18].tolist() == raw[0, h // 64 * 16:][:16].tolist()
    print("Q2_0 synthetic byte round-trip passed")


def convert(rt: Path, output: Path, verify: bool) -> None:
    dense = read_dense(rt)
    blob_bytes = 3 * H * FF // 64 * 18
    if (rt / "experts.bin").stat().st_size != NE * blob_bytes:
        raise ValueError("experts.bin: unexpected file size")
    experts = np.memmap(rt / "experts.bin", mode="r", dtype=np.uint8, shape=(NE, blob_bytes))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(output, "qwen4exp")
    writer.add_name("Qwen3.8-Flash-Next MTP block (Strata runtime)")
    writer.add_uint32("qwen4exp.nextn_predict_layers", 1)
    for t in dense:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.qtype)
    for which in ("gate", "up", "down"):
        shape = (NE, H, FF // 64 * 18) if which == "down" else (NE, FF, H // 64 * 18)
        writer.add_tensor_info(f"blk.48.ffn_{which}_exps.weight", shape, np.dtype("uint8"),
                               NE * H * FF // 64 * 18, raw_dtype=Q.Q2_0)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for t in dense:
            writer.write_tensor_data(t.data, tensor_endianess=gguf.GGUFEndian.LITTLE)
        for which in ("gate", "up", "down"):
            data = restore_experts(experts, which)
            writer.write_tensor_data(data, tensor_endianess=gguf.GGUFEndian.LITTLE)
            del data
    finally:
        writer.close()
    if verify:
        reader = gguf.GGUFReader(output)
        saved = {t.name: t for t in reader.tensors}
        for t in dense:
            actual = saved[t.name]
            if actual.tensor_type != t.qtype or not np.array_equal(actual.data.view(np.uint8).reshape(-1), t.data.reshape(-1)):
                raise ValueError(f"{t.name}: byte/type round-trip failed")
        verify_experts(experts, {k: saved[f"blk.48.ffn_{k}_exps.weight"].data for k in ("gate", "up", "down")})
        print("All dense bytes/types and all 512 expert blobs verified")
    print(f"{output}: {len(dense) + 3} tensors, {output.stat().st_size:,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rt", type=Path, help="Strata rt directory")
    parser.add_argument("--out", type=Path, help="output MTP-only GGUF")
    parser.add_argument("--verify", action="store_true", help="verify every source byte after writing")
    parser.add_argument("--self-test", action="store_true", help="run a tiny synthetic Q2_0 relayout test")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.rt is not None and args.out is not None:
        convert(args.rt, args.out, args.verify)
    elif not args.self_test or args.rt is not None or args.out is not None:
        parser.error("--rt and --out are required together")


if __name__ == "__main__":
    main()
