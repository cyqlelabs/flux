#!/usr/bin/env bash
# Builds the pinned llama.cpp backend that flux-native links against.
# CUDA architectures: 75 = RTX 20xx (2060), 86 = RTX 30xx (3060). Override with CUDA_ARCHS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
PIN="$(cat "$ROOT/backend.pin")"
HEAD="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$HEAD" != "$PIN" ]]; then
  echo "third_party/llama.cpp is at $HEAD, backend.pin requires $PIN" >&2
  exit 1
fi
# Flux extensions to the pinned revision (expert split for hot/cold MoE residency).
for patch in "$ROOT"/patches/llama.cpp/*.patch; do
  if git -C "$SRC" apply --reverse --check "$patch" 2>/dev/null; then
    continue
  fi
  git -C "$SRC" apply "$patch"
done
cmake -S "$SRC" -B "$SRC/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA="${GGML_CUDA:-ON}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS:-75;86}" \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON
cmake --build "$SRC/build" --config Release -j "$(nproc)" \
  --target llama llama-common ggml llama-server llama-bench llama-perplexity
