#!/usr/bin/env bash
# Builds a relocatable release: Flux binaries, the pinned llama.cpp libraries and tools, the converter,
# and a flux.toml pointing at them. Output: dist/flux-<version>-linux-x86_64.tar.gz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
VERSION="$(grep -m1 '^version' "$ROOT/Cargo.toml" | cut -d'"' -f2)"
NAME="flux-$VERSION-linux-x86_64"
STAGE="$ROOT/dist/$NAME"
rm -rf "$STAGE" && mkdir -p "$STAGE/bin" "$STAGE/llama.cpp"

"$ROOT/scripts/build-backend.sh"
# Installed llama.cpp finds its libraries next to itself.
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' >/dev/null
cmake --install "$SRC/build" --prefix "$STAGE/llama.cpp" >/dev/null
cp -r "$SRC/convert_hf_to_gguf.py" "$SRC/conversion" "$SRC/gguf-py" "$STAGE/llama.cpp/"

FLUX_RPATH='$ORIGIN/../llama.cpp/lib' cargo build --release --manifest-path "$ROOT/Cargo.toml" -p flux-cli -p flux-worker
cp "$ROOT/target/release/flux" "$ROOT/target/release/flux-worker" "$STAGE/bin/"
cp "$ROOT/backend.pin" "$STAGE/"
cat > "$STAGE/flux.toml" <<TOML
# Unpack to /opt, then copy this to ~/.config/flux/flux.toml (or set FLUX_CONFIG) and add cache_dir/models_dir.
llama_dir = "/opt/$NAME/llama.cpp"
TOML
tar -C "$ROOT/dist" -czf "$ROOT/dist/$NAME.tar.gz" "$NAME"
echo "$ROOT/dist/$NAME.tar.gz"
