#!/usr/bin/env bash
# Build a pinned llama.cpp for this machine and copy the binaries into bin/<os>-<arch>/.
# Pinning matters: llama.cpp renames flags (e.g. --no-mmap became -lm/--load-mode),
# and PocketLLM generates command lines for this exact version.
#
# usage: scripts/build_llamacpp.sh [backend]   backend: auto (default) | cpu | metal | cuda | vulkan
set -euo pipefail

LLAMA_COMMIT="${LLAMA_COMMIT:-caa8191df159f764fb0160e3d2fc6be008948aee}"   # verified 2026-10-26
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${LLAMA_SRC:-$ROOT/.build/llama.cpp}"
BACKEND="${1:-auto}"

case "$(uname -s)" in Darwin) OS=macos ;; Linux) OS=linux ;; *) OS=$(uname -s | tr A-Z a-z) ;; esac
case "$(uname -m)" in x86_64|amd64) ARCH=x64 ;; arm64|aarch64) ARCH=arm64 ;; *) ARCH=$(uname -m) ;; esac

if [ "$BACKEND" = auto ]; then
  if [ "$OS" = macos ]; then BACKEND=metal
  elif command -v nvcc >/dev/null 2>&1; then BACKEND=cuda
  elif command -v glslc >/dev/null 2>&1 && pkg-config --exists vulkan 2>/dev/null; then BACKEND=vulkan
  else BACKEND=cpu; fi
fi
FLAGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON -DGGML_NATIVE=ON -DGGML_RPC=ON)
case "$BACKEND" in
  metal)  FLAGS+=(-DGGML_METAL=ON) ;;
  cuda)   FLAGS+=(-DGGML_CUDA=ON) ;;
  vulkan) FLAGS+=(-DGGML_VULKAN=ON) ;;
  cpu)    [ "$OS" = macos ] && FLAGS+=(-DGGML_METAL=OFF) ;;
esac
echo "llama.cpp $LLAMA_COMMIT  backend=$BACKEND  target=$OS-$ARCH"

if [ ! -d "$SRC/.git" ]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$SRC"
fi
git -C "$SRC" fetch --depth 1 origin "$LLAMA_COMMIT" 2>/dev/null || git -C "$SRC" fetch origin
git -C "$SRC" checkout -q -f "$LLAMA_COMMIT"
git -C "$SRC" clean -fdq ggml/src ggml/include src
# PocketLLM expert-streaming engine (ggml-pocket): see patches/ and docs/ENGINE_KO.md
for p in "$ROOT"/patches/llama.cpp-*.patch; do
  echo "applying $(basename "$p")"
  git -C "$SRC" apply --whitespace=nowarn "$p"
done

cmake -S "$SRC" -B "$SRC/build-$BACKEND" "${FLAGS[@]}" || {
  echo "configure failed (missing curl dev headers?) - retrying without -hf download support"
  cmake -S "$SRC" -B "$SRC/build-$BACKEND" "${FLAGS[@]/-DLLAMA_CURL=ON/-DLLAMA_CURL=OFF}"
}
JOBS=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu)
cmake --build "$SRC/build-$BACKEND" -j "$JOBS" --target llama-server llama-cli llama-bench llama-gguf-split llama-quantize rpc-server

DEST="$ROOT/bin/$OS-$ARCH"
mkdir -p "$DEST"
cp -P "$SRC/build-$BACKEND/bin/"* "$DEST/"
echo "$LLAMA_COMMIT $BACKEND" > "$DEST/VERSION"
echo "installed into $DEST"
