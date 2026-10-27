#!/usr/bin/env bash
# Assemble PocketLLM drive(s).
#
#   one drive:            scripts/make_usb.sh /Volumes/POCKETLLM model.gguf [more.gguf ...]
#   striped over drives:  scripts/make_usb.sh --stripe /Volumes/USB1 /Volumes/USB2 [/Volumes/USB3] -- moe-model.gguf
#
# With --stripe the app goes on the first drive and the model's expert tensors are dealt
# across all drives (pocketllm stripe); PocketLLM finds the parts on whatever computer
# the drives are plugged into. Format drives as exFAT (FAT32 cannot hold files > 4 GB).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY=$(command -v python3 || command -v python)

install_app() {
  local dest="$1/PocketLLM"
  mkdir -p "$dest/models" "$dest/data"
  rm -rf "$dest/pocketllm" && cp -R "$ROOT/pocketllm" "$dest/" && find "$dest/pocketllm" -name __pycache__ -prune -exec rm -rf {} +
  cp "$ROOT/launch.sh" "$ROOT/launch.command" "$ROOT/launch.bat" "$ROOT/README.md" "$dest/"
  if [ -d "$ROOT/bin" ]; then cp -RP "$ROOT/bin" "$dest/"; fi
  if [ -d "$ROOT/python" ]; then cp -RP "$ROOT/python" "$dest/"; fi
  chmod +x "$dest/launch.sh" "$dest/launch.command"
  echo "app installed on $dest (binaries for: $(ls "$dest/bin" 2>/dev/null | tr '\n' ' '))"
}

if [ "${1:-}" = "--stripe" ]; then
  shift
  vols=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do vols+=("$1"); shift; done
  [ "${1:-}" = "--" ] && shift
  [ "${#vols[@]}" -ge 2 ] && [ "$#" -ge 1 ] || { echo "usage: $0 --stripe VOL1 VOL2 [...] -- model.gguf"; exit 1; }
  install_app "${vols[0]}"
  dirs=()
  for v in "${vols[@]}"; do mkdir -p "$v/PocketLLM/models"; dirs+=("$v/PocketLLM/models"); done
  for m in "$@"; do
    (cd "$ROOT" && "$PY" -m pocketllm stripe "$m" --drives "${dirs[@]}" --link-dir "${vols[0]}/PocketLLM/data/links/$(basename "$m" .gguf)")
  done
  echo "striped over ${#vols[@]} drives; plug all of them in, then run launch on ${vols[0]}/PocketLLM"
  exit 0
fi

[ "$#" -ge 1 ] || { echo "usage: $0 VOLUME [model.gguf ...]  |  $0 --stripe VOL1 VOL2 [...] -- model.gguf"; exit 1; }
VOL="$1"; shift
install_app "$VOL"
for m in "$@"; do
  echo "copying $(basename "$m") ..."
  cp "$m" "$VOL/PocketLLM/models/"
done
