#!/usr/bin/env bash
# Assemble a PocketLLM drive: copies the app, launchers and binaries (and
# optionally models) onto a mounted USB volume.
#
# usage: scripts/make_usb.sh /Volumes/POCKETLLM [model.gguf ...]
# The volume should be exFAT (FAT32 cannot hold files > 4 GB) or APFS/NTFS/ext4
# if it will only be used on one OS family.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$1/PocketLLM"; shift || true
mkdir -p "$DEST/models" "$DEST/data"
rsync -a --exclude '__pycache__' "$ROOT/pocketllm" "$DEST/"
cp "$ROOT/launch.sh" "$ROOT/launch.command" "$ROOT/launch.bat" "$ROOT/README.md" "$DEST/"
[ -d "$ROOT/bin" ] && rsync -a "$ROOT/bin" "$DEST/"
for m in "$@"; do
  echo "copying $(basename "$m") ..."
  rsync -a --progress "$m" "$DEST/models/"
done
chmod +x "$DEST/launch.sh" "$DEST/launch.command"
echo "drive ready: $DEST  (binaries for: $(ls "$DEST/bin" 2>/dev/null | tr '\n' ' '))"
