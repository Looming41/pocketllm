#!/usr/bin/env bash
# Update the PocketLLM app on every plugged-in PocketLLM drive (models are left as they are).
#
#   curl -fsSL https://raw.githubusercontent.com/Looming41/pocketllm/main/update.sh | bash
set -euo pipefail
REPO="https://github.com/Looming41/pocketllm"
DIR="${POCKETLLM_DIR:-$HOME/pocketllm}"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull -q --ff-only; else git clone -q "$REPO" "$DIR"; fi
n=0
for app in /Volumes/*/PocketLLM; do
  [ -f "$app/launch.sh" ] || continue
  rm -rf "$app/pocketllm" && cp -R "$DIR/pocketllm" "$app/" && find "$app/pocketllm" -name __pycache__ -prune -exec rm -rf {} +
  cp "$DIR/launch.sh" "$DIR/launch.command" "$DIR/launch.bat" "$DIR/README.md" "$app/"
  echo "updated $app"
  n=$((n + 1))
done
[ "$n" -gt 0 ] || echo "no PocketLLM drive found under /Volumes (plug in the drive with the app on it)"
echo "Close the PocketLLM terminal window if it is open, then start launch.command again."
