#!/usr/bin/env bash
# PocketLLM: make the USB drives plugged into this Mac into PocketLLM drives, in one command.
#
#   curl -fsSL https://raw.githubusercontent.com/Looming41/pocketllm/main/install.sh | bash
#
# Finds the USB drives (only drives on the USB bus; internal disks and disk images are never
# touched), installs what is missing (Xcode command line tools, cmake), gets PocketLLM into
# ~/pocketllm, picks the biggest model that fits the drives and this Mac, and runs
# scripts/setup_drives.sh. Nothing on the drives is erased. Afterwards, plugging the first drive
# into a Mac and double-clicking PocketLLM/launch.command starts it.
set -euo pipefail
REPO="https://github.com/Looming41/pocketllm"
DIR="${POCKETLLM_DIR:-$HOME/pocketllm}"
MODEL="${POCKETLLM_MODEL:-auto}"
say() { printf '\n== %s\n' "$*"; }
ask() { local a; read -r -p "$1 [y/N] " a </dev/tty; [ "$a" = y ] || [ "$a" = Y ]; }
free_gb() { df -Pk "$1" | awk 'NR==2{printf "%.1f", $4/1048576}'; }

[ "$(uname -s)" = Darwin ] || { echo "this installer is for macOS; on Linux run scripts/setup_drives.sh"; exit 1; }

say "USB drives"
USBS=()
for v in /Volumes/*; do
  [ -d "$v" ] || continue
  info=$(diskutil info "$v" 2>/dev/null || true)
  echo "$info" | grep -q "Protocol: *USB" || continue
  fs=$(echo "$info" | awk -F: '/File System Personality/{gsub(/^ +/,"",$2); print $2; exit}')
  echo "  $v  (${fs:-?}, $(free_gb "$v") GB free)"
  USBS+=("$v")
done
[ "${#USBS[@]}" -ge 1 ] || { echo "no USB drive found under /Volumes. Plug one in and run this again."; exit 1; }
BIG=(); for v in "${USBS[@]}"; do
  diskutil info "$v" | grep -qiE "Personality: *(MS-DOS|FAT)" || BIG+=("$v")
done
if [ "${#BIG[@]}" -lt "${#USBS[@]}" ]; then
  echo "  note: FAT32 drives cannot hold files over 4 GB; they only get the small model."
  echo "  (Disk Utility > Erase > ExFAT makes them usable, but erases them.)"
fi

say "tools"
if ! xcode-select -p >/dev/null 2>&1; then
  xcode-select --install || true
  echo "Install the Xcode command line tools in the window that opened, then run this command again."
  exit 1
fi
if ! command -v cmake >/dev/null; then
  if command -v brew >/dev/null; then brew install cmake
  else echo "cmake is needed: install Homebrew (https://brew.sh), then run this command again."; exit 1; fi
fi
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull -q --ff-only; else git clone -q "$REPO" "$DIR"; fi

if [ "$MODEL" = auto ]; then
  total=0; for v in ${BIG[@]+"${BIG[@]}"}; do total=$(python3 -c "print($total + $(free_gb "$v"))"); done
  mac=$(free_gb "$HOME")
  if   python3 -c "import sys; sys.exit(0 if $total >= 15 and $mac >= 34 else 1)"; then MODEL=q3
  elif python3 -c "import sys; sys.exit(0 if $total >= 12.5 and $mac >= 13 else 1)"; then MODEL=ud-q2
  else MODEL=small; fi
fi
case "$MODEL" in
  q3|ud-q2) [ "${#BIG[@]}" -ge 1 ] || { echo "every USB drive is FAT32: use POCKETLLM_MODEL=small, or reformat one as ExFAT"; exit 1; } ;;
esac
case "$MODEL" in
  q3) DRIVES=("${BIG[@]}"); WHAT="Qwen3-30B-A3B, Q3_K experts (14 GB, near-lossless; ~34 GB temporary space on this Mac)" ;;
  ud-q2) DRIVES=("${BIG[@]}"); WHAT="Qwen3-30B-A3B UD-Q2_K_XL (12 GB, acceptable quality)" ;;
  small) DRIVES=("${USBS[0]}"); WHAT="Llama 3.2 3B Instruct (2 GB)" ;;
  *) echo "POCKETLLM_MODEL must be q3, ud-q2, small or auto"; exit 1 ;;
esac
say "plan"
echo "  model : $WHAT"
echo "  drives: ${DRIVES[*]}  (app on ${DRIVES[0]}; existing files stay)"
ask "Go ahead?" || exit 1
"$DIR/scripts/setup_drives.sh" --model "$MODEL" "${DRIVES[@]}"
