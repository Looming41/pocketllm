#!/usr/bin/env bash
# One command to turn the USB drives plugged into this Mac (or Linux PC) into PocketLLM drives.
#
#   scripts/setup_drives.sh /Volumes/USB1 /Volumes/USB2              # Qwen3-30B-A3B, Q3_K experts (default)
#   scripts/setup_drives.sh --model ud-q2 /Volumes/USB1 /Volumes/USB2 # smaller download, no conversion
#   scripts/setup_drives.sh --model small /Volumes/USB1                # 2 GB Llama 3.2 3B, quick check
#   scripts/setup_drives.sh --file model.gguf /Volumes/USB1 /Volumes/USB2  # a GGUF you already have
#
# What it does (nothing is erased; existing files on the drives stay):
#   1. checks each drive's filesystem and free space (exFAT or APFS needed: FAT32 cannot hold >4 GB files)
#   2. builds llama.cpp + the PocketLLM engine for this computer (needs cmake and Xcode command line tools)
#   3. downloads the model, and for --model q3 converts its routed experts to Q3_K
#      (measured near-lossless: mean KLD 0.065 vs Q4_K_M, docs/RESULTS.md R17)
#   4. puts the app on the first drive and stripes the model over all drives (pocketllm stripe),
#      so the experts are read from every drive in parallel
#   5. adds a portable Python to the first drive, so it also runs on computers without Python
#
# Models:   q3     Qwen3-30B-A3B, Q3_K experts + Q8_0 rest, 14.1 GB on the drives
#                  (temporary space on this computer: ~33 GB for the download and the conversion)
#           ud-q2  Qwen3-30B-A3B, unsloth UD-Q2_K_XL, 11.8 GB (measured "acceptable": KLD 0.128)
#           small  Llama 3.2 3B Instruct Q4_K_M, about 2 GB (one drive is enough)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL=q3
WORK="${POCKETLLM_WORK:-$HOME/pocketllm-work}"
KEEP=0
FILE=""
VOLS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --file) FILE="$2"; MODEL=file; shift 2 ;;
    -h|--help) sed -n 2,24p "$0"; exit 0 ;;
    *) VOLS+=("${1%/}"); shift ;;
  esac
done
[ "${#VOLS[@]}" -ge 1 ] || { sed -n 2,24p "$0"; exit 1; }
case "$MODEL" in
  q3) NEED_GB=15; TMP_GB=34 ;;
  ud-q2) NEED_GB=12.5; TMP_GB=13 ;;
  small) NEED_GB=2.5; TMP_GB=3 ;;
  file) [ -f "$FILE" ] || { echo "no such file: $FILE"; exit 1; }
        NEED_GB=$(python3 -c "import os; print(round(os.path.getsize('$FILE') / 1e9 + 0.5, 1))"); TMP_GB=0 ;;
  *) echo "unknown --model $MODEL (q3, ud-q2, small)"; exit 1 ;;
esac

say() { printf '\n== %s\n' "$*"; }
free_gb() { df -Pk "$1" | awk 'NR==2{printf "%.1f", $4/1048576}'; }
fs_of() {
  if command -v diskutil >/dev/null; then
    diskutil info "$1" 2>/dev/null | awk -F: '/File System Personality|Type \(Bundle\)/{gsub(/^ +/,"",$2); print $2; exit}'
  else
    df -PT "$1" | awk 'NR==2{print $2}'
  fi
}

say "1/5 checking drives"
TOTAL_FREE=0
for v in "${VOLS[@]}"; do
  [ -d "$v" ] || { echo "not found: $v"; exit 1; }
  fs=$(fs_of "$v"); free=$(free_gb "$v")
  echo "$v: ${fs:-?}, ${free} GB free"
  case "$(echo "$fs" | tr '[:upper:]' '[:lower:]')" in
    *fat32*|*msdos*|vfat|"ms-dos fat32")
      echo "  $v is FAT32, which cannot hold files over 4 GB. Reformat it as exFAT in Disk Utility"
      echo "  (this ERASES the drive), or use --model small."
      [ "$MODEL" = small ] || exit 1 ;;
  esac
  TOTAL_FREE=$(python3 -c "print($TOTAL_FREE + $free)")
done
python3 -c "import sys; sys.exit(0 if $TOTAL_FREE >= $NEED_GB else 1)" || {
  echo "the drives have ${TOTAL_FREE} GB free together; --model $MODEL needs about ${NEED_GB} GB"; exit 1; }
mkdir -p "$WORK"
python3 -c "import sys; sys.exit(0 if $(free_gb "$WORK") >= $TMP_GB else 1)" || {
  echo "this computer needs about ${TMP_GB} GB free in $WORK for the download (free: $(free_gb "$WORK") GB)."
  [ "$MODEL" = q3 ] && echo "try --model ud-q2 (13 GB), or --work on a bigger disk"
  exit 1; }

say "2/5 llama.cpp + PocketLLM engine"
TAG=$(cd "$ROOT" && python3 -c 'from pocketllm import runtime; print(runtime.platform_tag())')
if [ -x "$ROOT/bin/$TAG/llama-server" ] && grep -q "$(grep -o '[0-9a-f]\{40\}' "$ROOT/scripts/build_llamacpp.sh" | head -1)" "$ROOT/bin/$TAG/VERSION" 2>/dev/null; then
  echo "already built: bin/$TAG"
else
  command -v cmake >/dev/null || { echo "cmake is missing: brew install cmake  (or install it from cmake.org)"; exit 1; }
  "$ROOT/scripts/build_llamacpp.sh" auto
fi
export POCKETLLM_BIN="$ROOT/bin/$TAG"

say "3/5 model"
python3 -m venv "$WORK/venv" >/dev/null 2>&1 || true
[ "$MODEL" = file ] || "$WORK/venv/bin/pip" install --quiet --upgrade huggingface_hub
case "$MODEL" in
  file) OUT="$FILE" ;;
  q3)
    SRC=$("$WORK/venv/bin/python" "$ROOT/scripts/fetch_model.py" qwen3-30b-a3b "$WORK/download" --max-gb 25)
    OUT="$WORK/Qwen3-30B-A3B-experts-Q3_K.gguf"
    [ -f "$OUT" ] || (cd "$ROOT" && python3 -m pocketllm requant "$SRC" --recipe experts-q3 --out "$OUT")
    [ "$KEEP" = 1 ] || rm -f "$SRC" ;;
  ud-q2)
    OUT=$("$WORK/venv/bin/python" "$ROOT/scripts/fetch_model.py" qwen3-30b-a3b-ud-q2 "$WORK/download" --max-gb 15) ;;
  small)
    OUT=$("$WORK/venv/bin/python" "$ROOT/scripts/fetch_model.py" llama-3.2-3b "$WORK/download" --max-gb 4) ;;
esac
echo "model: $OUT ($(du -h "$OUT" | cut -f1))"

say "4/5 app + model onto the drives"
if [ "${#VOLS[@]}" -ge 2 ] && [ "$MODEL" != small ]; then
  "$ROOT/scripts/make_usb.sh" --stripe "${VOLS[@]}" -- "$OUT"
else
  "$ROOT/scripts/make_usb.sh" "${VOLS[0]}" "$OUT"
fi

say "5/5 portable Python on ${VOLS[0]}"
"$WORK/venv/bin/python" "$ROOT/scripts/fetch_python.py" "${VOLS[0]}/PocketLLM" --platforms macos-arm64 linux-x64 windows-x64 \
  || echo "(portable Python skipped: this computer's Python will be used)"

[ "$KEEP" = 1 ] || [ "$MODEL" != q3 ] || rm -f "$OUT"
say "done"
for v in "${VOLS[@]}"; do echo "$v: $(du -sh "$v/PocketLLM" 2>/dev/null | cut -f1) PocketLLM files, $(free_gb "$v") GB free"; done
cat <<EOF

Start it: double-click ${VOLS[0]}/PocketLLM/launch.command  (first time: right-click > Open)
          or run: ${VOLS[0]}/PocketLLM/launch.sh
Keep all ${#VOLS[@]} drive(s) plugged in; the dashboard at http://127.0.0.1:8080 shows them and their read speed.
EOF
