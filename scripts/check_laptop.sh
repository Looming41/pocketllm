#!/usr/bin/env bash
# One command to run on the real laptop. Produces data/laptop_report.txt and
# data/profile.json - send those back and the plan can be finalised.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=$(command -v python3 || command -v python)
OUT=data/laptop_report.txt
mkdir -p data
{
  echo "== date: $(date)"
  echo "== pocketllm profile"
  "$PY" -m pocketllm profile
  echo
  echo "== pocketllm plan (catalog + local models, ctx 4096)"
  "$PY" -m pocketllm plan -v
  echo
  echo "== extra system info"
  uname -a
  if [ "$(uname -s)" = Darwin ]; then
    system_profiler SPHardwareDataType SPDisplaysDataType 2>/dev/null | grep -E "Chip|Model Name|Memory|Cores|Metal"
  else
    lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|Flags" | cut -c1-200
    command -v nvidia-smi >/dev/null && nvidia-smi
    command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | head -30
    free -h; df -h .
  fi
} 2>&1 | tee "$OUT"
echo "saved $OUT and data/profile.json"
