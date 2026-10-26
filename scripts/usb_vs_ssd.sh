#!/usr/bin/env bash
# Experiment E3: how much slower is running a model from the USB drive than
# from the internal SSD?  Measures, for the same GGUF file on both devices:
#   1) cold sequential read speed (page cache evicted where the OS allows it)
#   2) llama-bench load + generation with -lm none (copy into RAM) and -lm mmap
#
# usage: scripts/usb_vs_ssd.sh /Volumes/POCKETLLM/models/model.gguf ~/pocketllm-ssd-copy/
# Writes data/usb_vs_ssd.json. Needs free space on the SSD for one copy of the model.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USB_MODEL="$1"
SSD_DIR="${2:-$HOME/pocketllm-ssd}"
PY=$(command -v python3 || command -v python)
mkdir -p "$SSD_DIR" "$ROOT/data"
SSD_MODEL="$SSD_DIR/$(basename "$USB_MODEL")"
if [ ! -f "$SSD_MODEL" ]; then
  echo "copying model to SSD..."; t0=$(date +%s); cp "$USB_MODEL" "$SSD_MODEL"; echo "copy took $(( $(date +%s) - t0 ))s"
fi

cd "$ROOT"
"$PY" - "$USB_MODEL" "$SSD_MODEL" <<'EOF'
import json, sys, time, subprocess
from pocketllm import profiler, runtime, tuner
res = {}
for label, path in (("usb", sys.argv[1]), ("ssd", sys.argv[2])):
    r = {"storage": profiler.storage_info(path.rsplit("/", 1)[0]),
         "read": profiler.measure_read_speed(path, mb=1024)}
    for mode in ("none", "mmap"):
        # evict again so the load really comes from the device
        profiler.measure_read_speed(path, mb=1)
        r[f"bench_lm_{mode}"] = tuner.run_bench(path, {"load_mode": mode}, n_prompt=64, n_gen=32, reps=1)
    res[label] = r
    print(label, json.dumps({k: v for k, v in r.items() if k != "storage"}, indent=1))
print("NOTE: the OS may still cache part of the file; for a strict cold test reboot or unplug/replug the drive between runs.")
json.dump(res, open("data/usb_vs_ssd.json", "w"), indent=2)
EOF
