# PocketLLM

Plug a drive into a laptop and get a local LLM that is **sized to that machine automatically**: hardware profiling, a placement planner that says *before you download 60 GB* whether a 100B-class model will run (and how fast at best), measured auto-tuning, and a local chat UI with live speed and memory readouts.

> **What this prototype is, and is not.** The USB drive carries the software, the llama.cpp runtime and the model files. **All inference runs on the host computer's own CPU/GPU and RAM.** The drive adds no compute and no memory. A USB-attached accelerator is a separate, long-term hardware track (see `docs/ROADMAP_HW_KO.md`).

## What works today (tested)

| Component | Status | Where it was tested |
|---|---|---|
| GGUF header reader (sizes, MoE experts, KV cache/token, split files) | ✅ matches llama-bench byte-for-byte | Linux x64 container |
| Hardware profiler (CPU, RAM, cgroup limits, NVIDIA/AMD/Apple GPU, USB/removable detection, cold read speed) | ✅ Linux paths exercised; macOS/Windows paths written, **not yet run on real hardware** | Linux |
| Placement planner (full GPU / MoE expert split `-ncmoe` / partial `-ngl` / CPU / paging from drive) | ✅ 15 unit tests incl. synthetic laptop profiles | — |
| Measured tuner (llama-bench sweep, default vs tuned, noise-aware) | ✅ | CPU only (no GPU in the build box) |
| llama-server supervisor with fallback chain and stale-port protection | ✅ | Linux |
| Web UI (hardware, model verdicts, chat, TTFT/tok/s/RAM, drive + network badges) | ✅ Playwright end-to-end test | Linux + Chromium |
| Setup assistant via Meta Model API (cloud) with offline fallback | ✅ request format + fallback tested against a mock server; **not yet called against the real API** | — |

Real-model results: **none yet.** The build machine could not download model weights, so the pipeline was verified with random-weight test models (a dense and a 32-expert MoE GGUF with a real tokenizer). Their output text is meaningless. Their speed numbers only validate the tooling. See `docs/RESULTS.md`.

## Quick start (on the laptop)

```bash
scripts/build_llamacpp.sh            # pinned llama.cpp; auto-selects Metal / CUDA / Vulkan / CPU
scripts/check_laptop.sh              # hardware report + feasibility table -> data/laptop_report.txt
# put GGUF files in models/ (e.g. llama-server -hf <repo>:<quant> downloads, or copy them)
python3 -m pocketllm plan -v         # verdict + exact llama-server args for every model
python3 -m pocketllm tune models/<model>.gguf   # measured: defaults vs tuned
./launch.sh                          # UI at http://127.0.0.1:8080 (launch.command on macOS, launch.bat on Windows)
```

Build a drive: `scripts/make_usb.sh /Volumes/POCKETLLM models/*.gguf` (format the drive as **exFAT**. FAT32 cannot hold files over 4 GB).

## Architecture

```
USB drive (storage only)                      Host computer (does all compute)
┌──────────────────────────┐   USB 3.x    ┌──────────────────────────────────────────────────────┐
│ launch.sh/.command/.bat  │ ───────────▶ │ pocketllm (Python, stdlib only)                       │
│ pocketllm/  (app)        │   files      │  profiler ─▶ planner ─▶ tuner ─▶ runtime supervisor   │
│ bin/<os>-<arch>/ llama.cpp│             │      │          ▲          │            │             │
│ models/*.gguf            │              │      │   gguf header       │     llama-server (CPU/GPU)│
│ data/ (profiles, results)│              │      │   reader            │            │             │
└──────────────────────────┘              │  web UI (127.0.0.1) ◀──────┴── chat stream, metrics   │
                                          │  setup assistant ──(optional, internet)──▶ Meta Model API
                                          └──────────────────────────────────────────────────────┘
```

* **Load modes.** When the model fits in RAM, PocketLLM loads it fully (`-lm none`), so the drive can be unplugged after loading. When it does not fit, weights are memory-mapped (`-lm mmap`) and re-read from the drive while generating. The planner then reports the paged gigabytes and a storage-bound speed ceiling. In this mode the drive must stay connected.
* **Why a planner when llama.cpp has `--fit`?** llama.cpp's `--fit` (on by default) adjusts parameters to free memory at load time. PocketLLM decides *before* download or load, explains the decision, and verifies it with measurements.

## Meta technology used

* **Meta Model API** (`https://api.meta.ai/v1`, OpenAI-compatible chat completions; model id configurable, default `muse-spark-1.3`). It turns a natural-language goal ("biggest model that still feels real-time, offline") into constraints. Its output is only a proposal: values are clamped and every model choice is re-checked by the local planner against measured memory. Set `MODEL_API_KEY` to enable it. The UI labels it as cloud.
* **Meta open-weight models run locally**: Llama 3.x (default chat), **Llama 4 Scout (109B MoE)** as the 100B-class target, and Muse Glimmer once a GGUF is available (llama.cpp at the pinned commit has a `muse-glimmer` architecture).

## Honest limits

* Speeds marked `≤` are upper bounds from a bandwidth model, not measurements.
* No real 100B-class model has been run yet. On a typical 16–32 GB laptop, Llama 4 Scout Q4 needs paging from storage, and the planner estimates well under 1 tok/s (see `docs/feasibility_matrix.md`).
* The host needs Python 3 (preinstalled on most Linux; macOS needs Command Line Tools; Windows needs python.org). A single-binary launcher is on the roadmap.
* Binaries must be built per OS/arch/GPU backend; the drive can carry several in `bin/`.

Docs: `docs/RESEARCH_KO.md` (ecosystem + competitor research), `docs/TEST_PLAN_BIG_MODEL_KO.md`, `docs/HACKATHON_PLAN_KO.md`, `docs/ROADMAP_HW_KO.md`, `docs/RESULTS.md`.

License: MIT for PocketLLM code. llama.cpp is MIT. Models keep their own licenses (Llama Community Licenses, Apache 2.0 for gpt-oss and Muse Glimmer).
