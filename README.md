# PocketLLM

**Run 30B–100B+ Mixture-of-Experts models on laptops that don't have the RAM for them, fully offline, from a USB drive.**

PocketLLM combines ideas from roughly 50 open-source projects that run large models locally (see `docs/LANDSCAPE_KO.md`). It puts them on a drive you can plug into any laptop:

1. **Expert streaming engine (ggml-pocket).** This is a ~500-line patch to llama.cpp's CPU MoE kernel. When a model is bigger than RAM, it reads *only the experts the router picked*, as soon as it picks them, with concurrent large reads instead of page-fault-by-page-fault. In our emulated "low-RAM laptop + USB drive" lab this made decode **several times faster** than stock llama.cpp mmap (see Results).
2. **More drives, more bandwidth.** `pocketllm stripe` spreads the expert tensors over several drives, and the engine reads them in parallel.
3. **Planner + measured tuner.** Before anything is downloaded, PocketLLM reports whether a model will fit, where each part will live (GPU / RAM / streamed from the drive), and an upper bound on speed. A llama-bench sweep then measures the real speed.
4. **Air-gap mode for organisations that must not leak data** (defence, government, regulated industry). Cloud features are disabled by policy, network detection sends no packets, the manifest is signed and verified before anything runs, the local model is protected by a per-session key, and the audit log stores metadata only.
5. **Meta, in two zones.** Online provisioning uses the Meta Model API (Muse Spark) for fleet planning, synthetic eval sets and LLM-as-judge model qualification. The air-gapped field runs Meta open-weight models only (Llama 4 Scout MoE, Llama 3.x, Muse Glimmer). See `docs/META_KO.md`.
6. **Cluster mode.** It pools the memory of several machines ("compute sticks", mini-PCs, laptops) over a private link with llama.cpp RPC.

> **Honest scope.** A USB *storage* drive adds capacity and read bandwidth, **not compute**. All inference runs on the host's CPU/GPU. PocketLLM makes large *sparse* (MoE) models practical on small machines by streaming what each token needs. It cannot make a dense 70B model fast on a 16 GB laptop. Fanless laptops run cooler on a ~3B-active MoE than on a 7B dense model, because the work per token is smaller. Thermal profiles cap threads further.

## Results (measured, emulated hardware)

Emulated "16 GB-class laptop" (1.5 GiB RAM limit for a 3.5 GB MoE, 5.7B params, 64 experts top-8) with "USB drives" throttled to 300 MB/s each. Decode speed, mean ± sd of 3 cold runs (`docs/RESULTS.md` R5):

| | 1 drive | 2 drives | 3 drives |
|---|---|---|---|
| stock llama.cpp (mmap) | 0.63 ± 0.07 tok/s · 13.0 GB read | 0.58 ± 0.15 · 18.4 GB | 0.63 ± 0.09 · 21.4 GB |
| **PocketLLM engine** | **4.15 ± 0.07 tok/s (×6.6)** · 5.4 GB | **7.56 ± 0.36 (×12.9)** · 5.2 GB | **7.41 ± 0.35 (×11.8)** · 5.2 GB |

With slower drives (100 MB/s each, like cheap flash sticks): stock 0.08 tok/s, against PocketLLM **1.31 / 2.29 / 2.76 tok/s with 1 / 2 / 3 drives** (R5b).

On fast internal storage (about 2 GB/s), the gain shrinks to noise (×0.98–1.6, R5c). PocketLLM pays off when the weights come from slow drives.

### Real model: Qwen3-30B-A3B (18.6 GB) with 6 GiB of RAM, from 300 MB/s drives (CI lab, R12)

| | 1 drive | 2 drives |
|---|---|---|
| stock llama.cpp | 0.08 tok/s | 0.12 tok/s |
| PocketLLM v1 | 0.63 (×7.5) | 0.66 |
| **PocketLLM v2 (own expert cache)** | **0.82 (×9.6)** | **1.04 (×8.7)** |

Second run (R14), 1 drive: v2 + top-k 4 reaches **1.61 tok/s**. The model requantized to Q2_K experts + Q8_0 (18.6 → 11.2 GB) gives v2 2.01 and v2 + top-k 4 **3.51 tok/s**, about ×44 over stock llama.cpp. The quality cost of those last two steps is not measured yet.

On a real router, the v2 cache hits 64% of expert lookups, so it beats v1. On random-weight models it lost, which is why the default is now chosen by drive speed. On a fast unthrottled disk, v1 is ×1.21 over stock and v2 is slower (R13).

**Update (R15, idea from kimi-k3-in-c):** most of the drive traffic was llama.cpp reading the whole model file ahead at load, under memory pressure. On slow drives the planner now skips that, and reads ahead and locks only the dense weights: **0.94 tok/s with 1 drive, 1.31 with 2** (stock 0.08), and drive reads fall from 56 to 14 GB. The v2 cache no longer wins once this is in place.

### Ideas taken from Pulsar, SwiftLM and ExpertFlow (R8, R9, same lab, 1 drive)

| what (source) | decode | note |
|---|---|---|
| PocketLLM v1, Q4_K_M model | 4.15 tok/s | baseline for this table |
| + `pocketllm requant`: **Q2_K experts + Q8_0 rest** (Pulsar recipe) | **9.34 (×2.3)** | model 3.52 → 2.53 GB |
| + Q2_K experts + Q4_K rest | **13.76 (×3.3)** | model 2.18 GB |
| + top-k 4 of 8 experts per token (SwiftLM `SWIFTLM_TOP_K`) | **18.72 (×4.5)** | |
| at 8K context: q8_0 instead of f16 KV cache (SwiftLM's KV lesson; planner default when streaming) | 0.44 → **4.83 (×11)** | R11; f16 control with +192 MiB RAM: 1.30 |
| own LFU expert cache (Pulsar), random-weight model | 2.47–3.55 (slower) | on the **real** model it wins instead (R12), so it is now the default for slow drives |

Low-bit experts and top-k **change the model's output**. Unlike the streaming engine, they trade quality for speed and size, so a recipe goes on a drive only after `pocketllm qualify` passes on the provisioning station. When weights must be streamed, the planner also quantizes the KV cache to q8_0, following SwiftLM's long-context finding that every KV byte saved becomes page cache for weights. On Apple Silicon, MLX models run through SwiftLM's own `--stream-experts` backend.

The speedup has two sources. First, stock mmap read-around pulls in neighbouring *unused* experts, roughly 3× read amplification. Second, PocketLLM issues all routed-expert reads at once, so they overlap and spread across drives. Outputs are token-identical, because the engine changes only *when* weights are read.

## Quick start

```bash
scripts/build_llamacpp.sh             # pinned llama.cpp + PocketLLM engine patch + RPC; Metal/CUDA/Vulkan/CPU auto
# Windows: powershell -ExecutionPolicy Bypass -File scripts\build_llamacpp.ps1   (or download the CI artifact)
scripts/check_laptop.sh               # hardware report + feasibility table -> data/laptop_report.txt
python3 -m pocketllm plan -v          # verdict, placement, engine env and llama-server args per model
python3 -m pocketllm tune models/<model>.gguf    # measured: llama.cpp defaults vs PocketLLM
./launch.sh                           # UI on http://127.0.0.1:8080 (launch.command on macOS, launch.bat on Windows)
```

| Command | What it does |
|---|---|
| `profile`, `plan`, `inspect`, `tune`, `serve`, `thermal` | hardware, feasibility, GGUF facts, measured tuning, UI, power profiles |
| `stripe MODEL --drives D1 D2 D3 --link-dir DIR` | split a GGUF across drives for parallel expert reads (llama.cpp split format) |
| `seal --key K` / `verify --pubkey P [--full]` | sign the drive on the provisioning station / verify before use |
| `node --host IP` / `cluster MODEL --nodes a:p,b:p --run` | lend a machine's memory / run across machines (llama.cpp RPC) |
| `requant MODEL --recipe experts-q2 --out OUT` | low-bit routed experts, 8-bit attention: smaller model, fewer bytes streamed per token |
| `scripts/fetch_python.py DRIVE/PocketLLM` | portable Python on the drive, for computers without one (SHA-256 checked) |
| `fleet-plan`, `evalset`, `qualify` | provisioning with Meta Model API (refuses to run under an air-gap policy) |

## Architecture

```
USB drive(s)  ── storage + read bandwidth only ──▶  host laptop (does all compute)
┌──────────────────────────┐                      ┌───────────────────────────────────────────────────────┐
│ launch.* · pocketllm/     │                      │ profiler → planner → tuner → runtime supervisor       │
│ bin/<os>-<arch>/ llama.cpp│  experts streamed     │ llama-server (patched): MUL_MAT_ID hook ─▶ ggml-pocket │
│   + ggml-pocket engine    │ ◀──── on demand ──── │   prefetch routed experts · per-drive fds · pinning   │
│ models/*.gguf (striped)   │                      │ web UI · air-gap policy · integrity · audit · thermal │
│ manifest.json + .sig      │                      │ optional: --rpc nodes (cluster) · Meta API (provision) │
└──────────────────────────┘                      └───────────────────────────────────────────────────────┘
```

## Status and honest limits
- Most numbers above come from an emulated low-RAM machine with throttled loop devices standing in for USB drives, running a **random-weight** MoE model. The model's routing statistics differ from real models, so prediction and pinning still need testing on real models such as Qwen3-30B-A3B and Llama 4 Scout. The first real-model run (Qwen3-30B-A3B on a Linux CI runner, R12–R13) confirms the engine's gain on trained routing. Quality checks of requant recipes and real laptops with real USB drives are still to do.
- **CI (GitHub Actions) builds and tests on Linux, macOS (Apple Silicon) and Windows (MSVC)**: unit tests plus an engine check, in which the engine reports activity and its greedy output is identical with the engine off and on. CI also uploads per-OS binaries for the drive (`pocketllm-bin-<os>` artifacts). CI proves that the engine works on each OS. It does **not** measure speed on a real Mac or Windows laptop with USB drives. Windows has no equivalent of `MADV_RANDOM`, so the read-around fix does not apply there.
- CI found two real bugs that root-only local tests missed. First, an expert whose mlock failed under a normal user's limit was still marked as pinned, so it was never prefetched. Second, manifest paths written on Windows used backslashes, so a drive sealed on one OS failed verification on another. Both are fixed and covered by tests.
- llama.cpp RPC has no authentication or encryption. Use it only on a dedicated link.
- This is not a security-certified product.

Docs (Korean): `docs/LANDSCAPE_KO.md` (survey), `docs/ENGINE_KO.md`, `docs/META_KO.md`, `docs/SECURITY_KO.md`, `docs/RESULTS.md`, `docs/HACKATHON_PLAN_KO.md`, `docs/ROADMAP_HW_KO.md`.

License: MIT (PocketLLM, and the engine patch to MIT-licensed llama.cpp). Models keep their own licenses.
