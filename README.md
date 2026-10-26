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

The speedup has two sources. First, stock mmap read-around pulls in neighbouring *unused* experts, roughly 3× read amplification. Second, PocketLLM issues all routed-expert reads at once, so they overlap and spread across drives. Outputs are token-identical, because the engine changes only *when* weights are read.

## Quick start

```bash
scripts/build_llamacpp.sh             # pinned llama.cpp + PocketLLM engine patch + RPC; Metal/CUDA/Vulkan/CPU auto
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
- Every number above comes from an emulated low-RAM machine with throttled loop devices standing in for USB drives, running a **random-weight** MoE model. The model's routing statistics differ from real models, so prediction and pinning still need testing on real models such as Qwen3-30B-A3B and Llama 4 Scout. **No real model has been run yet** because model downloads were blocked in the build environment.
- The engine is tested on Linux. The macOS path (`F_RDADVISE`, or `POCKET_MOE_IO_THREADS`) compiles from the same code but **has not been run on a Mac yet**. Windows has no streaming engine yet; inference works there without it.
- llama.cpp RPC has no authentication or encryption. Use it only on a dedicated link.
- This is not a security-certified product.

Docs (Korean): `docs/LANDSCAPE_KO.md` (survey), `docs/ENGINE_KO.md`, `docs/META_KO.md`, `docs/SECURITY_KO.md`, `docs/RESULTS.md`, `docs/HACKATHON_PLAN_KO.md`, `docs/ROADMAP_HW_KO.md`.

License: MIT (PocketLLM, and the engine patch to MIT-licensed llama.cpp). Models keep their own licenses.
