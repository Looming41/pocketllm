"""Placement planner: decides *where* each part of a model lives (GPU / RAM /
paged from disk) and turns that into llama.cpp arguments plus an honest
feasibility verdict.

llama.cpp already has ``--fit`` (on by default), which shrinks unset
parameters until the model fits free device memory at load time. The planner
complements it rather than replacing it:

* it runs *before* anything is downloaded or loaded, from GGUF headers or
  catalog numbers alone, so a user learns "this 109B model will not fit / will
  page from the USB drive at ~0.2 tok/s" without waiting for a 60 GB load;
* it explains the decision (memory breakdown + bottleneck) in the UI;
* it emits explicit -ngl / -ncmoe / --no-mmap choices that the tuner then
  verifies with real llama-bench measurements.

All speed figures produced here are upper-bound estimates from a
bandwidth model, never measurements. The tuner produces measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

GB = 1e9
GIB = 2**30

# Headroom left for the OS / other apps, and for llama.cpp compute buffers.
OS_RESERVE_FRAC = 0.10
OS_RESERVE_MIN = 2 * GIB
GPU_RESERVE = 0.75 * GIB
COMPUTE_OVERHEAD = 0.5 * GIB


@dataclass
class ModelSpec:
    """What the planner needs to know about a model, from a GGUF file or the catalog."""
    name: str
    total_bytes: int
    n_layer: int
    layer_dense_bytes: list          # per-layer bytes excluding routed experts
    layer_expert_bytes: list         # per-layer routed-expert bytes (0 for dense)
    non_layer_bytes: int             # embeddings, output head, norms
    active_bytes: int                # weight bytes read per generated token
    kv_bytes_per_token: int
    is_moe: bool
    train_ctx: int = 0
    path: str | None = None
    source: str = "gguf"             # "gguf" (exact), "estimate" (catalog), "recipe" (catalog, made by requant), "mlx"

    @classmethod
    def from_gguf(cls, m, path=None):
        n = m.n_layer
        return cls(
            name=m.name, total_bytes=m.weight_bytes, n_layer=n,
            layer_dense_bytes=[m.layer_bytes(i, experts=False) for i in range(n)],
            layer_expert_bytes=[m.layer_bytes(i, experts=True) for i in range(n)],
            non_layer_bytes=m.non_layer_bytes(), active_bytes=m.active_bytes_per_token(),
            kv_bytes_per_token=m.kv_bytes_per_token(), is_moe=m.is_moe, train_ctx=m.context_length,
            path=path or m.paths[0], source="gguf",
        )

    @classmethod
    def estimate(cls, name, params_b, active_b, bpw, n_layer, kv_bytes_per_token, expert_frac=0.0, train_ctx=0,
                 expert_bpw=None):
        """Rough spec from parameter counts. expert_frac = share of weights in routed experts.
        expert_bpw: routed experts quantized differently from the rest (pocketllm requant)."""
        ebpw = expert_bpw or bpw
        p_exp = params_b * 1e9 * expert_frac
        p_dense = params_b * 1e9 - p_exp
        dense_b, exp_total = p_dense * bpw / 8, p_exp * ebpw / 8
        total = int(dense_b + exp_total)
        active = int(dense_b + max(0.0, active_b * 1e9 - p_dense) * ebpw / 8) if expert_frac else \
            int(active_b * 1e9 * bpw / 8)
        non_layer = int(dense_b * 0.02) if expert_frac else int(total * 0.02)
        dense_layer = (dense_b - non_layer) / n_layer if expert_frac else (total - non_layer) / n_layer
        exp = exp_total / n_layer
        return cls(name=name, total_bytes=total, n_layer=n_layer,
                   layer_dense_bytes=[int(dense_layer)] * n_layer,
                   layer_expert_bytes=[int(exp)] * n_layer, non_layer_bytes=non_layer,
                   active_bytes=active, kv_bytes_per_token=kv_bytes_per_token,
                   is_moe=expert_frac > 0, train_ctx=train_ctx, source="estimate")


@dataclass
class Plan:
    model: str
    strategy: str            # full_gpu | moe_hybrid | partial_gpu | cpu_ram | disk_paging | expert_stream | impossible
    verdict: str             # fits | slow | impractical | does_not_fit
    ctx: int
    gpu_bytes: int = 0
    ram_bytes: int = 0
    paged_bytes: int = 0     # weights that do not fit in RAM and must be re-read from storage
    kv_bytes: int = 0
    n_gpu_layers: int = 0
    n_cpu_moe: int = 0
    no_mmap: bool = False
    est_tok_s_upper: float | None = None
    bottleneck: str = ""
    notes: list = field(default_factory=list)
    args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)   # runtime environment (PocketLLM streaming engine)
    backend: str = "llama"                   # llama (llama-server + engine) | swiftlm (MLX on Apple Silicon)
    topk: int | None = None                  # routed experts per token, if reduced (speed vs quality)
    kv_type: str = "f16"                     # KV cache type (q8_0 when RAM is short)

    def to_dict(self):
        d = asdict(self)
        for k in ("gpu_bytes", "ram_bytes", "paged_bytes", "kv_bytes"):
            d[k.replace("_bytes", "_gb")] = round(d.pop(k) / GB, 2)
        return d


@dataclass
class Budget:
    ram_budget: int
    gpu_budget: int
    unified: bool
    ram_bw: float | None       # bytes/s
    gpu_bw: float | None
    disk_bw: float | None
    threads: int
    bw_is_lower_bound: bool = False


def budget_from_profile(profile: dict, disk_gbps: float | None = None, ram_gbps: float | None = None,
                        gpu_gbps: float | None = None) -> Budget:
    ram = profile.get("ram", {})
    total = ram.get("total_bytes") or 0
    avail = ram.get("available_bytes") or total
    reserve = max(OS_RESERVE_MIN, total * OS_RESERVE_FRAC)
    ram_budget = int(max(0, min(avail, total - reserve)))
    gpus = [g for g in profile.get("gpus", []) if g.get("vram_free_bytes")]
    gpu = max(gpus, key=lambda g: g["vram_free_bytes"], default=None)
    unified = bool(gpu and gpu.get("unified_memory"))
    gpu_budget = int(max(0, gpu["vram_free_bytes"] - GPU_RESERVE)) if gpu else 0
    if unified:
        # one physical pool: the GPU share comes out of the RAM budget
        gpu_budget = min(gpu_budget, ram_budget)
    bw_info = (ram.get("bandwidth") or {})
    lower = False
    if ram_gbps is None and bw_info.get("gb_per_s_1thread"):
        ram_gbps = bw_info["gb_per_s_1thread"]
        lower = True
    cores = profile.get("cpu", {}).get("physical_cores") or profile.get("cpu", {}).get("logical_cores") or 4
    if unified and gpu_gbps is None:
        gpu_gbps = ram_gbps  # same physical memory
    return Budget(ram_budget=ram_budget, gpu_budget=gpu_budget, unified=unified,
                  ram_bw=ram_gbps * GB if ram_gbps else None, gpu_bw=gpu_gbps * GB if gpu_gbps else None,
                  disk_bw=disk_gbps * GB if disk_gbps else None, threads=int(cores), bw_is_lower_bound=lower)


def _speed(cpu_active, paged_active, b: Budget, gpu_active=0):
    """Upper-bound tok/s from weight bytes moved per token, ignoring compute.

    Discrete-GPU bandwidth is not detected, so when the GPU reads most of the
    active bytes no number is given (the tuner measures it instead); when it
    reads a minority, its time is dropped, which keeps the result an upper bound."""
    parts = {}
    if gpu_active:
        if b.gpu_bw:
            parts["GPU memory bandwidth"] = gpu_active / b.gpu_bw
        elif gpu_active >= cpu_active + paged_active:
            return None, "GPU memory bandwidth (measure with tuner)"
    if cpu_active and b.ram_bw:
        parts["RAM bandwidth"] = cpu_active / b.ram_bw
    if paged_active:
        if b.disk_bw:
            parts["storage read speed"] = paged_active / b.disk_bw
        else:
            return None, "storage read speed (not measured)"
    if not parts:
        return None, "GPU (not modelled)" if not cpu_active else "RAM bandwidth (not measured)"
    t = sum(parts.values())
    return round(1.0 / t, 2), max(parts, key=parts.get)


KV_BYTES = {"f16": 2.0, "q8_0": 1.0625}  # bytes per element


def plan(spec: ModelSpec, b: Budget, ctx: int = 4096, allow_paging: bool = True, topk: int | None = None,
         n_expert_used: int | None = None, kv_type: str = "auto") -> Plan:
    """topk: run MoE models with at most this many routed experts per token (needs n_expert_used,
    the model's default); fewer experts means fewer bytes streamed per token.

    kv_type "auto": f16 KV when everything fits in memory, q8_0 KV otherwise. When weights are
    streamed, every byte of KV cache is a byte of page cache the weights lose (SwiftLM measured
    0.32 -> 4.16 tok/s at 40K context on a 126 GB model by compressing the KV cache)."""
    if kv_type == "auto":
        p = plan(spec, b, ctx, allow_paging, topk, n_expert_used, "f16")
        if p.strategy in ("expert_stream", "disk_paging") or p.verdict == "does_not_fit":
            q = plan(spec, b, ctx, allow_paging, topk, n_expert_used, "q8_0")
            if q.verdict != "does_not_fit" or p.verdict == "does_not_fit":
                q.notes.append(f"q8_0 KV cache: {p.kv_bytes / GB:.2f} -> {q.kv_bytes / GB:.2f} GB, "
                               "more RAM left for caching weights")
                return q
        return p
    if spec.train_ctx:
        ctx = min(ctx, spec.train_ctx)
    if topk and n_expert_used and spec.is_moe and topk < n_expert_used:
        experts = sum(spec.layer_expert_bytes)
        routed_active = max(0, spec.active_bytes - (spec.total_bytes - experts))
        spec = ModelSpec(**{**spec.__dict__,
                            "active_bytes": int(spec.active_bytes - routed_active * (1 - topk / n_expert_used))})
    else:
        topk = None
    kv = int(spec.kv_bytes_per_token * ctx * KV_BYTES[kv_type] / 2.0)
    p = Plan(model=spec.name, strategy="", verdict="", ctx=ctx, kv_bytes=kv, kv_type=kv_type)
    n = spec.n_layer
    dense_layers = sum(spec.layer_dense_bytes)
    experts = sum(spec.layer_expert_bytes)
    # fraction of each layer's bytes read per token (experts are sparsely read)
    exp_read = 0.0
    if spec.is_moe and experts:
        exp_read = max(0.0, (spec.active_bytes - (spec.total_bytes - experts)) / experts)
    exp_read = min(1.0, exp_read)

    def active_of(dense_b, exp_b):
        return dense_b + exp_b * exp_read

    gpu_ok = b.gpu_budget > 0
    gpu_free = b.gpu_budget - COMPUTE_OVERHEAD
    ram_for_weights = b.ram_budget - COMPUTE_OVERHEAD

    # ---- 1. everything on the GPU
    if gpu_ok and spec.total_bytes + kv <= gpu_free:
        p.strategy, p.n_gpu_layers = "full_gpu", n + 1
        p.gpu_bytes = spec.total_bytes + kv
        if b.unified:
            p.ram_bytes = 0
        p.est_tok_s_upper, p.bottleneck = _speed(0, 0, b, gpu_active=spec.active_bytes)
        p.verdict = "fits"
    # ---- 2. MoE: attention/shared weights + KV on GPU, routed experts split
    elif gpu_ok and spec.is_moe and dense_layers + spec.non_layer_bytes + kv <= gpu_free:
        room = gpu_free - (dense_layers + spec.non_layer_bytes + kv)
        on_gpu = 0
        # -ncmoe N keeps experts of the *first* N layers on the CPU, so fill from the last layer down
        for i in reversed(range(n)):
            if spec.layer_expert_bytes[i] <= room:
                room -= spec.layer_expert_bytes[i]
                on_gpu += 1
            else:
                break
        p.n_cpu_moe = n - on_gpu
        p.n_gpu_layers = n + 1
        cpu_exp = sum(spec.layer_expert_bytes[:p.n_cpu_moe])
        p.gpu_bytes = spec.total_bytes - cpu_exp + kv
        p.ram_bytes = cpu_exp
        p.strategy = "moe_hybrid" if p.n_cpu_moe else "full_gpu"
        cpu_active = cpu_exp * exp_read
        p.notes.append(f"routed experts of the first {p.n_cpu_moe}/{n} layers stay in system RAM (-ncmoe {p.n_cpu_moe})")
        # unified memory: the GPU share comes out of the same physical RAM
        ram_left = ram_for_weights - (p.gpu_bytes if b.unified else 0)
        _finish_cpu_side(p, spec, b, cpu_bytes=cpu_exp, cpu_active=cpu_active, ram_for_weights=ram_left,
                         allow_paging=allow_paging, gpu_active=spec.active_bytes - cpu_active)
    # ---- 3. dense (or MoE whose shared part is too big): offload whole layers
    elif gpu_ok:
        per_layer = [spec.layer_dense_bytes[i] + spec.layer_expert_bytes[i] + kv / n
                     for i in range(n)]
        room = gpu_free
        k = 0
        for i in reversed(range(n)):  # llama.cpp offloads the last -ngl layers
            if per_layer[i] <= room:
                room -= per_layer[i]
                k += 1
            else:
                break
        p.n_gpu_layers = k
        gpu_w = sum(spec.layer_dense_bytes[n - k:]) + sum(spec.layer_expert_bytes[n - k:])
        cpu_w = spec.total_bytes - gpu_w
        p.gpu_bytes = gpu_w + int(kv * k / n)
        p.strategy = "partial_gpu" if k else "cpu_ram"
        cpu_active = active_of(sum(spec.layer_dense_bytes[:n - k]), sum(spec.layer_expert_bytes[:n - k])) + spec.non_layer_bytes
        p.notes.append(f"{k}/{n} layers on GPU (-ngl {k})")
        ram_left = ram_for_weights - kv * (n - k) / n - (p.gpu_bytes if b.unified else 0)
        _finish_cpu_side(p, spec, b, cpu_bytes=cpu_w, cpu_active=cpu_active, ram_for_weights=ram_left,
                         allow_paging=allow_paging, kv_cpu=int(kv * (n - k) / n),
                         gpu_active=max(0, spec.active_bytes - cpu_active))
    # ---- 4. CPU only
    else:
        p.strategy = "cpu_ram"
        _finish_cpu_side(p, spec, b, cpu_bytes=spec.total_bytes, cpu_active=spec.active_bytes,
                         ram_for_weights=ram_for_weights - kv, allow_paging=allow_paging, kv_cpu=kv)

    p.topk = topk
    p.args = build_args(spec, p, b)
    return p


def _finish_cpu_side(p: Plan, spec, b: Budget, cpu_bytes, cpu_active, ram_for_weights, allow_paging, kv_cpu=0,
                     gpu_active=0):
    p.ram_bytes = cpu_bytes + kv_cpu
    if ram_for_weights <= 0:
        p.strategy, p.verdict = "impossible", "does_not_fit"
        p.notes.append("not enough RAM even for the KV cache and buffers; lower ctx")
        return
    if cpu_bytes <= ram_for_weights:
        p.verdict = "fits"
        p.est_tok_s_upper, p.bottleneck = _speed(cpu_active, 0, b, gpu_active)
        # loading fully into RAM means the USB drive can be unplugged after load
        p.no_mmap = True
        return
    # weights exceed RAM: llama.cpp mmaps the file and the OS pages weights in from storage
    paged = cpu_bytes - ram_for_weights
    p.paged_bytes = int(paged)
    p.ram_bytes = int(ram_for_weights + kv_cpu)
    if not allow_paging:
        p.strategy, p.verdict = "impossible", "does_not_fit"
        return
    p.strategy = "disk_paging"
    if spec.is_moe:
        # PocketLLM expert streaming (patched llama.cpp, ggml-pocket): read exactly the routed
        # experts, concurrently, with fault read-around off. Without it, llama.cpp's mmap
        # read-around fetches neighbouring unused experts (measured ~2.5-3x read amplification).
        p.strategy = "expert_stream"
        p.env = {"POCKET_MOE": "1", "POCKET_MOE_PREDICT": "0", "POCKET_MOE_NORA": "1"}
        p.notes.append("PocketLLM expert streaming: only routed experts are read from the drive")
    miss = paged / cpu_bytes  # share of CPU-side weights that are not resident at any moment
    p.est_tok_s_upper, p.bottleneck = _speed(cpu_active * (1 - miss), cpu_active * miss, b, gpu_active)
    p.notes.append(f"{paged / GB:.1f} GB of weights do not fit in RAM and are re-read from storage while generating")
    p.notes.append("mmap mode: the drive must stay connected; unplugging it crashes the runtime")
    if p.est_tok_s_upper is None:
        p.verdict = "slow"
    else:
        p.verdict = "slow" if p.est_tok_s_upper >= 1.0 else "impractical"


def build_args(spec: ModelSpec, p: Plan, b: Budget) -> list:
    if p.verdict == "does_not_fit":
        return []
    if spec.source == "mlx":
        # SwiftLM (MLX, Apple Silicon): streams experts from the SSD itself when asked to
        p.backend, p.env = "swiftlm", {}
        args = ["--model", spec.path, "--ctx-size", str(p.ctx)]
        if spec.is_moe and p.strategy in ("expert_stream", "disk_paging"):
            args.append("--stream-experts")
            p.notes.append("SwiftLM --stream-experts: routed experts are read from the drive per token")
        if p.topk:
            p.env["SWIFTLM_TOP_K"] = str(p.topk)
        return args
    if p.topk:
        p.env = dict(p.env, POCKET_MOE_TOPK=str(p.topk))
        p.notes.append(f"top-k {p.topk}: fewer experts per token - faster, slightly lower quality")
    # catalog entries are not on disk yet: llama.cpp can fetch them with -hf <repo>:<quant>
    src = ["-hf", spec.path] if spec.source == "estimate" else ["-m", spec.path or spec.name]
    if spec.source == "recipe":
        p.notes.append("build on the provisioning station: pocketllm requant <bf16/q8 gguf> --experts q2_k --rest q8_0")
    args = [*src, "-c", str(p.ctx), "-t", str(b.threads)]
    if p.strategy == "expert_stream" and b.unified:
        # Apple Silicon: until Metal + mmap'd experts larger than RAM is validated on a Mac,
        # stream on the CPU path (the engine hooks the CPU MoE kernel)
        p.n_gpu_layers, p.n_cpu_moe = 0, 0
        if "Metal hybrid (attention on GPU, experts streamed) not validated yet: CPU path" not in p.notes:
            p.notes.append("Metal hybrid (attention on GPU, experts streamed) not validated yet: CPU path")
    if p.n_gpu_layers:
        args += ["-ngl", str(p.n_gpu_layers)]
    else:
        args += ["-ngl", "0"]
    if p.n_cpu_moe:
        args += ["-ncmoe", str(p.n_cpu_moe)]
    # -lm none: read the whole model into RAM (USB can be unplugged after load);
    # -lm mmap: page weights in on demand (needed when the model exceeds RAM)
    if p.kv_type != "f16":
        # a quantized V cache needs flash attention in llama.cpp
        args += ["-fa", "on", "-ctk", p.kv_type, "-ctv", p.kv_type]
    args += ["-lm", "none" if p.no_mmap else "mmap"]
    if not p.no_mmap:
        args += ["--no-repack"]  # repacking copies weights into RAM and defeats streaming
    return args


def recommend(specs: list, b: Budget, ctx: int = 4096) -> list:
    """Rank candidate models: runnable first, then by size (bigger ~ more capable)."""
    order = {"fits": 0, "slow": 1, "impractical": 2, "does_not_fit": 3}
    plans = [(plan(s, b, ctx), s) for s in specs]
    plans.sort(key=lambda ps: (order.get(ps[0].verdict, 9), -ps[1].total_bytes))
    return plans
