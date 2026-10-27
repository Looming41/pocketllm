"""Download a model from Hugging Face for the CI lab, trying candidate repos in order.

usage: python scripts/fetch_model.py PRESET OUT_DIR [--max-gb N]
Prints the path of the downloaded GGUF (first split part) or MLX directory.

Presets list several community repos because names change; the first repo that
exists and has a matching file under the size cap wins.
"""
import argparse
import fnmatch
import os
import sys

PRESETS = {
    # small real MoE: real routing statistics in a few GB
    "olmoe": {"kind": "gguf", "repos": ["allenai/OLMoE-1B-7B-0125-Instruct-GGUF", "allenai/OLMoE-1B-7B-0924-Instruct-GGUF",
                                         "bartowski/OLMoE-1B-7B-0125-Instruct-GGUF"],
              "patterns": ["*Q4_K_M*.gguf", "*q4_k_m*.gguf", "*Q4*.gguf"]},
    # the 30B-class MoE that runs on 16 GB laptops when streamed
    "qwen3-30b-a3b": {"kind": "gguf", "repos": ["unsloth/Qwen3-30B-A3B-GGUF", "Qwen/Qwen3-30B-A3B-GGUF",
                                                 "bartowski/Qwen_Qwen3-30B-A3B-GGUF"],
                      "patterns": ["*Q4_K_M*.gguf", "*Q3_K_M*.gguf", "*Q2_K*.gguf", "*IQ2_XXS*.gguf"]},
    "gpt-oss-20b": {"kind": "gguf", "repos": ["ggml-org/gpt-oss-20b-GGUF", "unsloth/gpt-oss-20b-GGUF"],
                    "patterns": ["*mxfp4*.gguf", "*MXFP4*.gguf", "*Q4_K_M*.gguf"]},
    # Meta: Llama 4 Scout (109B MoE) - only for big runners
    "llama4-scout": {"kind": "gguf", "repos": ["unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF"],
                     "patterns": ["*IQ1_S*.gguf", "*IQ2_XXS*.gguf", "*Q2_K*.gguf"]},
    # MLX builds for SwiftLM on Apple Silicon
    "qwen3-30b-a3b-mlx": {"kind": "mlx", "repos": ["mlx-community/Qwen3-30B-A3B-4bit", "mlx-community/Qwen3-30B-A3B-3bit"]},
}


def pick_files(api, repo, patterns, max_bytes):
    info = api.model_info(repo, files_metadata=True)
    files = {s.rfilename: (s.size or 0) for s in info.siblings}
    for pat in patterns:
        hits = sorted(f for f in files if fnmatch.fnmatch(f.split("/")[-1], pat))
        if not hits:
            continue
        first = hits[0]
        # a split model: take every part of the same set
        stem = first.split("-00001-of-")[0] if "-00001-of-" in first else None
        group = [f for f in files if stem and f.startswith(stem) and f.endswith(".gguf")] if stem else [first]
        size = sum(files[f] for f in group)
        if size <= max_bytes:
            return group, size
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("out")
    ap.add_argument("--max-gb", type=float, default=1000)
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    p = PRESETS[a.preset]
    os.makedirs(a.out, exist_ok=True)
    for repo in p["repos"]:
        try:
            if p["kind"] == "mlx":
                info = api.model_info(repo, files_metadata=True)
                size = sum(s.size or 0 for s in info.siblings)
                if size > a.max_gb * 1e9:
                    print(f"skip {repo}: {size / 1e9:.1f} GB > cap", file=sys.stderr)
                    continue
                path = snapshot_download(repo, local_dir=os.path.join(a.out, repo.split("/")[1]))
                print(path)
                return
            files, size = pick_files(api, repo, p["patterns"], a.max_gb * 1e9)
            if not files:
                print(f"skip {repo}: no matching file under {a.max_gb} GB", file=sys.stderr)
                continue
            print(f"{repo}: {files} ({size / 1e9:.1f} GB)", file=sys.stderr)
            paths = [hf_hub_download(repo, f, local_dir=a.out) for f in sorted(files)]
            print(paths[0])
            return
        except Exception as e:  # repo missing, gated, network
            print(f"skip {repo}: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(f"no candidate for {a.preset} could be downloaded")


if __name__ == "__main__":
    main()
