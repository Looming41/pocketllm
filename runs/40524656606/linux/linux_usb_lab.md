
### results/linux_usb_lab.json
model Qwen3-30B-A3B-Q4_K_M.gguf (18.56 GB), RAM limit 6144 MiB, 300 MB/s per drive, 16 generated tokens, 4 threads

| config | 1 drive | 2 drives |
|---|---|---|
| expert prefetch + no read-around on experts | 0.70 ± 0.00 tok/s (54.8 GB read, n=2) | 0.63 ± 0.13 tok/s (62.9 GB read, n=2) |
| v2 cache (no lookahead) | 0.76 ± 0.03 tok/s ×1.1 (53.6 GB read, n=2) | 1.03 ± 0.02 tok/s ×1.6 (54.1 GB read, n=2) |
| v1 + warm start from routing census | 0.65 ± 0.00 tok/s ×0.9 (55.8 GB read, n=2) | 0.74 ± 0.04 tok/s ×1.2 (58.9 GB read, n=2) |
| v2 cache + top-k 4 (quality trade-off) | 1.61 ± 0.02 tok/s ×2.3 (46.0 GB read, n=2) | 2.22 ± 0.07 tok/s ×3.5 (46.2 GB read, n=2) |
