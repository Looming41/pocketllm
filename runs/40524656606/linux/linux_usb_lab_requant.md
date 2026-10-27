
### results/linux_usb_lab_requant.json
model requant-E2R8.gguf (11.19 GB), RAM limit 6144 MiB, 300 MB/s per drive, 16 generated tokens, 4 threads

| config | 1 drive |
|---|---|
| expert prefetch + no read-around on experts | 1.82 ± 0.60 tok/s (30.6 GB read, n=2) |
| v2 cache (no lookahead) | 2.01 ± 0.14 tok/s ×1.1 (29.8 GB read, n=2) |
| v2 cache + top-k 4 (quality trade-off) | 3.51 ± 0.01 tok/s ×1.9 (27.4 GB read, n=2) |
