
### results/linux_usb_lab_requant.json
model requant-E2R8.gguf (11.19 GB), RAM limit 6144 MiB, 300 MB/s per drive, 16 generated tokens, 4 threads

| config | 1 drive |
|---|---|
| expert prefetch + no read-around on experts | 1.79 ± 0.01 tok/s (9.7 GB read, n=2) |
| v2 cache (no lookahead) | 1.53 ± 0.01 tok/s ×0.9 (10.6 GB read, n=2) |
| v2 cache + top-k 4 (quality trade-off) | 2.46 ± 0.01 tok/s ×1.4 (7.8 GB read, n=2) |
