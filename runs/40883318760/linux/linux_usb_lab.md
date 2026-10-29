
### results/linux_usb_lab.json
model Qwen3-30B-A3B-Q4_K_M.gguf (18.56 GB), RAM limit 6144 MiB, 300 MB/s per drive, 16 generated tokens, 4 threads

| config | 1 drive | 2 drives |
|---|---|---|
| llama.cpp mmap (baseline) | 0.08 ± 0.00 tok/s (162.6 GB read, n=2) | 0.07 ± 0.00 tok/s (238.9 GB read, n=2) |
| expert prefetch + no read-around on experts | 0.83 ± 0.01 tok/s ×9.9 (15.4 GB read, n=2) | 1.18 ± 0.00 tok/s ×17.8 (15.3 GB read, n=2) |
| v1 + dense weights locked in RAM | 0.94 ± 0.00 tok/s ×11.1 (14.2 GB read, n=2) | 1.31 ± 0.00 tok/s ×19.7 (14.2 GB read, n=2) |
| v2 cache (no lookahead) | 0.74 ± 0.00 tok/s ×8.7 (16.8 GB read, n=2) | 1.03 ± 0.00 tok/s ×15.6 (16.6 GB read, n=2) |
| v2 cache + dense weights locked in RAM | 0.79 ± 0.01 tok/s ×9.3 (16.1 GB read, n=2) | 1.09 ± 0.01 tok/s ×16.4 (16.0 GB read, n=2) |
