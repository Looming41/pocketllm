
### results/linux_usb_lab.json
model Qwen3-30B-A3B-Q4_K_M.gguf (18.56 GB), RAM limit 6144 MiB, 300 MB/s per drive, 16 generated tokens, 4 threads

| config | 1 drive | 2 drives |
|---|---|---|
| llama.cpp mmap (baseline) | 0.08 ± 0.00 tok/s (161.9 GB read, n=2) | 0.12 ± 0.00 tok/s (157.9 GB read, n=2) |
| expert prefetch + no read-around on experts | 0.63 ± 0.09 tok/s ×7.5 (56.3 GB read, n=2) | 0.66 ± 0.18 tok/s ×5.5 (62.1 GB read, n=2) |
| v2 cache (no lookahead) | 0.82 ± 0.09 tok/s ×9.6 (52.8 GB read, n=2) | 1.04 ± 0.00 tok/s ×8.7 (53.9 GB read, n=2) |
| v2 cache + router lookahead | 0.65 ± 0.00 tok/s ×7.6 (55.6 GB read, n=2) | 0.91 ± 0.00 tok/s ×7.6 (55.6 GB read, n=2) |
