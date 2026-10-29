
### results/linux_usb_lab.json
model Qwen3-30B-A3B-Q4_K_M.gguf (18.56 GB), RAM limit 6144 MiB, 300 MB/s per drive, 16 generated tokens, 4 threads

| config | 1 drive | 2 drives |
|---|---|---|
| llama.cpp mmap (baseline) | 0.07 ± 0.02 tok/s (201.3 GB read, n=2) | 0.12 ± 0.00 tok/s (158.8 GB read, n=2) |
| expert prefetch + no read-around on experts | 0.64 ± 0.03 tok/s ×9.5 (55.3 GB read, n=2) | 0.65 ± 0.16 tok/s ×5.4 (61.5 GB read, n=2) |
| v1 + lazy load + dense weights locked (slow-drive default) | 0.94 ± 0.00 tok/s ×14.0 (14.2 GB read, n=2) | 1.31 ± 0.00 tok/s ×10.9 (14.2 GB read, n=2) |
| slow-drive default + top-k 4 (quality trade-off) | 1.47 ± 0.00 tok/s ×22.0 (9.4 GB read, n=2) | 2.07 ± 0.01 tok/s ×17.3 (9.4 GB read, n=2) |
