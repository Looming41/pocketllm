
### results/linux_usb_lab.json
model Qwen3-30B-A3B-Q4_K_M.gguf (18.56 GB), RAM limit 6144 MiB, 300 MB/s per drive, 16 generated tokens, 4 threads

| config | 1 drive | 2 drives |
|---|---|---|
| v1 + lazy load + dense weights locked (slow-drive default) | 0.94 ± 0.00 tok/s (14.2 GB read, n=2) | 1.31 ± 0.00 tok/s (14.2 GB read, n=2) |
| slow-drive default + top-k 6 (near-lossless, R17) | 1.14 ± 0.00 tok/s ×1.2 (11.9 GB read, n=2) | 1.59 ± 0.00 tok/s ×1.2 (11.9 GB read, n=2) |
