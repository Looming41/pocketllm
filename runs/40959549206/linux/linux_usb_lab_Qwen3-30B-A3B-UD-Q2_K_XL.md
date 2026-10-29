
### results/linux_usb_lab_Qwen3-30B-A3B-UD-Q2_K_XL.json
model Qwen3-30B-A3B-UD-Q2_K_XL.gguf (11.81 GB), RAM limit 6144 MiB, 300 MB/s per drive, 16 generated tokens, 4 threads

| config | 1 drive | 2 drives |
|---|---|---|
| v1 + lazy load + dense weights locked (slow-drive default) | 1.54 ± 0.00 tok/s (9.4 GB read, n=2) | 2.17 ± 0.00 tok/s (9.4 GB read, n=2) |
| slow-drive default + top-k 6 (near-lossless, R17) | 1.82 ± 0.00 tok/s ×1.2 (8.1 GB read, n=2) | 2.58 ± 0.01 tok/s ×1.2 (8.1 GB read, n=2) |
