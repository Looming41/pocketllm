
### results/linux_usb_lab_experts-q3.json
model experts-q3.gguf (14.14 GB), RAM limit 6144 MiB, 300 MB/s per drive, 16 generated tokens, 4 threads

| config | 1 drive | 2 drives |
|---|---|---|
| v1 + lazy load + dense weights locked (slow-drive default) | 1.40 ± 0.00 tok/s (12.0 GB read, n=2) | 1.96 ± 0.00 tok/s (12.0 GB read, n=2) |
| slow-drive default + top-k 6 (near-lossless, R17) | 1.67 ± 0.00 tok/s ×1.2 (10.5 GB read, n=2) | 2.35 ± 0.00 tok/s ×1.2 (10.5 GB read, n=2) |
