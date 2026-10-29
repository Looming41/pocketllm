
### results/linux_usb_lab_requant.json
model requant-E2R8.gguf (11.19 GB), RAM limit 6144 MiB, 300 MB/s per drive, 16 generated tokens, 4 threads

| config | 1 drive |
|---|---|
| expert prefetch + no read-around on experts | 1.87 ± 0.87 tok/s (30.7 GB read, n=2) |
| v1 + lazy load + dense weights locked (slow-drive default) | 1.86 ± 0.00 tok/s ×1.0 (9.9 GB read, n=2) |
| slow-drive default + top-k 4 (quality trade-off) | 2.83 ± 0.01 tok/s ×1.5 (7.6 GB read, n=2) |
