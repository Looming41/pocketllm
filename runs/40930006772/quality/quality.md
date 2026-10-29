reference Qwen3-30B-A3B-Q4_K_M.gguf, 16 x 512 tokens, HellaSwag 100 tasks

| variant | PPL | mean KLD | 99% KLD | same top token | HellaSwag |
|---|---|---|---|---|---|
| reference | 10.094 | – | – | – | 76.0% (100) |
| Q2_K experts + Q8_0 rest | 12.668 | 0.3361 | 3.687 | 77.1% | 65.0% (100) |
| Q2_K experts + Q4_K rest | 12.632 | 0.3400 | 3.801 | 76.6% | 64.0% (100) |
| Q3_K experts + Q8_0 rest | 10.534 | 0.0636 | 0.614 | 89.5% | 73.0% (100) |
| Q2_K experts + Q8_0 rest, top-k 4 | 17.964 | 0.7478 | 6.274 | 64.8% | 61.0% (100) |
| published UD-Q2_K_XL (unsloth) | 11.069 | 0.1266 | 1.478 | 86.2% | 72.0% (100) |
| top-k 6 | 10.236 | 0.0603 | 0.526 | 89.0% | 75.0% (100) |
| top-k 4 | 12.115 | 0.2991 | 2.540 | 77.0% | 71.0% (100) |
