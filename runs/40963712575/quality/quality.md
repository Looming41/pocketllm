reference Qwen3-30B-A3B-Q4_K_M.gguf, 16 x 512 tokens, HellaSwag 100 tasks

| variant | PPL | mean KLD | 99% KLD | same top token | HellaSwag |
|---|---|---|---|---|---|
| reference | 10.127 | – | – | – | 76.0% (100) |
| Q3_K experts + Q8_0 rest | 10.530 | 0.0649 | 0.660 | 89.2% | 71.0% (100) |
| Q3_K experts + Q8_0 rest, top-k 6 | 10.842 | 0.1194 | 1.220 | 85.2% | 75.0% (100) |
| published UD-Q2_K_XL (unsloth) | 11.068 | 0.1280 | 1.600 | 86.1% | 72.0% (100) |
| published UD-Q2_K_XL (unsloth), top-k 6 | 11.285 | 0.1707 | 1.941 | 84.0% | 69.0% (100) |
| top-k 6 | 10.268 | 0.0613 | 0.618 | 89.2% | 74.0% (100) |
