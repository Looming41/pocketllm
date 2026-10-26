# Feasibility matrix (planner estimates, not measurements)

`stream` = PocketLLM expert streaming from the drive (MoE larger than RAM), `paging` = dense model paged from the drive (no engine benefit). Speeds are upper bounds from a bandwidth model.

assumptions: RAM bandwidth 60.0 GB/s, storage 1.0 GB/s, ctx 4096, unified-memory GPUs read at the RAM bandwidth; discrete-GPU speed is not estimated (blank)

| model | size GB | MacBook Air 16GB | 16GB laptop, no dGPU | 32GB laptop, no dGPU | 32GB + 8GB GPU | 64GB + 16GB GPU | Mac 36GB unified | Mac 64GB unified | Mac 128GB unified | 128GB mini-PC / workstation |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b-instruct:Q4_K_M | 5 | OK (gpu) ≤12.2 | OK (ram) ≤12.2 | OK (ram) ≤12.2 | OK (gpu) | OK (gpu) | OK (gpu) ≤12.2 | OK (gpu) ≤12.2 | OK (gpu) ≤12.2 | OK (ram) ≤12.2 |
| llama-3.1-8b-instruct:Q8_0 | 9 | OK (gpu) ≤7.02 | OK (ram) ≤7.02 | OK (ram) ≤7.02 | OK (partial gpu) | OK (gpu) | OK (gpu) ≤7.02 | OK (gpu) ≤7.02 | OK (gpu) ≤7.02 | OK (ram) ≤7.02 |
| llama-3.3-70b-instruct:Q4_K_M | 43 | impract. (paging) ≤0.03 | impract. (paging) ≤0.03 | impract. (paging) ≤0.06 | impract. (paging) ≤0.11 | OK (partial gpu) ≤2.11 | impract. (paging) ≤0.08 | OK (gpu) ≤1.39 | OK (gpu) ≤1.39 | OK (ram) ≤1.39 |
| llama-3.3-70b-instruct:Q2_K | 28 | impract. (paging) ≤0.06 | impract. (paging) ≤0.06 | impract. (paging) ≤0.91 | OK (partial gpu) ≤2.84 | OK (partial gpu) | OK (partial gpu) ≤2.14 | OK (gpu) ≤2.14 | OK (gpu) ≤2.14 | OK (ram) ≤2.14 |
| llama-4-scout-17b-16e-instruct:Q4_K_M | 67 | impract. (stream) ≤0.41 | impract. (stream) ≤0.12 | impract. (stream) ≤0.16 | impract. (stream) ≤0.2 | OK (moe split) | impract. (stream) ≤0.6 | slow (stream) ≤1.71 | OK (gpu) ≤5.76 | OK (ram) ≤5.76 |
| llama-4-scout-17b-16e-instruct:Q3_K_M | 54 | impract. (stream) ≤0.53 | impract. (stream) ≤0.15 | impract. (stream) ≤0.24 | slow (stream) | OK (moe split) | impract. (stream) ≤0.9 | OK (moe split) ≤7.06 | OK (gpu) ≤7.06 | OK (ram) ≤7.06 |
| llama-4-scout-17b-16e-instruct:Q2_K | 43 | impract. (stream) ≤0.73 | impract. (stream) ≤0.21 | impract. (stream) ≤0.41 | slow (stream) | OK (moe split) | slow (stream) ≤1.65 | OK (gpu) ≤8.91 | OK (gpu) ≤8.91 | OK (ram) ≤8.91 |
| llama-4-scout-17b-16e-instruct:IQ2_XXS | 28 | slow (stream) ≤1.43 | impract. (stream) ≤0.43 | slow (stream) ≤9.59 | OK (moe split) | OK (moe split) | OK (moe split) ≤13.71 | OK (gpu) ≤13.71 | OK (gpu) ≤13.71 | OK (ram) ≤13.71 |
| llama-4-scout-17b-16e-instruct:IQ1_S | 21 | slow (stream) ≤2.54 | impract. (stream) ≤0.78 | OK (ram) ≤18.1 | OK (moe split) | OK (moe split) | OK (gpu) ≤18.1 | OK (gpu) ≤18.1 | OK (gpu) ≤18.1 | OK (ram) ≤18.1 |
| gpt-oss-120b:MXFP4 | 62 | slow (stream) ≤4.06 | impract. (stream) ≤0.47 | impract. (stream) ≤0.67 | slow (stream) | OK (moe split) | slow (stream) ≤5.88 | slow (stream) ≤15.68 | OK (gpu) ≤22.15 | OK (ram) ≤22.15 |
| qwen3-30b-a3b:Q4_K_M | 19 | slow (stream) ≤4.14 | slow (stream) ≤1.76 | OK (ram) ≤29.68 | OK (moe split) | OK (moe split) | OK (gpu) ≤29.68 | OK (gpu) ≤29.68 | OK (gpu) ≤29.68 | OK (ram) ≤29.68 |
| qwen3-30b-a3b:IQ2_XXS | 8 | OK (gpu) ≤70.61 | OK (ram) ≤70.61 | OK (ram) ≤70.61 | OK (moe split) | OK (gpu) | OK (gpu) ≤70.61 | OK (gpu) ≤70.61 | OK (gpu) ≤70.61 | OK (ram) ≤70.61 |
| gpt-oss-20b:MXFP4 | 11 | OK (moe split) ≤31.37 | OK (ram) ≤31.37 | OK (ram) ≤31.37 | OK (moe split) | OK (gpu) | OK (gpu) ≤31.37 | OK (gpu) ≤31.37 | OK (gpu) ≤31.37 | OK (ram) ≤31.37 |

## Same, with a fast internal/USB4 SSD (3 GB/s) and 100 GB/s memory (M-series Pro class)

assumptions: RAM bandwidth 100.0 GB/s, storage 3.0 GB/s, ctx 4096, unified-memory GPUs read at the RAM bandwidth; discrete-GPU speed is not estimated (blank)

| model | size GB | MacBook Air 16GB | 16GB laptop, no dGPU | 32GB laptop, no dGPU | 32GB + 8GB GPU | 64GB + 16GB GPU | Mac 36GB unified | Mac 64GB unified | Mac 128GB unified | 128GB mini-PC / workstation |
|---|---|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b-instruct:Q4_K_M | 5 | OK (gpu) ≤20.33 | OK (ram) ≤20.33 | OK (ram) ≤20.33 | OK (gpu) | OK (gpu) | OK (gpu) ≤20.33 | OK (gpu) ≤20.33 | OK (gpu) ≤20.33 | OK (ram) ≤20.33 |
| llama-3.1-8b-instruct:Q8_0 | 9 | OK (gpu) ≤11.69 | OK (ram) ≤11.69 | OK (ram) ≤11.69 | OK (partial gpu) | OK (gpu) | OK (gpu) ≤11.69 | OK (gpu) ≤11.69 | OK (gpu) ≤11.69 | OK (ram) ≤11.69 |
| llama-3.3-70b-instruct:Q4_K_M | 43 | impract. (paging) ≤0.1 | impract. (paging) ≤0.1 | impract. (paging) ≤0.18 | impract. (paging) ≤0.31 | OK (partial gpu) ≤3.52 | impract. (paging) ≤0.23 | OK (gpu) ≤2.31 | OK (gpu) ≤2.31 | OK (ram) ≤2.31 |
| llama-3.3-70b-instruct:Q2_K | 28 | impract. (paging) ≤0.19 | impract. (paging) ≤0.19 | slow (paging) ≤2.04 | OK (partial gpu) ≤4.73 | OK (partial gpu) | OK (partial gpu) ≤3.57 | OK (gpu) ≤3.57 | OK (gpu) ≤3.57 | OK (ram) ≤3.57 |
| llama-4-scout-17b-16e-instruct:Q4_K_M | 67 | slow (stream) ≤1.18 | impract. (stream) ≤0.36 | impract. (stream) ≤0.48 | impract. (stream) ≤0.58 | OK (moe split) | slow (stream) ≤1.68 | slow (stream) ≤4.18 | OK (gpu) ≤9.6 | OK (ram) ≤9.6 |
| llama-4-scout-17b-16e-instruct:Q3_K_M | 54 | slow (stream) ≤1.52 | impract. (stream) ≤0.46 | impract. (stream) ≤0.7 | slow (stream) | OK (moe split) | slow (stream) ≤2.47 | OK (moe split) ≤11.76 | OK (gpu) ≤11.76 | OK (ram) ≤11.76 |
| llama-4-scout-17b-16e-instruct:Q2_K | 43 | slow (stream) ≤2.07 | impract. (stream) ≤0.63 | slow (stream) ≤1.19 | slow (stream) | OK (moe split) | slow (stream) ≤4.35 | OK (gpu) ≤14.85 | OK (gpu) ≤14.85 | OK (ram) ≤14.85 |
| llama-4-scout-17b-16e-instruct:IQ2_XXS | 28 | slow (stream) ≤4 | slow (stream) ≤1.27 | slow (stream) ≤18.5 | OK (moe split) | OK (moe split) | OK (moe split) ≤22.84 | OK (gpu) ≤22.84 | OK (gpu) ≤22.84 | OK (ram) ≤22.84 |
| llama-4-scout-17b-16e-instruct:IQ1_S | 21 | slow (stream) ≤6.92 | slow (stream) ≤2.29 | OK (ram) ≤30.17 | OK (moe split) | OK (moe split) | OK (gpu) ≤30.17 | OK (gpu) ≤30.17 | OK (gpu) ≤30.17 | OK (ram) ≤30.17 |
| gpt-oss-120b:MXFP4 | 62 | slow (stream) ≤10.73 | slow (stream) ≤1.41 | slow (stream) ≤1.99 | slow (stream) | OK (moe split) | slow (stream) ≤14.67 | slow (stream) ≤30.1 | OK (gpu) ≤36.91 | OK (ram) ≤36.91 |
| qwen3-30b-a3b:Q4_K_M | 19 | slow (stream) ≤11.3 | slow (stream) ≤5.11 | OK (ram) ≤49.47 | OK (moe split) | OK (moe split) | OK (gpu) ≤49.47 | OK (gpu) ≤49.47 | OK (gpu) ≤49.47 | OK (ram) ≤49.47 |
| qwen3-30b-a3b:IQ2_XXS | 8 | OK (gpu) ≤117.68 | OK (ram) ≤117.68 | OK (ram) ≤117.68 | OK (moe split) | OK (gpu) | OK (gpu) ≤117.68 | OK (gpu) ≤117.68 | OK (gpu) ≤117.68 | OK (ram) ≤117.68 |
| gpt-oss-20b:MXFP4 | 11 | OK (moe split) ≤52.29 | OK (ram) ≤52.29 | OK (ram) ≤52.29 | OK (moe split) | OK (gpu) | OK (gpu) ≤52.29 | OK (gpu) ≤52.29 | OK (gpu) ≤52.29 | OK (ram) ≤52.29 |

### Sanity check against an independent measurement
TinyGiant ([jerryjokesalot/tinygiant](https://github.com/jerryjokesalot/tinygiant)) reports measured decode on an **Apple M1 16 GB** with Qwen3-30B-A3B Q4_K_M: 1.06 tok/s for stock llama.cpp CPU, and 2.68 tok/s for its expert-streaming build (project's own benchmark, not verified by us). The planner's upper bound for the "MacBook Air 16GB" tier is ≤4.1 tok/s at 1 GB/s storage and ≤11.3 tok/s at 3 GB/s. The measured value sits below both bounds, as it should, and stock llama.cpp sits well below them, consistent with the read amplification we measured (RESULTS R5).
