assumptions: RAM bandwidth 60.0 GB/s, storage 1.0 GB/s, ctx 4096, unified-memory GPUs read at the RAM bandwidth; discrete-GPU speed is not estimated (blank)

| model | size GB | 16GB laptop, no dGPU | 32GB laptop, no dGPU | 32GB + 8GB GPU | 64GB + 16GB GPU | Mac 36GB unified | Mac 64GB unified | Mac 128GB unified | 128GB mini-PC / workstation |
|---|---|---|---|---|---|---|---|---|---|
| llama-3.1-8b-instruct:Q4_K_M | 5 | OK (cpu ram) ≤12.2 | OK (cpu ram) ≤12.2 | OK (full gpu) | OK (full gpu) | OK (full gpu) ≤12.2 | OK (full gpu) ≤12.2 | OK (full gpu) ≤12.2 | OK (cpu ram) ≤12.2 |
| llama-3.1-8b-instruct:Q8_0 | 9 | OK (cpu ram) ≤7.02 | OK (cpu ram) ≤7.02 | OK (partial gpu) | OK (full gpu) | OK (full gpu) ≤7.02 | OK (full gpu) ≤7.02 | OK (full gpu) ≤7.02 | OK (cpu ram) ≤7.02 |
| llama-4-scout-17b-16e-instruct:Q4_K_M | 67 | impract. (disk paging) ≤0.12 | impract. (disk paging) ≤0.16 | impract. (disk paging) ≤0.2 | OK (moe hybrid) | impract. (disk paging) ≤0.6 | slow (disk paging) ≤1.71 | OK (full gpu) ≤5.76 | OK (cpu ram) ≤5.76 |
| llama-4-scout-17b-16e-instruct:Q3_K_M | 54 | impract. (disk paging) ≤0.15 | impract. (disk paging) ≤0.24 | slow (disk paging) | OK (moe hybrid) | impract. (disk paging) ≤0.9 | OK (moe hybrid) ≤7.06 | OK (full gpu) ≤7.06 | OK (cpu ram) ≤7.06 |
| llama-4-scout-17b-16e-instruct:Q2_K | 43 | impract. (disk paging) ≤0.21 | impract. (disk paging) ≤0.41 | slow (disk paging) | OK (moe hybrid) | slow (disk paging) ≤1.65 | OK (full gpu) ≤8.91 | OK (full gpu) ≤8.91 | OK (cpu ram) ≤8.91 |
| llama-4-scout-17b-16e-instruct:IQ2_XXS | 28 | impract. (disk paging) ≤0.43 | slow (disk paging) ≤9.59 | OK (moe hybrid) | OK (moe hybrid) | OK (moe hybrid) ≤13.71 | OK (full gpu) ≤13.71 | OK (full gpu) ≤13.71 | OK (cpu ram) ≤13.71 |
| llama-4-scout-17b-16e-instruct:IQ1_S | 21 | impract. (disk paging) ≤0.78 | OK (cpu ram) ≤18.1 | OK (moe hybrid) | OK (moe hybrid) | OK (full gpu) ≤18.1 | OK (full gpu) ≤18.1 | OK (full gpu) ≤18.1 | OK (cpu ram) ≤18.1 |
| gpt-oss-120b:MXFP4 | 62 | impract. (disk paging) ≤0.47 | impract. (disk paging) ≤0.67 | slow (disk paging) | OK (moe hybrid) | slow (disk paging) ≤5.88 | slow (disk paging) ≤15.68 | OK (full gpu) ≤22.15 | OK (cpu ram) ≤22.15 |
