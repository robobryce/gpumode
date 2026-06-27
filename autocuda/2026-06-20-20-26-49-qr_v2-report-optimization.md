# Optimization Report: qr_v2

## Executive Summary

| Metric | Value |
|---|---|
| Final accepted best | `39e59cea` (`73896.201171`) |
| Best end-to-end speedup | `1.7789x` |
| Optimization rows | `653` |
| Kept / failed rows | `603` / `50` |
| Main failure | Manager spent hours on low-upside micro/routing/zero-fill variants before pursuing generator-aware structure. |

The run eventually produced a valid accepted result, but it underperformed the user objective because the optimization search was too conservative for too long. The late generator-aware rank-1 QR path was the only clear strategic breakthrough.

## Most Impactful Optimization Techniques

| Technique | Peak contribution | Prevalence | Representative commits |
|---|---:|---|---|
| Custom CUDA/Triton Householder kernels | 1.4486x | 228 branches, 244 attempts, 18 kept/1 failed | `8802b14f`, `8639d1fc`, `7b93bf8b`, `be26e71c`, `ba966d0d` |
| Combining accepted branches | 1.1878x | 228 branches, 1673 attempts, 52 kept/0 failed | `5e877ad1`, `50996395`, `2238308e`, `5910f00d`, `f1e1a44b` |
| n512 device-side mixed dispatcher | 1.1828x | 83 branches, 109 attempts, 26 kept/3 failed | `282510be`, `1d2b3f0e`, `546fcb39`, `110a3606`, `6de29ae8` |
| Dense blocked/panel QR replacements | 1.1774x | 42 branches, 62 attempts, 50 kept/7 failed | `568c1feb`, `5982a98e`, `f8a841a8`, `d8da6de9`, `dbe19f39` |
| Local routing/allocation changes | 1.1766x | 236 branches, 4235 attempts, 284 kept/16 failed | `f995f1bb`, `bdeea3fa`, `876c4ff9`, `ec760be8`, `29aab2d5` |
| n1024 structured prefix reductions | 1.1539x | 77 branches, 164 attempts, 23 kept/1 failed | `c8a9c275`, `bad50345`, `25a5ce9b`, `be88f140`, `6e872fd5` |
| Static-shape graph executors | 1.1286x | 49 branches, 190 attempts, 37 kept/11 failed | `229324ea`, `7d9d45f8`, `ba13e335`, `5ce011e0`, `ba6f4e9b` |
| Compact-output write-order assembly | 1.0199x | 149 branches, 351 attempts, 100 kept/4 failed | `788213b9`, `3318db42`, `3ef45d94`, `ac65c643`, `eacb52c7` |
| Generator-aware structured QR | 1.0109x | 4 branches, 7 attempts, 5 kept/0 failed | `39e59cea`, `ba6dda97`, `ce16bb4a`, `39e59cea`, `ba6dda97` |
| Library substitution routes | 1.0030x | 5 branches, 15 attempts, 8 kept/7 failed | `f6357871`, `56bc0d81`, `51f5d668`, `9c0e4559`, `234772c3` |

Generator-aware structured QR produced the best real marginal improvement: `ba6dda97` took the parent from `74861.362275` to `74050.971646`, and `39e59cea` improved further to `73896.201171`. Earlier dispatcher, graph, and write-order families produced valid but small deltas.

## Failed Optimization Techniques

Dense custom QR, cuSOLVER/cuBLAS substitutions, graph-slot tweaks, Python mask cleanups, and inactive-tail write-order variants repeatedly validated but regressed. The core lesson is that most failures were not bold enough: they changed implementation details around the dominant algorithms instead of replacing the algorithms or exploiting generator structure.

## Unexplored Areas

- Closed-form compact H/tau emitters for banded, row-scaled, clustered, and rank-deficient generators.
- Real Blackwell blocked-WY QR with CuTe/Triton tensor-core trailing updates.
- A per-shape speed-of-light model for n176/n352/n512/n1024/n2048/n4096.
- Aggressive validation-safe tolerance exploitation for compact output semantics.
- Reviewer-enforced 2x-upside screening from the first brief.
