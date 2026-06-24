# Behavior Report: qr_v2

## Executive Summary

The fleet generated `485` trials across `238` branches, with `83` no-trial branches and `205` detected gaps. The run maintained worker activity but failed strategically: active workers were often assigned low-upside briefs, so throughput did not become leaderboard progress until the late generator-aware branch.

## Throughput

| Branch | Trials | Duration | Trials/hour | Improvement % |
|---|---:|---:|---:|---:|
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-3-brief-0 | 10 | 8049s | 4.03 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-0-brief-6 | 6 | 4098s | 4.39 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-1-brief-0 | 6 | 3064s | 5.87 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-1-brief-3 | 6 | 4187s | 4.30 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-2-brief-1 | 6 | 3012s | 5.98 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-2-brief-3 | 6 | 3180s | 5.66 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-3-brief-1 | 6 | 3378s | 5.33 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-3-brief-2 | 6 | 3694s | 4.87 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-3-brief-7 | 6 | 2962s | 6.08 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-0-brief-48 | 5 | 2399s | 6.00 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-0-brief-49 | 5 | 2247s | 6.41 | +0.00 |
| autocuda/optimize/2026-06-20-20-26-49-qr_v2/worker-0-brief-51 | 5 | 2810s | 5.12 | +0.00 |

Throughput was adequate; selection quality was not. Many branches produced validated regressions from local implementation changes.

## Gaps And Stalls

Total detected gaps: `205`; maximum branch gap: `1595` seconds. The biggest practical stall was the plateau around `74861`, where numerous micro-briefs ran without plausible 2x upside.

## Early-Termination Attempts

The run ended only after explicit graceful-termination instruction. Before termination, user interruptions corrected steering and demanded bolder briefs; they were not convergence proof.

## Other Aberrant Behavior

The manager failed to apply reviewer-before-brief discipline early and repeatedly selected micro-optimization briefs despite optimize-tree’s diversity/macro guidance. This is the primary explanation for weak results.
