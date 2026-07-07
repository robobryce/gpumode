---
name: feedback_lock_gpu_clocks_for_small_n
description: ALWAYS lock GPU clocks before histogram A/B benchmarking — unlocked clocks corrupt small-N (1M) ratios by 10-17%
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 10d12530-9310-490a-ba41-c6ae41a28d09
---

On the B200, **lock the SM clock before any A/B benchmark comparison**, especially at small element counts (N=1M):

```
sudo nvidia-smi -lgc 1800,1800   # lock both min,max to 1800 MHz (max is 1965)
# ... run benchmarks ...
sudo nvidia-smi -rgc             # reset when done
```

**Why:** The GPU idles at 120 MHz and boosts to ~1965 MHz under load. At N=1M (F64 = 8 MB, ~4 us/pass) the workload is too short for boost to stabilize, so measured throughput depends on the boost-ramp state, which in turn depends on what ran *before* the cell (benchmark ordering, warmup). Measured proof: the SAME cell (1M b16 F64 concentrated:1.0, same binary) read **1.019 from a 2-shape axis but 0.894 from a 10-shape axis** — a 17% swing in absolute GiB/s (492 vs 575) purely from run context. NVBench's per-shape Noise% was 4-14% at 1M unlocked.

**Effect of locking:** at 1800 MHz the 1M b16 F64 range A/B became uniformly 1.006-1.083 (median 1.015) across all 10 shapes — the apparent "0.85-0.93x 1M regression" VANISHED. It was 100% a clock artifact, not a kernel regression.

**How to apply:** Lock clocks for EVERY ratio comparison, not just 1M (saturated cells are less sensitive but still benefit). Re-validate any prior unlocked conclusion — e.g. the rejection of 640 threads for F64 range (1M b16=0.874 unlocked) must be re-checked locked before trusting it. Confirm the lock holds under load with `nvidia-smi --query-gpu=clocks.sm --format=csv,noheader` mid-run (should read 1800, not boost/idle). Related: [[reference_range_f64_lowbin_root_cause]].
