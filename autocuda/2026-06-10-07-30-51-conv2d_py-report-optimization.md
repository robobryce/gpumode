# Optimization Report: `2026-06-10-07-30-51-conv2d_py`

This run optimized `pmpp_v2/conv2d_py` on B200 with a 4-worker optimize-tree loop. The final safe submitted source is commit `bc053cc626b666751c18871efc6d01674bcc6241`.

## Leaderboard Submissions

- Baseline `f47cc6c232bec176711608e8c66001de701a2aaa`: GPUMODE `conv2d_v2` / B200 submission `785293`, passed, ranked around 52.9 ms.
- Safe candidate `bc053cc626b666751c18871efc6d01674bcc6241`: submission `785294`, passed public/secret tests and ranked around 6.45 ms.
- Cadence resubmission of the same safe source: submission `785314`, passed public + secret test/benchmark/leaderboard.

## Main Valid Optimization

The accepted B200-safe candidate combines:

- a k8 c64 CUDA graph path with cache keys including input shape/stride, kernel shape/stride, and `kernel.data_ptr()`;
- a c64/k16 `im2col` + expanded-weight `torch.bmm` path;
- cuDNN fallback for c128 shapes with TF32 disabled.

## Explored And Rejected

- H100 FFT+sparse-correction champion (`robobryce/autocuda-gpumode` commit `6bf0aaa`) was investigated. Raw port built and public-validated but failed B200 benchmark recheck; guarded/minimal ports were safe only by falling back to cuDNN and were slower.
- TF32 and split-matmul ideas for c128/k32 failed official tolerance.
- Direct CUDA c64/k16 was correct but far slower than `im2col+bmm`.
- c64/k16 column caching was proven unsafe under same-pointer input mutation.
- Multiple Python dispatch, import, graph-output, and cuDNN wrapper variants were either noisy, slower, or not reproducibly better.

## CSV

The companion CSV contains 585 optimization rows across worker branches.
