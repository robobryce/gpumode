# Final selection note — linalg/eigh_py (run 2026-06-27-13-12-42)

## Selected candidate

- **Commit:** `40344f9150ad8d81ff8cde925f7cad2a35538043`
  (branch `autocuda/optimize/2026-06-27-13-12-42/worker-1-brief-8`)
- **Kernel:** batched cuSOLVER symmetric eigensolver
  (`cusolverDnXsyevBatched`, FP32, CUDA 13) — a clean, minimal (122-line),
  leaderboard-portable `submission.py` (links only `-lcusolver`).

## Local performance

- **Geomean: 50,915 µs = 1.10× over the 56,255 µs `torch.linalg.eigh` baseline.**
- All 39 local test shapes pass. Per-gate worst-case scaled residuals (audited):
  eigen 1.64 / 200 (122×), reconstruction 2.91 / 400 (137×),
  orthogonality 2.47 / 100 (40×), ascending-sort all pass — comfortable
  headroom, no near-miss, so no accuracy fallback is needed.

## Remote leaderboard record (mandatory evidence)

- **Verdict: ACCEPTED** (first attempt, no retry needed).
  `bash harness/submit.sh linalg/eigh_py` → `verdict=ACCEPTED`, `cli_exit=0`.
- Remote: ✅ Testing successful, ✅ Benchmarking successful,
  ✅ Leaderboard run successful, **Passed 39/39 tests** on NVIDIA B200 (Modal).
- Submitting user: `badelsteinlelbach`. Recent submission id ~`839603`
  (popcorn-cli reported Status=done; the numeric score is read from standings).
- Remote per-gate residuals match local (worst orthogonality ~1.03 at n=32).

## Final standings (eigh / B200, read post-submission)

- **Rank: #7 of 16.  My score: 49,576.300 µs.**
- **#1 holder: `az` at 33,954.506 µs.  Gap to #1: +46.0%.**
- Frontier cluster: #1 az 33,954; #2 Sinatras 35,579; #3 revolutionaryspaces
  35,697; #4 msaroufim (triton_diagonal_fast_path) 38,305; #5 alazarr 40,616.

## Rationale (one line)

Batched `cusolverDnXsyevBatched` (1.10×) is the **legitimate performance ceiling**
on this B200: ~12 independent custom families were built and measured slower
across the fleet (this worker alone built and benchmarked 8 — Jacobi, rank
fast-path, reduced-precision two-stage, large-n band-reduction + bulge-chase +
a custom bisection/inverse-iteration tridiagonal solver, a WY-blocked reduction
speedup, and a fused small-n eigensolver — all correct but slower, because
cuSOLVER's tuned batched kernels are very hard to beat in PyTorch/custom-CUDA
here). The 1.47–1.66× public-leaderboard cluster (#1–#5, ~34–41k µs) is **not a
legitimate kernel**: it exploits the `eval.py` input-reuse / timer reward-hack
(the harness reuses the same `data_list` across timed repeats and only rechecks
the cached output), which the run's red-team flagged as forbidden. The selected
commit is the clean, defensible, robust answer of record.
