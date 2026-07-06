# eigh optimize-tree run 2026-06-27-13-12-42 — submission notes

## Submission 1 (baseline) — 13:16
- Commit 424ba1894 (stock torch.linalg.eigh), local geomean 56,255 µs.
- verdict=ACCEPTED, 39/39 tests. Establishes run validity.

## Submission 2 (1.10x best) — 15:39
- Commit 2332f1a0 (batched cusolverDnXsyevBatched load_inline C++ extension), local geomean 50,998 µs (1.10x).
- verdict=ACCEPTED, 39/39 tests. KEY RESULT: a custom load_inline C++/cuSOLVER extension BUILDS AND RUNS on the remote Modal harness — de-risks all custom-kernel briefs.
- Leaderboard standings (B200 eigh): #1 az best.py 33,954 µs; my rank 6 (badelsteinlelbach, 49,739 µs from a pre-existing submission faster than this run's 50,998). Gap to #1 ~46%. Frontier unchanged — fleet must keep pushing custom kernels into the dense interior.
