# Brief 100 hosted profiles

- Requested: `2026-07-21T00:07:44.998331+00:00`
- Completed: `2026-07-21T00:13:34.975598+00:00`
- Commit: `5f048a1c1afc2a3d7d5ab0963aba6e9e93a912ef`
- Benchmark index: `10` (`batch=1, n=4096, cond=2, seed=48096`)
- Job: `f614742501984c25883334e3df0005f5` (`succeeded`)
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- env POPCORN_BREV_PROFILER_URL=https://http--brev-profiler-proxy--dxfjds728w5v.code.run popcorn-cli submit --no-tui --leaderboard cholesky --gpu B200 --mode profile --profile-brev --benchmark-index 10 submission.py`
- Archive: `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-100-trial-4-5f048a1c1afc2a3d7d5ab0963aba6e9e93a912ef.ncu-rep`
- SHA-256: `5e60042a515fce7d3db351bbf8cf65b3ac2fabca27a8371fc516011cd9663e76`

## Tenth-trial cadence waiver

- Approval denied after trial 8 because all three run-wide hosted slots were active (Brief 100 index 10, Brief 101 index 14, and Brief 98 index 0).
- Manager instruction: continue without waiting, make no request before approximately `2026-07-21T01:08:00Z`, and re-request index 6 with the exact committed SHA only if it remains valuable.

## Conditional index-6 waiver

- Conditional approval was granted after `01:08Z` for exact commit `2500c0f8e7f76879b539f23acf662b74c5c9864a`, index 6, only if its infrastructure retry passed and remained useful.
- The retry passed but measured `787.847382 us` full-grid with all four target cells slower than the stronger donor, so the approved request was not invoked and the slot remained unused.

## Fifteenth-trial cadence waiver

- Index-4 profiling of exact run-leading commit `15c9e8384de94cf42e9ec9088bd9aeba6d364ceb` was denied until approximately `2026-07-21T01:54:00Z` because Brief 98's 00:53 capture, Brief 102's 01:22 capture, and Brief 99's approved request filled or reserved the rolling-window capacity.
- Manager instruction: continue without waiting and re-request only with the then-best exact SHA.

## Twentieth-trial hosted profile

- Requested: `2026-07-21T02:52:13.674867+00:00`
- Completed: `2026-07-21T02:56:01.194086+00:00`
- Commit: `06cbd50b0b01e81997be858765235eceb4cd55c2`
- Benchmark index: `8` (`batch=2, n=2048, cond=2, seed=44048`)
- Job: `3a4b4061c9004170bd955e713341b118` (`succeeded`)
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- env POPCORN_BREV_PROFILER_URL=https://http--brev-profiler-proxy--dxfjds728w5v.code.run popcorn-cli submit --no-tui --leaderboard cholesky --gpu B200 --mode profile --profile-brev --benchmark-index 8 submission.py`
- Archive: `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-100-trial-20-06cbd50b0b01e81997be858765235eceb4cd55c2.ncu-rep`
- SHA-256: `5c3b96832d895fca0f8136e0979b1d9412b3f261eee27de269f96fdfe641023c`
