# Leaderboard submissions and profile evidence — 2026-07-19-01-44-06-cholesky-resumed-local
Reconstructed as one continued experiment from the three source-tag segments below. Brief references use the combined global IDs.
## Continuation segment 1

- Original source tag: `2026-07-19-01-44-06-cholesky-resumed-local-unmerged`
- Combined brief IDs: `0–97`

### Baseline

- Commit: `52427ff70e52ec6262c50b0bdbdbf9531ac09e42`
- Command: `bash harness/submit.sh linalg/cholesky_py`
- Local Modal benchmark: 2153.690486 us (logged as 2153.690 us)
- Submission: 886480, accepted on B200
- Public leaderboard score: 2069.097787 us
- Secret leaderboard score: 2104.407305 us
- Standings after submission: badelsteinlelbach rank 45/51; 557.5% gap to #1 viridale at 314.669 us
- Submitted: 2026-07-19T01:56:27Z; verdict completed at approximately 2026-07-19T01:58:14Z

### Hosted profiling attempts

- 2026-07-19T01:59:40Z, baseline commit `52427ff70e52ec6262c50b0bdbdbf9531ac09e42`, benchmark index 5 (`batch=640, n=512`): infrastructure failure before submission because neither `POPCORN_BREV_PROFILER_URL` nor `BREV_PROFILER_URL` is configured. No profile artifact was produced.
- 2026-07-19T16:53:31Z, brief-52 commit `d3d552f6401fd10069bb61dae8412a775d3eeba0`, benchmark index 5 (`batch=640, n=512`): `popcorn-cli submit --no-tui --leaderboard cholesky --gpu B200 --mode profile --profile-brev --benchmark-index 5 problems/linalg/cholesky_py/submission.py` exited 1 before submission with `Application error: POPCORN_BREV_PROFILER_URL or BREV_PROFILER_URL is not set. Configure a hardened Brev profiler endpoint before using --profile-brev.` No Nsight artifact was produced. Plain profile submission 888017 had already run all indices and all 15 jobs failed; 888018 completed with no recorded run; 888023 remained pending at inspection time.
- 2026-07-19T16:53:24Z, brief-50 selected n=512 commit `1806068ad883f308bc5d97b5a4d0763ca104a591`: plain `--mode profile --benchmark-index 5` was accepted on B200, but plain mode does not honor `--benchmark-index` (the CLI documents it only for `--profile-brev`) and attempted all 15 cells. Every Nsight Compute launch connected to Python and then exited 9 with `Failed to initialize the profiler: LibraryNotLoaded. Check that a compatible driver library is loaded.` The returned environment was Modal under gVisor; no `.ncu-rep` artifact or usable counters were produced. The Brev URL variables were confirmed unset at launch.
- 2026-07-19T16:53:25Z, brief-53 trial-0 commit `b2367f60eeac6434d1911b10a9d10fd4dde2f6d6`, benchmark index 5 (`batch=640, n=512`): the `--profile-brev` request exited 1 before submission because both Brev endpoint variables were unset. A plain `--mode profile --benchmark-index 5` request then ran from 16:53:29Z to 17:00:23Z; plain mode ignored the index and made 15 Nsight Compute launches. All 15 connected to Python but exited 9 with `Failed to initialize the profiler: LibraryNotLoaded` under Modal's gVisor kernel, so no artifact or counters were produced. This was the final request begun before the manager announced that the hardened Brev endpoint had been restored.
- 2026-07-19, exact global leader commit `c213c7a0ab26ce17de1b9c3d3fd98e4bb86af350`, benchmark index 5 (`batch=640, n=512`): hardened Brev Nsight Compute capture succeeded at `autocuda/worktrees/leaderboard-c213c7a/profile.5-batch-640-n-512-cond-2-seed-510512/`. `_dx_potrf` measured 2.74 ms with 77.44% memory, 90.38% L1/TEX, and 27.77% compute throughput. Shared loads averaged 3.3-way conflicts (133,260,418 conflicts; 52.14% estimated speedup) and stores 3.1-way (41,779,200; 52.76%); short-scoreboard/MIO stalls consumed 6.7 of 15.4 cycles (43.9%), 73.54% of cycles had no eligible warp, and achieved occupancy was 25.43% versus 43.75% theoretical at 72 registers/thread and 24.58 KiB shared. The preceding `zeros_like` fill measured 141.18 us. Saved artifacts: `ncu-details.txt`, `ncu-details.csv`, and `profile.ncu-rep`.
- 2026-07-19T18:03:46Z, brief-53 padded winner commit `8607a22c8bb1cbf4dc5d6aa845002d79695840a4`, benchmark index 5 (`batch=640, n=512`), Brev job `2d7f9a820153454f81f5c390ff4700e9`: hosted Nsight capture succeeded and was saved under `autocuda/worktrees/optimize-2026-07-19-01-44-06-cholesky-resumed-local-brief-53/profile.5-batch-640-n-512-cond-2-seed-510512/`. Relative to the parent capture, `_dx_potrf` fell from 2.74 to 2.28 ms; shared loads improved from 3.3- to 2.8-way conflicts and stores from 3.1- to 2.0-way, no-eligible cycles fell from 73.54% to 66.89%, registers dropped from 72 to 64, and theoretical occupancy rose from 43.75% to 50%. The remaining profile is still L1/shared limited: 74.53% memory and 85.75% L1/TEX throughput, 100,431,255 shared-load conflicts, 15,250,209 shared-store conflicts, and short-scoreboard stalls at 5.6 of 12.5 cycles (44.8%).
- 2026-07-19T19:15:39Z, brief-50 compact padded-sidecar commit `e7f243ffd064aeeacb955defc40273f7ffbd8b7d`, benchmark index 5 (`batch=640, n=512`), Brev job `22d4b158068a4812bcdab7c1e08d8d17`: hosted Nsight capture succeeded and was saved under `autocuda/worktrees/profile-brief50-e7f243f/profile.5-batch-640-n-512-cond-2-seed-510512/`. `_dx_potrf` measured 1.66 ms with 70.57% memory, 81.28% L1/TEX, 41.49% compute, 15.84% DRAM, and 26.75% L2 hit rate. Shared loads fell to 1.6-way conflicts (16,953,943; 12.02% estimated speedup) and stores to 2.4-way (17,672,711; 33.63%); no-eligible cycles fell to 58.05%. The kernel used 64 registers/thread and 29.20 KiB shared, with 43.75% theoretical and 25.65% achieved occupancy; shared memory is now the occupancy limiter. Nsight still reports 9% excessive global sectors (8.554% estimate), 21% excessive shared wavefronts (17.86%), and a separate 140.86 us `zeros_like` fill. Saved artifacts: `ncu-details.txt`, `ncu-details.csv`, and `profile.ncu-rep`.
- 2026-07-19T16:52:55Z, brief-49 commit `6c037ef6c9ec82e643a7eea817035b4f5fe91d76`, benchmark index 7 (`batch=60, n=1024`): the intended Brev request failed locally in 20 ms because `POPCORN_BREV_PROFILER_URL or BREV_PROFILER_URL is not set`. A plain profile request then ran from 16:53:00Z to 16:59:59Z; plain mode ignored the index and all 15 Nsight launches failed under Modal/gVisor with `Failed to initialize the profiler: LibraryNotLoaded` and exit 9. No usable artifact was produced by either path.
- 2026-07-19T18:05:45Z, brief-49 commit `6c037ef6c9ec82e643a7eea817035b4f5fe91d76`, benchmark index 7 (`batch=60, n=1024`), Brev job `717d2326e03a4ff8b94ef43926d9b776`: hosted Nsight succeeded. The lower copy measured 58.43 us with 75.19% no-eligible cycles and a 272-block partial wave; PyTorch's factor-input copy measured 34.78 us with only 4/32 bytes used per L1 sector and 78% excessive sectors. The following cuSOLVER factor kernels were tiny-grid/launch-latency dominated. Canonical artifacts: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-49-trial-5-6c037ef6c9ec82e643a7eea817035b4f5fe91d76.{ncu-rep,ncu-details.txt,ncu-details.csv}`.
- 2026-07-19T19:34:57Z, brief-49 two-CTA MathDx commit `0b40a8cf222934c5e2825b033b8f370dada8b34c`, benchmark index 7 (`batch=60, n=1024`), Brev job `69e11c6a158846929fd3323604d8791a`: hosted Nsight succeeded. The compact gather/scatter measured 12.10/12.64 us and the two-CTA factor measured 220.19 us, with only 8.26% compute, 90.57% no-eligible cycles, 168 registers/thread, 49.16 KiB dynamic shared, and one resident block/SM (6.25% occupancy); its 120-block grid leaves 28 of 148 SMs idle. Shared loads/stores averaged 1.7-way conflicts (1,370,880/230,400 conflicts, about 6.3% estimates), and CTA barriers accounted for 30.5% of issue intervals. The first TRSM measured 139.14 us at 78.67% memory throughput. Canonical artifacts: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-49-trial-16-0b40a8cf222934c5e2825b033b8f370dada8b34c.{ncu-rep,ncu-details.txt,ncu-details.csv}`.

### Direct cuSOLVER BF16x9 candidate

- Commit: `20835bf52073a315bb5519970faee1ef15eb4eb8`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1597.563015 us (logged as 1597.563 us)
- Submission: 886602, accepted on B200
- Public leaderboard score: 1601.649672 us
- Secret leaderboard score: 1598.300320 us
- Standings after submission: badelsteinlelbach rank 35/52; 409.0% gap to #1 viridale at 314.669 us
- Verdict completed: 2026-07-19T03:15:25Z

### Direct/mixed crossover candidate

- Commit: `55d801d04a2e66cc7dd4bfda655580e02f567e63`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1429.045 us
- Submission: 886669, accepted on B200
- Public leaderboard score: 2368.424365 us (regressed versus submission 886602)
- Secret leaderboard score: 1435.163419 us (consistent with the run metric)
- Standings after submission retained the prior best: badelsteinlelbach rank 35/52 at 1601.650 us; 409.0% gap to #1 viridale at 314.669 us
- Evidence assessment: accepted and strong on the secret set, but not a public leaderboard improvement because of the large public/secret split
- Verdict completed: 2026-07-19T03:44:47Z

### Full-stack small/direct/mixed candidate

- Commit: `9283c406c8c22216b7a1a69f09f2bacac055b46e`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1292.701 us
- Submission: 886720, accepted on B200
- Public leaderboard score: 1290.970582 us
- Secret leaderboard score: 1296.269836 us
- Standings after submission: badelsteinlelbach rank 23/52; 310.3% gap to #1 viridale at 314.669 us
- Evidence assessment: public, secret, and run metrics agree closely; this is the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T04:18:51Z

### Fast-math small-kernel full-stack candidate

- Commit: `340f5481b24e0ba0ef95b05db413e1f0b8e534a3`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1235.523 us
- Submission: 886808, accepted on B200
- Public leaderboard score: 1236.014746 us
- Secret leaderboard score: 1235.773275 us
- Standings after submission: badelsteinlelbach rank 19/52; 292.8% gap to #1 viridale at 314.669 us
- Evidence assessment: public, secret, and run metrics agree closely; this supersedes commit `9283c406` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T05:12:48Z

### Register-resident n=64 full-stack candidate

- Commit: `b81f7a94a1e0f4a4b4cd51bc3e9eb28b0d41918c`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1184.177 us
- Submission: 886828, accepted on B200
- Public leaderboard score: 1185.927611 us
- Secret leaderboard score: 1186.622299 us
- Standings after submission: badelsteinlelbach rank 19/52; 276.9% gap to #1 viridale at 314.669 us
- Evidence assessment: public, secret, and run metrics agree closely; this supersedes commit `340f5481` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T05:20:03Z

### Four-thread register-resident n=128 candidate

- Commit: `1428d299cca30b21303a58328c1cc004316355cd`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1148.535 us
- Submission: 886866, accepted on B200
- Public leaderboard score: 1147.618199 us
- Secret leaderboard score: 1145.658053 us
- Standings after submission: badelsteinlelbach rank 19/52; 264.7% gap to #1 viridale at 314.669 us
- Evidence assessment: public, secret, and run metrics agree closely; this supersedes commit `b81f7a94` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T05:34:09Z

### Simplified coalesced small-kernel staging candidate

- Commit: `635089248d0da1ddb04dc698f7c4eb8d2af4b0f4`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1086.905 us (confirmation 1092.505 us)
- Submission: 887001, accepted on B200
- Public leaderboard score: 1088.072248 us
- Secret leaderboard score: 1088.332025 us
- Standings after submission: badelsteinlelbach rank 19/52; 245.8% gap to #1 viridale at 314.669 us
- Evidence assessment: public, secret, and repeated run metrics agree closely; this supersedes commit `1428d299` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T06:25:02Z

### Cached lower-seeding CuTe full-stack candidate

- Commit: `d977c8e722b3097556d80bcb9c78daa0124959c3`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1068.764 us
- Submission: 887070, accepted on B200
- Public leaderboard score: 1067.690226 us
- Secret leaderboard score: 1063.148212 us
- Standings after submission: badelsteinlelbach rank 18/53; 497.2% gap to new #1 yanchi_72526 at 178.772 us
- Evidence assessment: public, secret, and run metrics agree closely; this supersedes commit `63508924` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T07:10:35Z

### Tiled lower-factor copy CuTe candidate

- Commit: `f041bc224f6b6d4caf9123bcd2c8148863f299a0`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1058.933 us
- Submission: 887098, accepted on B200
- Public leaderboard score: 1058.191589 us
- Secret leaderboard score: 1064.986702 us
- Standings after submission: badelsteinlelbach rank 18/53; 491.9% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public, secret, and run metrics agree within expected run variation; this supersedes commit `d977c8e7` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T07:21:17Z

### Caller-current MathDx n=512 full-stack candidate

- Commit: `644d05ec12309b591a5f6ee2dcfed9b6323fcda8`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1027.399 us
- Submission: 887187, accepted on B200
- Public leaderboard score: 1036.556506 us
- Secret leaderboard score: 1031.257386 us
- Standings after submission: badelsteinlelbach rank 18/53; 479.8% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores are consistent with the run within measured variance; this supersedes commit `f041bc22` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T08:04:34Z

### MathDx n=512 plus grouped n=2048 candidate

- Commit: `9d7af0d6d089c2276ab89ef22cf5d1eac7e8f94b`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 1000.488 us
- Submission: 887204, accepted on B200
- Public leaderboard score: 1006.765152 us
- Secret leaderboard score: 1001.543543 us
- Standings after submission: badelsteinlelbach rank 18/53; 463.2% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public, secret, and run metrics agree closely; this supersedes commit `644d05ec` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T08:16:16Z

### Dedicated cached MathDx n=256 full-stack candidate

- Commit: `2ebbdd3e54fe96ed7f74b9a306b79c18ec973cde`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 970.669 us
- Submission: 887388, accepted on B200
- Public leaderboard score: 979.209212 us
- Secret leaderboard score: 977.495431 us
- Standings after submission: badelsteinlelbach rank 16/54; 447.7% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public, secret, and run metrics agree closely; this supersedes commit `9d7af0d6` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T09:55:40Z

### Strided-panel MathDx n=2048 full-stack candidate

- Commit: `be139ba00c1c12446ffb050091ec0510e67018dd`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 959.482 us
- Submission: 887455, accepted on B200
- Public leaderboard score: 956.076234 us
- Secret leaderboard score: 961.967215 us
- Standings after submission: badelsteinlelbach rank 15/54; 434.8% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public, secret, and run metrics agree closely; this supersedes commit `2ebbdd3e` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T10:30:01Z

### Cooperative n=256 plus L1 MathDx n=2048 candidate

- Commit: `3e8ba7a655ccf4cf99a27f0c7266f549486dcd11`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 942.649 us
- Submission: 887539, accepted on B200
- Public leaderboard score: 943.665906 us
- Secret leaderboard score: 949.555404 us
- Standings after submission: badelsteinlelbach rank 15/54; 427.9% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public, secret, and run metrics agree closely; this supersedes commit `be139ba0` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T11:33:27Z

### Pair-local n=256 plus L1 MathDx n=2048 candidate

- Commit: `0b37b0ce7d12293ade1a43593de8bd3da116ecee`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 933.198 us
- Submission: 887604, accepted on B200
- Public leaderboard score: 940.720939 us
- Secret leaderboard score: 933.276874 us
- Standings after submission: badelsteinlelbach rank 16/54; 426.2% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores bracket the local run within observed hosted variance; this supersedes commit `3e8ba7a` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T12:14:17Z

### Compile-time-specialized n=1024 full-stack candidate

- Commit: `134a89f4746ab79b940af7079e82f0fe15d549dd`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 924.311 us
- Submission: 887691, accepted on B200
- Public leaderboard score: 937.473344 us
- Secret leaderboard score: 934.731824 us
- Standings after submission: badelsteinlelbach rank 16/56; 424.4% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores are consistent with the run within observed hosted variance; this supersedes commit `0b37b0ce` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T13:32:06Z

### Prefetched cp.async n=512 plus direct-FP32 n=2048 candidate

- Commit: `6808886817ccc85b267727f15114b3e516181a16`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 917.179 us
- Submission: 887863, accepted on B200
- Public leaderboard score: 928.726823 us
- Secret leaderboard score: 917.377453 us
- Standings after submission: badelsteinlelbach rank 16/56; 419.5% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: the secret score closely reproduces the local run and the public score improves the prior accepted result by 8.746521 us; this supersedes commit `134a89f4` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T15:30:11Z

### Zero-cached fused-WMMA n=128 plus tile-major n=512 candidate

- Commit: `c213c7a0ab26ce17de1b9c3d3fd98e4bb86af350`
- Command: `bash harness/submit.sh linalg/cholesky_py` from a detached worktree at that exact commit
- Local Modal benchmark: 909.588 us
- Submission: 888011, accepted on B200
- Public leaderboard score: 919.039713 us
- Secret leaderboard score: 917.531365 us
- Standings after submission: badelsteinlelbach rank 16/56; 414.1% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores agree within hosted variance and the public score improves the prior accepted result by 9.687110 us; this supersedes commit `68088868` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T16:48:36Z

### Hosted Nsight Compute profiles for `c213c7a`

- 2026-07-19T16:56:01Z, benchmark index 2 (`batch=256, n=128`), profile job `c6d1b567edc8461691e3910eaa571303`: succeeded; downloaded `ncu-details.txt`, `ncu-details.csv`, and `profile.ncu-rep` under `autocuda/worktrees/leaderboard-c213c7a/profile.2-batch-256-n-128-cond-2-seed-41128/`.
- 2026-07-19T16:58:11Z, benchmark index 5 (`batch=640, n=512`), profile job `a0e78c94b2194431a36aeeffa08dd898`: succeeded; downloaded artifacts under `autocuda/worktrees/leaderboard-c213c7a/profile.5-batch-640-n-512-cond-2-seed-510512/`.
- 2026-07-19T17:01:25Z, benchmark index 9 (`batch=8, n=2048`), profile job `1ae2f38f9a4042ab8954fd5412b49f3b`: succeeded; downloaded artifacts under `autocuda/worktrees/leaderboard-c213c7a/profile.9-batch-8-n-2048-cond-2-seed-512048/`.
- Profiler evidence: n=128 and n=512 are dominated by shared-memory bank conflicts and scheduler stalls; n=2048 exposes only eight diagonal-panel blocks, uses 255 registers per thread, and reaches 6.27% occupancy. These findings now steer the active briefs.
- Earlier plain `--mode profile` submissions 888017 and 888018 produced no usable artifacts: the Modal/gVisor path failed Nsight initialization with `LibraryNotLoaded`; hosted `--profile-brev` is the working path.

### Aligned cp.async n=512 candidate

- Commit: `66bc1676cbd19608d4b8701c7d35c569a3d33654`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a detached worktree at that exact commit
- Local Modal benchmark: 898.198 us
- Submission: 888123, accepted on B200
- Public leaderboard score: 906.844894 us
- Secret leaderboard score: 899.659259 us
- Standings after submission: badelsteinlelbach rank 16/57; 407.3% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: the secret score closely reproduces the local run and the public score improves the prior accepted result by 12.194819 us; this supersedes commit `c213c7a0` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T18:03:57Z

### Hosted Nsight Compute profiles for brief 54

- 2026-07-19T18:02:17Z, commit `b4cdcc39496caafb770858507f17c05cedba5dba`, benchmark index 5 (`batch=640, n=512`), profile job `9eb26eca618142b69fb0f1761445e091`: succeeded. Artifact: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-54-trial-0-b4cdcc39496caafb770858507f17c05cedba5dba.ncu-rep`.
- 2026-07-19T18:14:18Z, commit `68b86372d3e2dfc0c2f3f6098a3f7fd05de0ebce`, benchmark index 5 (`batch=640, n=512`), profile job `e9426bd29147400a867acfbc91235535`: succeeded. Artifact: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-54-trial-1-68b86372d3e2dfc0c2f3f6098a3f7fd05de0ebce.ncu-rep`.
- FP32 LD36 on top of FP16 LD40 reduced the captured `_dx_potrf` duration from 2.16 to 1.73 ms, shared-load conflicts from 2.1-way to 1.6-way, shared-store conflicts from 3.7-way to 2.2-way, no-eligible cycles from 69.93% to 63.17%, and short-scoreboard stalls from 40.5% to 35.0%; registers remained 72/thread and achieved occupancy remained 25.51%.
- Global quota reconciliation reported at least six requests across workers between 17:56 and 18:14. Brief 54 will issue no further hosted request until the manager explicitly reconciles and grants one.

### Hosted Nsight Compute profiles for brief 43

- 2026-07-19T17:56:32Z, commit `2029fe557df26d1fdd566b0268699fa351502bfe`, benchmark index 9 (`batch=8, n=2048`), profile job `9928cea6b37149779b6a4608c9b8d335`: succeeded. Artifacts are under `autocuda/worktrees/profile-2026-07-19-01-44-06-cholesky-resumed-local-brief-43-2029fe5/profile.9-batch-8-n-2048-cond-2-seed-512048/`; Nsight's ten-launch window ended at `grouped_upper_copy`, so it did not capture the panel kernel.
- 2026-07-19T18:08:40Z, commit `44fe6af9f8e56230215c6e522b780e999140da75`, benchmark index 9, profile job `caa7aa26f98741fa8f81f71820b6c812`: succeeded. Artifacts are under `autocuda/worktrees/optimize-2026-07-19-01-44-06-cholesky-resumed-local-brief-43/profile.9-batch-8-n-2048-cond-2-seed-512048/`.
- Profiler evidence: the redesigned `_g2048_potrf_multi_cta` measured 163.07 us versus 250.98 us for the original one-CTA panel kernel, with grid size 128 versus 8 and 80 registers/thread versus 255. It still measured 6.25% occupancy, 96.62% no-eligible cycles, and 61.89% of average warp delay at CTA barriers; shared loads/stores averaged 2.1-/1.3-way conflicts. This evidence motivated the validated 256-thread follow-up `f9e0d256966f79179f34140ca76f1978d3ad8207` at 915.063 us aggregate / 1936.066 us target.
- No further hosted request is authorized until the manager explicitly reconciles the run-wide rolling-hour ledger and grants one.

### Specialized n=512 GEMM block-dimension candidate

- Commit: `fd7190dd6aaa0393a6bf615728a548495ac5379d`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a detached worktree at that exact commit
- Submission: 888222, accepted on B200
- Public leaderboard score: 885.248222 us
- Secret leaderboard score: 883.767075 us
- Standings after submission: badelsteinlelbach rank 15/57; 395.2% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores agree closely, and the public score improves the prior accepted result by 21.596672 us; this supersedes commit `66bc1676` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T19:02:53Z

### Compact padded n=512 factor-sidecar candidate

- Commit: `e7f243ffd064aeeacb955defc40273f7ffbd8b7d`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a detached worktree at that exact commit
- Submission: 888236, accepted on B200
- Public leaderboard score: 890.158636 us
- Secret leaderboard score: 884.066582 us
- Standings after submission: badelsteinlelbach remains rank 15/57 with the retained 885.248222 us best; 395.2% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: the candidate's public score is 4.910414 us slower than commit `fd7190dd`, so it does not supersede the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T19:14:15Z

### Hosted Nsight Compute profile for brief 55

- 2026-07-19T19:22:21Z, exact commit `9f6707156762d1e82ea42898771395860c302349`, benchmark index 9 (`batch=8, n=2048`), profile job `f3d82c7afb244ee6aa2bbd5acf6219d2`: succeeded through the required `POPCORN_BREV_PROFILER_URL` endpoint.
- Artifact: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-55-trial-0-9f6707156762d1e82ea42898771395860c302349.ncu-rep`; full text/CSV details remain under `autocuda/worktrees/profile-2026-07-19-01-44-06-cholesky-resumed-local-brief-55-9f67071/profile.9-batch-8-n-2048-cond-2-seed-512048/`.
- Profiler evidence: the 256-thread `_g2048_potrf_multi_cta` measured 165.89 us, 63 registers/thread, 12.54% achieved occupancy, 94.85% no-eligible cycles, 71.72% CTA-barrier delay, and 2.4-/1.3-way shared load/store conflicts. Grid size remained 128 blocks (0.86 waves across 148 SMs).
- 2026-07-19T20:18:13Z, exact right-looking commit `42aa42d6526c14fe9db50b1d2eaee60693afb183`, benchmark index 9, profile job `60a98fa1b62b4a7e8d5f712e01801ae5`: succeeded. Artifact: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-55-trial-9-42aa42d6526c14fe9db50b1d2eaee60693afb183.ncu-rep`.
- Right-looking evidence: `_g2048_potrf_multi_cta` fell to 156.77 us, 56 registers/thread, 92.12% no-eligible cycles, and 62.74% CTA-barrier delay at 12.46% occupancy. Shared conflicts measured 2.3-/1.3-way; the 128-block grid remains residency-safe, while the separately measured 152-block software-barrier grid deadlocked because only 148 blocks can reside concurrently.
- 2026-07-19T20:50:42Z request (hosted start 20:52:22Z), exact LD24/static right-looking commit `ee80504f3d090311cb6bcd96320d02729a44281d`, benchmark index 9, profile job `4ad1e28553d64451acc2e22fb08df8f8`: succeeded. Artifact: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-55-trial-15-ee80504f3d090311cb6bcd96320d02729a44281d.ncu-rep`.
- LD24/static evidence: panel time was noise-flat at 157.18 us with 56 registers/thread and 12.48% occupancy. Shared loads improved from 2.3- to 2.2-way conflicts, while no-eligible cycles/barrier delay were 92.30%/63.01%; the padding did not materially change the profiled kernel.

### Hosted Nsight Compute profile for brief 56

- Request start `2026-07-19T20:18:19Z`; hosted job start `2026-07-19T20:20:38.827430Z`; end `2026-07-19T20:22:45.281172Z`. Exact commit `01e2d6e0053d9f6f54869ac3b02ccd3bb850cd2a`, benchmark index 5 (`batch=640, n=512`), profile job `1c2f93bfa61f47319e9de602317dcbb5`: succeeded through the required `POPCORN_BREV_PROFILER_URL` endpoint and `autocuda run slice`.
- Canonical artifacts: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-56-trial-10-01e2d6e0053d9f6f54869ac3b02ccd3bb850cd2a.ncu-rep`, `.ncu-details.txt`, and `.ncu-details.csv`; downloaded source artifacts remain under `autocuda/worktrees/optimize-2026-07-19-01-44-06-cholesky-resumed-local-brief-56/profile.5-batch-640-n-512-cond-2-seed-510512/`.
- Profiler evidence: `_dx_potrf` measured 1.63 ms, 56 registers/thread, 43.75% theoretical and 25.96% achieved occupancy, 58.89% no-eligible cycles, and 32.5% short-scoreboard delay. Shared accesses remained the main issue at 1.6-way loads and 2.4-way stores (32,931,840 excessive wavefronts, 21% of total), while global accesses had 3,686,400 excessive sectors (9% of total). Relative to the prior LD40/LD36 capture at 1.73 ms / 72 registers / 63.17% no-eligible / 35.0% short-scoreboard, compact strength-reduced addressing reduced duration and register pressure, but shared-store conflicts remain the next edit target.

### Padded n=2048 half-panel candidate

- Commit: `1a06e92d3da85ff798884aa69b9a623d23a73691`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a detached worktree at that exact commit
- Submission: 888372, accepted on B200
- Public leaderboard score: 873.786348 us
- Secret leaderboard score: 875.246703 us
- Standings after submission: badelsteinlelbach rank 15/57; 388.8% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores agree closely, and the public score improves the prior accepted result by 11.461874 us; this supersedes commit `fd7190dd` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T20:44:39Z

### Hosted Nsight Compute profile for brief 49 trial 25

- Request start `2026-07-19T21:53:27Z`; hosted job start `2026-07-19T21:57:44.525311Z`; end `2026-07-19T22:01:54.828694Z`. Exact commit `2e67c8690572532b410d5ffc6ee773fdd4ad8a5b`, benchmark index 7 (`batch=60, n=1024`), profile job `0c928b55b6a34f7eaa5c450803d750a2`: succeeded through the required `POPCORN_BREV_PROFILER_URL` endpoint and `autocuda run slice`.
- Canonical artifacts: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-49-trial-25-2e67c8690572532b410d5ffc6ee773fdd4ad8a5b.ncu-rep`, `.ncu-details.txt`, and `.ncu-details.csv`.
- Profiler evidence: the padded 256-thread `_batch1024_potrf_two_cta_padded` factor measured 154.08 us, down from the prior 128-thread compact kernel's 220.19 us. Registers fell from 168 to 64/thread, achieved occupancy doubled from 6.25% to 12.49%, compute throughput rose from 8.26% to 14.12%, and no-eligible cycles fell from 90.57% to 83.74%. Shared-load conflicts no longer triggered a rule; shared stores improved from 1.7-way/230,400 conflicts to 1.3-way/76,800 conflicts, with excessive shared wavefronts falling to 6% of total. CTA-barrier delay remains the main measured factor-kernel opening at 38.29% of issue intervals; the 120-block grid still leaves 28 of 148 SMs idle. Gather/scatter measured 11.78/12.42 us and the first TRSM 137.76 us.

### Batch-1024 half-LD80 candidate

- Commit: `0345b5df4723e7cb4edd13be544e290cd38cd1e3`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a detached worktree at that exact commit
- Submission: 888556, accepted on B200
- Public leaderboard score: 851.309672 us
- Secret leaderboard score: 857.676459 us
- Standings after submission: badelsteinlelbach rank 14/57; 376.2% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores agree within hosted variance, and the public score improves the prior accepted result by 22.476676 us; this supersedes commit `1a06e92d` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T22:06:14Z

### Compact n=512 bulk-path combined candidate

- Commit: `913a3f334c4b2580a93bdb09c4df4e789caa9a07`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a detached worktree at that exact commit
- Submission: 888589, accepted on B200
- Public leaderboard score: 845.686671 us
- Secret leaderboard score: 850.740887 us
- Standings after submission: badelsteinlelbach rank 14/57; 373.1% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores agree within hosted variance, and the public score improves the prior accepted result by 5.623001 us; this supersedes commit `0345b5df` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T22:24:19Z

### Aliased n=2048 update-operands candidate

- Commit: `efc046578218f20fe6da9001f6e1dfc499fafddc`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a clean detached worktree at that exact commit
- Submission: 888634, accepted on B200
- Public leaderboard score: 854.459046 us
- Secret leaderboard score: 847.037790 us
- Standings after submission: badelsteinlelbach remains rank 14/57 with the retained 845.686671 us best; 373.1% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: the candidate's public score is 8.772375 us slower than commit `913a3f33`, so it does not supersede the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T22:54:21Z

### Hosted Nsight Compute profile for brief 58 trial 5

- Request start `2026-07-19T22:55:09Z`; hosted job start `2026-07-19T22:55:11.374519Z`; end `2026-07-19T22:59:02.534310Z`. Exact commit `4b3b45fac10b801d3f6e93d99e6982e8c7ba0c07`, benchmark index 11 (`batch=2, n=4096`), profile job `8129d669066a4b9b981dcc1e027aa556`: succeeded through the required `POPCORN_BREV_PROFILER_URL` endpoint and `autocuda run slice`.
- Canonical artifacts: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-58-trial-5-4b3b45fac10b801d3f6e93d99e6982e8c7ba0c07.ncu-rep`, `.ncu-details.txt`, and `.ncu-details.csv`; downloaded source artifacts remain under `autocuda/worktrees/profile-2026-07-19-01-44-06-cholesky-resumed-local-brief-58-4b3b45f/profile.11-batch-2-n-4096-cond-2-seed-514096/`.
- Profiler evidence: the first four 256-wide `_b4096_factor_solve_stage` launches measured 245.28, 246.34, 243.94, and 245.18 us. The first used 168 registers/thread, achieved 6.25% occupancy, spent 91.98% of scheduler cycles with no eligible warp and 41.01% of average warp delay at CTA barriers, and incurred 1.8-/1.7-way shared load/store conflicts (39% excessive shared wavefronts). Its 128-block grid leaves 20 of 148 SMs idle. The intervening triangular CuTe rank-k launches measured only 46.30, 39.62, and 38.50 us; the first used 78 registers/thread at 9.64% achieved occupancy, with 78.67% no-eligible cycles, 50.3% short-scoreboard delay, and 1.8-/1.5-way shared load/store conflicts. The resident factor/solve stage, especially its barrier-heavy high-register path, is therefore the dominant measured opening.
- No additional hosted request was launched by brief 58; this request occupies its run-wide rolling-hour slot until `2026-07-19T23:55:09Z`.

### Hosted Nsight Compute profile for brief 60 trial 3

- Request start `2026-07-19T22:55:14Z`; hosted job start `2026-07-19T22:59:02.534578Z`; end `2026-07-19T23:01:30.220596Z`. Exact commit `2682f769cccdc6391036a9f964e5bfe820f64362`, benchmark index 9 (`batch=8, n=2048`), profile job `40b207152f3948ca9a19d34c2397602d`: succeeded through the required `POPCORN_BREV_PROFILER_URL` endpoint and `autocuda run slice`.
- Canonical artifacts: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-60-trial-3-2682f769cccdc6391036a9f964e5bfe820f64362.ncu-rep`, `.ncu-details.txt`, and `.ncu-details.csv`; downloaded source artifacts remain under `autocuda/worktrees/optimize-2026-07-19-01-44-06-cholesky-resumed-local-brief-60/profile.9-batch-8-n-2048-cond-2-seed-512048/`.
- Profiler evidence: the aliased, coordinator-rotated LD20 `_g2048_potrf_multi_cta` measured 159.30 us, used 56 registers/thread and 6.15 KiB dynamic shared, and achieved 12.54% occupancy. Its 128-block grid remains a 0.86-wave launch that leaves 20 of 148 SMs idle; the 8.19 KiB shared-memory configuration still limits each SM to one resident block despite the reduced allocation. Scheduler cycles had no eligible warp 92.01% of the time, CTA barriers consumed 62.72% of average warp delay, and shared loads/stores averaged 2.2-/1.3-way conflicts (632,320/64,512 conflicts). Relative to the preceding static LD24 capture at 157.18 us, 12.48% occupancy, 92.30% no-eligible cycles, 63.01% barrier delay, and 2.2-/1.3-way conflicts, operand aliasing retained essentially the same kernel profile while reducing allocation and validating the 16-CTA fallback.
- Brief 60 launched no additional hosted request.
- Final selected branch commit `408b762d3d78f7ecb642212a28e5951883fbb57c` is byte-identical to profiled commit `2682f769`; the saved capture was reused without another hosted request and canonicalized as `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-60-trial-9-408b762d3d78f7ecb642212a28e5951883fbb57c.{ncu-rep,ncu-details.txt,ncu-details.csv}`.

### Combined profiled stack with blocked n=1024 solve

- Commit: `ab51eb653751a95679b417d7caf52eb40f5d6ecf`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a clean detached worktree at that exact commit
- Submission: 888671, accepted on B200
- Public leaderboard score: 833.266001 us
- Secret leaderboard score: 825.389479 us
- Standings after submission: badelsteinlelbach rank 14/57; 366.1% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores agree within hosted variance, and the public score improves the prior accepted best by 12.420670 us; this supersedes commit `913a3f33` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-19T23:21:14Z

### Doubled n=4096 factor-solve threads

- Commit: `76d8f735c47bf32e42edc0c0b63e6674bbb4ed94`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a clean detached worktree at that exact commit
- Submission: 888888, accepted on B200
- Public leaderboard score: 799.783123 us
- Secret leaderboard score: 811.071481 us
- Standings after submission: badelsteinlelbach rank 14/57; 347.4% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: the public score improves the prior accepted best by 33.482878 us; this supersedes commit `ab51eb65` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T01:12:02Z

### Split-role 256-thread n=4096 candidate

- Commit: `26e6471dbfc8fb929dd8c058cdc4d585bbc1de9e`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a clean detached worktree at that exact commit
- Submission: 888978, accepted on B200
- Public leaderboard score: 808.661270 us
- Secret leaderboard score: 792.910969 us
- Standings after submission: badelsteinlelbach remains rank 14/57 with the retained 799.783123 us best; 347.4% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: the candidate's public score is 8.878147 us slower than commit `76d8f735`, so it does not supersede the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T02:09:53Z

### Hosted Nsight Compute profile for brief 68 trial 2

- Request lock acquired `2026-07-20T02:29:40.611575Z`; hosted job start `2026-07-20T02:29:42.852670Z`; end `2026-07-20T02:35:26.377065Z`. Exact commit `de2dbc67d406baed80b4508931269c2de8629653`, benchmark index 7 (`batch=60, n=1024`), profile job `a70c33a7a07a4fdab660338fff498c44`: succeeded through the required `POPCORN_BREV_PROFILER_URL` endpoint and `autocuda run slice`.
- Canonical artifacts: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-68-trial-2-de2dbc67d406baed80b4508931269c2de8629653.{ncu-rep,ncu-details.txt,ncu-details.csv,ncu.zip}`; downloaded source artifacts remain under `autocuda/worktrees/profile-brief-68-de2dbc67/profile.7-batch-60-n-1024-cond-2-seed-511024/`.
- Profiler evidence: the cap-72 aliased `_batch1024_panel_trsm` measured 210.75 us at grid 720/block 256, 72 registers/thread, 35.84 KiB dynamic shared, three-block register limit, 37.50% theoretical and 31.43% achieved occupancy, and 47.56% no-eligible cycles. Relative to the uncapped aliased capture's 218.94 us, 96 registers, 22.70% achieved occupancy, and 51.90% no-eligible cycles, the cap raised residency and reduced time, but generated 138.24 KiB of local spill requests: local accesses were 11.11% of L1 sectors, all local loads/stores were spills, and the report estimates a 22.57% local-memory opportunity. The launch is 1.62 waves with a 276-block partial wave. This makes a slightly higher cap within the same three-block residency class and source-level live-range reduction the measured next directions.
- Brief 68 launched exactly this one authorized hosted request and will launch no additional request.

### Async-loaded large-shape factor copies

- Commit: `1a1ae967047cfe6e0bebef7a0735931e6bfac14e`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py` from a clean detached worktree at that exact commit; no hosted profile request was used
- Submission: 889045, accepted on B200
- Public leaderboard score: 790.379608 us
- Secret leaderboard score: 798.981343 us
- Standings after submission: badelsteinlelbach rank 14/57; 342.1% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores agree within hosted variance, and the public score improves the prior accepted best by 9.403515 us; this supersedes commit `76d8f735` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T02:54:30Z

### Hosted Nsight Compute profile for brief 69 selected final

- Request lock acquired `2026-07-20T03:31:44.112520Z`; hosted job start `2026-07-20T03:32:32.371007Z`; end `2026-07-20T03:35:04.553632Z`. Exact source-identical selected commit `163fbd3d82a27eddbd35f8e5746b19a6d6fcd7f9`, benchmark index 9 (`batch=8, n=2048`), profile job `28c4112d57fe425bb651142d4b0a667c`: succeeded through the required `POPCORN_BREV_PROFILER_URL` endpoint and `autocuda run slice`.
- Canonical artifacts: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-69-final-163fbd3d82a27eddbd35f8e5746b19a6d6fcd7f9.{ncu-rep,ncu-details.txt,ncu-details.csv,profile.zip}`; downloaded source artifacts remain under `autocuda/worktrees/optimize-2026-07-19-01-44-06-cholesky-resumed-local-brief-69/profile.9-batch-8-n-2048-cond-2-seed-512048/`.
- Profiler evidence: the first split `_g2048_factor_stage` measured 11.10 us at grid 8/block 256, 39 registers/thread, 6.15 KiB dynamic shared (8.19 KiB configured), 12.70% achieved occupancy, 72.14% no-eligible cycles, no local/shared spills, and only 0.05 waves across 148 SMs. Shared accesses incurred 192 excessive wavefronts, 2% of 9,224. The preceding grouped upper copy measured 34.72 us. The hosted ten-launch window was consumed by four cold pooled-output fills, four pointer-table kernels, upper copy, and this first factor node, so it did not reach the new panel/update nodes; the saved exact report still establishes the factor launch's tiny-grid latency/occupancy opening for a follow-up graph design.
- Brief 69 launched exactly this one manager-authorized hosted request and will launch no additional request.

### Direct n=256 stage-zero read candidate

- Commit: `6080f5d26352a496b63163ba6455f17e43dd8660`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched commit blob `21e11ceddccfd90fba924b583029c169eeeab8f8`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889137, accepted on B200
- Public leaderboard score: 791.958741 us
- Secret leaderboard score: 783.923867 us
- Standings after submission: badelsteinlelbach remains rank 14/58 with retained submission 889045 at 790.379608 us; 342.1% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: the candidate's public score is 1.579133 us slower than commit `1a1ae967`, so it does not supersede the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T03:53:33Z

### Specialized final n=256 update-factor candidate

- Commit: `d6e0a3cc0bc30242d073b37b662efaef9fd58e7e`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched the committed file byte-for-byte (SHA-256 `e4f4b2b53dc0eed3920a407b3d4f2a7a470832ffacfc54425b3d99c27466a52e`)
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --leaderboard cholesky --gpu B200 --mode leaderboard problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889235, accepted on B200
- Public leaderboard score: 776.461869 us
- Secret leaderboard score: 783.907802 us
- Standings after submission: badelsteinlelbach rank 14/59; 334.3% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores agree within hosted variance, and the public score improves retained submission 889045 by 13.917739 us; this supersedes commit `1a1ae967` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T04:47:36Z

### Async FP32 graph-stage tile-load candidate

- Commit: `2f94e5ff83fd10248cef377146aaa9092110b4b7`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched commit blob `29e37c6cab2bd5197e88217b93a1cf3b4488f9d6`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889335, accepted on B200
- Public leaderboard score: 765.667079 us
- Secret leaderboard score: 770.083914 us
- Standings after submission: badelsteinlelbach rank 14/59; 328.3% gap to #1 yanchi_72526 at 178.772 us
- Evidence assessment: public and secret scores agree within hosted variance, and the public score improves retained submission 889235 by 10.794790 us; this supersedes commit `d6e0a3cc` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T05:20:04Z

### Async-loaded graph update accumulators candidate

- Commit: `d6762e56144be42eb0f60cdf5bcc7dc7b016a713`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched commit blob `a1e488c88e12b24893f7b50999cca57c6420f54b`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --mode leaderboard --leaderboard cholesky problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889432, accepted on B200
- Public leaderboard score: 768.980517 us
- Secret leaderboard score: 767.962013 us
- Standings after submission: candidate was not retained because it did not improve the account's public best; badelsteinlelbach remains rank 14/59 with retained submission 889335 at 765.667079 us; 328.3% gap to #1 yanchi_72526 at 178.771770 us
- Evidence assessment: the candidate's public score is 3.313438 us slower than retained commit `2f94e5ff`, so it does not supersede the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T05:44:25Z

### Hosted Nsight Compute profile for brief 76 trial 3

- Request start `2026-07-20T06:07:45.317558Z`; hosted job start `2026-07-20T06:07:48.476256Z`; end `2026-07-20T06:11:27.673407Z`. Exact commit `faf58c840af228079b9989009f903d59460bc444`, benchmark index 9 (`batch=8, n=2048`), profile job `2f355f3d5b3142319f33d3f0df98ef14`: succeeded through the required `POPCORN_BREV_PROFILER_URL` endpoint and `autocuda run slice`.
- Canonical artifacts: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-76-trial-3-faf58c840af228079b9989009f903d59460bc444.{ncu-rep,ncu-details.txt,ncu-details.csv,ncu.zip}`; downloaded source artifacts remain under `autocuda/worktrees/optimize-2026-07-19-01-44-06-cholesky-resumed-local-brief-76/profile.9-batch-8-n-2048-cond-2-seed-512048/`.
- Profiler evidence: the standalone 35.78-us grouped upper-copy launch is absent. After four cold pooled-output fills, the direct-source first factor measured 11.49 us versus the parent's 11.17 us. The first float4-fused panel/update waves measured 36.70/69.22 us versus the parent's 7.01/7.14 us; their 14-/56-block grids reached only 51.33/96.71 GB/s, 11.07/7.07% achieved occupancy, and 91.15/88.21% no-eligible cycles, with 42/32 registers and zero spills. The next unfused factor/panel/update returned to 11.42/7.36/7.17 us. This directly motivates widening only the first panel/update grids with copy-only CTAs while preserving every later launch.
- Brief 76 used exactly this one manager-granted hosted request and will launch no additional request without a new grant.

### LD72 shared graph-update operands candidate

- Commit: `2503506c32b1bfd4a8affc5ddadf3cf0fd6cb974`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched the committed file byte-for-byte (SHA-256 `71e615079c09bd85e556d07dde39758abafe73c13f0b16d1ddafa3f32e157bcd`)
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --leaderboard cholesky --mode leaderboard problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889648, accepted on B200
- Public leaderboard score: 775.767869 us
- Secret leaderboard score: 769.848826 us
- Standings after submission: candidate was not retained because it did not improve the account's public best; badelsteinlelbach remains rank 14/60 with retained submission 889335 at 765.667079 us; 328.3% gap to #1 yanchi_72526 at 178.771770 us
- Evidence assessment: the candidate's public score is 10.100790 us slower than retained commit `2f94e5ff`, so it does not supersede the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T07:09:52Z

### Hosted Nsight Compute profile for brief 76 selected final

- Request lock acquired `2026-07-20T07:07:56.465621Z`; hosted job start `2026-07-20T07:08:45.111572Z`; end `2026-07-20T07:12:11.121528Z`. Exact selected commit `0224ad48705863f8d80778521570605a1f86796c`, benchmark index 9 (`batch=8, n=2048`), profile job `9d5d6db956824e87be9fdbe4a5554e5f`: succeeded through the required `POPCORN_BREV_PROFILER_URL` endpoint and `autocuda run slice`.
- Canonical artifacts: `autocuda/profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-76-trial-11-0224ad48705863f8d80778521570605a1f86796c.{ncu-rep,ncu-details.txt,ncu-details.csv,ncu.zip}`; downloaded source artifacts remain under `autocuda/worktrees/optimize-2026-07-19-01-44-06-cholesky-resumed-local-brief-76/profile.9-batch-8-n-2048-cond-2-seed-512048/` as the timestamp-suffixed second capture.
- Profiler evidence: the parent's 35.78-us grouped upper-copy launch is absent. After four cold pooled-output fills, the direct-source first factor measured 11.46 us. The final 74-CTA first panel wave materialized the in-place STRSM panel and solved the internal tiles in 9.92 us at 190.04 GB/s, 44 registers, 11.32% achieved occupancy, 88.88% no-eligible cycles, and zero spills. The first 56-CTA internal update returned to 7.42 us at 32 registers and 12.51% achieved occupancy because the first outer rank-k now reads original-input C directly instead of copying the 1792x1792 trailing upper triangle. The next factor/panel/update measured 11.26/7.39/7.30 us. Relative to the exact parent sequence of 35.78 + 11.17 + 7.01 + 7.14 us, the corresponding selected sequence is 11.46 + 9.92 + 7.42 us, removing about 32.3 us before overlap.
- Brief 76 launched exactly the two separately manager-granted requests recorded in this file and will launch no additional request.

### Fused final-panel store and half-conversion candidate

- Commit: `99c3ededac07c9a3190d9c51464387998aa283b1`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched commit blob `552c49344aba9a2e9839fcfca7477dc7c062dd4d`
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --mode leaderboard --leaderboard cholesky problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889663, accepted on B200
- Public leaderboard score: 772.182025 us
- Secret leaderboard score: 764.347421 us
- Standings after submission: candidate was not retained because it did not improve the account's public best; badelsteinlelbach remains rank 14/60 with retained submission 889335 at 765.667079 us; 328.3% gap to #1 yanchi_72526 at 178.771770 us
- Evidence assessment: the candidate's public score is 6.514946 us slower than retained commit `2f94e5ff`, so it does not supersede the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T07:18:54Z

### Prefetched final accumulator during panel drain candidate

- Commit: `1d28e315916043ab74d7c5e0a9909613845ee92d`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched commit blob `2402c79050e0ce36a30b4738bb891562ec945f67` byte-for-byte (SHA-256 `3fd2727283c6bbb53810dd6ed11d742f7dd9a953cbaaa73029e689d3ea019568`)
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --mode leaderboard --leaderboard cholesky problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889676, accepted on B200
- Public leaderboard score: 764.712545 us
- Secret leaderboard score: 769.682187 us
- Standings after submission: badelsteinlelbach rank 14/60; 327.8% gap to #1 yanchi_72526 at 178.771770 us
- Evidence assessment: public and secret scores agree within hosted variance, and the public score improves retained submission 889335 by 0.954534 us; this supersedes commit `2f94e5ff` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T07:22:48Z

### Folded initial factor into graph solves candidate

- Commit: `ca5edb0ed732d4a83bd30280b1c8c6993bef7818`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched commit blob `c01375d3c35592a145c3fdef68845fa7a076d272` byte-for-byte (SHA-256 `3c88df9f7fc95a3e43d0c9a20b871a971a3605add00b44e9b84c73311ba74e39`)
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --mode leaderboard --leaderboard cholesky problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889691, accepted on B200
- Public leaderboard score: 766.051934 us
- Secret leaderboard score: 776.832967 us
- Standings after submission: candidate was not retained because it did not improve the account's public best; badelsteinlelbach remains rank 14/60 with retained submission 889676 at 764.712545 us; 327.8% gap to #1 yanchi_72526 at 178.771770 us
- Evidence assessment: the candidate's public score is 1.339389 us slower than retained commit `1d28e315`, so it does not supersede the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T07:34:45Z

### Combined graph n=256 and pipelined n=1024 candidate

- Commit: `f9085ac5c3a277f87fc5baa00c9decdd31308887`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched commit blob `70e079950804ecad83164b3310847cca5b407e2c` byte-for-byte (SHA-256 `4766df7893928b6d478572215d4f12fdc5b868e0c1fed55d7de3272fbcf24278`)
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --mode leaderboard --leaderboard cholesky problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889762, accepted on B200
- Public leaderboard score: 769.410798 us
- Secret leaderboard score: 779.923731 us
- Standings after submission: candidate was not retained because it did not improve the account's public best; badelsteinlelbach remains rank 14/60 with retained submission 889676 at 764.712545 us; 327.8% gap to #1 yanchi_72526 at 178.771770 us
- Evidence assessment: the candidate's public score is 4.698253 us slower than retained commit `1d28e315`, so it does not supersede the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T08:25:24Z

### Increased n=2048 cleanup-wave exposure candidate

- Commit: `000651cf03679fef8f7f70f00859f072898247a9`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched commit blob `2bd6a2c5fc43cd73797e6c90e5e0e938fccdb04c` byte-for-byte (SHA-256 `23a7583ad6cbc2bc7ba51c16004efdbadd4516f7d862114b966b08cdbd94bfdc`)
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --mode leaderboard --leaderboard cholesky problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889820, accepted on B200
- Public leaderboard score: 757.521944 us
- Secret leaderboard score: 751.175897 us
- Standings after submission: badelsteinlelbach rank 14/60; 323.7% gap to #1 yanchi_72526 at 178.771770 us
- Evidence assessment: public and secret scores differ by 6.346047 us, and the public score improves retained submission 889676 by 7.190601 us (0.949%); this supersedes commit `1d28e315` as the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T08:51:48Z

### Specialized batch-1024 graph update depths candidate

- Commit: `b68c3ddd53c28e45f0622bc1986f9687bd265d89`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched commit blob `6cd6ec799fdd71228c3f6742c3f6228cc6d84b3c` byte-for-byte (SHA-256 `94769779e0d94176f23b4bf6d2bd3e4c5453c40b83db6ca658c98730262cba65`)
- Command: `autocuda run slice --data-dir /home/ubuntu/gpumode/autocuda -- popcorn-cli submit --no-tui --gpu B200 --mode leaderboard --leaderboard cholesky problems/linalg/cholesky_py/submission.py`; no hosted profile request was used
- Submission: 889858, accepted on B200
- Public leaderboard score: 760.841231 us
- Secret leaderboard score: 766.045450 us
- Standings after submission: candidate was not retained because it did not improve the account's public best; badelsteinlelbach remains rank 14/60 with retained submission 889820 at 757.521944 us; 323.7% gap to #1 yanchi_72526 at 178.771770 us
- Evidence assessment: the candidate's public score is 3.319287 us slower than retained commit `000651cf`, so it does not supersede the current accepted leaderboard candidate
- Verdict completed: 2026-07-20T09:12:50Z

### Split n=2048 outer-copy candidate

- Commit: `cf2cf5a716932f40445a372e07f02fec0de495fc`
- Source verification: clean detached worktree at that exact commit; `problems/linalg/cholesky_py/submission.py` matched commit blob `ee29786e79a584cde56425dc8b66347ce73aafd5` byte-for-byte (SHA-256 `3a8cb869071dfaeaa6af476f1d2d35d8502855b4eea06e60c252998898a908b0`, 255051 bytes)
- Local Modal benchmark: 759.287 us (brief 81 trial 22; wrapped build, all 17 tests, fresh-input guard, and all 15 benchmark cells passed)
- Command: `GPUMODE_GPU=B200 bash harness/submit.sh linalg/cholesky_py` from the clean detached worktree; accepted on the first attempt, with no retry or hosted profile request
- Submission: 890077, accepted on B200 (`cli_exit=0`, helper verdict `ACCEPTED`)
- Public leaderboard score: 752.637332 us
- Secret leaderboard score: 750.896201 us
- Standings after submission: badelsteinlelbach rank 14/61 at retained submission 890077; 321.0% gap to #1 yanchi_72526 at 178.771770 us
- Evidence assessment: public and secret scores differ by 1.741131 us, and the public score improves retained submission 889820 by 4.884612 us (0.645%); this supersedes commit `000651cf` as the current accepted leaderboard candidate
- Submitted: 2026-07-20T11:46:05Z; verdict completed: 2026-07-20T11:52:20Z; standings queried: 2026-07-20T11:52:45Z

### Balanced n=2048 panel shared and L1 carveout candidate

- Commit: `f7249293ad84805b0b690be3035d14bfd5d594fe`
- Source verification: before submission, a clean detached temporary worktree at that exact commit had `problems/linalg/cholesky_py/submission.py` matching expected and actual Git blob `d6f005c8595d3306cb818fddb13eb88ce95e76c3` and expected and actual SHA-256 `454b9e3ceaacf843446fa480831556bb799216b1f268d0e2f6c89740e0479062`; the post-submission HEAD, blob, SHA-256, and clean status were unchanged, and the temporary worktree was removed
- Command: `GPUMODE_GPU=B200 bash harness/submit.sh linalg/cholesky_py` from that clean detached worktree
- Attempt evidence: attempt 1 started `2026-07-20T12:47:17Z` and completed `2026-07-20T12:56:19Z`; it succeeded with no timeout, retry, or hard failure
- Submission: 890163, accepted on B200 (`cli_exit=0`, helper verdict `ACCEPTED`); all returned public and secret test, benchmark, and leaderboard runs passed
- Public leaderboard score: 758.375307231631 us (`0.0007583753072316308` s)
- Secret leaderboard score: 757.196347062311 us (`0.000757196347062311` s)
- Standings after submission: candidate was not retained because it did not improve the account's public best; badelsteinlelbach remains rank 14/61 with retained submission 890077 at 752.637331889246 us; exact gap 321.004575065188% to #1 yanchi_72526 at 178.771770300289 us
- Evidence assessment: public and secret scores differ by 1.178960169320 us, and the candidate's public score is 5.737975342385 us slower than retained commit `cf2cf5a7`, so it does not supersede the current accepted leaderboard candidate
- Submitted: service run started `2026-07-20T12:47:21.942591Z`; public leaderboard verdict completed `2026-07-20T12:56:13.998549Z`; standings queried `2026-07-20T12:57:14Z`–`2026-07-20T12:57:28Z`

### Vector panel conversion ABI candidate

- Commit: `99366726a12118bec4fc637d7892c22ab71e4562`
- Source verification: before submission, a clean detached temporary worktree at that exact commit had `problems/linalg/cholesky_py/submission.py` matching expected and actual Git blob `a8bd63e5dc8b3e943afb80c746a6db165c749be5` and expected and actual SHA-256 `9d43a14bb8b026de14054df76e126544e6f6f3c150a3919503f098987f01cafa` byte-for-byte (263455 bytes); the post-submission HEAD, blob, SHA-256, and clean status were unchanged, and the temporary worktree was removed at `2026-07-20T14:42:26Z` without a commit or shared-branch change
- Command: `GPUMODE_GPU=B200 bash harness/submit.sh linalg/cholesky_py` from that clean detached worktree
- Attempt evidence: attempt 1 started `2026-07-20T14:31:42Z` and completed `2026-07-20T14:41:29Z`; it succeeded with no timeout, retry, hard failure, or hosted profile request
- Submission: 890281, accepted on B200 (`cli_exit=0`, helper verdict `ACCEPTED`); all returned public and secret test, benchmark, and leaderboard runs passed
- Public leaderboard score: 755.970570414264 us (`0.0007559705704142638` s)
- Secret leaderboard score: 751.545428404890 us (`0.0007515454284048896` s)
- Standings after submission: queried with the `leaderboard-rankings` script; the candidate was not retained because it did not improve the account's public best; badelsteinlelbach remains rank 14/61 with retained submission 890077 at 752.637331889246 us; exact gap 321.004575065188% to #1 yanchi_72526 at 178.771770300289 us
- Evidence assessment: public and secret scores differ by 4.425142009374 us, and the candidate's public score is 3.333238525018 us slower than retained commit `cf2cf5a7`, so it does not supersede the current accepted leaderboard candidate
- Submitted: public service test started `2026-07-20T14:31:49.251348Z`; public leaderboard verdict completed `2026-07-20T14:41:22.916832Z`; standings queried `2026-07-20T14:42:20Z`

### Grouped four padded n=64 factors per CTA candidate

- Commit: `9eb5fd3c38b05df81e2b2c7ba151b13c4ab339f8`
- Source verification: before submission, a clean detached temporary worktree at that exact commit had `problems/linalg/cholesky_py/submission.py` matching expected and actual Git blob `94436af80ee7e43a45551ebd0955575f32c984df` and expected and actual SHA-256 `a0a048a0d030d6989602180117c07d3110bbbc63359ebc2b25b52d9c9424bce9` byte-for-byte (264746 bytes); the post-submission HEAD, blob, SHA-256, and clean status were unchanged, the primary checkout remained at `8179fd08b3ba9114d74f67944da22ff883da7308`, and the temporary worktree was removed at `2026-07-20T16:22:46Z` without a commit or shared-branch/worktree change
- Command: `GPUMODE_GPU=B200 bash harness/submit.sh linalg/cholesky_py` from that clean detached worktree; this resolved to `popcorn-cli submit --no-tui --leaderboard cholesky --gpu B200 --mode leaderboard problems/linalg/cholesky_py/submission.py`
- Attempt evidence: attempt 1 was accepted by the service as submission 890443 at `2026-07-20T16:15:40.735978Z` and its final secret leaderboard run completed at `2026-07-20T16:21:54.499817Z`; it succeeded with no timeout, retry, hard failure, or hosted profile request
- Submission: 890443, accepted on B200 (`cli_exit=0`, helper verdict `ACCEPTED`); all six returned public and secret test, benchmark, and leaderboard runs passed, and the service job status was `succeeded` with `error=null`
- Public leaderboard score: 749.739809546865 us (`0.0007497398095468645` s)
- Secret leaderboard score: 750.837417424890 us (`0.00075083741742489` s)
- Standings after submission: queried with the `leaderboard-rankings` script; submission 890443 is the retained account best, and badelsteinlelbach is rank 15/63 at 749.739809546865 us, with an exact 319.383780944554% gap to #1 yanchi_72526 at 178.771770300289 us
- Evidence assessment: public and secret scores differ by 1.097607878026 us, and the public score improves retained submission 890077 (`cf2cf5a7`) by 2.897522342381 us (0.384982543333%); this supersedes commit `cf2cf5a7` as the current accepted leaderboard candidate
- Transcript evidence: `/tmp/popcorn-submit-J8genZ.log` was 4610 bytes with SHA-256 `972730b19d39dbb95c6aca0f2ecbd31b7b1e40fa06e89eacec00af0c3a673ea9` and contained no rejection, timeout, or hard-failure marker
- Service timing: public test `2026-07-20T16:15:49.616501Z`–`2026-07-20T16:16:37.359699Z`; public benchmark through `2026-07-20T16:18:58.770971Z`; public leaderboard through `2026-07-20T16:21:30.146923Z`; secret test began `2026-07-20T16:15:47.799658Z`; secret leaderboard completed `2026-07-20T16:21:54.499817Z`; standings queried `2026-07-20T16:22:34Z`

### Inverse WGMMA for larger panels candidate

- Commit: `c651e5807dedd7a294e547ff695542c3b740e1dc`
- Source verification: before submission, a clean detached temporary worktree at that exact commit had `problems/linalg/cholesky_py/submission.py` matching expected and actual Git blob `f585054c0a947121974265e07712161234a3ac75` and expected and actual SHA-256 `c800c7585070502eed2634240f8ec6815047f1ab79f527043aa115a24b032ee9` byte-for-byte (276981 bytes); the post-submission HEAD, blob, SHA-256, and clean status were unchanged, the primary checkout remained at `8179fd08b3ba9114d74f67944da22ff883da7308`, and the temporary worktree was removed at `2026-07-20T17:48:07Z` without a commit or shared-branch/worktree change
- Command: `GPUMODE_GPU=B200 bash harness/submit.sh linalg/cholesky_py` from that clean detached worktree; this resolved to `popcorn-cli submit --no-tui --leaderboard cholesky --gpu B200 --mode leaderboard problems/linalg/cholesky_py/submission.py`
- Attempt evidence: attempt 1 was accepted by the service as submission 890543 at `2026-07-20T17:35:00.661543Z` and the final public leaderboard run completed at `2026-07-20T17:47:15.480687Z`; it succeeded with no timeout, retry, hard failure, or hosted profile request
- Submission: 890543, accepted on B200 (`cli_exit=0`, helper verdict `ACCEPTED`); all six returned public and secret test, benchmark, and leaderboard runs passed, and the service job status was `succeeded` with `error=null`
- Public leaderboard score: 743.695546528557 us (`0.000743695546528557` s)
- Secret leaderboard score: 748.557336316234 us (`0.0007485573363162341` s)
- Standings after submission: queried with the `leaderboard-rankings` script; submission 890543 is the retained account best, and badelsteinlelbach is rank 15/63 at 743.695546528557 us, with an exact 316.002786837847% gap to #1 yanchi_72526 at 178.771770300289 us
- Evidence assessment: public and secret scores differ by 4.861789787677 us, and the public score improves retained submission 890443 (`9eb5fd3c`) by 6.044263018308 us (0.806181416719%); this supersedes commit `9eb5fd3c` as the current accepted leaderboard candidate
- Transcript evidence: `/tmp/popcorn-submit-zLxZ4S.log` was 7048 bytes with SHA-256 `0b347759cd35717c6bb81e2f21c398f509663408c048893285f425fe2a887af5` and contained no rejection, timeout, or hard-failure marker
- Service timing: secret test `2026-07-20T17:35:11.127178Z`–`2026-07-20T17:37:03.753339Z`; secret benchmark through `2026-07-20T17:41:57.458416Z`; secret leaderboard through `2026-07-20T17:46:38.866785Z`; public test `2026-07-20T17:36:21.167200Z`–`2026-07-20T17:38:07.672012Z`; public benchmark through `2026-07-20T17:42:36.697813Z`; public leaderboard through `2026-07-20T17:47:15.480687Z`; standings queried `2026-07-20T17:47:36Z`–`2026-07-20T17:47:50Z`

### Retained direct solves for 4096-row tails candidate

- Commit: `01366b200a486990c6990a223f26d91357bb8758`
- Source verification: before submission, a clean detached temporary worktree at that exact commit had `problems/linalg/cholesky_py/submission.py` matching expected and actual Git blob `ffdececd32adc4b79127878bd7226edd364dceca` and expected and actual SHA-256 `538541f4c346c4e236dbae5fa7fa4f5d28f043c8f5b391a1de94806104d0a47a` byte-for-byte (276994 bytes); the post-submission HEAD, blob, SHA-256, and clean status were unchanged, the primary checkout remained at `8179fd08b3ba9114d74f67944da22ff883da7308`, and the temporary worktree was removed at `2026-07-20T18:00:13Z` without a commit or shared-branch/worktree change
- Command: `bash harness/submit.sh linalg/cholesky_py` from that clean detached worktree; because B200 is the task's only supported GPU, this resolved to `popcorn-cli submit --no-tui --leaderboard cholesky --gpu B200 --mode leaderboard problems/linalg/cholesky_py/submission.py`
- Attempt evidence: attempt 1 was accepted by the service as submission 890577 at `2026-07-20T17:50:24.850066Z` and its final secret leaderboard run completed at `2026-07-20T17:59:30.240303Z`; it succeeded with no timeout, retry, hard failure, or hosted profile request
- Submission: 890577, accepted on B200 (`cli_exit=0`, helper verdict `ACCEPTED`); all six returned public and secret test, benchmark, and leaderboard runs passed, and the service job status was `succeeded` with `error=null`
- Public leaderboard score: 735.196476364953 us (`0.0007351964763649528` s)
- Secret leaderboard score: 742.221385865336 us (`0.0007422213858653356` s)
- Standings after submission: queried with the `leaderboard-rankings` script; submission 890577 is the retained account best, and badelsteinlelbach is rank 15/63 at 735.196476364953 us, with an exact 311.248641287167% gap to #1 yanchi_72526 at 178.771770300289 us
- Evidence assessment: public and secret scores differ by 7.024909500383 us, and the public score improves retained submission 890543 (`c651e580`) by 8.499070163604 us (1.142815793812%); this supersedes commit `c651e580` as the current accepted leaderboard candidate
- Transcript evidence: `/tmp/popcorn-submit-BWCdrz.log` was 6558 bytes with SHA-256 `5fe152f4955789a22a8d96f845a9ce85b956ef296f51b998d03a3ae6f702fdd1` and contained no rejection, timeout, or hard-failure marker
- Service timing: public test `2026-07-20T17:50:27.065292Z`–`2026-07-20T17:51:15.740543Z`; public benchmark through `2026-07-20T17:53:44.560559Z`; public leaderboard through `2026-07-20T17:56:28.223662Z`; secret test `2026-07-20T17:50:26.898920Z`–`2026-07-20T17:51:09.404433Z`; secret benchmark through `2026-07-20T17:55:12.849688Z`; secret leaderboard through `2026-07-20T17:59:30.240303Z`; standings queried `2026-07-20T18:00:26Z`–`2026-07-20T18:00:29Z`

### Removed zero-beta solve input candidate

- Commit: `6d5995fe722665c88016535042942d72664144db`
- Source verification: before submission, a clean detached temporary worktree at that exact commit had `problems/linalg/cholesky_py/submission.py` matching expected and actual Git blob `051d19fcabd5e17e401dac84187977409cf52f05` and expected and actual SHA-256 `aada31f5dd2e2f8725fd9162d090896358a3e471a20f3e79706da3e525a53df7` byte-for-byte (277020 bytes); the post-submission HEAD, blob, SHA-256, byte count, and clean status were unchanged, the primary checkout remained at `8179fd08b3ba9114d74f67944da22ff883da7308`, and the temporary worktree was removed at `2026-07-20T20:17:53Z` without a commit or shared optimization branch/worktree change
- Command: `bash harness/submit.sh linalg/cholesky_py` from that clean detached worktree; because B200 is the task's only supported GPU, this resolved to `popcorn-cli submit --no-tui --leaderboard cholesky --gpu B200 --mode leaderboard problems/linalg/cholesky_py/submission.py`
- Attempt evidence: attempt 1 started at `2026-07-20T20:08:37Z`, was accepted by the service as submission 890826 at `2026-07-20T20:08:40.008039Z`, and completed at `2026-07-20T20:17:00Z`; it succeeded on the first attempt with no timeout, retry, hard failure, or hosted profile request
- Submission: 890826, accepted on B200 (`cli_exit=0`, helper verdict `ACCEPTED`); all six returned public and secret test, benchmark, and leaderboard runs passed, and the service job status was `succeeded` with `error=null`
- Public leaderboard score: 735.327886007728 us (`0.0007353278860077278` s)
- Secret leaderboard score: 739.562837696820 us (`0.0007395628376968204` s)
- Standings after submission: queried with the `leaderboard-rankings` script; the candidate was not retained because it did not improve the account's public best; badelsteinlelbach remains rank 15/64 with retained submission 890577 at 735.196476364953 us, with an exact 311.248641287167% gap to #1 yanchi_72526 at 178.771770300289 us
- Evidence assessment: public and secret scores differ by 4.234951689093 us, and the candidate's public score is 0.131409642775 us (0.017874084955%) slower than retained commit `01366b20`, so it does not supersede the current accepted leaderboard candidate
- Transcript evidence: `/tmp/popcorn-submit-Sp70Oi.log` was 5954 bytes with SHA-256 `bc899deaaf9aa85553c551dd001dca1b38ee033b6b31d91c35ac4924f4cb6eab`; wrapper output `/tmp/gpumode-submit-6d5995fe-attempt1.log` was 7228 bytes with SHA-256 `d60fa4a2843488cd8fa213a10b153897d355b0b617189efdf413be48583da7e6`; neither contained a timeout, rejection, or hard-failure marker
- Service timing: public test `2026-07-20T20:08:42.567764Z`–`2026-07-20T20:09:29.827205Z`; public benchmark through `2026-07-20T20:11:54.347952Z`; public leaderboard through `2026-07-20T20:14:31.522403Z`; secret test `2026-07-20T20:08:48.416014Z`–`2026-07-20T20:10:02.952055Z`; secret benchmark through `2026-07-20T20:13:34.948704Z`; secret leaderboard through `2026-07-20T20:16:57.720457Z`; standings queried `2026-07-20T20:17:37Z`–`2026-07-20T20:17:39Z`
## Continuation segment 2

- Original source tag: `2026-07-20-23-00-46-cholesky-unmerged`
- Combined brief IDs: `98–105`

### Continuation checkpoint (source baseline)

- Tag: `2026-07-19-01-44-06-cholesky-resumed-local`
- Commit: `6d5995fe722665c88016535042942d72664144db`
- Local full-grid metric: `781.912897 us`
- Submission source blob: `051d19fcabd5e17e401dac84187977409cf52f05`
- Submission source SHA-256: `aada31f5dd2e2f8725fd9162d090896358a3e471a20f3e79706da3e525a53df7`
- Command: `bash harness/submit.sh linalg/cholesky_py` from the detached manager worktree at the exact commit above.
- Submission: `891223`, accepted on B200; all public and secret test, benchmark, and leaderboard phases passed with service status `succeeded` and `error=null`.
- Public leaderboard score: `732.3556153902046 us`.
- Secret leaderboard score: `733.8762812461586 us`.
- Standing after submission: rank `15/64`; #1 was `yanchi_72526` at `178.771770300289 us`; exact gap to #1 was `309.6595419735689%` (`4.096595419735689x`).
- Hosted baseline profile: job `ffbd155bd07541b6acf445e60b0e618a`, created `2026-07-20T23:15:20.917010Z`, benchmark index `14` (`batch=1, n=32768`), succeeded and was archived under `profiles/2026-07-19-01-44-06-cholesky-resumed-local/baseline-6d5995fe722665c88016535042942d72664144db.*`.

### Brief 101 direct 2048-wide panels

- Commit: `ba07e3fb2f06e6ec2b201ed9930dfb982a7a3711`.
- Local full-grid metric: `773.223374 us`.
- Submission source blob: `615f3727d5d4d04d38910302c79845bd24f01e7f`.
- Submission source SHA-256: `dd2839782d9758cfb8c29cb1842b7f651d30c13369713bb756a10bf358780da8`.
- Submission: `891311`, accepted on B200; all public and secret phases passed.
- Public leaderboard score: `761.8465440595186 us`.
- Secret leaderboard score: `762.8584991228829 us`.
- Evidence assessment: valid but not retained because it was slower than baseline submission `891223`.

### Brief 100 recursive low-batch panels

- Commit: `041dafa88b8e63352cc6ae5889c24b8ff277486e`.
- Local full-grid metric: `754.512961 us`.
- Submission source blob: `a47d39a8adb14cf918e1322e92c61744cc1b260f`.
- Submission source SHA-256: `3a0a9ae099f355a330796197731289445357a1622b0b58c3548390aa3d523cbd`.
- Submission: `891313`, accepted on B200; all public and secret phases passed.
- Public leaderboard score: `718.7745252137361 us`.
- Secret leaderboard score: `708.809157126694 us`.
- Standing after submission: rank `15/64`; #1 remained `178.771770300289 us`; exact gap was `302.0626545267109%` (`4.020626545267109x`).
- Evidence assessment: retained account best, improving baseline submission `891223` by `1.8544392766391737%` publicly.
- Hosted profile: job `f614742501984c25883334e3df0005f5`, created `2026-07-21T00:07:44.998331Z`, benchmark index `10` (`batch=1, n=4096`), exact commit `5f048a1c1afc2a3d7d5ab0963aba6e9e93a912ef`; primary artifact archived as `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-100-trial-4-5f048a1c1afc2a3d7d5ab0963aba6e9e93a912ef.ncu-rep`.

### Brief 99 fixed-role n=256 queue

- Commit: `a831249d64e4529a46f1935d0f19bde5c40ecd73`.
- Local full-grid metric: `753.349 us`.
- Submission source blob: `82c56ea712b4535f2fb711f7c9ffb408734ce0d9`.
- Submission source SHA-256: `f529c9b1209b50bb2a7cbb359de7a054aa9e1365e157ca8db9085576c9230458`.
- Submission: `891341`, accepted on B200; all public and secret phases passed.
- Public leaderboard score: `742.9088308478604 us`.
- Secret leaderboard score: `743.796108743652 us`.
- Evidence assessment: valid but not retained because it was slower than Brief 100 submission `891313`.
- Hosted profile: job `b8c1ca37b48b42da94d3e12477a44fcb`, created `2026-07-20T23:51:18Z`, benchmark index `3` (`batch=64, n=256`), exact commit `a831249d64e4529a46f1935d0f19bde5c40ecd73`; artifacts archived as `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-99-trial-2-a831249d64e4529a46f1935d0f19bde5c40ecd73.*`.

### Brief 101 direct-upper large path profile

- Candidate commit: `fc15d8ebedbc14cf274877195fbd595ce8c9956e`.
- Local full-grid metric: `756.804160 us`; selected as a large-shape lineage donor but not submitted in isolation because it did not beat the run-wide local/submitted leader.
- Hosted profile: job `34d8940a507148479a153eab5a924929`, created `2026-07-21T00:17:27.199851Z`, ended `2026-07-21T00:19:34.614286Z`, benchmark index `14` (`batch=1, n=32768, cond=2, seed=48368`); artifacts archived as `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-101-trial-4-fc15d8ebedbc14cf274877195fbd595ce8c9956e.*`.

### Brief 99 expanded n=256 queue residency

- Commit: `dccf2459d135da81338b83ba3c4eafa5dacbe92d`.
- Local full-grid metric: `748.014 us`.
- Submission source blob: `4a133f7e09d7b503be1c93dc01662e53674beddd`.
- Submission source SHA-256: `74404dd595f847ffb2ef742b31d352af80bfda27e00b2fe949e1d6c1e0902d09`.
- Submission: `891401`, accepted on B200 after `635 s`; all public and secret phases passed.
- Public leaderboard score: `743.1693983770589 us`.
- Secret leaderboard score: `740.1183213503164 us`.
- Evidence assessment: valid but not retained because it was slower than Brief 100 submission `891313`.

### Brief 98 small-matrix profile

- Candidate commit: `fa33d3743bea2c3f33691d1fa7a1ef966f26b2e5`.
- Local full-grid metric: `756.698 us`; retained as a small-shape donor but not submitted in isolation because it did not beat the run-wide local/submitted leader.
- Hosted profile: job `6e1f723303e4444981ff5a719e77b227`, created `2026-07-21T00:53:38Z`, ran `00:55:02Z`-`00:56:28Z`, benchmark index `0`; artifacts archived as `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-98-trial-8-fa33d374*`.
- Report SHA-256: `92d76056337a722b40f6f84cf6cdcaafbab2a55310ddb60fc3ae87d299421e6b`.

### Brief 99 batched n=2048 stages

- Commit: `960b2a103d84b98c4ea57c736ef22158cfb22155`.
- Local full-grid metric: `745.304 us`.
- Submission source blob: `1f988ea21291b5946ce4f1cbad741b0f2d690b6d`.
- Submission source SHA-256: `7a17506f6e9338942fa5627a66ef42b947a97491ac9903fcae97ed4c6d3fd538`.
- Submission: `891467`, accepted on B200 after `540 s`; all public and secret phases passed.
- Public leaderboard score: `749.0294061515697 us`.
- Secret leaderboard score: `742.5020023846213 us`.
- Evidence assessment: valid but not retained because it was slower than Brief 100 submission `891313`.
- Hosted profile: job `0d9ee2450bfb4309a0388c5664f4ddcc`, benchmark index `9` (`batch=8, n=2048`), exact commit `960b2a103d84b98c4ea57c736ef22158cfb22155`; artifacts archived as `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-99-trial-6-960b2a103d84b98c4ea57c736ef22158cfb22155.*`.

### Brief 102 graph-captured FP32 frontier profile

- Candidate commit: `598b992842d5e15bdf46702064c1656d4a8b108d`.
- Local full-grid metric: `764.157766 us`; retained as a medium-shape combine donor but not submitted in isolation because it did not beat the run-wide local/submitted leader.
- Hosted profile: job `133df6c953d34d6b9f346ed8a7468713`, created `2026-07-21T01:22:45Z`, benchmark index `6`, exact commit `598b992842d5e15bdf46702064c1656d4a8b108d`; artifacts archived as `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-102-trial-14-598b992842d5e15bdf46702064c1656d4a8b108d.*`.

### Brief 103 consolidated shape paths

- Commit: `e5e246995cf46eeba6f1c9cbeecba82e0cdb93b7`.
- Local full-grid metric: `744.366141 us`.
- Submission source blob: `6c08f6194da14bc673d5ff7041aa54e745727e15`.
- Submission source SHA-256: `55c6dd696a3e25e14ff900cac70af4a7c18a27635532d6ddf8ba7d1f215c81a0`.
- Submission: `891569`, accepted on B200 after `390 s`; all public and secret phases passed.
- Public leaderboard score: `718.6781008918352 us`.
- Secret leaderboard score: `726.5451416681947 us`.
- Evidence assessment: retained account best, improving submission `891313` by `0.09642432190082673 us` (`0.013415100079145658%`) publicly.

### Brief 100 dedicated n=512 and shape-specific frontiers

- Commit: `15c9e8384de94cf42e9ec9088bd9aeba6d364ceb`.
- Local full-grid metric: `741.324487 us`.
- Submission source blob: `c99c133e816fd73f50e0fa91823d31d87d87ed32`.
- Submission source SHA-256: `b45e6b520e078b5686d3f94458ada6f086d3b5ce856a4e46841eff69f4fe064a`.
- Submission: `891589`, accepted on B200 after `355 s`; all public and secret phases passed.
- Public leaderboard score: `711.9910269393262 us`.
- Secret leaderboard score: `719.9446840117271 us`.
- Evidence assessment: retained account best, improving submission `891569` by `6.687073952509081 us` (`0.9304685845040825%`) publicly.
- Standing after subsequent submissions settled: rank `15/64`; #1 remained `178.771770300289 us`; exact gap was `298.26815259667154%` (`3.9826815259667154x`).

### Brief 103 batched n=512 consolidated path

- Commit: `76b595e4ae47ded70ac030f56de2c76acb9c12d4`.
- Local full-grid metric: `725.182621 us`.
- Submission source blob: `79633812b324cffb98188b49705ed76197f0290e`.
- Submission source SHA-256: `2bfc0bc771ffd17d9288ccc05c9d519bc6d519d38c929d8f938d826f7e9d07dc`.
- Submission: `891603`, accepted on B200 after `470 s`; all public and secret phases passed.
- Public leaderboard score: `716.9497204769473 us`.
- Secret leaderboard score: `724.420615802057 us`.
- Evidence assessment: valid but not retained because it was slower than Brief 100 submission `891589`.
- Hosted profile: job `055e5995c108450582ddff5c6c57928e`, ran `2026-07-21T01:55:20Z`-`01:58:45Z`, benchmark index `4` (`batch=16, n=512`), exact commit `76b595e4ae47ded70ac030f56de2c76acb9c12d4`; artifacts archived as `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-103-trial-2-76b595e4*`.

### Brief 104 combined n=2048 profile

- Candidate commit: `282d391f895dbcba85d1108061d9d3fb82e66c06`.
- Hosted profile: job `0a98a796807f4bee982757baa7d70675`, ran `2026-07-21T03:18:48Z`-`03:22:17Z`, benchmark index `9` (`batch=8, n=2048`); artifacts archived as `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-104-trial-3-282d391*`.

### Brief 104 expanded persistent n=2048 queue

- Commit: `32e1f1ee3ee3f89b1b56605d4f3b7e97c424598c`.
- Local full-grid metric: `724.898460 us`.
- Submission source blob: `792cd99c571fd8b20c12e50ca93654c7b30e3960`.
- Submission source SHA-256: `c95dbb3b2243147a7a599fb26960333a2d49d20e946a4f6e1afe145ddd27c105`.
- Submission: `891832`, accepted on B200 after `660 s`; all public and secret phases passed.
- Public leaderboard score: `719.17754030126 us`.
- Secret leaderboard score: `724.9978914696428 us`.
- Evidence assessment: valid but not retained because it was slower than Brief 100 submission `891589`.

### Brief 102 coalesced frontier panel transfers

- Commit: `cf32ecafdd0ea673ece7244bbdb9aaf6876fd5b8`.
- Local full-grid metric: `717.665 us` (Brief 102, Trial 31; fully validated before submission).
- Submission source blob: `7a5feb4435e7640e2af5cc55cced75357cdb8ef8`.
- Submission source SHA-256: `1c42e2ba8fba10f63bf046c9b524d71ef8b44d39e0e0e4ea7758c8adc87146de`.
- Command: `bash harness/submit.sh linalg/cholesky_py` from a clean detached worktree at the exact commit above.
- Submission: `891885`, accepted on B200 after `385 s`; all public and secret test, benchmark, and leaderboard phases passed with service status `succeeded` and `error=null`.
- Public leaderboard score: `712.3096230492157 us`.
- Secret leaderboard score: `713.0217021238968 us`.
- Evidence assessment: valid but not retained publicly because it was `0.3185961098895 us` (`0.044747208075%`) slower than Brief 100 submission `891589`; its secret score was lower than submission `891589` by `6.9229818878303 us`.

### Brief 102 final reciprocal-square-root frontier factors

- Commit: `fd3d941d86d5d052f95e1e7ad9d92480bf47d77b`.
- Local full-grid metric: `715.900 us` (Brief 102 final; fresh official validation passed before submission).
- Submission source blob: `ffa5b6f996a74e384b69561b98da85746637470f`.
- Submission source SHA-256: `d88d8943f04124f466ffce7f3f557750e84a1217c0a74d11d9673a09ab1bda12`.
- Command: `bash harness/submit.sh linalg/cholesky_py` from a clean detached worktree at the exact commit above.
- Submission: `891904`, accepted on B200 after `595 s`; all public and secret test, benchmark, and leaderboard phases passed with service status `succeeded` and `error=null`.
- Public leaderboard score: `714.868511560764 us`.
- Secret leaderboard score: `707.4869422391948 us`.
- Evidence assessment: valid but not retained publicly because it was `2.8774846214378 us` (`0.404146191814722%`) slower than Brief 100 submission `891589`; its secret score was lower than submission `891589` by `12.4577417725322 us`.
- Live standing after both final submissions settled: retained submission `891589` at `711.9910269393262 us`, rank `15/65`; #1 was `yanchi_72526` at `178.771770300289 us`; exact gap to #1 was `298.268152596671541%` (`3.982681525966715x`).
## Continuation segment 3

- Original source tag: `2026-07-21-04-34-16-cholesky-unmerged`
- Combined brief IDs: `106–111`

- Run tag: `2026-07-19-01-44-06-cholesky-resumed-local`
- Benchmark: `linalg/cholesky_py`
- GPU: `B200`

### Continuation checkpoint (source baseline)

- Commit: `15c9e8384de94cf42e9ec9088bd9aeba6d364ceb`
- Official validation and fresh-input guard: PASS
- Local full-grid geomean: `751.039470 us`
- Submission command: `bash harness/submit.sh linalg/cholesky_py`
- Submission: `891942`, accepted; all public and secret phases passed
- Public score: `725.1037490980791 us`
- Secret score: `715.5940930527288 us`
- Retained account best: `711.9910269393262 us`, rank `15/65`
- Current #1: Ravi Theja, `112.611 us`, file `sc2cap_hffull16_1.py`
- Gap to #1: `+532.3%`

### Baseline hosted profile

- Job: `a7b028f76ee84e57931d849d3330708c`
- Commit: `15c9e8384de94cf42e9ec9088bd9aeba6d364ceb`
- Benchmark index: `7` (`batch=60, n=1024`)
- Ran: `2026-07-21T04:59:00.766144Z` to `2026-07-21T05:24:53.308799Z`
- Archived report: `profiles/2026-07-19-01-44-06-cholesky-resumed-local/baseline-15c9e8384de94cf42e9ec9088bd9aeba6d364ceb.ncu-rep`
- Report SHA-256: `bc401a1b37818a446e2671e8e13eb8d32377480f8055bd18cd9d7463b3cb0606`
- Dominant factor kernel: `191.23 us`, 120 CTAs, 168 registers/thread, 6.27% achieved occupancy, 90.12% no-eligible cycles, 0.75% DRAM throughput, and 9.37% SM throughput.
- Full lower-copy kernel: `60.45 us`, 888 CTAs, 53.18% DRAM throughput.
- Three bulk panel-solve launches: `39.58-40.70 us` each, about 65.5% SM throughput and 56% no-eligible cycles.

### Local leader submission: first-stage input read

- Commit: `95ebfa272d02993bf15348dd000629b807ea17af`
- Local full-grid geomean: `731.209 us`
- Isolated submission checkout: detached at the exact commit and clean before submission; `submission.py` blob `f4eedfc24fc6781e3dc270059041f009db143201`
- Command: `bash harness/submit.sh linalg/cholesky_py`
- Attempt: one, started `2026-07-21T05:39:56Z`
- Submission: `892057`, accepted (`cli_exit=0`, verdict `ACCEPTED`)
- Public phases: test PASS (`05:40:11.976500Z`–`05:41:04.434831Z`), benchmark PASS (`05:41:04.434994Z`–`05:44:03.707659Z`), leaderboard PASS (`05:44:03.707803Z`–`05:47:54.369758Z`)
- Public score: `712.7614303158045 us`
- Secret phases: test PASS (`05:40:20.655585Z`–`05:42:20.475180Z`), benchmark PASS (`05:42:20.475931Z`–`05:47:11.173714Z`), leaderboard PASS (`05:47:11.174910Z`–`05:52:50.729220Z`)
- Secret score: `719.78114456795 us`
- Live B200 standings after completion: retained account best `711.9910269393262 us`, rank `15/66`
- Current #1: Ravi Theja, `55.818 us`, file `submission_003.py`
- Gap to #1: `+1175.6%`

### Brief 108 compliant single-queue profile

- Job: `e8e80355692c4c14b8982f9af21728de`
- Commit: `44ee0a782f0b114bebb6622e2f768d1db0c7b669`
- Benchmark index: `8` (`batch=2, n=2048`)
- Requested: `2026-07-21T05:36:25Z`; ran `2026-07-21T05:49:25Z` to `2026-07-21T05:52:41Z`
- Archived report: `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-108-trial-0-44ee0a782f0b114bebb6622e2f768d1db0c7b669.ncu-rep`
- Findings: factor grid 2 takes `11.52/11.36 us`; first panel grid 296 takes `10.34 us`; update grid 112 takes `6.98 us`; next panel grid 24 takes `8.38 us`; four initial fill kernels each take about `11.3-11.9 us`. The small grids expose only `0.01-0.76` waves/SM, identifying launch/dependency overhead as the primary fusion target.

### Brief 110 recursive lower-only profile

- Job: `951c2f34d3794dbdbf72d029816aef04`
- Commit: `bda239818a665e516cd9cbf29066bc9605fa3c27`
- Benchmark index: `14` (`batch=1, n=32768`)
- Requested: `2026-07-21T06:40:59Z`; ran `2026-07-21T06:41:01.848578Z` to `2026-07-21T06:43:32.221482Z`
- Archived report: `profiles/2026-07-19-01-44-06-cholesky-resumed-local/brief-110-trial-9-bda239818a665e516cd9cbf29066bc9605fa3c27.ncu-rep`
- Report SHA-256: `63d31adfb9f7a838a8a37a5b3c392fe7d6d338637442f5d5de83d4848e010395`
- Archived details: matching `-ncu-details.{csv,txt}` files and `-profile.zip` in the same directory.
- Findings from the captured first-stage launches: the 4096-square cuSOLVER factor kernel dominates at `2.38 ms`, reaches only `36.15%` SM throughput and `12.36%` achieved occupancy, and spends `57.54%` of cycles with no eligible warp. The full lower copy takes `768.03 us` at `83.27%` DRAM throughput. Compact-leaf inverse TRSM takes `160.45 us`; its following tcgen05 GEMM takes `11.97 us`. The factor-panel transpose copy is only `35.71 us`. The next structural target is therefore replacing or recursively decomposing the 4096 POTRF base, not further panel-copy tuning.

### Batch-eight 2048 wavefront candidate submission

- Commit: `6740b296f78ec4068d92a000b4e0bc80f6e534db`
- Local full-grid geomean: `731.158498 us`
- Causal target-cell change: `batch=8, n=2048` improved from `1797.576 us` to `1046.071 us`
- Isolated submission checkout: detached at the exact commit and clean before submission
- Command: `bash harness/submit.sh linalg/cholesky_py`
- Attempt: one
- Submission: `892117`, accepted (`cli_exit=0`, verdict `ACCEPTED`)
- Public test: PASS (`2026-07-21T06:19:03.715379Z`–`2026-07-21T06:19:52.026238Z`)
- Public benchmark: PASS (`2026-07-21T06:19:52.026443Z`–`2026-07-21T06:22:27.372245Z`)
- Public leaderboard: PASS (`2026-07-21T06:22:27.372592Z`–`2026-07-21T06:25:51.105888Z`)
- Public score: `720.8194706302248 us`
- Secret test: PASS (`2026-07-21T06:19:04.586924Z`–`2026-07-21T06:19:54.253679Z`)
- Secret benchmark: PASS (`2026-07-21T06:19:54.253947Z`–`2026-07-21T06:22:48.385932Z`)
- Secret leaderboard: PASS (`2026-07-21T06:22:48.386326Z`–`2026-07-21T06:26:25.168798Z`)
- Secret score: `722.0986042570636 us`
- Result: did not replace the retained account best
- Live B200 standings after completion: retained account best `711.9910269393262 us`, rank `15/66`
- Current #1: Ravi Theja, `55.818 us`, file `submission_003.py`
- Gap to #1: `+1175.6%`

### Batch-sixteen N512 epoch-specialized submission

- Commit: `8c6e54cd1ac30b763e620b540a1cbc28ff0c315c`
- Local full-grid geomean: `691.790270 us`
- Causal target-cell change: `batch=16, n=512` improved from `508.062 us` to `175.995 us`
- Isolated submission checkout: detached at the exact commit and clean before and after submission; `submission.py` blob `5944ab5d706907f1fb0fae9c598094e2323c44aa`
- Command: `bash harness/submit.sh linalg/cholesky_py`
- Attempt: one, started `2026-07-21T06:24:38Z`; no retry was needed
- Submission: `892130`, accepted (`cli_exit=0`, verdict `ACCEPTED`)
- Public test: PASS (`2026-07-21T06:24:42.857497Z`–`2026-07-21T06:26:21.639565Z`)
- Public benchmark: PASS (`2026-07-21T06:26:21.639770Z`–`2026-07-21T06:30:30.352885Z`)
- Public leaderboard: PASS (`2026-07-21T06:30:30.353133Z`–`2026-07-21T06:35:44.264887Z`)
- Public score: `672.4353258919637 us`
- Secret test: PASS (`2026-07-21T06:24:42.845671Z`–`2026-07-21T06:26:05.205509Z`)
- Secret benchmark: PASS (`2026-07-21T06:26:05.205709Z`–`2026-07-21T06:29:50.827423Z`)
- Secret leaderboard: PASS (`2026-07-21T06:29:50.827593Z`–`2026-07-21T06:34:34.642391Z`)
- Secret score: `671.1689096524961 us`
- Result: replaced the retained account best
- Live B200 standings after completion: retained account best `672.4353258919637 us`, rank `13/66`
- Current #1: Ravi Theja, `55.818231694033744 us`, file `submission_003.py`
- Gap to #1: `+1104.687618%` (`12.046876x`)

### Packed N512 sidecar local-leader submission

- Commit: `0b18f368ac286b8646e796cd0736552c48e11a41`
- Local full-grid geomean: `684.643213 us`
- Target cells: `n=256` at `71.474 us`; `batch=16, n=512` at `179.002 us`; `batch=640, n=512` at `1025.284 us`
- Isolated submission checkout: detached at the exact commit and clean before and after submission; `submission.py` blob `e6ecc68ea7e5b3594c43f0758d16d235583aed47`
- Command: `bash harness/submit.sh linalg/cholesky_py`
- Attempt: one, started `2026-07-21T07:30:15Z`; no retry was needed
- Submission: `892267`, accepted (`cli_exit=0`, verdict `ACCEPTED`)
- Public test: PASS (`2026-07-21T07:30:32.260971Z`–`2026-07-21T07:32:33.481674Z`)
- Public benchmark: PASS (`2026-07-21T07:32:33.482064Z`–`2026-07-21T07:38:01.799799Z`)
- Public leaderboard: PASS (`2026-07-21T07:38:01.800134Z`–`2026-07-21T07:44:25.777472Z`)
- Public score: `674.5945587986847 us`
- Secret test: PASS (`2026-07-21T07:30:20.419938Z`–`2026-07-21T07:31:17.049645Z`)
- Secret benchmark: PASS (`2026-07-21T07:31:17.050030Z`–`2026-07-21T07:34:02.072440Z`)
- Secret leaderboard: PASS (`2026-07-21T07:34:02.072617Z`–`2026-07-21T07:37:45.429881Z`)
- Secret score: `667.9239111917014 us`
- Result: did not replace the retained account best
- Live B200 standings queried at `2026-07-21T07:46:26Z`: retained account best `672.4353258919637 us`, rank `14/66`
- Current #1: Yukariko, `21.167764995065733 us`, file `submission.py`
- Gap to #1: `+3076.694970%` (`31.766950x`)

### Final-combine global-leader submission

- Commit: `fd956a9d484692c4dc7a4ec20d94bc371d50e7ed`
- Final independently revalidated local full-grid geomean: `670.920127 us`; all 15 cells passed
- Initial validated target cells: `batch=16, n=512` at `184.191 us`; `batch=640, n=512` at `1082.129 us`; `batch=60, n=1024` at `881.801 us`; `batch=8, n=2048` at `971.318 us`; `batch=2, n=4096` at `1686.342 us`; `n=8192/16384/32768` at `4430.623/11984.624/35581.746 us`
- Isolated submission checkout: detached at the exact commit and clean before and after submission; `submission.py` blob `59b1b4ba92925305e2cd7b407bc2cbf98b9ef9e7`
- Command: `bash harness/submit.sh linalg/cholesky_py`
- Attempt: one, started `2026-07-21T07:48:18Z`; no retry was needed
- Submission: `892316`, accepted (`cli_exit=0`, verdict `ACCEPTED`)
- Public test: PASS (`2026-07-21T07:48:29.888882Z`–`2026-07-21T07:49:28.034930Z`)
- Public benchmark: PASS (`2026-07-21T07:49:28.035045Z`–`2026-07-21T07:52:20.202229Z`)
- Public leaderboard: PASS (`2026-07-21T07:52:20.202620Z`–`2026-07-21T07:56:14.433153Z`)
- Public score: `656.4454378590116 us`
- Secret test: PASS (`2026-07-21T07:48:29.828249Z`–`2026-07-21T07:49:32.517499Z`)
- Secret benchmark: PASS (`2026-07-21T07:49:32.517661Z`–`2026-07-21T07:52:36.904288Z`)
- Secret leaderboard: PASS (`2026-07-21T07:52:36.904518Z`–`2026-07-21T07:56:35.843304Z`)
- Secret score: `655.9620875210324 us`
- Result: replaced the retained account best
- Live B200 standings queried at `2026-07-21T07:56:54Z`: account best `656.4454378590116 us`, rank `13/66`
- Current #1: Yukariko, `21.167764995065733 us`, file `submission.py`
- Gap to #1: `+3001.156112%` (`31.011561x`)

### Resume checkpoint: final-combine leader resubmission

- Commit: `fd956a9d484692c4dc7a4ec20d94bc371d50e7ed`
- Command: `autocuda run slice --data-dir /home/shadeform/.local/share/autocuda/gpumode/2026-07-19-01-44-06-cholesky-resumed-local --tag 2026-07-19-01-44-06-cholesky-resumed-local -- bash harness/submit.sh linalg/cholesky_py`
- Attempt: one, started `2026-07-29T15:11:20Z`; no retry was needed
- Submission: `926808`, accepted on B200 (`cli_exit=0`, helper verdict `ACCEPTED`); all public and secret test, benchmark, and leaderboard phases passed with service status `succeeded` and `error=null`
- Public leaderboard score: `658.6568103074733 us`
- Secret leaderboard score: `656.6180169954428 us`
- Result: did not replace retained account best submission `892316` at `656.4454378590116 us`
- Live B200 standings queried after completion: badelsteinlelbach rank `33/107`; retained score `656.445 us`; gap to #1 zhongmingee at `201.531 us` is `+225.7%`

### Resume cadence checkpoint: final-combine leader

- Commit: `fd956a9d484692c4dc7a4ec20d94bc371d50e7ed`
- Attempt: one, started `2026-07-29T16:02:03Z`; no retry was needed
- Submission: `927043`, accepted on B200; all public and secret test, benchmark, and leaderboard phases passed with service status `succeeded` and `error=null`
- Public leaderboard score: `660.0095904909605 us`
- Secret leaderboard score: `653.4613376598011 us`
- Result: did not replace retained account best submission `892316` at `656.4454378590116 us`
