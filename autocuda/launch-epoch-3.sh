#!/usr/bin/env bash
set -euo pipefail

export PATH=/home/shadeform/.local/bin:/home/shadeform/.local/aab-bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin

data_dir=/home/shadeform/.local/share/autocuda/gpumode/2026-07-19-01-44-06-cholesky-resumed-local
tag=2026-07-19-01-44-06-cholesky-resumed-local
repo=/home/shadeform/gpumode
log="$data_dir/epoch-3-launch.log"
manager_log="$data_dir/$tag-optimize-tree-manager-log.csv"
sentinel="$data_dir/epoch-3-briefs.logged"

exec 9>"$data_dir/.epoch-3-launch.lock"
flock -n 9 || exit 0

if [[ -e "$sentinel" ]]; then
  exit 0
fi

cd "$repo"

last_epoch="$(tail -n 1 "$manager_log" | cut -d, -f2)"
last_brief="$(tail -n 1 "$manager_log" | cut -d, -f3)"
if [[ "$last_epoch" == 3 && "$last_brief" == 126 ]]; then
  touch "$sentinel"
  exit 0
fi
if ! { [[ "$last_epoch" == 2 && "$last_brief" == 121 ]] || \
       [[ "$last_epoch" == 3 && "$last_brief" -ge 122 && "$last_brief" -le 125 ]]; }; then
  printf '[%s] refusing unexpected manager tail epoch=%s brief=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$last_epoch" "$last_brief" >> "$log"
  exit 1
fi

if [[ "$last_epoch" == 2 ]]; then
  status="$(autocuda status --data-dir "$data_dir" --tag "$tag")"
  if [[ "$(jq -r '.progress.new_epoch' <<<"$status")" != true ]]; then
    printf '[%s] cooldown still closed\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$log"
    exit 0
  fi
fi

common=(--data-dir "$data_dir" --tag "$tag" --brief-kind macro --min-macro .25 --min-micro 0 --min-combine 0 --min-simplify 0)

if [[ "$last_brief" -lt 122 ]]; then
  autocuda log optimize-tree brief "${common[@]}" --new-epoch \
    --parent-commit 52427ff70e52ec6262c50b0bdbdbf9531ac09e42 \
    --description "Build a genuinely different shape-dispatched Cholesky from the baseline around compensated FP8/BF16 tensor-core products: keep FP32 POTRF/TRSM on diagonal frontiers, use one block-scaled FP8 product for far trailing tiles, add bounded residual terms only on next-diagonal/frontier tiles, and use communication-avoiding recursive supernodes with selective FP32 repair across the full benchmark."
  last_brief=122
fi

if [[ "$last_brief" -lt 123 ]]; then
  autocuda log optimize-tree brief "${common[@]}" \
    --parent-commit fd956a9d484692c4dc7a4ec20d94bc371d50e7ed \
    --description "Replace the leader's low-batch library islands at batch16/n512, batch4/n1024, batch2/n2048, and batch1/n4096 with one shape-scoped recursive portfolio. Reintegrate the current-run staggered copy/factor relay, fused batch4 first wave, and batch8-only batched graph while preserving the leader's neighboring dispatches and ordered fresh-input state."
  last_brief=123
fi

if [[ "$last_brief" -lt 124 ]]; then
  autocuda log optimize-tree brief "${common[@]}" \
    --parent-commit 0b18f368ac286b8646e796cd0736552c48e11a41 \
    --description "Rebuild both n512 regimes as coarse two-CTA SM100 clusters: one factor/panel owner and one update owner share diagonal and panel tiles through DSM or cluster multicast, retain correction accumulators in TMEM, and keep compact LD32 global sidecars without row-wise expansion or fine-grained global task queues."
  last_brief=124
fi

if [[ "$last_brief" -lt 125 ]]; then
  autocuda log optimize-tree brief "${common[@]}" \
    --parent-commit fd956a9d484692c4dc7a4ec20d94bc371d50e7ed \
    --description "Replace the low-batch n2048/n4096 graph frontier with correctness-complete next-diagonal dependency counters. Count every Schur contribution to the next diagonal tile, publish a diagonal-ready generation before off-diagonal completion, factor it immediately, and use 128x128 tcgen05 updates with per-matrix barriers rather than whole-grid synchronization."
  last_brief=125
fi

if [[ "$last_brief" -lt 126 ]]; then
  autocuda log optimize-tree brief "${common[@]}" \
    --parent-commit 96be23f43abd103dc13f7ebd3229d1715b939be8 \
    --description "Replace large-shape FP16-history trailing products with compensated FP8 or BF16 two/three-term lower-triangular products while retaining FP32 POTRF/TRSM. Specialize n8192/n16384/n32768 schedules, keep the 4096 supernode interface, use a single block-scaled FP8 product on far tiles, and spend extra decomposition terms only on the next diagonal and frontier so the dominant products retain Blackwell's higher-throughput mode."
fi

tail_epoch="$(tail -n 1 "$manager_log" | cut -d, -f2)"
tail_brief="$(tail -n 1 "$manager_log" | cut -d, -f3)"
if [[ "$tail_epoch" != 3 || "$tail_brief" != 126 ]]; then
  printf '[%s] incomplete launch tail epoch=%s brief=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$tail_epoch" "$tail_brief" >> "$log"
  exit 1
fi

touch "$sentinel"
printf '[%s] logged epoch 3 briefs 122-126\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$log"
