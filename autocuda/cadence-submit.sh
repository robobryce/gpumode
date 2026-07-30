#!/usr/bin/env bash
set -u

export PATH=/home/shadeform/.local/bin:/home/shadeform/.local/aab-bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin

data_dir=/home/shadeform/.local/share/autocuda/gpumode/2026-07-19-01-44-06-cholesky-resumed-local
tag=2026-07-19-01-44-06-cholesky-resumed-local
worktree="$data_dir/worktrees/submission-fd956a9-20260729T151120Z"
log="$data_dir/cadence-submit.log"
expected_commit=fd956a9d484692c4dc7a4ec20d94bc371d50e7ed
expected_blob=59b1b4ba92925305e2cd7b407bc2cbf98b9ef9e7

cd "$worktree" || exit 1
printf '\n[%s] cadence submission attempt\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$log"
actual_commit="$(git rev-parse HEAD)"
actual_blob="$(git hash-object problems/linalg/cholesky_py/submission.py)"
if [[ "$actual_commit" != "$expected_commit" || "$actual_blob" != "$expected_blob" ]]; then
  printf 'source guard failed: commit=%s blob=%s\n' "$actual_commit" "$actual_blob" >> "$log"
  exit 1
fi
autocuda run slice --data-dir "$data_dir" --tag "$tag" -- \
  bash harness/submit.sh linalg/cholesky_py >> "$log" 2>&1
exit 0
