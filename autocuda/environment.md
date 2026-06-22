# Environment

## Timeouts

- Full build from clean: ~1 minute for the first import/runtime compile; many submissions are pure PyTorch and build in ~1 second.
- Incremental rebuild after a source change: ~1-45 seconds depending on whether `torch.utils.cpp_extension` recompiles CUDA.
- Test suite: ~3 minutes budget (`task.yml` test timeout is 180 seconds).
- Single benchmark run: ~3 minutes budget (`task.yml` benchmark timeout is 180 seconds).
- Benchmark under `nsys`: ~5 minutes budget; use the profile-mode harness command below.
- Benchmark under `ncu`: ~10 minutes budget; filter/limit launches if full capture is too slow.

## GPU and environment

- GPU: NVIDIA B200, compute capability 10.0.
- Driver: 580.126.09.
- CUDA toolkit: `/usr/local/cuda`, `nvcc` 12.8.93.
- Python: `/home/shadeform/gpumode/.venv/bin/python` from `~/.config/gpumode/gpumode.env`.
- Host CPUs: 30 logical CPUs.
- Build model: GPU MODE Python submissions compile CUDA at runtime through PyTorch extensions; `harness/env.sh` sets `MAX_JOBS=3` for four concurrent workers.
- Required env config: source `~/.config/gpumode/gpumode.env` through the harness scripts; do not set `GPUMODE_PROBLEM` manually.

## Profiling

- nsys command: `nsys profile --stats=true --force-overwrite=true -o <output-without-extension> bash harness/profile_ncu.sh <set>/<problem>`.
- ncu command: `ncu --set full --force-overwrite -o <output-without-extension> bash harness/profile_ncu.sh <set>/<problem>`.
- Remediations applied: None recorded yet; if profiler permissions fail, retry the same invocation through `sudo -E` as described in the optimize-tree profiler reference.
- Environment variables: use the harness defaults from `~/.config/gpumode/gpumode.env`; profile through `harness/profile_ncu.sh` so the generated profile spec is used.
