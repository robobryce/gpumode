"""sort_v2 submission - GPU MODE leaderboard `sort_v2`.

Problem: sort a 1-D float32 tensor ascending, matching torch.sort exactly.

Strategy (COUNTING SORT over a narrow integer range). The ranked 100M input is
positive floats spanning a NARROW value range (~6246-16257), so after
reinterpreting as int32 (positive floats are monotonic as signed int32) and
subtracting the device min, every key fits in a 24-bit integer range (< 2^24
distinct values). We then sort by COUNTING, not comparison/radix:

  1. device min/max of the int32 keys (one fused pass);
  2. if (max-min) < 2^24 and min >= 0  ->  counting-sort fast path:
       a. histogram the (key-min) values into a 2^24-bin counter array;
       b. exclusive prefix-sum the histogram (bin start offsets);
       c. reconstruct: write each value (bin+min) in its contiguous output run.
     This moves only ~2*N*4 bytes of key traffic (vs ~6*N*4 for a 3-pass radix),
     and the output writes are contiguous/coalesced (NO data scatter), so it is
     markedly faster than radix on the ranked shape.
  3. otherwise (wide range, negatives, denormal-spanning, inf/nan) ->  universal
     fallback: a full-width 32-bit cub::DeviceRadixSort, bit-exact for EVERY
     float (CUB applies the IEEE->sortable transform internally).

Correctness: the fast path is exact because within a non-negative narrow range
the int32 value IS the sort key and each histogram bin is an EXACT value, so the
reconstruction reproduces torch.sort's ascending order with all duplicates. The
range/sign gate is recomputed every call from the actual data (no cached pivot),
so it is immune to allocator pointer reuse. All device scratch (the 2^24-bin
histogram = 64 MB, temp, min/max) is pre-allocated ONCE at import (sort_init),
so no timed call pays an allocation cost. All work is library calls
(cub::DeviceScan / cub::DeviceRadixSort) plus three plain <<<>>> kernels on the
default execution path, with no per-call allocation.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

CS_BITS = 24            # counting-sort bin count = 2^CS_BITS
CS_FAST_GATE = 2_000_000      # below this, full sort is already instant
CS_MAX_N = 100_000_000

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef CS_BITS
#define CS_BITS 24
#endif
#ifndef CS_FAST_GATE
#define CS_FAST_GATE 2000000
#endif
#ifndef CS_MAX_N
#define CS_MAX_N 100000000LL
#endif
#define NBINS (1u << CS_BITS)

// ---- min/max of int32 keys (int4-vectorized, fused) -------------------------
__global__ void minmax_kernel(const int32_t* __restrict__ d, int n,
                              int* __restrict__ out /*[min,max]*/) {
  __shared__ int smn, smx;
  if (threadIdx.x == 0) { smn = 0x7fffffff; smx = (int)0x80000000; }
  __syncthreads();
  int lmn = 0x7fffffff, lmx = (int)0x80000000;
  const long n4 = n / 4;
  const int4* d4 = reinterpret_cast<const int4*>(d);
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += (long)gridDim.x * blockDim.x) {
    int4 v = d4[i];
    int mn = min(min(v.x, v.y), min(v.z, v.w));
    int mx = max(max(v.x, v.y), max(v.z, v.w));
    if (mn < lmn) lmn = mn;
    if (mx > lmx) lmx = mx;
  }
  if (blockIdx.x == 0) {
    for (long i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x) {
      int v = d[i]; if (v < lmn) lmn = v; if (v > lmx) lmx = v;
    }
  }
  atomicMin(&smn, lmn); atomicMax(&smx, lmx);
  __syncthreads();
  if (threadIdx.x == 0) { atomicMin(&out[0], smn); atomicMax(&out[1], smx); }
}

// ---- histogram of (key - min), int4-vectorized loads ------------------------
__global__ void hist_kernel(const int32_t* __restrict__ d, int n, int mn,
                            unsigned int* __restrict__ hist) {
  const long n4 = n / 4;
  const int4* d4 = reinterpret_cast<const int4*>(d);
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += (long)gridDim.x * blockDim.x) {
    int4 v = d4[i];
    atomicAdd(&hist[(unsigned)(v.x - mn)], 1u);
    atomicAdd(&hist[(unsigned)(v.y - mn)], 1u);
    atomicAdd(&hist[(unsigned)(v.z - mn)], 1u);
    atomicAdd(&hist[(unsigned)(v.w - mn)], 1u);
  }
  if (blockIdx.x == 0) {
    for (long i = n4 * 4 + threadIdx.x; i < n; i += blockDim.x)
      atomicAdd(&hist[(unsigned)(d[i] - mn)], 1u);
  }
}

// ---- reconstruct: write value (bin+min) into out[prefix[b], prefix[b+1]) ----
// int4-vectorized run writes for long runs (duplicates) -> fewer store insns.
__global__ void reconstruct_kernel(const unsigned int* __restrict__ prefix,
                                   unsigned nbins, int mn,
                                   int32_t* __restrict__ out) {
  for (long b = (long)blockIdx.x * blockDim.x + threadIdx.x; b < (long)nbins;
       b += (long)gridDim.x * blockDim.x) {
    unsigned start = prefix[b], end = prefix[b + 1];
    if (end <= start) continue;
    int val = (int)b + mn;
    unsigned p = start;
    while ((p & 3u) && p < end) { out[p] = val; p++; }
    int4 v4 = make_int4(val, val, val, val);
    for (; p + 4 <= end; p += 4) *reinterpret_cast<int4*>(out + p) = v4;
    for (; p < end; p++) out[p] = val;
  }
}

// ---- persistent scratch (pre-allocated at import) ---------------------------
static void*         g_temp       = nullptr;
static size_t        g_temp_bytes = 0;
static unsigned int* g_hist       = nullptr;   // NBINS+1
static int*          g_mm         = nullptr;   // [min,max]
static int           g_ready      = 0;

static inline void ensure_temp(size_t need){ if(need>g_temp_bytes){ if(g_temp)cudaFree(g_temp); cudaMalloc(&g_temp,need); g_temp_bytes=need; } }

static void sort_setup() {
  if (g_ready) return;
  cudaFree(0);
  size_t nf = 0, ns = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, nf, (const float*)nullptr,
                                 (float*)nullptr, (int)CS_MAX_N, 0, 32);
  cub::DeviceScan::ExclusiveSum(nullptr, ns, (unsigned*)nullptr,
                                (unsigned*)nullptr, (int)NBINS + 1);
  ensure_temp(nf > ns ? nf : ns);
  if (!g_hist) cudaMalloc(&g_hist, ((size_t)NBINS + 1) * sizeof(unsigned));
  if (!g_mm)   cudaMalloc(&g_mm, 2 * sizeof(int));
  cudaDeviceSynchronize();
  g_ready = 1;
}
void sort_init() { sort_setup(); }

static inline void full_sort(const float* d_in, float* d_out, int n) {
  size_t need = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, need, d_in, d_out, n, 0, 32);
  ensure_temp(need);
  cub::DeviceRadixSort::SortKeys(g_temp, need, d_in, d_out, n, 0, 32);
}

void sort_keys(torch::Tensor input, torch::Tensor output) {
  const int n = (int)input.numel();
  const float* d_in = input.data_ptr<float>();
  float* d_out = output.data_ptr<float>();
  if (n <= 0) return;
  sort_setup();

  if (n < (int)CS_FAST_GATE) { full_sort(d_in, d_out, n); return; }

  const int32_t* ki = reinterpret_cast<const int32_t*>(d_in);
  int32_t* ko = reinterpret_cast<int32_t*>(d_out);

  // device min/max
  int init[2] = {0x7fffffff, (int)0x80000000};
  cudaMemcpy(g_mm, init, 2 * sizeof(int), cudaMemcpyHostToDevice);
  minmax_kernel<<<2048, 256>>>(ki, n, g_mm);
  int mm[2];
  cudaMemcpy(mm, g_mm, 2 * sizeof(int), cudaMemcpyDeviceToHost);
  int mn = mm[0], mx = mm[1];

  // gate: non-negative (min >= 0 => sign bit 0 => int32 order == value order)
  // and range fits in NBINS distinct integer values.
  long range = (long)mx - (long)mn;
  if (mn < 0 || range < 0 || range >= (long)NBINS) {
    full_sort(d_in, d_out, n);
    return;
  }

  unsigned nbins_used = (unsigned)range + 1;  // bins [0 .. range]
  // histogram (key - min) -> g_hist[0 .. nbins_used)
  cudaMemset(g_hist, 0, ((size_t)nbins_used + 1) * sizeof(unsigned));
  hist_kernel<<<2048, 256>>>(ki, n, mn, g_hist);
  // exclusive prefix-sum over nbins_used+1 -> prefix[b] = start offset of bin b
  size_t need = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, need, g_hist, g_hist, (int)nbins_used + 1);
  ensure_temp(need);
  cub::DeviceScan::ExclusiveSum(g_temp, need, g_hist, g_hist, (int)nbins_used + 1);
  // reconstruct sorted output
  reconstruct_kernel<<<4096, 256>>>(g_hist, nbins_used, mn, ko);
}
"""

_CPP_SRC = (
    "void sort_keys(torch::Tensor input, torch::Tensor output);\n"
    "void sort_init();\n"
)

_mod = load_inline(
    name="sort_v2_counting",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["sort_keys", "sort_init"],
    extra_cuda_cflags=[
        "-O3",
        f"-DCS_BITS={CS_BITS}",
        f"-DCS_FAST_GATE={CS_FAST_GATE}",
        f"-DCS_MAX_N={CS_MAX_N}LL",
    ],
    verbose=False,
)

_mod.sort_init()


def custom_kernel(data: input_t) -> output_t:
    inp, output = data
    _mod.sort_keys(inp, output)
    return output
