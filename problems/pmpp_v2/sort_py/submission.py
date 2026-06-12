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
SMEM_BINS = 12288       # privatized smem histogram window per block (48KB)

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

// ---- min/max of int32 keys (fused) ------------------------------------------
__global__ void minmax_kernel(const int32_t* __restrict__ d, int n,
                              int* __restrict__ out /*[min,max]*/) {
  __shared__ int smn, smx;
  if (threadIdx.x == 0) { smn = 0x7fffffff; smx = (int)0x80000000; }
  __syncthreads();
  int lmn = 0x7fffffff, lmx = (int)0x80000000;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (long)gridDim.x * blockDim.x) {
    int v = d[i];
    if (v < lmn) lmn = v;
    if (v > lmx) lmx = v;
  }
  atomicMin(&smn, lmn); atomicMax(&smx, lmx);
  __syncthreads();
  if (threadIdx.x == 0) { atomicMin(&out[0], smn); atomicMax(&out[1], smx); }
}

// ---- histogram of (key - min): privatized shared-memory sub-histogram -------
// The ranked data is generated row-major (each ~10000-long row ~ N(row_mean,1)),
// so a CONTIGUOUS chunk of the input clusters into a NARROW bin sub-range. Each
// block takes a contiguous chunk, finds its local [lo,hi] bin window, and if it
// fits SMEM_BINS accumulates into a smem histogram (fast smem atomics, no global
// contention), then merges only the non-empty smem bins to global with ONE
// atomic each. Chunks whose window exceeds SMEM_BINS fall back to direct global
// atomics. This removes most of the 100M global atomics that bottleneck the
// naive one-atomic-per-key histogram on the clustered distribution.
#ifndef SMEM_BINS
#define SMEM_BINS 12288
#endif
extern __shared__ unsigned int s_hist[];
__global__ void hist_kernel(const int32_t* __restrict__ d, int n, int mn,
                            unsigned int* __restrict__ hist, long chunk) {
  long lo_i = (long)blockIdx.x * chunk;
  long hi_i = lo_i + chunk; if (hi_i > n) hi_i = n;
  if (lo_i >= hi_i) return;

  // block-local min/max of the chunk's (key-mn) values (one read).
  unsigned lmin = 0xffffffffu, lmax = 0u;
  for (long i = lo_i + threadIdx.x; i < hi_i; i += blockDim.x) {
    unsigned b = (unsigned)(d[i] - mn);
    lmin = min(lmin, b); lmax = max(lmax, b);
  }
  // block reduce min/max
  for (int o = 16; o > 0; o >>= 1) {
    lmin = min(lmin, __shfl_down_sync(0xffffffffu, lmin, o));
    lmax = max(lmax, __shfl_down_sync(0xffffffffu, lmax, o));
  }
  __shared__ unsigned int rmn[32], rmx[32], s_base, s_span;
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) { rmn[w] = lmin; rmx[w] = lmax; }
  __syncthreads();
  if (w == 0) {
    int nw = (blockDim.x + 31) >> 5;
    unsigned a = (lane < nw) ? rmn[lane] : 0xffffffffu;
    unsigned b = (lane < nw) ? rmx[lane] : 0u;
    for (int o = 16; o > 0; o >>= 1) {
      a = min(a, __shfl_down_sync(0xffffffffu, a, o));
      b = max(b, __shfl_down_sync(0xffffffffu, b, o));
    }
    if (lane == 0) { s_base = a; s_span = b - a + 1u; }
  }
  __syncthreads();
  unsigned base = s_base, span = s_span;

  if (span <= (unsigned)SMEM_BINS) {
    for (int t = threadIdx.x; t < (int)span; t += blockDim.x) s_hist[t] = 0u;
    __syncthreads();
    for (long i = lo_i + threadIdx.x; i < hi_i; i += blockDim.x) {
      unsigned b = (unsigned)(d[i] - mn) - base;
      atomicAdd(&s_hist[b], 1u);
    }
    __syncthreads();
    for (int t = threadIdx.x; t < (int)span; t += blockDim.x) {
      unsigned c = s_hist[t];
      if (c) atomicAdd(&hist[base + t], c);
    }
  } else {
    // wide chunk (crosses many rows): direct global atomics.
    for (long i = lo_i + threadIdx.x; i < hi_i; i += blockDim.x)
      atomicAdd(&hist[(unsigned)(d[i] - mn)], 1u);
  }
}

// ---- reconstruct: write value (bin+min) into out[prefix[b], prefix[b+1]) ----
// One thread per bin, contiguous run write. (A position-parallel binary-search
// variant was measured SLOWER: 2M threads each binary-searching the 64MB prefix
// array thrash the cache far more than the per-bin loop's imbalance costs.)
__global__ void reconstruct_kernel(const unsigned int* __restrict__ prefix,
                                   unsigned nbins, int mn,
                                   int32_t* __restrict__ out) {
  for (long b = (long)blockIdx.x * blockDim.x + threadIdx.x; b < (long)nbins;
       b += (long)gridDim.x * blockDim.x) {
    unsigned start = prefix[b], end = prefix[b + 1];
    if (end > start) {
      int val = (int)b + mn;
      for (unsigned p = start; p < end; p++) out[p] = val;
    }
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
  cudaFuncSetAttribute(hist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)(SMEM_BINS * sizeof(unsigned)));
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
  {
    int nblk = 16384;                         // small contiguous chunks -> narrow
    long chunk = ((long)n + nblk - 1) / nblk; // bin window per block (fits smem)
    int smem = SMEM_BINS * (int)sizeof(unsigned);
    hist_kernel<<<nblk, 256, smem>>>(ki, n, mn, g_hist, chunk);
  }
  // exclusive prefix-sum over nbins_used+1 -> prefix[b] = start offset of bin b
  size_t need = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, need, g_hist, g_hist, (int)nbins_used + 1);
  ensure_temp(need);
  cub::DeviceScan::ExclusiveSum(g_temp, need, g_hist, g_hist, (int)nbins_used + 1);
  // reconstruct sorted output (one thread per bin, contiguous run)
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
        f"-DSMEM_BINS={SMEM_BINS}",
    ],
    verbose=False,
)

_mod.sort_init()


def custom_kernel(data: input_t) -> output_t:
    inp, output = data
    _mod.sort_keys(inp, output)
    return output

