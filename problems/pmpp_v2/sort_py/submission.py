"""
Warp-Level Register Bitonic Sort + torch.sort Merge.
Each warp of 32 threads sorts 256 items (8 items/thread) entirely in registers
using __shfl_xor_sync bitonic merge network. Grid-stride loop over all data.
Sorted chunks are then fully sorted via torch.sort (CUB Onesweep internally).

This eliminates shared-memory overhead — sort happens in registers during
the load phase, producing partially-sorted output tiles. The final torch.sort
merge then has reduced work since 256-item tiles are already sorted internally.
"""
import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t

ITEMS_PER_THREAD = 8
WARP_SIZE = 32
CHUNK_SIZE = WARP_SIZE * ITEMS_PER_THREAD  # 256

sort_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cstdio>

#define WARP_SIZE 32
#define ITEMS_PER_THREAD 8
#define CHUNK_SIZE (WARP_SIZE * ITEMS_PER_THREAD)

// ---------------------------------------------------------------------------
// Register bitonic sort of 8 items (ascending within thread)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void bitonic_sort_8_asc(float *r) {
    #pragma unroll
    for (int k = 2; k <= 8; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool asc = ((i & k) == 0);
                    float a = r[i], b = r[ixj];
                    if ((a > b) == asc) { r[i] = b; r[ixj] = a; }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Merge two sorted 8-item arrays. Store lower or upper half in 'a'.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void merge_8_split(float *a, const float *b, bool keep_lower) {
    float merged[16];
    int i = 0, j = 0;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        bool take_a = (i < 8) && (j >= 8 || a[i] <= b[j]);
        merged[k] = take_a ? a[i++] : b[j++];
    }
    int offset = keep_lower ? 0 : 8;
    #pragma unroll
    for (int m = 0; m < 8; m++) a[m] = merged[m + offset];
}

// ---------------------------------------------------------------------------
// Warp-level register bitonic sort using __shfl_xor_sync
// 32 threads, each with 8 items = 256 items per warp-sort
// Grid-stride loop: each warp processes one chunk per iteration
// 256 threads/block, 8 warps -> 8 chunks per block per iteration
// ---------------------------------------------------------------------------
__global__ void warp_bitonic_sort_kernel(
    const float * __restrict__ input,
    float * __restrict__ output,
    int64_t n,
    int num_chunks,
    int chunks_per_block) {

    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int warp_id = tid >> 5;

    float r[ITEMS_PER_THREAD];

    for (int cid = blockIdx.x * chunks_per_block + warp_id;
         cid < num_chunks;
         cid += gridDim.x * (blockDim.x >> 5) * chunks_per_block) {

        int64_t chunk_start = static_cast<int64_t>(cid) * CHUNK_SIZE;

        // Load items; pad partial chunks with +inf
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int64_t idx = chunk_start + lane_id * ITEMS_PER_THREAD + i;
            r[i] = (idx < n) ? input[idx] : INFINITY;
        }

        // Per-thread ascending sort
        bitonic_sort_8_asc(r);

        // Warp-level bitonic merge
        #pragma unroll
        for (int stage = 1; stage <= 5; stage++) {
            #pragma unroll
            for (int step = stage; step >= 1; step--) {
                int mask = 1 << (step - 1);
                float partner[ITEMS_PER_THREAD];
                #pragma unroll
                for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                    partner[i] = __shfl_xor_sync(0xFFFFFFFF, r[i], mask);
                }
                bool ascending = ((lane_id & (1 << stage)) == 0);
                bool lower_thread = ((lane_id & mask) == 0);
                bool keep_lower = (ascending == lower_thread);
                merge_8_split(r, partner, keep_lower);
            }
        }

        // Store sorted items
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int64_t idx = chunk_start + lane_id * ITEMS_PER_THREAD + i;
            if (idx < n) {
                output[idx] = r[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry: warp bitonic sort -> buf, then rely on torch.sort for final merge
// ---------------------------------------------------------------------------
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor buf) {
    int64_t n = input.numel();
    int num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    if (num_chunks < 1) num_chunks = 1;

    int threads_per_block = 256;
    int warps_per_block = threads_per_block >> 5;    // 8
    int chunks_per_block = warps_per_block;            // 8

    int blocks = (num_chunks + chunks_per_block - 1) / chunks_per_block;
    if (blocks < 1) blocks = 1;
    if (blocks > 65535) blocks = 65535;

    warp_bitonic_sort_kernel<<<blocks, threads_per_block>>>(
        input.const_data_ptr<float>(),
        buf.data_ptr<float>(),
        n, num_chunks, chunks_per_block);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after warp sort: %s\\n", cudaGetErrorString(err));
    }

    return buf;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor buf);
"""

sort_module = load_inline(
    name='sort_cuda_warp_bitonic_v2',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Phase 1: warp-level register bitonic sort (256-item chunks in registers)
    Phase 2: torch.sort (CUB Onesweep) to produce fully sorted output
    """
    input_tensor, output_tensor = data
    input_contig = input_tensor.contiguous()
    n = input_contig.numel()

    buf = torch.empty(n, dtype=torch.float32, device='cuda')
    sort_module.sort_cuda(input_contig, buf)
    output_tensor[...] = torch.sort(buf)[0]
    return output_tensor