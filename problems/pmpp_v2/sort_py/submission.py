"""
Warp-merge sort via cub::BlockRadixSort + pairwise merge tree.
Targets brief objective #3: warp-merge sort with cub::WarpMergeSort
for local sort, eliminating global histogram for sub-problems.

Approach:
1. Block-level sort: cub::BlockRadixSort sorts 2048 items per block
   (256 threads, 8 items/thread) → 1 kernel launch
2. Pairwise merge tree: log2(num_blocks) levels of merge kernels
   Each merge kernel merges two adjacent sorted runs into one.
   Uses merge-path for parallel merge within each block.

Total kernel launches: 1 + log2(num_blocks) (vs parent's 12).
For 100M: 1 + 16 = 17 launches.
For 100K: 1 + 6 = 7 launches.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

radix_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cstdint>
#include <cstdio>

constexpr int ITEMS_PER_THREAD = 8;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int ITEMS_PER_BLOCK = ITEMS_PER_THREAD * THREADS_PER_BLOCK;  // 2048
constexpr int RADIX_BITS = 8;

// Persistent scratch memory
static torch::Tensor d_scratch0 = {};
static torch::Tensor d_scratch1 = {};

void init_persistent() {
    int64_t n = 100'000'000;
    d_scratch0 = torch::empty({n},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    d_scratch1 = torch::empty({n},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

//------------------------------------------------------------------------------
// Kernel 1: Block-level radix sort using cub::BlockRadixSort.
// Each block sorts its ITEMS_PER_BLOCK items in-place.
// Sorted output is written to d_out.
//------------------------------------------------------------------------------
__global__ void block_radix_sort_kernel(
    const uint32_t* __restrict__ d_in,
    uint32_t* __restrict__ d_out,
    int64_t num_items,
    int num_blocks)
{
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int64_t block_start = (int64_t)block_id * ITEMS_PER_BLOCK;

    // Load items into registers
    uint32_t items[ITEMS_PER_THREAD];
    int valid_count = 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int64_t idx = block_start + threadIdx.x + (int64_t)i * THREADS_PER_BLOCK;
        if (idx < num_items) {
            items[i] = d_in[idx];
            valid_count++;
        } else {
            items[i] = 0xFFFFFFFFu;  // sentinel: largest possible uint32
        }
    }

    // BlockRadixSort: sorts ITEMS_PER_THREAD items per thread, THREADS_PER_BLOCK threads
    using BlockRadixSortT = cub::BlockRadixSort<uint32_t, THREADS_PER_BLOCK, ITEMS_PER_THREAD>;
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;
    BlockRadixSortT(temp_storage).Sort(items);

    // Write sorted items to output. After BlockRadixSort, thread t holds
    // elements at sorted ranks [t*8, t*8+1, ..., t*8+7]. Write sequentially.
    int64_t out_base = block_start + (int64_t)threadIdx.x * ITEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int64_t out_idx = out_base + (int64_t)i;
        if (out_idx < num_items && items[i] != 0xFFFFFFFFu) {
            d_out[out_idx] = items[i];
        }
    }
}

//------------------------------------------------------------------------------
// Kernel 2: Merge kernel using standard merge-path decomposition.
// Reference: "Merge Path - A Visually Intuitive Approach to Parallel Merging"
// (Green, McColl, Bader). Each thread handles its assigned output slice
// after computing start positions via binary search.
//------------------------------------------------------------------------------
__global__ void merge_kernel(
    const uint32_t* __restrict__ d_in,
    uint32_t* __restrict__ d_out,
    int64_t num_items,
    int64_t stride,
    int num_merges)
{
    int merge_id = blockIdx.x;
    if (merge_id >= num_merges) return;

    int64_t half = stride / 2;
    int64_t A_start = (int64_t)merge_id * stride;
    int64_t B_start = A_start + half;
    int64_t B_end = (merge_id + 1) * stride;
    int64_t A_end = B_start;
    if (A_end > num_items) A_end = num_items;
    if (B_end > num_items) B_end = num_items;
    int64_t lenA = (A_end > A_start) ? (A_end - A_start) : 0;
    int64_t lenB = (B_end > B_start) ? (B_end - B_start) : 0;
    int64_t total = lenA + lenB;
    if (total == 0) return;

    // Each thread handles a slice of the merged output.
    int64_t diag = (int64_t)threadIdx.x * total / THREADS_PER_BLOCK;
    int64_t next_diag = (int64_t)(threadIdx.x + 1) * total / THREADS_PER_BLOCK;
    if (diag >= total) return;

    // Standard merge-path: find a = number of A elements in merged prefix[0..diag).
    // Compares A[a] (first A NOT in prefix) vs B[diag-1-a] (last B IN prefix).
    int64_t a_lo = diag > lenB ? diag - lenB : 0;  // max(0, diag-lenB)
    int64_t a_hi = diag < lenA ? diag : lenA;      // min(diag, lenA)
    while (a_lo < a_hi) {
        int64_t a = a_lo + (a_hi - a_lo) / 2;
        int64_t b = diag - 1 - a;                 // index of last B element in prefix
        if (b < 0 || d_in[A_start + a] <= d_in[B_start + b]) {
            a_lo = a + 1;  // can include A[a]
        } else {
            a_hi = a;
        }
    }
    int64_t a_idx = a_lo;
    int64_t b_idx = diag - a_lo;

    // Linear merge for this thread's assigned output slice
    for (int64_t out_pos = A_start + diag; out_pos < A_start + next_diag; out_pos++) {
        bool take_a = (a_idx < lenA && (b_idx >= lenB || d_in[A_start + a_idx] <= d_in[B_start + b_idx]));
        d_out[out_pos] = take_a ? d_in[A_start + (a_idx++)] : d_in[B_start + (b_idx++)];
    }
}

//------------------------------------------------------------------------------
// Top-level: block sort + merge tree
//------------------------------------------------------------------------------
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    int64_t num_items = input.numel();
    int blocks = (int)((num_items + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const uint32_t* d_in = reinterpret_cast<const uint32_t*>(input.const_data_ptr<float>());
    uint32_t* d_out = reinterpret_cast<uint32_t*>(output.data_ptr<float>());
    uint32_t* d_s0 = reinterpret_cast<uint32_t*>(d_scratch0.data_ptr<int32_t>());
    uint32_t* d_s1 = reinterpret_cast<uint32_t*>(d_scratch1.data_ptr<int32_t>());

    // Step 1: Block-level sort → d_s0
    block_radix_sort_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_in, d_s0, num_items, blocks);

    // Step 2: Merge tree
    // Start with runs of ITEMS_PER_BLOCK elements, double stride each level
    int64_t stride = ITEMS_PER_BLOCK * 2;  // after first merge, runs are 2*ITEMS_PER_BLOCK
    const uint32_t* src = d_s0;
    uint32_t* dst = d_s1;
    int level = 0;

    while (stride / 2 < num_items) {
        int merges = (int)((num_items + stride - 1) / stride);
        merge_kernel<<<merges, THREADS_PER_BLOCK, 0, stream>>>(
            src, dst, num_items, stride, merges);

        // Swap buffers for next level
        const uint32_t* tmp_src = src;
        src = dst;
        dst = const_cast<uint32_t*>(tmp_src);
        stride *= 2;
        level++;
    }

    // Final result: copy from src to output if needed
    if (src != d_out) {
        cudaMemcpyAsync(d_out, src, num_items * sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice, stream);
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error after warp-merge sort: %s\n", cudaGetErrorString(err));
    }

    return output;
}
"""

radix_cpp = r"""
#include <torch/extension.h>

void init_persistent();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

radix_module = load_inline(
    name='custom_warp_merge_sort',
    cpp_sources=radix_cpp,
    cuda_sources=radix_source,
    functions=['sort_cuda', 'init_persistent'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

radix_module.init_persistent()


def custom_kernel(data: input_t) -> output_t:
    """
    Warp-merge sort: cub::BlockRadixSort for block-level sort + pairwise
    merge tree for global ordering. Eliminates global histogram.
    """
    input_tensor, output_tensor = data
    radix_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor