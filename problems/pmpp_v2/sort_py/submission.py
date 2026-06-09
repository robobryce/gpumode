"""
Per-row BlockRadixSort + global merge via DeviceRadixSort.
Input is a flattened sqrt(N)xsqrt(N) matrix where each row is randn centered
at (seed+i). Each row is sorted via BlockRadixSort in shared memory.
Then a global DeviceRadixSort merges all row-sorted results.
"""
import math
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cpp_source = """
#include <torch/extension.h>
void sort_rows_init(int64_t max_items);
torch::Tensor sort_rows(torch::Tensor input, torch::Tensor output, torch::Tensor temp);
"""

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <cfloat>

// BlockRadixSort: 256 threads x 32 items = 8192 items/block (power of 2)
constexpr int BLOCK_THREADS = 256;
constexpr int ITEMS_PER_THREAD = 32;
constexpr int BLOCK_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;  // 8192
using BlockSortT = cub::BlockRadixSort<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>;

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void sort_rows_init(int64_t max_items) {
    if (persistent_temp.defined()) return;
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int64_t>(max_items),
        0, 32);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

// BlockRadixSort kernel: each block sorts one chunk of a row.
// Input: blocked arrangement - thread t owns [t*ITEMS_PER_THREAD, (t+1)*ITEMS_PER_THREAD-1].
// Output: blocked ascending - thread 0 holds smallest ITEMS_PER_THREAD keys.
__global__ void sort_rows_kernel(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int64_t num_rows,
    int64_t row_stride,
    int64_t* row_counts,
    int64_t blocks_per_row)
{
    int global_block_id = blockIdx.x;
    int row = global_block_id / blocks_per_row;
    int block_in_row = global_block_id % blocks_per_row;

    if (row >= num_rows) return;

    int64_t row_count = row_counts[row];
    int64_t row_offset = int64_t(row) * row_stride;
    int64_t block_start = int64_t(block_in_row) * BLOCK_ITEMS;
    const int32_t* row_in = input + row_offset;
    int32_t* row_out = output + row_offset;

    int32_t keys[ITEMS_PER_THREAD];
    int tid = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        int64_t idx = block_start + int64_t(tid) * ITEMS_PER_THREAD + j;
        keys[j] = (idx < row_count) ? row_in[idx] : INT_MAX;
    }

    __shared__ typename BlockSortT::TempStorage temp;
    BlockSortT(temp).Sort(keys);

    #pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        int64_t idx = block_start + int64_t(tid) * ITEMS_PER_THREAD + j;
        if (idx < row_count) {
            row_out[idx] = keys[j];
        }
    }
}

torch::Tensor sort_rows(torch::Tensor input, torch::Tensor output, torch::Tensor temp) {
    int64_t N = input.numel();
    int64_t num_rows = static_cast<int64_t>(std::sqrt(static_cast<double>(N)));
    if (num_rows < 1) num_rows = 1;
    int64_t row_stride = (N + num_rows - 1) / num_rows;
    int64_t blocks_per_row = (row_stride + BLOCK_ITEMS - 1) / BLOCK_ITEMS;

    auto input_c = input.contiguous();
    const int32_t* in_ptr = reinterpret_cast<const int32_t*>(
        input_c.const_data_ptr<float>());
    int32_t* temp_ptr = reinterpret_cast<int32_t*>(temp.data_ptr<float>());
    int32_t* out_ptr = reinterpret_cast<int32_t*>(output.data_ptr<float>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Compute per-row element counts on host, copy to device
    std::vector<int64_t> row_counts_h(num_rows);
    for (int64_t r = 0; r < num_rows - 1; r++) {
        row_counts_h[r] = row_stride;
    }
    row_counts_h[num_rows - 1] = N - (num_rows - 1) * row_stride;

    auto row_counts_t = torch::from_blob(
        row_counts_h.data(), {num_rows},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto row_counts_d = row_counts_t.to(torch::kCUDA);
    int64_t* row_counts_ptr = reinterpret_cast<int64_t*>(row_counts_d.data_ptr());

    // Phase 1: BlockRadixSort per row
    int64_t total_blocks = num_rows * blocks_per_row;
    sort_rows_kernel<<<total_blocks, BLOCK_THREADS, 0, stream>>>(
        in_ptr, temp_ptr, num_rows, row_stride,
        row_counts_ptr, blocks_per_row);

    // Phase 2: Global CUB DeviceRadixSort interleaves the row-sorted chunks
    size_t tb = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), tb,
        temp_ptr, out_ptr, N,
        0, 32, stream);

    return output;
}
"""

sort_module = load_inline(
    name='sort_rows_blockradix',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_rows', 'sort_rows_init'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.sort_rows_init(100_000_000)

_temp_tensor = None


def custom_kernel(data: input_t) -> output_t:
    """
    Phase 1: Per-row BlockRadixSort in shared memory (256x32=8192).
    Phase 2: Global CUB DeviceRadixSort merges row-sorted data.
    """
    global _temp_tensor
    input_tensor, output_tensor = data
    N = input_tensor.numel()
    if _temp_tensor is None or _temp_tensor.numel() < N:
        _temp_tensor = torch.empty(N, dtype=torch.float32, device='cuda')
    sort_module.sort_rows(input_tensor, output_tensor, _temp_tensor)
    return output_tensor
