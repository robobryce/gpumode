"""
BlockRadixSort with grid-stride loop + DeviceMergeSort merge.
Each block sorts 2048 float32 items using CUB BlockRadixSort (int32 bitcast),
producing globally-sorted runs. DeviceMergeSort merges runs into final sorted output.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cstdint>
#include <algorithm>

constexpr int BLOCK_THREADS = 256;
constexpr int ITEMS_PER_THREAD = 8;
constexpr int CHUNK_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;  // 2048

using BlockRadixSortT = cub::BlockRadixSort<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>;

// Block-level radix sort with grid-stride loop.
// Each block cooperatively sorts CHUNK_SIZE items at a time using shared memory.
// Writes sorted chunks in-place into the data array.
__global__ void block_sort_kernel(float* data, int64_t n) {
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;

    int32_t thread_keys[ITEMS_PER_THREAD];

    for (int64_t chunk_start = blockIdx.x * static_cast<int64_t>(CHUNK_SIZE);
         chunk_start < n;
         chunk_start += gridDim.x * static_cast<int64_t>(CHUNK_SIZE)) {

        int64_t chunk_items = min(static_cast<int64_t>(CHUNK_SIZE), n - chunk_start);

        // Load items from global memory, pad with INT_MAX sentinel
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int64_t idx = chunk_start + threadIdx.x + static_cast<int64_t>(i) * BLOCK_THREADS;
            if (idx < chunk_start + chunk_items) {
                // Positive floats: raw IEEE 754 bits preserve sort order
                thread_keys[i] = __float_as_int(data[idx]);
            } else {
                thread_keys[i] = 2147483647;  // INT_MAX sentinel
            }
        }

        // Cooperative sort via shared memory
        BlockRadixSortT(temp_storage).Sort(thread_keys);

        // Write sorted items back to global memory
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int chunk_local_rank = threadIdx.x + i * BLOCK_THREADS;
            if (chunk_local_rank < chunk_items) {
                data[chunk_start + chunk_local_rank] = __int_as_float(thread_keys[i]);
            }
        }

        __syncthreads();
    }
}

// Persistent temp storage, allocated once at module init
static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void init_temp_storage() {
    if (persistent_temp.defined()) return;

    int64_t max_n = 100'000'000;

    // Query temp storage for DeviceMergeSort on float keys
    float* dummy = nullptr;
    cub::DeviceMergeSort::SortKeys(
        nullptr, persistent_temp_bytes,
        dummy, static_cast<int64_t>(max_n),
        cub::Less{});

    // Add 15% padding
    persistent_temp_bytes = (persistent_temp_bytes * 23 + 19) / 20;

    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Step 1: Copy input to output (we'll sort in-place in output)
    cudaMemcpyAsync(
        output.data_ptr<float>(),
        input.const_data_ptr<float>(),
        num_items * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream);

    if (num_items > 0) {
        // Step 2: Block-level sort chunks
        int num_chunks = (num_items + CHUNK_SIZE - 1) / CHUNK_SIZE;
        int grid_blocks = min(num_chunks, 228);  // B200 has 228 SMs, use all of them

        block_sort_kernel<<<grid_blocks, BLOCK_THREADS, 0, stream>>>(
            output.data_ptr<float>(), num_items);

        // Step 3: Merge-sort the partially-sorted output to produce globally sorted result
        size_t temp_bytes = persistent_temp_bytes;
        cub::DeviceMergeSort::SortKeys(
            persistent_temp.data_ptr(), temp_bytes,
            output.data_ptr<float>(), num_items,
            cub::Less{}, stream);
    }

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_temp_storage();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='block_radix_sort_merge',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_temp_storage'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_temp_storage()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB BlockRadixSort (grid-stride) + DeviceMergeSort merge.
    Each block sorts 2048 items in smem, then DeviceMergeSort merges runs.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor