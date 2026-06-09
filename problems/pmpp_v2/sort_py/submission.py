"""
BlockRadixSort (256t x 8 = 2048/block) grid-stride + pairwise merge tree.
Phase 1: block-sort chunks in-place (1R+1W per element).
Phase 2: pairwise merge tree via cub::DeviceMerge::MergeKeys (no values).
All temp storage allocated once at module init.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_merge.cuh>
#include <cstdint>
#include <algorithm>

struct FloatLess {
    __device__ bool operator()(float a, float b) const { return a < b; }
};

constexpr int BLOCK_THREADS = 256;
constexpr int ITEMS_PER_THREAD = 8;
constexpr int CHUNK_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;  // 2048

using BlockRadixSortT = cub::BlockRadixSort<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>;

__global__ void block_sort_kernel(float* data, int64_t n) {
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;
    int32_t thread_keys[ITEMS_PER_THREAD];

    for (int64_t chunk_start = static_cast<int64_t>(blockIdx.x) * CHUNK_SIZE;
         chunk_start < n;
         chunk_start += static_cast<int64_t>(gridDim.x) * CHUNK_SIZE) {

        int64_t chunk_items = n - chunk_start;
        if (chunk_items > CHUNK_SIZE) chunk_items = CHUNK_SIZE;

        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int64_t idx = chunk_start + threadIdx.x + static_cast<int64_t>(i) * BLOCK_THREADS;
            if (idx < chunk_start + chunk_items) {
                thread_keys[i] = __float_as_int(data[idx]);
            } else {
                thread_keys[i] = 2147483647;
            }
        }

        BlockRadixSortT(temp_storage).Sort(thread_keys);

        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int rank = threadIdx.x + i * BLOCK_THREADS;
            if (rank < chunk_items) {
                data[chunk_start + rank] = __int_as_float(thread_keys[i]);
            }
        }
        __syncthreads();
    }
}

static torch::Tensor merge_temp_storage = {};
static size_t merge_temp_bytes = 0;
static torch::Tensor scratch_buf = {};
static torch::Tensor merge_buf = {};

void init_temp_storage() {
    if (merge_temp_storage.defined()) return;
    int64_t max_n = 100'000'000;

    size_t bytes = 0;
    float* d = nullptr;

    // Query temp for DeviceMerge on largest pair (two halves of max_n)
    cub::DeviceMerge::MergeKeys(
        nullptr, bytes,
        d, static_cast<int>(max_n/2),
        d, static_cast<int>(max_n - max_n/2),
        d,
        FloatLess{});

    bytes = (bytes * 11 + 9) / 10;
    merge_temp_bytes = bytes;
    merge_temp_storage = torch::empty(
        {static_cast<int64_t>(bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    scratch_buf = torch::empty(
        {max_n},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    merge_buf = torch::empty(
        {max_n},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
}

void do_merge_tree(
    float* src_buf, float* dst_buf, float* output_buf,
    int64_t n, int num_chunks, int chunk_size,
    cudaStream_t stream)
{
    float* src = src_buf;
    float* dst = dst_buf;
    int cur_chunks = num_chunks;
    int cur_chunk_sz = chunk_size;

    while (cur_chunks > 1) {
        int next_chunks = (cur_chunks + 1) / 2;
        int pairs = cur_chunks / 2;

        for (int p = 0; p < pairs; p++) {
            int64_t base1 = static_cast<int64_t>(p * 2) * cur_chunk_sz;
            int64_t base2 = static_cast<int64_t>(p * 2 + 1) * cur_chunk_sz;
            int64_t base_out = static_cast<int64_t>(p * 2) * cur_chunk_sz;

            int n1 = (base1 + cur_chunk_sz <= n) ? cur_chunk_sz : static_cast<int>(n - base1);
            int n2 = (base2 + cur_chunk_sz <= n) ? cur_chunk_sz : static_cast<int>(n - base2);
            if (n2 < 0) n2 = 0;
            if (n1 < 0) n1 = 0;

            if (n1 > 0 && n2 > 0) {
                size_t tb = merge_temp_bytes;
                cub::DeviceMerge::MergeKeys(
                    merge_temp_storage.data_ptr(), tb,
                    src + base1, n1,
                    src + base2, n2,
                    dst + base_out,
                    FloatLess{},
                    stream);
            } else if (n1 > 0) {
                cudaMemcpyAsync(dst + base_out, src + base1,
                    n1 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
        }

        // Copy last odd chunk
        if (cur_chunks % 2 == 1) {
            int last = cur_chunks - 1;
            int64_t base = static_cast<int64_t>(last) * cur_chunk_sz;
            int64_t sz = n - base;
            if (sz > 0) {
                cudaMemcpyAsync(dst + base, src + base,
                    sz * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
        }

        float* tmp = src;
        src = dst;
        dst = tmp;
        cur_chunks = next_chunks;
        cur_chunk_sz *= 2;
    }

    // Result is in 'src'
    if (src != output_buf) {
        cudaMemcpyAsync(output_buf, src,
            n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto n = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (n == 0) return output;

    // Copy input -> scratch
    cudaMemcpyAsync(scratch_buf.data_ptr<float>(),
        input.const_data_ptr<float>(),
        n * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // Block-sort chunks in scratch
    int num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int grid = num_chunks < 228 ? num_chunks : 228;

    block_sort_kernel<<<grid, BLOCK_THREADS, 0, stream>>>(
        scratch_buf.data_ptr<float>(), n);

    // Merge tree: scratch -> output
    do_merge_tree(
        scratch_buf.data_ptr<float>(),
        merge_buf.data_ptr<float>(),
        output.data_ptr<float>(),
        n, num_chunks, CHUNK_SIZE, stream);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_temp_storage();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='block_radix_sort_merge_tree_v2',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_temp_storage'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    extra_cuda_cflags=['--expt-relaxed-constexpr'],
    verbose=False,
)

sort_module.init_temp_storage()


def custom_kernel(data: input_t) -> output_t:
    """BlockRadixSort + pairwise merge tree."""
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor