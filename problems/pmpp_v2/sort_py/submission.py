"""
BlockRadixSort + batched tile merge kernel (single-kernel per tree level).
Block sort: 2048/block via BlockRadixSort (256 threads x 8 items/thread).
Merge tree: each pair of segments merged by one tile_merge_kernel block.
Key insight: block-level merge path with binary search + serial merge segment.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_radix_sort.cuh>
#include <cstdint>
#include <algorithm>

constexpr int BLOCK_THREADS = 256;
constexpr int ITEMS_PER_THREAD = 8;
constexpr int CHUNK_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;  // 2048
constexpr int MERGE_THREADS = 256;

using BlockRadixSortT = cub::BlockRadixSort<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>;

__global__ void block_sort_kernel(float* data, int64_t n) {
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;
    int32_t thread_keys[ITEMS_PER_THREAD];

    for (int64_t chunk_start = blockIdx.x * CHUNK_SIZE;
         chunk_start < n;
         chunk_start += gridDim.x * CHUNK_SIZE) {
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

// Merge two sorted segments into one using merge path algorithm.
// Each thread binary-searches for its partition start, then does serial merge.
__global__ void tile_merge_kernel(
    const float* __restrict__ left,
    const float* __restrict__ right,
    float* __restrict__ out,
    int n_left,
    int n_right)
{
    int total = n_left + n_right;
    int tid = threadIdx.x;

    // Each thread handles a segment of the output: [my_start, my_end)
    int my_start = static_cast<int>(static_cast<int64_t>(tid) * total / MERGE_THREADS);
    int my_end = static_cast<int>(static_cast<int64_t>(tid + 1) * total / MERGE_THREADS);

    // Binary search in left for start position
    int low = (my_start > n_right) ? (my_start - n_right) : 0;
    int high = (my_start < n_left) ? my_start : n_left;

    while (low < high) {
        int mid = (low + high) >> 1;
        int r = my_start - mid - 1;
        bool take_left;
        if (r < 0) {
            take_left = false;
        } else if (r >= n_right) {
            take_left = true;
        } else if (mid >= n_left) {
            take_left = false;
        } else {
            take_left = __float_as_int(left[mid]) <= __float_as_int(right[r]);
        }
        if (take_left) low = mid + 1;
        else high = mid;
    }
    int li = low;
    int ri = my_start - low;

    // Serial merge
    for (int pos = my_start; pos < my_end; pos++) {
        bool take_left;
        if (li >= n_left) take_left = false;
        else if (ri >= n_right) take_left = true;
        else take_left = __float_as_int(left[li]) <= __float_as_int(right[ri]);

        out[pos] = take_left ? left[li++] : right[ri++];
    }
}

static torch::Tensor scratch_buf;
static torch::Tensor merge_buf;

void init_temp_storage() {
    if (scratch_buf.defined()) return;
    int64_t max_n = 100'000'000;
    scratch_buf = torch::empty({max_n},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    merge_buf = torch::empty({max_n},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto n = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (n == 0) return output;

    float* scratch = scratch_buf.data_ptr<float>();
    float* merges = merge_buf.data_ptr<float>();
    float* out = output.data_ptr<float>();

    // Copy input -> scratch
    cudaMemcpyAsync(scratch, input.const_data_ptr<float>(),
        n * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // Block sort chunks
    int num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int grid = num_chunks < 228 ? num_chunks : 228;
    block_sort_kernel<<<grid, BLOCK_THREADS, 0, stream>>>(scratch, n);

    if (num_chunks <= 1) {
        cudaMemcpyAsync(out, scratch, n * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
        return output;
    }

    // Merge tree
    float* src = scratch;
    float* dst = merges;
    int cur_chunks = num_chunks;
    int64_t seg_size = CHUNK_SIZE;

    while (cur_chunks > 1) {
        int pairs = cur_chunks / 2;

        for (int p = 0; p < pairs; p++) {
            int64_t base = p * 2 * seg_size;
            float* left = src + base;
            float* right_ptr = src + base + seg_size;
            float* out_ptr = dst + base;

            int n_left = seg_size;
            int64_t right_end = n - (base + seg_size);
            int n_right = (right_end < seg_size) ? static_cast<int>(right_end) : seg_size;
            if (n_right < 0) n_right = 0;

            if (n_right <= 0) {
                cudaMemcpyAsync(out_ptr, left, n_left * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
            } else {
                tile_merge_kernel<<<1, MERGE_THREADS, 0, stream>>>(
                    left, right_ptr, out_ptr, n_left, n_right);
            }
        }

        if (cur_chunks % 2 == 1) {
            int64_t base = static_cast<int64_t>(cur_chunks - 1) * seg_size;
            int64_t sz = n - base;
            if (sz > 0)
                cudaMemcpyAsync(dst + base, src + base,
                    sz * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }

        float* tmp = src; src = dst; dst = tmp;
        cur_chunks = (cur_chunks + 1) / 2;
        seg_size *= 2;
    }

    if (src != out)
        cudaMemcpyAsync(out, src, n * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
void init_temp_storage();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='batch_tile_merge_v3',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_temp_storage'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_temp_storage()

def custom_kernel(data: input_t) -> output_t:
    """BlockRadixSort + batched tile-merge tree (1 kernel launch per level)."""
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor