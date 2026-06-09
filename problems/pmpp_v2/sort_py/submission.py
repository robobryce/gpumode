"""
Per-row DeviceSegmentedRadixSort + integer-bucket global sort.
Input is a flattened sqrt(N)xsqrt(N) matrix where each row is randn centered
at (seed+i). DeviceSegmentedRadixSort sorts each row independently in one call,
then a global DeviceRadixSort merges all rows.
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
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cstdint>
#include <cuda_runtime_api.h>

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;
static torch::Tensor persistent_offsets = {};
static int64_t cached_num_rows = 0;
static int64_t cached_row_stride = 0;
static int64_t cached_N = 0;

void sort_rows_init(int64_t max_items) {
    if (persistent_temp.defined()) return;
    // Temp storage for the global DeviceRadixSort (worst case size)
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

__global__ void fill_segment_offsets_kernel(
    int64_t* __restrict__ begin_offsets,
    int64_t* __restrict__ end_offsets,
    int64_t num_rows,
    int64_t row_stride,
    int64_t N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    int64_t start = idx * row_stride;
    int64_t end = (idx + 1) * row_stride;
    if (end > N) end = N;
    begin_offsets[idx] = start;
    end_offsets[idx] = end;
}

torch::Tensor sort_rows(torch::Tensor input, torch::Tensor output, torch::Tensor temp) {
    int64_t N = input.numel();
    int64_t num_rows = static_cast<int64_t>(std::sqrt(static_cast<double>(N)));
    if (num_rows < 1) num_rows = 1;
    int64_t row_stride = (N + num_rows - 1) / num_rows;

    auto input_c = input.contiguous();
    const int32_t* in_ptr = reinterpret_cast<const int32_t*>(
        input_c.const_data_ptr<float>());
    int32_t* temp_ptr = reinterpret_cast<int32_t*>(temp.data_ptr<float>());
    int32_t* out_ptr = reinterpret_cast<int32_t*>(output.data_ptr<float>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Allocate or reuse segment offsets on GPU
    if (persistent_offsets.defined() && cached_num_rows < num_rows) {
        // Need larger offsets array
        persistent_offsets = torch::empty(
            {num_rows * 2},  // begin + end offsets
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
        cached_num_rows = num_rows;
    } else if (!persistent_offsets.defined()) {
        persistent_offsets = torch::empty(
            {num_rows * 2},
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
        cached_num_rows = num_rows;
    }

    int64_t* offsets_ptr = reinterpret_cast<int64_t*>(persistent_offsets.data_ptr());
    int64_t* begin_offsets = offsets_ptr;
    int64_t* end_offsets = offsets_ptr + num_rows;

    int blocks_needed = (num_rows + 255) / 256;
    fill_segment_offsets_kernel<<<blocks_needed, 256, 0, stream>>>(
        begin_offsets, end_offsets, num_rows, row_stride, N);

    // Phase 1: DeviceSegmentedRadixSort — sort each row independently
    // First, determine temp storage size for segmented sort
    size_t seg_temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, seg_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int64_t>(N),
        static_cast<int64_t>(num_rows),
        static_cast<const int64_t*>(nullptr),
        static_cast<const int64_t*>(nullptr),
        0, 32);

    // Allocate segmented sort temp storage
    size_t total_temp = std::max(persistent_temp_bytes, seg_temp_bytes);
    if (persistent_temp.numel() < static_cast<int64_t>(total_temp)) {
        persistent_temp = torch::empty(
            {static_cast<int64_t>(total_temp)},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
        persistent_temp_bytes = total_temp;
    }

    // Re-read pointers in case persistent_temp was reallocated
    void* temp_storage = persistent_temp.data_ptr();

    cub::DeviceSegmentedRadixSort::SortKeys(
        temp_storage, seg_temp_bytes,
        in_ptr, temp_ptr,
        N, num_rows,
        begin_offsets, end_offsets,
        0, 32, stream);

    // Phase 2: Global DeviceRadixSort merges the row-sorted data
    cub::DeviceRadixSort::SortKeys(
        temp_storage, persistent_temp_bytes,
        temp_ptr, out_ptr, N,
        0, 32, stream);

    return output;
}
"""

sort_module = load_inline(
    name='sort_rows_segmented',
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
    Phase 1: DeviceSegmentedRadixSort sorts each row independently.
    Phase 2: Global DeviceRadixSort merges row-sorted data into final output.
    """
    global _temp_tensor
    input_tensor, output_tensor = data
    N = input_tensor.numel()
    if _temp_tensor is None or _temp_tensor.numel() < N:
        _temp_tensor = torch.empty(N, dtype=torch.float32, device='cuda')
    sort_module.sort_rows(input_tensor, output_tensor, _temp_tensor)
    return output_tensor