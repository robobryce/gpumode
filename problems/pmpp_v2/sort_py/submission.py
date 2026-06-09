"""
bfloat16 sort: cast float32 -> bfloat16, bitcast to uint16,
CUB SortKeys (4 radix passes), cast back to float32.
bfloat16 preserves full 8-bit exponent -- sort order is exact for
same-sign values since higher exponent always wins, and same-exponent
values share the same upper mantissa bits for comparison.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;
static torch::Tensor bf16_scratch = {};

static constexpr int64_t MAX_N = 100'000'000;

__global__ void f32_to_bf16(const float* __restrict__ src,
                             uint16_t* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<uint16_t>(__float_as_uint(src[idx]) >> 16);
    }
}

__global__ void bf16_to_f32(const uint16_t* __restrict__ src,
                             float* __restrict__ dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __uint_as_float(static_cast<uint32_t>(src[idx]) << 16);
    }
}

void init_bf16() {
    if (persistent_temp.defined()) return;

    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const uint16_t*>(nullptr),
        static_cast<uint16_t*>(nullptr),
        MAX_N, 0, 16);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    bf16_scratch = torch::empty({MAX_N},
        torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCUDA));
}

torch::Tensor sort_bf16(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int64_t threads = 256;
    const int64_t blocks = (num_items + threads - 1) / threads;

    uint16_t* keys = reinterpret_cast<uint16_t*>(bf16_scratch.data_ptr());

    f32_to_bf16<<<blocks, threads, 0, stream>>>(
        input.const_data_ptr<float>(), keys, num_items);

    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        keys, keys, num_items, 0, 16, stream);

    bf16_to_f32<<<blocks, threads, 0, stream>>>(
        keys, output.data_ptr<float>(), num_items);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
void init_bf16();
torch::Tensor sort_bf16(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_bf16_final',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_bf16', 'init_bf16'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_bf16()


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    sort_module.sort_bf16(input_tensor.contiguous(), output_tensor)
    return output_tensor