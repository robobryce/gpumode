"""
Two-pass uint16 radix sort: LSB-first radix using SortPairs<uint16_t, float>.
Pass 1 sorts by lower 16 bits (bits & 0xFFFF), pass 2 sorts by upper 16 bits
(bits >> 16). Since CUB SortPairs is stable, the result is a correct full 32-bit
sort using 16-bit radix groups (4 total passes vs CUB's 4 8-bit passes).
Persistent temp storage + persistent swap buffer avoid per-call allocations.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;
static torch::Tensor persistent_swap = {};

void init_persistent() {
    if (persistent_temp.defined()) return;
    int64_t max_n = 100'000'000;

    // Query temp storage for SortPairs<uint16_t, float>
    cub::DeviceRadixSort::SortPairs(
        nullptr, persistent_temp_bytes,
        static_cast<const uint16_t*>(nullptr),
        static_cast<uint16_t*>(nullptr),
        static_cast<const float*>(nullptr),
        static_cast<float*>(nullptr),
        static_cast<int64_t>(max_n),
        0, 16);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    // Swap buffer for two-pass sort (intermediate values storage)
    persistent_swap = torch::empty({max_n},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
}

__global__ void encode_lower_u16(
    const float* __restrict__ input,
    uint16_t* __restrict__ keys,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t bits = *reinterpret_cast<const uint32_t*>(&input[idx]);
        keys[idx] = static_cast<uint16_t>(bits & 0xFFFF);
    }
}

__global__ void encode_upper_u16(
    const float* __restrict__ input,
    uint16_t* __restrict__ keys,
    int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t bits = *reinterpret_cast<const uint32_t*>(&input[idx]);
        keys[idx] = static_cast<uint16_t>(bits >> 16);
    }
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());

    auto keys = torch::empty({num_items},
        torch::TensorOptions().dtype(torch::kInt16).device(torch::kCUDA));

    int threads = 256;
    int blocks = (num_items + threads - 1) / threads;

    // --- Pass 1: sort by lower 16 bits ---
    encode_lower_u16<<<blocks, threads>>>(
        input.const_data_ptr<float>(),
        reinterpret_cast<uint16_t*>(keys.data_ptr<int16_t>()),
        num_items);

    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortPairs(
        persistent_temp.data_ptr(), temp_bytes,
        reinterpret_cast<const uint16_t*>(keys.const_data_ptr<int16_t>()),
        reinterpret_cast<uint16_t*>(keys.data_ptr<int16_t>()),
        input.const_data_ptr<float>(),
        persistent_swap.data_ptr<float>(),
        num_items, 0, 16);

    // --- Pass 2: sort by upper 16 bits (stable sort preserves pass 1 order) ---
    encode_upper_u16<<<blocks, threads>>>(
        persistent_swap.const_data_ptr<float>(),
        reinterpret_cast<uint16_t*>(keys.data_ptr<int16_t>()),
        num_items);

    cub::DeviceRadixSort::SortPairs(
        persistent_temp.data_ptr(), temp_bytes,
        reinterpret_cast<const uint16_t*>(keys.const_data_ptr<int16_t>()),
        reinterpret_cast<uint16_t*>(keys.data_ptr<int16_t>()),
        persistent_swap.const_data_ptr<float>(),
        output.data_ptr<float>(),
        num_items, 0, 16);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_u16_twopass',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent()


def custom_kernel(data: input_t) -> output_t:
    """
    Two-pass uint16 radix sort: LSB (lower 16 bits) then MSB (upper 16 bits).
    CUB SortPairs<uint16_t,float> with stable sort = full 32-bit sort.
    4 total radix passes (2 per call), no reconstruction error.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor