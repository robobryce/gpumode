"""CUB DeviceRadixSort with branchless encode/decode and fused copy.

Encode and decode use identical arithmetic (no branches, no conditionals).
Copy+encode is fused into one kernel. Decode is separate.
Only one temp buffer needed (for the CUB workspace).
"""

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_src = r"""
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>

// Encode: float -> sortable int32
//   v ^ ((v >> 31) | 0x80000000)
//   Positive: sign=0 -> mask=0x80000000 -> flip sign bit
//   Negative: sign=-1 -> mask=0xFFFFFFFF -> flip all bits
// Decode: reverse of encode
//   v ^ (~(v >> 31) | 0x80000000)
//   Encoded-positive (MSB=1): ~(-1) | 0x8... = 0|0x8... = 0x80000000 -> flip sign
//   Encoded-negative (MSB=0): ~(0) | 0x8... = 0xFFFFFFFF -> flip all bits
//
// Both are 4-op: load, shift, arith, xor, store. Decode uses complement.

__global__ void copy_encode_kernel(
    const float* __restrict__ input,
    int32_t* __restrict__ keys,
    int n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += gridDim.x * blockDim.x) {
        int32_t v = __float_as_int(input[idx]);
        keys[idx] = v ^ ((v >> 31) | (int32_t)(1u << 31));
    }
}

__global__ void decode_store_kernel(
    const int32_t* __restrict__ keys,
    float* __restrict__ output,
    int n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += gridDim.x * blockDim.x) {
        int32_t v = keys[idx];
        output[idx] = __int_as_float(v ^ (~(v >> 31) | (int32_t)(1u << 31)));
    }
}

torch::Tensor custom_kernel_fn(torch::Tensor data, torch::Tensor output) {
    int n = data.numel();
    if (n == 0) return output;

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(data.device());
    auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(data.device());
    auto keys = torch::empty({n}, opts_i32);

    int threads = 256;
    int blocks  = std::min((n + threads - 1) / threads, 65535);
    copy_encode_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float*>(data.data_ptr<float>()),
        keys.data_ptr<int>(), n);

    // CUB DeviceRadixSort (in-place on keys)
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        n);
    auto cub_temp = torch::empty(
        {static_cast<int64_t>(temp_bytes)}, opts_u8);
    cub::DeviceRadixSort::SortKeys(
        cub_temp.data_ptr(), temp_bytes,
        keys.data_ptr<int>(),
        keys.data_ptr<int>(),
        n, 0, 32);

    decode_store_kernel<<<blocks, threads>>>(
        keys.data_ptr<int>(),
        output.data_ptr<float>(), n);

    return output;
}
"""

cpp_src = r"""
#include <torch/extension.h>
torch::Tensor custom_kernel_fn(torch::Tensor data, torch::Tensor output);
"""

module = load_inline(
    name="cub_sort_branchless",
    cuda_sources=[cuda_src],
    cpp_sources=[cpp_src],
    functions=["custom_kernel_fn"],
    extra_cuda_cflags=["--expt-relaxed-constexpr"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    data, output = data
    module.custom_kernel_fn(data, output)
    return output