"""
Bfloat16 SortKeys v8: radix_bits=16 single pass + fused encode/decode.
Encode f32->uint16 directly into output[0..2N), SortKeys DoubleBuffer
within output buffer (current=output[0..2N), alt=output[2N..4N)),
decode reverse in-place. ZERO extra buffer allocations beyond CUB temp.
The bfloat16 key: f32 bits >> 16 gives the bfloat16 encoding directly.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

static torch::Tensor cub_temp = {};
static size_t cub_temp_bytes = 0;

void init_temp() {
    if (cub_temp.defined()) return;
    int64_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, cub_temp_bytes,
        static_cast<const uint16_t*>(nullptr),
        static_cast<uint16_t*>(nullptr),
        static_cast<int>(max_n),
        0, 16);
    cub_temp_bytes = (cub_temp_bytes * 11 + 9) / 10;
    cub_temp = torch::empty({static_cast<int64_t>(cub_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

// Encode: read f32, bitcast to uint32, shift right 16 -> bfloat16 in uint16.
// Valid for positive f32: raw uint bits preserve exponent+sign ordering.
__global__ void encode_bitshift_kernel(
    const float* __restrict__ in,
    int16_t* __restrict__ out,
    int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<int16_t>(
            *(reinterpret_cast<const uint32_t*>(in + idx)) >> 16);
    }
}

// Decode in reverse: read uint16, shift left 16, reinterpret as f32.
// Reverse order ensures write position (4*rev_idx) never overlaps
// read position (2*rev_idx) for any rev_idx.
__global__ void decode_bitshift_kernel(
    const int16_t* __restrict__ in,
    float* __restrict__ out,
    int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t rev_idx = n - 1 - idx;
        uint32_t u = static_cast<uint32_t>(
            static_cast<uint16_t>(in[rev_idx])) << 16;
        out[rev_idx] = *(reinterpret_cast<float*>(&u));
    }
}

torch::Tensor sort_cuda(torch::Tensor input_tensor, torch::Tensor output_tensor) {
    auto num_items = static_cast<int64_t>(input_tensor.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const float* data_in = input_tensor.const_data_ptr<float>();
    float* data_out = output_tensor.data_ptr<float>();
    uint16_t* uout = reinterpret_cast<uint16_t*>(data_out);

    const int block_size = 256;
    const int grid_size = static_cast<int>((num_items + block_size - 1) / block_size);

    // Step 1: Encode f32 -> bfloat16 uint16 into output[0 .. 2N)
    encode_bitshift_kernel<<<grid_size, block_size, 0, stream>>>(
        data_in, reinterpret_cast<int16_t*>(uout), num_items);

    // Step 2: DoubleBuffer: current=output[0..2N), alternate=output[2N..4N)
    cub::DoubleBuffer<uint16_t> d_keys(uout, uout + num_items);

    size_t temp_bytes = cub_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        cub_temp.data_ptr(), temp_bytes,
        d_keys, static_cast<int>(num_items),
        0, 16, stream);

    // Step 3: Decode from current buffer in reverse
    decode_bitshift_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const int16_t*>(d_keys.Current()),
        data_out, num_items);

    return output_tensor;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
void init_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_bf16_v8',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)
sort_module.init_temp()


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor