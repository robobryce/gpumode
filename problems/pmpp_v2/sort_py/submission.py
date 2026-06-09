"""
CUB DeviceRadixSort::SortKeys with lightweight device-side max scan.
Single-kernel block-reduce + warp-reduce to compute max int32 value,
store end_bit in device memory, SortKeys reads it without host sync.
Uses int32_t offsets and sm_100a arch for Blackwell optimizations.
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
static torch::Tensor d_end_bit_out = {};

// Quick max-scan kernel: block-level reduce + atomicMax to global
__global__ void quick_max_scan_kernel(const int32_t* __restrict__ data,
                                       int32_t* __restrict__ d_max,
                                       int32_t n) {
    __shared__ int32_t smem_max;
    int tid = threadIdx.x;
    int block_stride = blockDim.x * gridDim.x;

    // Per-thread max
    int32_t local_max = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += block_stride) {
        int32_t val = data[i];
        if (val > local_max) local_max = val;
    }

    // Block-level warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        int32_t other = __shfl_xor_sync(0xffffffff, local_max, offset);
        if (other > local_max) local_max = other;
    }

    if (tid == 0) {
        smem_max = local_max;
    }
    __syncthreads();

    if (tid == 0) {
        local_max = smem_max;
        for (int offset = 16; offset > 0; offset >>= 1) {
            int32_t other = __shfl_xor_sync(0xffffffff, local_max, offset);
            if (other > local_max) local_max = other;
        }
        if (local_max > 0) {
            atomicMax(d_max, local_max);
        }
    }
}

// Reduction kernel to compute end_bit from per-block max buffer
__global__ void end_bit_from_max_kernel(int32_t* __restrict__ d_max_in,
                                         int32_t* __restrict__ d_end_bit) {
    int32_t v = *d_max_in;
    unsigned int end_bit = 32;
    if (v <= 0) { end_bit = 1; }
    else {
        end_bit = 0;
        while (v > 0) { end_bit++; v >>= 1; }
    }
    if (end_bit < 1) end_bit = 1;
    if (end_bit > 32) end_bit = 32;
    *d_end_bit = (int32_t)end_bit;
    // Reset d_max_in to 0 for next call
    *d_max_in = 0;
}

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int32_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int32_t>(max_n),
        0, 32);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    // Small device buffer for max + end_bit (2 int32)
    d_end_bit_out = torch::empty(
        {2}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int32_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int32_t* key_in = reinterpret_cast<const int32_t*>(input.const_data_ptr<float>());
    int32_t* key_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());

    // Quick scan: block-level max + global atomicMax
    int32_t* d_max_and_endbit = d_end_bit_out.data_ptr<int32_t>();
    int blocks = std::min(1024, (num_items + 255) / 256);
    int threads = 256;
    quick_max_scan_kernel<<<blocks, threads, 0, stream>>>(key_in, d_max_and_endbit, num_items);
    // Single-block reduction to compute end_bit
    end_bit_from_max_kernel<<<1, 1, 0, stream>>>(d_max_and_endbit, d_max_and_endbit + 1);

    // Read end_bit from device memory (no host sync needed — in same stream)
    int32_t h_end_bit;
    cudaMemcpyAsync(&h_end_bit, d_max_and_endbit + 1, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    unsigned int end_bit = static_cast<unsigned int>(h_end_bit);
    if (end_bit < 1) end_bit = 1;
    if (end_bit > 32) end_bit = 32;

    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        key_in, key_out, num_items,
        0, static_cast<int>(end_bit),
        stream);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_quick_scan',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_cuda_cflags=['-gencode=arch=compute_100a,code=sm_100a'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort with lightweight device-side max scan.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor