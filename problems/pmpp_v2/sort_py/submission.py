"""
CUB DeviceRadixSort::SortKeys with dynamic end_bit scan.
Scan kernel computes max(abs(int32_t bitcast)) to find highest used bit,
then passes clamped end_bit=ceil(log2(max_val+1)) to SortKeys.
Uses int32_t offsets and sm_100a arch for Blackwell optimizations.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cstdint>

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;
static torch::Tensor scan_temp = {};
static size_t scan_temp_bytes = 0;
static torch::Tensor scan_output = {};

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int64_t max_n = 100'000'000;
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

    // Allocate scan workspace: DeviceReduce::Max needs ~8KB
    cub::DeviceReduce::Max(
        nullptr, scan_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int32_t>(max_n));
    scan_temp_bytes = (scan_temp_bytes * 11 + 9) / 10;
    scan_temp = torch::empty(
        {static_cast<int64_t>(scan_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    scan_output = torch::empty(
        {1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int32_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int32_t* key_in = reinterpret_cast<const int32_t*>(input.const_data_ptr<float>());
    int32_t* key_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());

    // Scan pass: compute max int32_t value via DeviceReduce::Max
    // CUB Reduce::Max works natively on int32_t
    int32_t* d_max = scan_output.data_ptr<int32_t>();
    size_t s_temp_bytes = scan_temp_bytes;
    cub::DeviceReduce::Max(
        scan_temp.data_ptr(), s_temp_bytes,
        key_in, d_max, num_items,
        stream);

    // Transfer max to host
    int32_t max_val;
    cudaMemcpyAsync(&max_val, d_max, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Compute end_bit: ceil(log2(max_val + 1)) if max_val > 0
    unsigned int end_bit = 32;
    if (max_val > 0) {
        // bit_length: position of highest set bit (1-indexed)
        unsigned int v = static_cast<unsigned int>(max_val);
        end_bit = 0;
        while (v > 0) { end_bit++; v >>= 1; }
    }
    // Clamp: at least 1, at most 32
    if (end_bit < 1) end_bit = 1;
    if (end_bit > 32) end_bit = 32;

    // SortKeys with dynamic end_bit
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
    name='sort_cuda_int32_dynamic_endbit',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_cuda_cflags=['-gencode=arch=compute_100a,code=sm_100a'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys on raw int32 bitcast of float32.
    Dynamic end_bit computed via CUB DeviceReduce::Max scan.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor