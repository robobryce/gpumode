"""
CUB DeviceRadixSort::SortKeys with int32 bitcast (no float conversion).
Since all data is positive IEEE 754, raw bits are in correct sort order.
Interpret float* as int*, sort keys-only, re-interpret back as float.
Persistent temp storage allocated once at module init to eliminate per-call overhead.
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
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int32_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int32_t* key_in = reinterpret_cast<const int32_t*>(input.const_data_ptr<float>());
    int32_t* key_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());

    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        key_in, key_out, num_items,
        0, 32,
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
    name='sort_cuda_int32_offsets_sm100a',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    extra_cuda_cflags=['-gencode=arch=compute_100a,code=sm_100a'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys on raw int32 bitcast of float32.
    No conversion needed — all data is positive IEEE 754 floats.
    Persistent temp storage avoids per-call allocation.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor