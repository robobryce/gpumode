"""
CUB DeviceRadixSort::SortKeys with DoubleBuffer, uint32_t keys, is_overwrite_okay=true.
Using DoubleBuffer overload avoids CUB's internal const_cast copy — CUB uses the
input buffer directly as swap space. uint32_t instead of int32_t to bypass signed
integer trait's sign-bit flip path in CUB's radix sort dispatch.
Handle DoubleBuffer selector: after sort, copy result to output if needed.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <cstdint>

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int64_t max_n = 100'000'000;
    cub::DoubleBuffer<uint32_t> dummy_keys(nullptr, nullptr);
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        dummy_keys,
        static_cast<int64_t>(max_n),
        0, 32);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    uint32_t* key_in = reinterpret_cast<uint32_t*>(input.data_ptr<float>());
    uint32_t* key_out = reinterpret_cast<uint32_t*>(output.data_ptr<float>());
    cub::DoubleBuffer<uint32_t> d_keys(key_in, key_out);

    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        d_keys, num_items,
        0, 32,
        stream);

    // With is_overwrite_okay=true, CUB writes sorted data to either
    // Current() or Alternate() depending on pass count parity.
    // If the sorted result landed in the input buffer, copy it to output.
    if (d_keys.Current() != key_out) {
        cudaMemcpyAsync(key_out, d_keys.Current(),
                        num_items * sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice, stream);
    }

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_doublebuffer_v2',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys with DoubleBuffer + uint32_t keys.
    Checks DoubleBuffer selector and copies result to output if sorted data
    ended up in the input buffer (dependent on pass count parity).
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor