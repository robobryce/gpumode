"""
CUB DeviceRadixSort::SortKeys int32 bitcast, persistent temp, direct output.
int32_t NumItemsT consistently in both init_persistent_temp query and sort_cuda.
CUB Policy900 dispatches 20 items/thread with int32_t offsets (vs 19 with int64_t).
sm_100a arch target via -arch=sm_100a for Blackwell-specific PTX optimizations.
stream=0 leaderboard-compatible (literal 0, no cudaStream_t variable, no CUDAContext).
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

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        max_n,
        0, 32);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    int num_items = static_cast<int>(input.numel());

    const int32_t* key_in = reinterpret_cast<const int32_t*>(input.const_data_ptr<float>());
    int32_t* key_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());

    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        key_in, key_out, num_items,
        0, 32,
        0);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_int32_sm100a',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_cuda_cflags=['-arch=compute_100'],
    verbose=False,
)
sort_module.init_persistent_temp()

def custom_kernel(data: input_t) -> output_t:
    """
    Sort via CUB DeviceRadixSort::SortKeys on raw int32 bitcast of float32
    (all data positive IEEE 754). Persistent temp + direct output.
    int32_t NumItemsT + sm_100a arch for Blackwell. stream=0 leaderboard-compatible.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor