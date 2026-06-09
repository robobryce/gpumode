import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

void sort_cuda_inplace(torch::Tensor data_ref) {
    TORCH_CHECK(data_ref.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(data_ref.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(data_ref.is_contiguous(), "Input must be contiguous");

    auto num_items = static_cast<int64_t>(data_ref.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Step 1: query temp storage size
    size_t temp_storage_bytes = 0;
    float* d_data = data_ref.data_ptr<float>();
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_storage_bytes,
        d_data, d_data,
        num_items,
        0, sizeof(float) * 8,
        stream);

    // Step 2: allocate temp storage
    auto temp_storage = torch::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(data_ref.device()));

    // Step 3: run the sort in-place
    cub::DeviceRadixSort::SortKeys(
        temp_storage.data_ptr(),
        temp_storage_bytes,
        d_data, d_data,
        num_items,
        0, sizeof(float) * 8,
        stream);
}
"""

sort_cpp_source = """
#include <torch/extension.h>
void sort_cuda_inplace(torch::Tensor data_ref);
"""

sort_module = load_inline(
    name='sort_cuda_onesweep',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda_inplace'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Sort using CUB DeviceRadixSort::SortKeys in-place.
    Copy input to output, then sort output in-place.
    """
    input_tensor, output_tensor = data
    output_tensor.copy_(input_tensor)
    sort_module.sort_cuda_inplace(output_tensor)
    return output_tensor