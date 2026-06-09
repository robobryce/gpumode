import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


cub_sort_source = """
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

torch::Tensor cub_sort_keys(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(output.numel() >= input.numel(), "output must be at least as large as input");

    int num_items = input.numel();

    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, temp_storage_bytes,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_items
    );

    // Allocate temporary storage
    auto temp_storage = torch::empty({static_cast<int64_t>(temp_storage_bytes)},
                                     torch::dtype(torch::kUInt8).device(input.device()));

    // Run the sort
    cub::DeviceRadixSort::SortKeys(
        temp_storage.data_ptr<uint8_t>(), temp_storage_bytes,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_items
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUB SortKeys failed: ", cudaGetErrorString(err));

    return output;
}
"""

cub_sort_cpp_source = """
#include <torch/extension.h>
torch::Tensor cub_sort_keys(torch::Tensor input, torch::Tensor output);
"""

cub_sort_module = load_inline(
    name='cub_sort_keys',
    cpp_sources=cub_sort_cpp_source,
    cuda_sources=cub_sort_source,
    functions=['cub_sort_keys'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Sort using CUB DeviceRadixSort::SortKeys directly, bypassing torch.sort
    overhead (index computation, scatter, elementwise comparison/negation).
    """
    input_tensor, output_tensor = data
    cub_sort_module.cub_sort_keys(input_tensor, output_tensor)
    return output_tensor