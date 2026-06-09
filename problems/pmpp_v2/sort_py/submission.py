import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <cstdint>

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "Output must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "Output must be float32");
    TORCH_CHECK(input.sizes() == output.sizes(), "Input and output must have same size");

    auto num_items = static_cast<int64_t>(input.numel());

    // Copy input to output, then sort in-place
    output.copy_(input);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Use thrust::sort with device execution policy on raw pointers
    // thrust::sort on GPU uses merge sort, operating directly on float32
    auto policy = thrust::cuda::par.on(stream);
    thrust::sort(policy,
                 output.data_ptr<float>(),
                 output.data_ptr<float>() + num_items);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_thrust',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Sort using Thrust thrust::sort with device execution policy (merge sort on GPU),
    operating directly on float32 without uint32 bit-trick conversion.
    """
    input_tensor, output_tensor = data
    output_tensor[...] = sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor