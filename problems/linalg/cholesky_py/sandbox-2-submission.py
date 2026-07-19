import torch
from torch.utils.cpp_extension import load_inline

from task import input_t, output_t


_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor small_cholesky_cuda(torch::Tensor input);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void batched_potrf(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    extern __shared__ float matrix[];
    const int tid = threadIdx.x;
    const int size = n * n;
    const long long base = static_cast<long long>(blockIdx.x) * size;
    const float* src = input + base;
    float* dst = output + base;

    for (int index = tid; index < size; index += blockDim.x) {
        const int row = index / n;
        const int col = index - row * n;
        matrix[index] = row >= col ? src[index] : 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < n; ++k) {
        if (tid == 0) {
            float diagonal = matrix[k * n + k];
            for (int j = 0; j < k; ++j) {
                const float value = matrix[k * n + j];
                diagonal = fmaf(-value, value, diagonal);
            }
            matrix[k * n + k] = sqrtf(fmaxf(diagonal, 0.0f));
        }
        __syncthreads();

        const float diagonal = matrix[k * n + k];
        for (int row = k + 1 + tid; row < n; row += blockDim.x) {
            float value = matrix[row * n + k];
            for (int j = 0; j < k; ++j) {
                value = fmaf(
                    -matrix[row * n + j], matrix[k * n + j], value
                );
            }
            matrix[row * n + k] = value / diagonal;
        }
        __syncthreads();
    }

    for (int index = tid; index < size; index += blockDim.x) {
        dst[index] = matrix[index];
    }
}

torch::Tensor small_cholesky_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must have shape (batch, n, n)");
    TORCH_CHECK(input.size(1) == input.size(2), "input matrices must be square");

    const int n = static_cast<int>(input.size(1));
    TORCH_CHECK(n > 0 && n <= 128, "small kernel supports 1 <= n <= 128");

    auto output = torch::empty_like(input);
    const int threads = n == 32 ? 32 : n;
    const size_t shared_bytes = static_cast<size_t>(n) * n * sizeof(float);

    if (shared_bytes > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            batched_potrf,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)
        ));
    }

    batched_potrf<<<input.size(0), threads, shared_bytes>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""

_EXTENSION = load_inline(
    name="cholesky_small_potrf_inline",
    cpp_sources=_CPP_SOURCE,
    cuda_sources=_CUDA_SOURCE,
    functions=["small_cholesky_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    with_cuda=True,
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    n = data.shape[-1]
    if n <= 128:
        return _EXTENSION.small_cholesky_cuda(data)
    return torch.linalg.cholesky_ex(data, check_errors=False).L
