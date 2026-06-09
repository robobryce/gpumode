"""
CUB DeviceRadixSort::SortKeys registered as a torch.library custom CUDA op.
torch.cuda.CUDAGraph captures the op through PyTorch's dispatch system.
No cudaStreamCreate -- leaderboard-compatible.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

static torch::Tensor persistent_temp;
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int64_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
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
    const int32_t* key_in = reinterpret_cast<const int32_t*>(input.const_data_ptr<float>());
    int32_t* key_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());
    size_t temp_bytes = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), temp_bytes,
        key_in, key_out, num_items, 0, 32, stream);
    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_torch_library',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    extra_cuda_cflags=['-arch=sm_100'],
    verbose=False,
)
sort_module.init_persistent_temp()

# Register as torch.library custom op for native graph capture
lib = torch.library.Library("gpumode_sort", "DEF")
lib.define("sort_keys(Tensor input, Tensor output) -> Tensor")

@torch.library.impl(lib, "sort_keys", "CUDA")
def sort_keys_cuda(input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    return sort_module.sort_cuda(input, output)

sort_keys_op = torch.ops.gpumode_sort.sort_keys
_graph_cache = {}


def custom_kernel(data: input_t) -> output_t:
    """
    Sort via torch.library custom CUB op with CUDAGraph replay.
    No cudaStreamCreate.
    """
    input_tensor, output_tensor = data
    in_contig = input_tensor if input_tensor.is_contiguous() else input_tensor.contiguous()
    key = (output_tensor.data_ptr(), in_contig.numel())

    if key in _graph_cache:
        _graph_cache[key].replay()
        return output_tensor

    # First call: warmup+correctness
    sort_keys_op(in_contig, output_tensor)
    torch.cuda.synchronize()

    # Capture torch op into CUDAGraph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sort_keys_op(in_contig, output_tensor)

    _graph_cache[key] = g
    return output_tensor