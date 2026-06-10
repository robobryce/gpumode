"""
Diagnostic: per-row DeviceSegmentedRadixSort + global DeviceRadixSort (SortKeys).
Measures whether per-row pre-sorting improves CUB Onesweep's memory locality
due to the per-row-normal input structure.

Phase 1: Per-row segmented SortKeys → data sorted within each row
Phase 2: Global SortKeys → final globally sorted output

Total: this adds one extra SortKeys call vs the parent. Compare overhead.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_source = r"""
#include <torch/extension.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime.h>

torch::Tensor per_row_plus_global_sort(torch::Tensor input, torch::Tensor output) {
    int N = (int)input.numel();
    if (N <= 1) { if (N==1) output[0]=input[0].item<float>(); return output; }

    cudaStream_t stream = (cudaStream_t)0;

    int num_rows = (int)sqrt((double)N);
    int row_len = (N + num_rows - 1) / num_rows;

    std::vector<int> ro(num_rows + 1);
    for (int i = 0; i < num_rows; i++)
        ro[i] = std::min(i * row_len, N);
    ro[num_rows] = N;

    auto d_ro = torch::from_blob(ro.data(), {num_rows+1},
        torch::dtype(torch::kInt32)).clone().to(torch::kCUDA);

    // Phase 1: Per-row segmented sort
    auto tmp = torch::empty_like(input);

    size_t tb_seg = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, tb_seg,
        input.const_data_ptr<float>(), tmp.data_ptr<float>(),
        N, num_rows,
        d_ro.const_data_ptr<int>(), d_ro.const_data_ptr<int>()+1,
        0, 32);
    auto d_ts = torch::empty({(int64_t)tb_seg},
        torch::dtype(torch::kUInt8).device(torch::kCUDA));
    cub::DeviceSegmentedRadixSort::SortKeys(
        d_ts.data_ptr(), tb_seg,
        input.const_data_ptr<float>(), tmp.data_ptr<float>(),
        N, num_rows,
        d_ro.const_data_ptr<int>(), d_ro.const_data_ptr<int>()+1,
        0, 32, stream);
    cudaStreamSynchronize(stream);

    // Phase 2: Global SortKeys on the per-row-sorted data
    size_t tb_global = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, tb_global,
        static_cast<const float*>(nullptr),
        static_cast<float*>(nullptr),
        static_cast<int64_t>(N),
        0, 32);
    auto d_tg = torch::empty({(int64_t)tb_global},
        torch::dtype(torch::kUInt8).device(torch::kCUDA));
    cub::DeviceRadixSort::SortKeys(
        d_tg.data_ptr(), tb_global,
        tmp.const_data_ptr<float>(), output.data_ptr<float>(),
        (int64_t)N, 0, 32, stream);
    cudaStreamSynchronize(stream);

    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor per_row_plus_global_sort(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='per_row_plus_global',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['per_row_plus_global_sort'],
    extra_cuda_cflags=['-O3'],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    inp = input_tensor.contiguous()
    sort_module.per_row_plus_global_sort(inp, output_tensor)
    return output_tensor