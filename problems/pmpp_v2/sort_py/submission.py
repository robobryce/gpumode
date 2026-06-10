"""
Integer-floor bucket sort.
Strategy:
1. Compute integer floor for each element via simple kernel
2. Histogram: count elements per floor (atomicAdd)
3. Prefix sum: per-floor output offsets
4. Scatter: write elements to contiguous per-floor buffers
5. CUB DeviceSegmentedRadixSort per floor: sort each bucket independently

Key insight: even without per-row sort, the floor-based partition isolates
non-overlapping ranges. Within each floor, elements are from ~7-8 adjacent
rows — a narrow range that sorts quickly.

Total memory traffic: ~5N vs CUB Onesweep ~8N (5 kernel launches vs 4).
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cuda_source = r"""
#include <torch/extension.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// ---------------------------------------------------------------------------
// Kernel 1: compute integer floor for each element
// ---------------------------------------------------------------------------
__global__ void compute_floor_kernel(
    const float* __restrict__ data,
    int N,
    int* __restrict__ floors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    floors[i] = (int)__float2int_rz(data[i]);
}

// ---------------------------------------------------------------------------
// Kernel 2: histogram — count elements per floor
// ---------------------------------------------------------------------------
__global__ void floor_histogram_kernel(
    const int* __restrict__ floors,
    int N,
    int min_floor,
    int* __restrict__ hist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int f = floors[i] - min_floor;
    atomicAdd(&hist[f], 1);
}

// ---------------------------------------------------------------------------
// Kernel 3: scatter elements to per-floor contiguous regions
// ---------------------------------------------------------------------------
__global__ void floor_scatter_kernel(
    const float* __restrict__ data,
    const int* __restrict__ floors,
    int N,
    int min_floor,
    const int* __restrict__ offsets,     // [num_floors]
    int* __restrict__ counters,          // [num_floors] mutable copy for atomic positions
    float* __restrict__ output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int f = floors[i] - min_floor;
    int pos = atomicAdd(&counters[f], 1);
    output[offsets[f] + pos] = data[i];
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
torch::Tensor floor_bucket_sort(torch::Tensor input, torch::Tensor output) {
    int N = (int)input.numel();
    if (N <= 1) {
        if (N == 1) output[0] = input[0].item<float>();
        return output;
    }

    cudaStream_t stream = (cudaStream_t)0;

    // --- Step 1: Compute floors ---
    auto d_floors = torch::empty({N},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    {
        int blk = (N + 255) / 256;
        compute_floor_kernel<<<blk, 256, 0, stream>>>(
            input.const_data_ptr<float>(), N,
            d_floors.data_ptr<int>());
    }
    cudaStreamSynchronize(stream);

    // Need min/max floor for histogram sizing.  Do a quick GPU reduction
    // or just sample.  For correctness, read the full floor array to CPU
    // (small overhead for int32)
    auto h_floors = d_floors.cpu();
    auto* hf = h_floors.const_data_ptr<int>();
    int min_floor = INT_MAX, max_floor = INT_MIN;
    for (int i = 0; i < N; i++) {
        int f = hf[i];
        if (f < min_floor) min_floor = f;
        if (f > max_floor) max_floor = f;
    }
    int num_floors = max_floor - min_floor + 1;

    // --- Step 2: Histogram ---
    auto d_hist = torch::zeros({num_floors},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    {
        int blk = (N + 255) / 256;
        floor_histogram_kernel<<<blk, 256, 0, stream>>>(
            d_floors.const_data_ptr<int>(), N, min_floor,
            d_hist.data_ptr<int>());
    }
    cudaStreamSynchronize(stream);

    // --- Step 3: Prefix sum for per-floor offsets ---
    auto h_hist = d_hist.cpu();
    auto* hh = h_hist.const_data_ptr<int>();
    std::vector<int> floor_offs(num_floors + 1, 0);
    for (int f = 0; f < num_floors; f++)
        floor_offs[f + 1] = floor_offs[f] + hh[f];

    if (floor_offs[num_floors] != N) {
        printf("ERROR: floor total %d != N %d\n", floor_offs[num_floors], N);
        return output;
    }

    auto d_floor_offs = torch::from_blob(floor_offs.data(), {num_floors + 1},
        torch::dtype(torch::kInt32)).clone().to(torch::kCUDA);

    // --- Step 4: Scatter to floor buffers ---
    auto scatter_buf = torch::empty({N},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto d_counters = torch::zeros({num_floors},
        torch::dtype(torch::kInt32).device(torch::kCUDA));
    {
        int blk = (N + 255) / 256;
        floor_scatter_kernel<<<blk, 256, 0, stream>>>(
            input.const_data_ptr<float>(),
            d_floors.const_data_ptr<int>(), N, min_floor,
            d_floor_offs.const_data_ptr<int>(),
            d_counters.data_ptr<int>(),
            scatter_buf.data_ptr<float>());
    }
    cudaStreamSynchronize(stream);

    // --- Step 5: Per-floor segmented sort ---
    {
        size_t tb = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, tb,
            scatter_buf.const_data_ptr<float>(), output.data_ptr<float>(),
            N, num_floors,
            d_floor_offs.const_data_ptr<int>(), d_floor_offs.const_data_ptr<int>() + 1,
            0, 32);
        auto d_tmp = torch::empty({(int64_t)tb},
            torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cub::DeviceSegmentedRadixSort::SortKeys(
            d_tmp.data_ptr(), tb,
            scatter_buf.const_data_ptr<float>(), output.data_ptr<float>(),
            N, num_floors,
            d_floor_offs.const_data_ptr<int>(), d_floor_offs.const_data_ptr<int>() + 1,
            0, 32, stream);
    }
    cudaStreamSynchronize(stream);

    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor floor_bucket_sort(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='floor_bucket_sort',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['floor_bucket_sort'],
    extra_cuda_cflags=['-O3'],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    inp = input_tensor.contiguous()
    sort_module.floor_bucket_sort(inp, output_tensor)
    return output_tensor