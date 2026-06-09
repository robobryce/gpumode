"""
Coarse bucket sort: partition into a small number of equal-range buckets using
analytically-estimated range (seed-based, no sampling overhead), then sort
each bucket with CUB SortKeys.

Insight: the data has sqrt(N) rows with means seed..seed+sqrt(N). The value
range is compact: [seed-4sigma, seed+sqrt(N)+4sigma]. Use estimated min/max
to compute bucket boundaries with NO sampling or binary search.
Bucket index = (val - est_min) * NUM_BUCKETS / est_range (clamped).
"""
import math
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/functional>
#include <cstdint>

constexpr int NUM_BUCKETS = 4;
constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS_PER_BLOCK = 2048;

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    size_t bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, bytes,
        static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
        static_cast<int64_t>(50'000'000),  // half N max
        0, 32);
    persistent_temp_bytes = (bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

// GPU-side exclusive scan for NUM_BUCKETS elements (trivial, 1 block)
__global__ void gpu_scan_kernel(
    const int* __restrict__ histogram,
    int* __restrict__ offsets)
{
    __shared__ int s[NUM_BUCKETS + 1];
    int tid = threadIdx.x;

    // Load histogram
    int val = (tid < NUM_BUCKETS) ? histogram[tid] : 0;
    // Inclusive scan via warp-level (single warp, NUM_BUCKETS < 32)
    // Simple serial scan for NUM_BUCKETS=4
    if (tid == 0) {
        s[0] = 0;
        for (int i = 0; i < NUM_BUCKETS; i++) {
            s[i + 1] = s[i] + histogram[i];
        }
    }
    __syncthreads();

    if (tid <= NUM_BUCKETS) {
        offsets[tid] = s[tid];
    }
}

// Histogram: shared memory per block, atomicAdd to global
__global__ void histogram_kernel(
    const int32_t* __restrict__ keys,
    int* __restrict__ global_histogram,
    float est_min, float inv_range,
    int num_elements)
{
    __shared__ int local_h[NUM_BUCKETS];
    int tid = threadIdx.x;
    for (int i = tid; i < NUM_BUCKETS; i += blockDim.x) local_h[i] = 0;
    __syncthreads();

    int start = blockIdx.x * ITEMS_PER_BLOCK;
    int end = min(start + ITEMS_PER_BLOCK, num_elements);

    for (int i = start + tid; i < end; i += blockDim.x) {
        float v = __int_as_float(keys[i]);
        // Linear bucket compute (no binary search)
        int b = (int)((v - est_min) * inv_range);
        if (b < 0) b = 0;
        if (b >= NUM_BUCKETS) b = NUM_BUCKETS - 1;
        atomicAdd(&local_h[b], 1);
    }
    __syncthreads();

    for (int i = tid; i < NUM_BUCKETS; i += blockDim.x) {
        if (local_h[i] > 0) atomicAdd(&global_histogram[i], local_h[i]);
    }
}

// Scatter: atomicAdd on per-bucket counters
__global__ void scatter_kernel(
    const int32_t* __restrict__ keys,
    int32_t* __restrict__ scatter_buf,
    int* __restrict__ counters,
    float est_min, float inv_range,
    int num_elements)
{
    int tid = threadIdx.x;
    int start = blockIdx.x * ITEMS_PER_BLOCK;
    int end = min(start + ITEMS_PER_BLOCK, num_elements);

    for (int i = start + tid; i < end; i += blockDim.x) {
        float v = __int_as_float(keys[i]);
        int b = (int)((v - est_min) * inv_range);
        if (b < 0) b = 0;
        if (b >= NUM_BUCKETS) b = NUM_BUCKETS - 1;
        int pos = atomicAdd(&counters[b], 1);
        scatter_buf[pos] = keys[i];
    }
}

torch::Tensor sort_cuda(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor histogram,
    torch::Tensor bucket_offsets,
    torch::Tensor scatter_buf,
    float est_min, float est_max)
{
    int64_t N = input.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const int32_t* key_in  = reinterpret_cast<const int32_t*>(input.const_data_ptr<float>());
    int32_t*       key_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());
    int*           d_hist   = histogram.data_ptr<int>();
    int*           d_offsets = bucket_offsets.data_ptr<int>();
    int32_t*       d_scatter = scatter_buf.data_ptr<int32_t>();

    float range = est_max - est_min;
    float inv_range = (range > 0.0f) ? (float)NUM_BUCKETS / range : 0.0f;

    // small-N fast path
    if (N <= 65536) {
        size_t bytes = persistent_temp_bytes;
        cub::DeviceRadixSort::SortKeys(persistent_temp.data_ptr(), bytes,
            key_in, key_out, N, 0, 32, stream);
        cudaStreamSynchronize(stream);
        return output;
    }

    int num_blocks = min(
        (static_cast<int>(N) + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK, 65535);

    // Step 1: Histogram
    cudaMemsetAsync(d_hist, 0, NUM_BUCKETS * sizeof(int), stream);
    histogram_kernel<<<num_blocks, BLOCK_SIZE, NUM_BUCKETS * sizeof(int), stream>>>(
        key_in, d_hist, est_min, inv_range, static_cast<int>(N));

    // Step 2: GPU-side exclusive scan (1 block, ~1us)
    gpu_scan_kernel<<<1, NUM_BUCKETS + 1, 0, stream>>>(d_hist, d_offsets);

    // Step 3: Scatter (use offsets as mutable counters)
    // Copy offsets to avoid clobbering
    // gpu_scan_kernel wrote offsets[0..NUM_BUCKETS]; we need to use them as counters
    // The scatter kernel increments them, so after scatter: offsets[b] = original_start_of_bucket + count_b
    scatter_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        key_in, d_scatter, d_offsets, est_min, inv_range, static_cast<int>(N));

    // Step 4: Read per-bucket sizes from the GPU
    cudaStreamSynchronize(stream);

    int h_offsets[NUM_BUCKETS + 1];
    cudaMemcpy(h_offsets, d_offsets, (NUM_BUCKETS + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    int h_sizes[NUM_BUCKETS];
    // h_offsets[b] is start + count, but we lost the starts.
    // We need to reread. Let me instead store the scan result separately.
    // Actually, we can just read histogram which tells us the sizes.
    int h_hist[NUM_BUCKETS];
    cudaMemcpy(h_hist, d_hist, NUM_BUCKETS * sizeof(int), cudaMemcpyDeviceToHost);

    // But we need the START positions. The scan wrote them to d_offsets before scatter.
    // After scatter, d_offsets[b] = start_b + size_b.
    // So start_b = (d_offsets after scan) which we lost.
    // Alternative: recalculate starts from histogram
    int running = 0;
    int h_starts[NUM_BUCKETS];
    for (int b = 0; b < NUM_BUCKETS; b++) {
        h_starts[b] = running;
        running += h_hist[b];
    }

    for (int b = 0; b < NUM_BUCKETS; b++) {
        int bsize = h_hist[b];
        if (bsize <= 0) continue;
        int32_t* src = d_scatter + h_starts[b];
        int32_t* dst = key_out + h_starts[b];

        size_t bytes = persistent_temp_bytes;
        cub::DeviceRadixSort::SortKeys(persistent_temp.data_ptr(), bytes,
            src, dst, static_cast<int64_t>(bsize), 0, 32, stream);
    }

    cudaStreamSynchronize(stream);
    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output,
                        torch::Tensor histogram,
                        torch::Tensor bucket_offsets,
                        torch::Tensor scatter_buf,
                        float est_min, float est_max);
"""

sort_module = load_inline(
    name='sort_cuda_coarse_bucket_v1',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    """
    Coarse bucket sort: 4 buckets with analytically-estimated range.
    No sampling overhead, fast GPU-only scan, minimal bucket sort overhead.
    """
    input_tensor, output_tensor = data
    input_tensor = input_tensor.contiguous()
    N = int(input_tensor.numel())

    # Estimate range from problem parameters: data has sqrt(N) rows with means
    # seed..seed+rows-1, each with std=1 normal. Use 5-sigma coverage.
    # We don't know the seed at Python level, but we can estimate min/max
    # from the data or from a conservative bound.
    #
    # Conservative range: sample a few elements to get a rough range.
    # For efficiency, use first 256 elements as a quick sample.
    flat = input_tensor.flatten()
    sample_n = min(256, N)
    sample = flat[:sample_n]
    sample_min = float(sample.min())
    sample_max = float(sample.max())

    # Expand range by 20% to account for sample being non-representative
    # (first 256 are from the first row, which has a narrow distribution)
    # Better: use seed-based estimation from the problem.
    # Since we can't access the seed, use the data itself.
    # Take samples from start and end of the array (row 0 and last row)
    rows = int(math.sqrt(N))
    if rows > 0:
        last_row_start = rows * (rows)  # approximate
        if last_row_start < N:
            end_sample = flat[max(0, N - 256):N]
            est_min = float(sample.min()) - 10.0
            est_max = float(end_sample.max()) + 10.0
        else:
            est_min = float(sample.min()) - 10.0
            est_max = float(sample.max()) + 10.0
    else:
        est_min = sample_min - 10.0
        est_max = sample_max + 10.0

    # Ensure range is positive
    if est_max <= est_min:
        est_max = est_min + 1.0

    # GPU buffers
    histogram = torch.zeros(4, dtype=torch.int32, device='cuda')
    bucket_offsets = torch.empty(5, dtype=torch.int32, device='cuda')
    scatter_buf = torch.empty(N, dtype=torch.int32, device='cuda')

    sort_module.sort_cuda(
        input_tensor, output_tensor,
        histogram, bucket_offsets, scatter_buf,
        est_min, est_max,
    )
    return output_tensor