"""
Custom LSD radix sort with CORRECT bin indexing.
Each lane owns bins at stride WARP_SIZE: lane_id -> bins {lane_id, lane_id+32, lane_id+64, ...}
BINS_PER_LANE = 8 (= 256/32), so lanes collectively own all 256 bins.
Shuffle exchange: lane broadcasts its digits, other lanes count if they own that bin.
The "owner" of bin d is d % WARP_SIZE, and the position within lane is d / WARP_SIZE.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

radix_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

constexpr int ITEMS_PER_THREAD = 8;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int ITEMS_PER_BLOCK = ITEMS_PER_THREAD * THREADS_PER_BLOCK;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;
constexpr int NUM_BINS = 256;
constexpr int BINS_PER_LANE = NUM_BINS / WARP_SIZE;
constexpr int RADIX_BITS = 8;
constexpr int NUM_PASSES = 4;

static torch::Tensor d_scratch0 = {};
static torch::Tensor d_scratch1 = {};
static torch::Tensor d_histogram = {};

void init_persistent() {
    int64_t n = 100'000'000;
    int64_t max_blocks = (n + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
    d_scratch0 = torch::empty({n},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    d_scratch1 = torch::empty({n},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    d_histogram = torch::zeros({max_blocks * NUM_BINS},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
}

// Kernel 1: Histogram with warp-aggregated shuffle reduction.
// Each lane owns BINS_PER_LANE bins at stride WARP_SIZE.
// lane_id owns bins: lane_id, lane_id+WARP_SIZE, lane_id+2*WARP_SIZE, ...
// Within lane_hist array: lane_hist[i] = count for bin (i*WARP_SIZE + lane_id)
__global__ void histogram_kernel(
    const uint32_t* d_in, uint32_t* d_histogram,
    int64_t num_items, int pass, int num_blocks)
{
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    int shift = pass * RADIX_BITS;
    int64_t block_start = (int64_t)block_id * ITEMS_PER_BLOCK;

    __shared__ uint32_t warp_hists[WARPS_PER_BLOCK * NUM_BINS];
    int warp_id  = threadIdx.x / WARP_SIZE;
    int lane_id  = threadIdx.x % WARP_SIZE;

    uint32_t* my_warp_hist = warp_hists + warp_id * NUM_BINS;
    for (int i = lane_id; i < NUM_BINS; i += WARP_SIZE) my_warp_hist[i] = 0;
    __syncthreads();

    uint32_t items[ITEMS_PER_THREAD];
    int digits_arr[ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int64_t idx = block_start + threadIdx.x + (int64_t)i * THREADS_PER_BLOCK;
        if (idx < num_items) {
            items[i] = d_in[idx];
            digits_arr[i] = (int)(items[i] >> shift) & 0xFF;
        } else {
            digits_arr[i] = -1;
        }
    }

    // Lane-local histogram: lane_hist[i] corresponds to bin (i*32 + lane_id)
    uint32_t lane_hist[BINS_PER_LANE] = {0};
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int d = digits_arr[i];
        if (d >= 0) {
            int owner = d % WARP_SIZE;
            if (owner == lane_id) lane_hist[d / WARP_SIZE]++;
        }
    }

    // Shuffle exchange: each lane broadcasts its digits, other lanes count
    uint32_t lane_mask = __activemask();
    for (int src_lane = 0; src_lane < WARP_SIZE; src_lane++) {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int d = __shfl_sync(lane_mask, digits_arr[i], src_lane);
            if (d >= 0) {
                int owner = d % WARP_SIZE;
                if (owner == lane_id && src_lane != lane_id) {
                    lane_hist[d / WARP_SIZE]++;
                }
            }
        }
    }

    // Accumulate lane counts into per-warp histogram
    // bin = i * WARP_SIZE + lane_id (correct indexing)
    #pragma unroll
    for (int i = 0; i < BINS_PER_LANE; i++) {
        if (lane_hist[i] > 0) {
            int bin = i * WARP_SIZE + lane_id;  // FIXED: b*WARP_SIZE+lane_id
            atomicAdd(&my_warp_hist[bin], lane_hist[i]);
        }
    }
    __syncthreads();

    // Reduce warps to global histogram
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += THREADS_PER_BLOCK) {
        uint32_t sum = 0;
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            sum += warp_hists[w * NUM_BINS + bin];
        }
        d_histogram[(int64_t)block_id * NUM_BINS + bin] = sum;
    }
}

// Kernel 2: Device-level exclusive prefix sum per bin
__global__ void prefix_sum_kernel(uint32_t* d_histogram, int num_blocks) {
    int bin = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ uint32_t s_data[THREADS_PER_BLOCK];
    __shared__ uint32_t s_warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ uint32_t s_chunk_total;

    uint32_t carry = 0;

    for (int chunk_start = 0; chunk_start < num_blocks; chunk_start += THREADS_PER_BLOCK) {
        int chunk_sz = THREADS_PER_BLOCK;
        if (chunk_start + chunk_sz > num_blocks) chunk_sz = num_blocks - chunk_start;

        if (tid < chunk_sz)
            s_data[tid] = d_histogram[((int64_t)(chunk_start + tid) * NUM_BINS) + bin];
        else
            s_data[tid] = 0;
        __syncthreads();

        uint32_t val = s_data[tid];
        unsigned active = __ballot_sync(0xFFFFFFFF, tid < chunk_sz);

        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            uint32_t n = __shfl_up_sync(active, val, offset);
            if (tid < chunk_sz && (tid % WARP_SIZE) >= offset) val += n;
        }

        if (tid % WARP_SIZE == WARP_SIZE - 1 && tid < chunk_sz)
            s_warp_sums[tid / WARP_SIZE] = val;
        __syncthreads();

        uint32_t warp_prefix = 0;
        int my_warp = tid / WARP_SIZE;
        if (my_warp > 0 && tid < chunk_sz) {
            for (int w = 0; w < my_warp; w++) warp_prefix += s_warp_sums[w];
        }

        uint32_t scanned = (tid < chunk_sz) ? val + warp_prefix : 0;

        if (tid < chunk_sz)
            d_histogram[((int64_t)(chunk_start + tid) * NUM_BINS) + bin] = carry + scanned - s_data[tid];

        if (tid == chunk_sz - 1) s_chunk_total = scanned;
        __syncthreads();
        carry += s_chunk_total;
        __syncthreads();
    }
}

// Kernel 3: Scatter with warp-aggregated histogram and per-warp atomic scatter
__global__ void scatter_kernel(
    const uint32_t* d_in, uint32_t* d_out,
    const uint32_t* d_prefix,
    int64_t num_items, int pass, int num_blocks)
{
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    int shift = pass * RADIX_BITS;
    int64_t block_start = (int64_t)block_id * ITEMS_PER_BLOCK;
    int warp_id  = threadIdx.x / WARP_SIZE;
    int lane_id  = threadIdx.x % WARP_SIZE;
    uint32_t lane_mask = __activemask();

    __shared__ uint32_t warp_data[WARPS_PER_BLOCK * NUM_BINS];
    __shared__ uint32_t warp_ctrs[WARPS_PER_BLOCK * NUM_BINS];

    uint32_t items[ITEMS_PER_THREAD];
    int digits_arr[ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int64_t idx = block_start + threadIdx.x + (int64_t)i * THREADS_PER_BLOCK;
        if (idx < num_items) {
            items[i] = d_in[idx];
            digits_arr[i] = (int)(items[i] >> shift) & 0xFF;
        } else {
            digits_arr[i] = -1;
        }
    }

    // A) Build per-warp histogram (same as histogram_kernel, fix bin indexing)
    uint32_t* my_warp = warp_data + warp_id * NUM_BINS;
    for (int i = lane_id; i < NUM_BINS; i += WARP_SIZE) my_warp[i] = 0;
    __syncthreads();

    uint32_t lane_hist[BINS_PER_LANE] = {0};
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int d = digits_arr[i];
        if (d >= 0) {
            int owner = d % WARP_SIZE;
            if (owner == lane_id) lane_hist[d / WARP_SIZE]++;
        }
    }
    for (int src_lane = 0; src_lane < WARP_SIZE; src_lane++) {
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int d = __shfl_sync(lane_mask, digits_arr[i], src_lane);
            if (d >= 0) {
                int owner = d % WARP_SIZE;
                if (owner == lane_id && src_lane != lane_id) {
                    lane_hist[d / WARP_SIZE]++;
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < BINS_PER_LANE; i++) {
        if (lane_hist[i] > 0) {
            int bin = i * WARP_SIZE + lane_id;  // FIXED: b*WARP_SIZE+lane_id
            atomicAdd(&my_warp[bin], lane_hist[i]);
        }
    }
    __syncthreads();

    // B) Compute per-warp per-bin base offsets from global prefix
    int64_t pref_base = (int64_t)block_id * NUM_BINS;
    if (threadIdx.x < NUM_BINS) {
        int bin = threadIdx.x;
        uint32_t running = d_prefix[pref_base + bin];
        for (int w = 0; w < WARPS_PER_BLOCK; w++) {
            uint32_t cnt = warp_data[w * NUM_BINS + bin];
            warp_data[w * NUM_BINS + bin] = running;
            running += cnt;
        }
    }
    __syncthreads();

    // C) Load per-warp base offsets into per-warp counters
    uint32_t* my_ctr = warp_ctrs + warp_id * NUM_BINS;
    for (int i = lane_id; i < NUM_BINS; i += WARP_SIZE) {
        my_ctr[i] = warp_data[warp_id * NUM_BINS + i];
    }
    __syncwarp();

    // D) Scatter using per-warp atomicAdd counters
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int64_t idx = block_start + threadIdx.x + (int64_t)i * THREADS_PER_BLOCK;
        if (idx < num_items) {
            int bin = digits_arr[i];
            uint32_t pos = atomicAdd(&my_ctr[bin], 1);
            d_out[pos] = items[i];
        }
    }
}

// Top-level orchestrator
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    int64_t num_items = input.numel();
    int blocks = (int)((num_items + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const uint32_t* d_in = reinterpret_cast<const uint32_t*>(input.const_data_ptr<float>());
    uint32_t* d_out = reinterpret_cast<uint32_t*>(output.data_ptr<float>());
    uint32_t* d_s0 = reinterpret_cast<uint32_t*>(d_scratch0.data_ptr<int32_t>());
    uint32_t* d_s1 = reinterpret_cast<uint32_t*>(d_scratch1.data_ptr<int32_t>());
    uint32_t* d_hist = reinterpret_cast<uint32_t*>(d_histogram.data_ptr<int32_t>());

    for (int pass = 0; pass < NUM_PASSES; pass++) {
        const uint32_t* src = (pass == 0) ? d_in : ((pass == 2) ? d_s1 : d_s0);
        uint32_t* dst = (pass == 3) ? d_out : ((pass == 1) ? d_s1 : d_s0);

        histogram_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            src, d_hist, num_items, pass, blocks);
        prefix_sum_kernel<<<NUM_BINS, THREADS_PER_BLOCK, 0, stream>>>(
            d_hist, blocks);
        scatter_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            src, dst, d_hist, num_items, pass, blocks);
    }

    cudaStreamSynchronize(stream);
    return output;
}
"""

radix_cpp = r"""
#include <torch/extension.h>
void init_persistent();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

radix_module = load_inline(
    name='custom_radix_sort_bin_fix',
    cpp_sources=radix_cpp,
    cuda_sources=radix_source,
    functions=['sort_cuda', 'init_persistent'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

radix_module.init_persistent()


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    radix_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor