"""
Custom ONESWEEP-style radix sort kernel targeting sm_100/B200.
256 threads, 8 items/thread (2048 items/tile), 4-bit radix (16 bins).
Per-thread register-based histogram, warp-level shuffle reduction.
ld.global.nc (via __ldg) for reads, st.global.wb (inline PTX) for writes.
"""
import torch
from torch.utils.cpp_extension import load_inline

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

#define RADIX_BITS 4
#define RADIX_BINS (1 << RADIX_BITS)           // 16
#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 8
#define TILE_SIZE  (BLOCK_THREADS * ITEMS_PER_THREAD)  // 2048
#define WARPS      (BLOCK_THREADS / 32)                // 8

__device__ __forceinline__ void stwb_u32(uint32_t* addr, uint32_t val) {
    asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(addr), "r"(val) : "memory");
}

// ================================================================
// Histogram kernel: grid-stride over tiles, block-level histogram
// ================================================================
__global__ void hist_kernel(
    const uint32_t* __restrict__ d_in,
    uint32_t*       __restrict__ d_block_hists, // [num_blocks * 16]
    int64_t num_items,
    int shift
) {
    const int block_id   = blockIdx.x;
    const int num_blocks = gridDim.x;
    const int tid        = threadIdx.x;
    const int lane_id    = tid & 31;
    const int warp_id    = tid >> 5;

    __shared__ uint32_t s_data[TILE_SIZE];
    __shared__ uint32_t s_hist[RADIX_BINS];

    if (tid < RADIX_BINS) s_hist[tid] = 0;
    __syncthreads();

    uint32_t warp_acc[RADIX_BINS] = {0};

    int64_t total_tiles = (num_items + TILE_SIZE - 1) / TILE_SIZE;
    for (int64_t t = block_id; t < total_tiles; t += num_blocks) {
        int64_t start  = t * TILE_SIZE;
        int64_t end    = (start + TILE_SIZE < num_items) ? start + TILE_SIZE : num_items;
        int64_t nitems = end - start;

        for (int i = tid; i < (int)nitems; i += BLOCK_THREADS)
            s_data[i] = __ldg(d_in + start + i);
        __syncthreads();

        uint32_t hist[RADIX_BINS] = {0};
        int base = tid * ITEMS_PER_THREAD;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int j = base + i;
            if (j < (int)nitems) {
                int bin = (s_data[j] >> shift) & (RADIX_BINS - 1);
                hist[bin]++;
            }
        }

        #pragma unroll
        for (int bin = 0; bin < RADIX_BINS; bin++) {
            uint32_t v = hist[bin];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (lane_id == 0) warp_acc[bin] += v;
        }
        __syncthreads();
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int bin = 0; bin < RADIX_BINS; bin++)
            atomicAdd(&s_hist[bin], warp_acc[bin]);
    }
    __syncthreads();

    if (tid < RADIX_BINS)
        d_block_hists[block_id * RADIX_BINS + tid] = s_hist[tid];
}

// ================================================================
// Prefix-sum kernel: computes (a) per-block-per-bin exclusive prefix
// and (b) global cross-bin offsets into the output array
// ================================================================
__global__ void prefix_kernel(
    const uint32_t* __restrict__ d_hists,
    uint32_t*       __restrict__ d_offsets,     // [num_blocks * 16] per-block prefix
    uint32_t*       __restrict__ d_bin_offsets, // [16] cross-bin global prefix
    int num_blocks
) {
    int tid = threadIdx.x;
    __shared__ uint32_t s_running[RADIX_BINS];
    __shared__ uint32_t s_total[RADIX_BINS];

    if (tid < RADIX_BINS) {
        s_running[tid] = 0;
        s_total[tid] = 0;
    }
    __syncthreads();

    // Phase 1: per-block exclusive prefix within each bin
    for (int blk = 0; blk < num_blocks; blk++) {
        int bin = tid;
        if (bin < RADIX_BINS) {
            uint32_t cnt = d_hists[blk * RADIX_BINS + bin];
            d_offsets[blk * RADIX_BINS + bin] = s_running[bin];
            s_running[bin] += cnt;
        }
    }
    __syncthreads();

    // At this point s_running[bin] = total count of elements in bin 'bin'

    if (tid < RADIX_BINS) s_total[tid] = s_running[tid];
    __syncthreads();

    // Phase 2: cross-bin prefix sum (exclusive scan across bins)
    // d_bin_offsets[bin] = sum of totals of bins 0..bin-1
    if (tid == 0) {
        uint32_t run = 0;
        for (int bin = 0; bin < RADIX_BINS; bin++) {
            d_bin_offsets[bin] = run;
            run += s_total[bin];
        }
    }
    // single-threaded is fine for 16 bins
}

// ================================================================
// Scatter kernel: one __ldg read per tile, st.global.wb writes.
// Warp-level exclusive scan for scatter positions within a tile.
// ================================================================
__global__ void scatter_kernel(
    const uint32_t* __restrict__ d_in,
    uint32_t*       __restrict__ d_out,
    const uint32_t* __restrict__ d_blk_offs,    // [num_blocks * 16] per-block per-bin exclusive prefix
    const uint32_t* __restrict__ d_bin_offsets, // [16] cross-bin global prefix
    int64_t num_items,
    int shift
) {
    const int block_id   = blockIdx.x;
    const int num_blocks = gridDim.x;
    const int tid        = threadIdx.x;
    const int lane_id    = tid & 31;
    const int warp_id    = tid >> 5;

    __shared__ uint32_t s_data[TILE_SIZE];
    __shared__ uint32_t s_warp_total [WARPS][RADIX_BINS];
    __shared__ uint32_t s_warp_prefix[WARPS][RADIX_BINS];

    // Load this block's per-bin offsets + global bin offsets
    __shared__ uint32_t s_blk_off[RADIX_BINS];
    __shared__ uint32_t s_bin_off[RADIX_BINS];
    if (tid < RADIX_BINS) {
        s_blk_off[tid] = d_blk_offs[block_id * RADIX_BINS + tid];
        s_bin_off[tid] = d_bin_offsets[tid];
    }
    __syncthreads();

    int64_t total_tiles = (num_items + TILE_SIZE - 1) / TILE_SIZE;
    for (int64_t t = block_id; t < total_tiles; t += num_blocks) {
        int64_t start  = t * TILE_SIZE;
        int64_t end    = (start + TILE_SIZE < num_items) ? start + TILE_SIZE : num_items;
        int64_t nitems = end - start;

        // A: load tile using __ldg (ld.global.nc)
        for (int i = tid; i < (int)nitems; i += BLOCK_THREADS)
            s_data[i] = __ldg(d_in + start + i);
        __syncthreads();

        // B: zero s_warp_total
        for (int i = tid; i < WARPS * RADIX_BINS; i += BLOCK_THREADS)
            s_warp_total[i / RADIX_BINS][i % RADIX_BINS] = 0;
        __syncthreads();

        // C: per-thread histogram + intra-warp inclusive scan
        uint32_t hist[RADIX_BINS] = {0};
        int base = tid * ITEMS_PER_THREAD;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int j = base + i;
            if (j < (int)nitems) {
                int bin = (s_data[j] >> shift) & (RADIX_BINS - 1);
                hist[bin]++;
            }
        }

        uint32_t exc[RADIX_BINS];
        #pragma unroll
        for (int bin = 0; bin < RADIX_BINS; bin++) {
            uint32_t v = hist[bin];
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                uint32_t u = __shfl_up_sync(0xffffffff, v, off);
                if (lane_id >= off) v += u;
            }
            uint32_t inc = v;
            exc[bin]     = inc - hist[bin];
            if (lane_id == 31) s_warp_total[warp_id][bin] = inc;
        }
        __syncthreads();

        // D: inter-warp exclusive prefix + save old block offsets
        uint32_t reg_blk_off[RADIX_BINS];
        if (tid < RADIX_BINS) reg_blk_off[tid] = s_blk_off[tid];

        if (lane_id < RADIX_BINS) {
            uint32_t run = 0;
            #pragma unroll
            for (int w = 0; w < WARPS; w++) {
                s_warp_prefix[w][lane_id] = run;
                run += s_warp_total[w][lane_id];
            }
            s_blk_off[lane_id] += run;
        }
        __syncthreads();

        __shared__ uint32_t s_old_blk_off[RADIX_BINS];
        if (tid < RADIX_BINS) s_old_blk_off[tid] = reg_blk_off[tid];
        __syncthreads();

        // E: scatter — base position = bin_off + blk_off + warp_off + lane_off
        uint32_t base_pos[RADIX_BINS];
        uint32_t local_cnt[RADIX_BINS] = {0};
        #pragma unroll
        for (int bin = 0; bin < RADIX_BINS; bin++) {
            base_pos[bin] = s_bin_off[bin] + s_old_blk_off[bin]
                          + s_warp_prefix[warp_id][bin] + exc[bin];
        }

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int j = base + i;
            if (j < (int)nitems) {
                uint32_t key = s_data[j];
                int bin = (key >> shift) & (RADIX_BINS - 1);
                uint32_t addr = base_pos[bin] + local_cnt[bin];
                local_cnt[bin]++;
                stwb_u32(d_out + addr, key);
            }
        }
        __syncthreads();
    }
}

// ================================================================
// Host driver
// ================================================================
static torch::Tensor d_temp;
static size_t       temp_sz = 0;

static void ensure_temp(int64_t nb, size_t data_sz) {
    size_t need = nb * RADIX_BINS * sizeof(uint32_t) * 2   // hists + blk_offsets
                + RADIX_BINS * sizeof(uint32_t)             // bin_offsets
                + data_sz + 4096;
    if (need <= temp_sz && d_temp.defined()) return;
    d_temp = torch::empty({(int64_t)need},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    temp_sz = need;
}

torch::Tensor sort_onesweep(torch::Tensor input, torch::Tensor output) {
    int64_t N = input.numel();
    if (N <= 0) return output;

    int64_t tiles    = (N + TILE_SIZE - 1) / TILE_SIZE;
    int     nb       = (tiles < 65535) ? (int)tiles : 65535;
    if (nb < 1) nb = 1;

    size_t data_sz = (size_t)N * sizeof(uint32_t);
    ensure_temp((int64_t)nb, data_sz);

    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    uint8_t* p = d_temp.data_ptr<uint8_t>();

    uint32_t* d_hists      = (uint32_t*)(p);
    uint32_t* d_blk_offs   = (uint32_t*)(p + nb * RADIX_BINS * sizeof(uint32_t));
    uint32_t* d_bin_offs   = (uint32_t*)(p + nb * RADIX_BINS * sizeof(uint32_t) * 2);
    uint32_t* d_buf        = (uint32_t*)(p + nb * RADIX_BINS * sizeof(uint32_t) * 2
                                              + RADIX_BINS * sizeof(uint32_t));

    const uint32_t* src = (const uint32_t*)input.const_data_ptr<float>();
    uint32_t* dst       = (uint32_t*)output.data_ptr<float>();

    // Copy input → temp buffer
    cudaMemcpyAsync(d_buf, src, data_sz, cudaMemcpyDeviceToDevice, st);

    const uint32_t* d_read = d_buf;
    uint32_t* d_write       = dst;

    for (int pass = 0; pass < 8; pass++) {
        int shift = pass * RADIX_BITS;
        hist_kernel   <<<nb, BLOCK_THREADS, 0, st>>>(d_read, d_hists, N, shift);
        prefix_kernel <<<1,  BLOCK_THREADS, 0, st>>>(d_hists, d_blk_offs, d_bin_offs, nb);
        scatter_kernel<<<nb, BLOCK_THREADS, 0, st>>>(d_read, d_write, d_blk_offs, d_bin_offs, N, shift);

        const uint32_t* tmp = d_read;
        d_read = d_write;
        d_write = (uint32_t*)tmp;
    }

    if (d_read == d_buf)
        cudaMemcpyAsync(dst, d_buf, data_sz, cudaMemcpyDeviceToDevice, st);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
torch::Tensor sort_onesweep(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_onesweep_custom',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_onesweep'],
    extra_cuda_cflags=['-O3', '-lineinfo'],
    verbose=False,
)


def custom_kernel(data):
    input_tensor, output_tensor = data
    sort_module.sort_onesweep(input_tensor.contiguous(), output_tensor)
    return output_tensor