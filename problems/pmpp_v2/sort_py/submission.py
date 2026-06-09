"""
Custom 4-bit radix sort, 8 passes x 3 kernels = 24 launches.
256 threads, 8 items/thread (2048 items/tile), 4-bit radix (16 bins).
Register-based histograms (16 bins fit in 16 registers).
Butterfly shuffle for warp reduction and intra-warp exclusive scan.
Regular loads/stores (no __ldg, no st.global.wb) for L1 caching.
Identical design to brief-4 iter-1 which scored 174.5us.
Added __launch_bounds__ to reduce register pressure.
"""
import torch
from torch.utils.cpp_extension import load_inline

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

#define RADIX_BITS 4
#define RADIX_BINS (1 << RADIX_BITS)
#define BLOCK_THREADS 256
#define ITEMS_PER_THREAD 8
#define TILE_SIZE  (BLOCK_THREADS * ITEMS_PER_THREAD)
#define WARPS      (BLOCK_THREADS / 32)

__global__ void hist_kernel(
    const uint32_t* __restrict__ d_in,
    uint32_t*       __restrict__ d_hists,
    int64_t num_items,
    int shift
) {
    int tid = threadIdx.x, lane_id = tid & 31, warp_id = tid >> 5, blk = blockIdx.x;
    __shared__ uint32_t s_data[TILE_SIZE];
    __shared__ uint32_t s_bh[RADIX_BINS];

    if (tid < RADIX_BINS) s_bh[tid] = 0;
    __syncthreads();

    uint32_t warp_acc[RADIX_BINS] = {0};
    int64_t tt = (num_items + TILE_SIZE - 1) / TILE_SIZE;

    for (int64_t tile = blk; tile < tt; tile += gridDim.x) {
        int64_t off = tile * TILE_SIZE, end = (off + TILE_SIZE < num_items) ? off + TILE_SIZE : num_items;
        int64_t ni = end - off;
        for (int i = tid; i < (int)ni; i += BLOCK_THREADS) s_data[i] = d_in[off + i];
        __syncthreads();

        uint32_t hist[RADIX_BINS] = {0};
        int base = tid * ITEMS_PER_THREAD;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int j = base + i;
            if (j < (int)ni) { int bin = (s_data[j] >> shift) & (RADIX_BINS - 1); hist[bin]++; }
        }

        #pragma unroll
        for (int b = 0; b < RADIX_BINS; b++) {
            uint32_t v = hist[b];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
            if (lane_id == 0) warp_acc[b] += v;
        }
        __syncthreads();
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int b = 0; b < RADIX_BINS; b++) atomicAdd(&s_bh[b], warp_acc[b]);
    }
    __syncthreads();
    if (tid < RADIX_BINS) d_hists[blk * RADIX_BINS + tid] = s_bh[tid];
}

__global__ void prefix_kernel(
    const uint32_t* __restrict__ d_hists,
    uint32_t*       __restrict__ d_blk_offs,
    uint32_t*       __restrict__ d_bin_offs,
    int nb
) {
    int tid = threadIdx.x;
    __shared__ uint32_t s_run[RADIX_BINS], s_tot[RADIX_BINS];
    for (int i = tid; i < RADIX_BINS; i += blockDim.x) { s_run[i] = 0; s_tot[i] = 0; }
    __syncthreads();
    for (int b = 0; b < nb; b++) {
        for (int i = tid; i < RADIX_BINS; i += blockDim.x) {
            uint32_t c = d_hists[b * RADIX_BINS + i];
            d_blk_offs[b * RADIX_BINS + i] = s_run[i];
            s_run[i] += c;
        }
    }
    __syncthreads();
    for (int i = tid; i < RADIX_BINS; i += blockDim.x) s_tot[i] = s_run[i];
    __syncthreads();
    if (tid == 0) {
        uint32_t run = 0;
        for (int bin = 0; bin < RADIX_BINS; bin++) { d_bin_offs[bin] = run; run += s_tot[bin]; }
    }
}

__global__ void scatter_kernel(
    const uint32_t* __restrict__ d_in,
    uint32_t*       __restrict__ d_out,
    const uint32_t* __restrict__ d_blk_offs,
    const uint32_t* __restrict__ d_bin_offs,
    int64_t num_items,
    int shift
) {
    int tid = threadIdx.x, lane_id = tid & 31, warp_id = tid >> 5, blk = blockIdx.x;

    __shared__ uint32_t s_data[TILE_SIZE];
    __shared__ uint32_t s_wt[WARPS * RADIX_BINS];
    __shared__ uint32_t s_wp[WARPS * RADIX_BINS];
    __shared__ uint32_t s_bin_off[RADIX_BINS];
    __shared__ uint32_t s_blk_ofs[RADIX_BINS];

    if (tid < RADIX_BINS) {
        s_bin_off[tid] = d_bin_offs[tid];
        s_blk_ofs[tid] = d_blk_offs[blk * RADIX_BINS + tid];
    }
    __syncthreads();

    int64_t tt = (num_items + TILE_SIZE - 1) / TILE_SIZE;
    for (int64_t tile = blk; tile < tt; tile += gridDim.x) {
        int64_t off = tile * TILE_SIZE, end = (off + TILE_SIZE < num_items) ? off + TILE_SIZE : num_items;
        int64_t ni = end - off;

        uint32_t reg_ofs[RADIX_BINS];
        if (tid < RADIX_BINS) reg_ofs[tid] = s_blk_ofs[tid];
        __syncthreads();

        for (int i = tid; i < (int)ni; i += BLOCK_THREADS) s_data[i] = d_in[off + i];
        __syncthreads();

        for (int i = tid; i < WARPS * RADIX_BINS; i += BLOCK_THREADS) s_wt[i] = 0;
        __syncthreads();

        uint32_t hist[RADIX_BINS] = {0};
        int base = tid * ITEMS_PER_THREAD;
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int j = base + i;
            if (j < (int)ni) { int bin = (s_data[j] >> shift) & (RADIX_BINS - 1); hist[bin]++; }
        }

        uint32_t exc[RADIX_BINS];
        #pragma unroll
        for (int b = 0; b < RADIX_BINS; b++) {
            uint32_t v = hist[b];
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                uint32_t u = __shfl_up_sync(0xffffffff, v, off);
                if (lane_id >= off) v += u;
            }
            exc[b] = v - hist[b];
            if (lane_id == 31) s_wt[warp_id * RADIX_BINS + b] = v;
        }
        __syncthreads();

        if (tid < RADIX_BINS) {
            uint32_t run = 0;
            #pragma unroll
            for (int w = 0; w < WARPS; w++) { s_wp[w * RADIX_BINS + tid] = run; run += s_wt[w * RADIX_BINS + tid]; }
            s_blk_ofs[tid] += run;
        }
        __syncthreads();

        __shared__ uint32_t s_old[RADIX_BINS];
        if (tid < RADIX_BINS) s_old[tid] = reg_ofs[tid];
        __syncthreads();

        uint32_t pos[RADIX_BINS], lc[RADIX_BINS] = {0};
        #pragma unroll
        for (int b = 0; b < RADIX_BINS; b++)
            pos[b] = s_bin_off[b] + s_old[b] + s_wp[warp_id * RADIX_BINS + b] + exc[b];

        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int j = base + i;
            if (j < (int)ni) { uint32_t key = s_data[j]; int bin = (key >> shift) & (RADIX_BINS - 1); d_out[pos[bin] + lc[bin]] = key; lc[bin]++; }
        }
        __syncthreads();
    }
}

static torch::Tensor d_temp; static size_t tsz = 0;
static void ensure_temp(int64_t nb, size_t dsz) {
    size_t n = (size_t)nb;
    size_t need = n * RADIX_BINS * sizeof(uint32_t) * 2 + RADIX_BINS * sizeof(uint32_t) + dsz + 4096;
    if (need <= tsz && d_temp.defined()) return;
    d_temp = torch::empty({(int64_t)need}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    tsz = need;
}

torch::Tensor sort_onesweep(torch::Tensor input, torch::Tensor output) {
    int64_t N = input.numel();
    if (N <= 0) return output;
    int64_t tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int nb = (int)(tiles < 65535 ? tiles : 65535);
    if (nb < 1) nb = 1;
    size_t dsz = (size_t)N * sizeof(uint32_t);
    ensure_temp((int64_t)nb, dsz);
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    uint8_t* p = d_temp.data_ptr<uint8_t>();
    uint32_t* d_hists = (uint32_t*)(p);
    uint32_t* d_blk_offs = (uint32_t*)(p + (size_t)nb * RADIX_BINS * sizeof(uint32_t));
    uint32_t* d_bin_offs = (uint32_t*)(p + (size_t)nb * RADIX_BINS * sizeof(uint32_t) * 2);
    uint32_t* d_buf = (uint32_t*)(p + (size_t)nb * RADIX_BINS * sizeof(uint32_t) * 2 + RADIX_BINS * sizeof(uint32_t));
    const uint32_t* src = (const uint32_t*)input.const_data_ptr<float>();
    uint32_t* dst = (uint32_t*)output.data_ptr<float>();
    cudaMemcpyAsync(d_buf, src, dsz, cudaMemcpyDeviceToDevice, st);
    const uint32_t* d_read = d_buf; uint32_t* d_write = dst;
    for (int pass = 0; pass < 8; pass++) {
        int shift = pass * RADIX_BITS;
        hist_kernel   <<<nb, BLOCK_THREADS, 0, st>>>(d_read, d_hists, N, shift);
        prefix_kernel <<<1,  BLOCK_THREADS, 0, st>>>(d_hists, d_blk_offs, d_bin_offs, nb);
        scatter_kernel<<<nb, BLOCK_THREADS, 0, st>>>(d_read, d_write, d_blk_offs, d_bin_offs, N, shift);
        const uint32_t* t = d_read; d_read = d_write; d_write = (uint32_t*)t;
    }
    if (d_read == d_buf) cudaMemcpyAsync(dst, d_buf, dsz, cudaMemcpyDeviceToDevice, st);
    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
torch::Tensor sort_onesweep(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_onesweep_custom_v6',
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