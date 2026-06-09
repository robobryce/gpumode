"""
Cooperative-groups single-kernel radix sort: all 4 radix passes in one cooperative launch.
Eliminates CUB's 4 internal kernel launches by using grid-level sync (this_grid().sync()).
Uses cub::BlockRadixSort for per-block sorting within each pass.
This is the structural answer to the brief's question: can we beat 174us without graph capture?
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/block/block_radix_sort.cuh>
#include <cooperative_groups.h>
#include <cstdint>

static constexpr int RADIX_BITS = 8;
static constexpr int RADIX_BINS = 256;
static constexpr int BLOCK_THREADS = 256;
static constexpr int ITEMS_PER_THREAD = 4;

using BlockRadixSort = cub::BlockRadixSort<
    int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>;

__global__ void cooperative_radix_sort_kernel(
    const int32_t* __restrict__ input,
    int32_t* __restrict__ output,
    int32_t* __restrict__ temp,
    uint32_t* __restrict__ global_histogram,
    uint32_t* __restrict__ global_offsets,
    uint64_t num_items)
{
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();

    __shared__ typename BlockRadixSort::TempStorage sort_temp;
    __shared__ uint32_t smem_hist[RADIX_BINS];
    __shared__ uint32_t smem_offsets[RADIX_BINS];

    int32_t keys[ITEMS_PER_THREAD];
    int32_t sorted_keys[ITEMS_PER_THREAD];

    uint64_t total_blocks = gridDim.x;
    uint64_t block_id = blockIdx.x;
    uint64_t items_per_block = (num_items + total_blocks - 1) / total_blocks;
    uint64_t start = block_id * items_per_block;
    uint64_t end = (start + items_per_block > num_items) ? num_items : start + items_per_block;
    uint64_t block_items = end - start;

    const int32_t* src = input;
    int32_t* dst = temp;

    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * RADIX_BITS;
        uint32_t mask = (1u << RADIX_BITS) - 1;

        // PHASE 1: Load data into registers
        uint64_t valid_count = 0;
        for (uint64_t idx = start + threadIdx.x; idx < end; idx += BLOCK_THREADS) {
            if (idx < num_items) {
                keys[valid_count] = src[idx];
                valid_count++;
            }
        }
        // Fill remaining with max int32
        for (uint64_t i = valid_count; i < ITEMS_PER_THREAD; i++) {
            keys[i] = 0x7FFFFFFF;
        }

        // PHASE 2: Block-level radix sort to compute per-block histogram
        // Clear local histogram
        for (int i = threadIdx.x; i < RADIX_BINS; i += BLOCK_THREADS) {
            smem_hist[i] = 0;
        }
        block.sync();

        for (uint64_t idx = start + threadIdx.x; idx < end; idx += BLOCK_THREADS) {
            if (idx < num_items) {
                uint8_t digit = (static_cast<uint32_t>(src[idx]) >> shift) & mask;
                atomicAdd(&smem_hist[digit], 1);
            }
        }
        block.sync();

        // PHASE 3: Write per-block histogram to global memory
        if (threadIdx.x < RADIX_BINS) {
            global_histogram[block_id * RADIX_BINS + threadIdx.x] = smem_hist[threadIdx.x];
        }

        // PHASE 4: Grid sync — wait for all blocks to write their histograms
        grid.sync();

        // PHASE 5: Block 0 computes global prefix sum
        if (block_id == 0) {
            // Compute global histogram
            for (int i = threadIdx.x; i < RADIX_BINS; i += BLOCK_THREADS) {
                uint32_t total = 0;
                for (uint64_t b = 0; b < total_blocks; b++) {
                    total += global_histogram[b * RADIX_BINS + i];
                }
                global_histogram[i] = total; // reuse first block's histogram slots
            }
            block.sync();

            // Exclusive prefix sum
            uint32_t running = 0;
            for (int i = threadIdx.x; i < RADIX_BINS; i += BLOCK_THREADS) {
                uint32_t count = global_histogram[i];
                global_offsets[i] = running;
                running += count;
            }
            block.sync();
        }

        // PHASE 6: Grid sync — wait for block 0 to finish prefix sum
        grid.sync();

        // PHASE 7: Write block-level offsets
        if (threadIdx.x < RADIX_BINS) {
            smem_offsets[threadIdx.x] = 0;
        }
        block.sync();

        // Accumulate block-local offsets from global base
        for (int bin = threadIdx.x; bin < RADIX_BINS; bin += BLOCK_THREADS) {
            smem_offsets[bin] = global_offsets[bin];
            if (bin > 0) {
                uint32_t prev_accum = 0;
                for (uint64_t b = 0; b < block_id; b++) {
                    prev_accum += global_histogram[b * total_blocks + (bin - 1)];
                }
            }
            // Add prior blocks' counts for this bin
            for (uint64_t b = 0; b < block_id; b++) {
                smem_offsets[bin] += global_histogram[b * RADIX_BINS + bin];
            }
        }
        block.sync();

        // PHASE 8: Scatter
        for (uint64_t idx = start + threadIdx.x; idx < end; idx += BLOCK_THREADS) {
            if (idx < num_items) {
                uint8_t digit = (static_cast<uint32_t>(src[idx]) >> shift) & mask;
                uint32_t pos = atomicAdd(&smem_offsets[digit], 1);
                dst[pos] = src[idx];
            }
        }

        // PHASE 9: Grid sync — ensure all blocks finish scatter
        grid.sync();

        // Swap buffers for next pass
        const int32_t* tmp_src = src;
        src = dst;
        dst = const_cast<int32_t*>(tmp_src);
    }

    // After all passes, copy result to output
    for (uint64_t idx = start + threadIdx.x; idx < end; idx += BLOCK_THREADS) {
        if (idx < num_items) {
            output[idx] = src[idx];
        }
    }
}

torch::Tensor sort_cooperative(torch::Tensor input, torch::Tensor output) {
    auto num_items = static_cast<int64_t>(input.numel());
    int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int max_blocks = num_sms; // 1 block per SM for cooperative launch

    // Pre-allocate temp buffers (persistent)
    static torch::Tensor temp_buf = torch::empty({num_items},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    static torch::Tensor global_hist = torch::empty({max_blocks * 256},
        torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA));
    static torch::Tensor global_offsets = torch::empty({256},
        torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA));

    const int32_t* key_in = reinterpret_cast<const int32_t*>(input.const_data_ptr<float>());
    int32_t* key_out = reinterpret_cast<int32_t*>(output.data_ptr<float>());
    int32_t* temp_ptr = temp_buf.data_ptr<int32_t>();
    uint32_t* hist_ptr = global_hist.data_ptr<uint32_t>();
    uint32_t* offs_ptr = global_offsets.data_ptr<uint32_t>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    void* args[] = {&key_in, &key_out, &temp_ptr, &hist_ptr, &offs_ptr, &num_items};
    dim3 grid(max_blocks);
    dim3 block(BLOCK_THREADS);

    cudaLaunchCooperativeKernel(
        (void*)cooperative_radix_sort_kernel,
        grid, block, args, 0, stream);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>
torch::Tensor sort_cooperative(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cooperative_radix_v2',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cooperative'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    extra_cuda_cflags=['--expt-extended-lambda'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Cooperative-groups single-kernel radix sort.
    All 4 radix passes in one cooperative launch using grid-level sync.
    Eliminates CUB's 4 internal kernel launches without using graph capture.
    """
    input_tensor, output_tensor = data
    sort_module.sort_cooperative(input_tensor.contiguous(), output_tensor)
    return output_tensor