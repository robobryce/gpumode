import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = """
#include <torch/extension.h>
torch::Tensor segment_sort(torch::Tensor input, torch::Tensor output);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

// ---------------------------------------------------------------------------
// copy kernel
// ---------------------------------------------------------------------------
__global__ void copy_kernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// ---------------------------------------------------------------------------
// Phase 2 merge kernel:
// Uses cub::DeviceMerge::MergeKeys under the hood per pair, but we batch
// pairs by launching at a coarse level.
//
// Instead of many tiny MergeKeys calls, we do a multi-way merge:
// at each level L, merge pairs (2i, 2i+1) using a single grid that
// partitions the work across all blocks.
// ---------------------------------------------------------------------------
#define MRG_THREADS 256

__global__ void merge_batched_kernel(
    const float* __restrict__ src,
    const int*    __restrict__ pair_data,  // [num_pairs * 4]: a_start, lenA, b_start, lenB
    float*        __restrict__ dst,
    const int*    __restrict__ dst_offsets, // [num_pairs]
    int           num_pairs)
{
    int p = blockIdx.y;
    if (p >= num_pairs) return;

    int a_off   = pair_data[4 * p];
    int a_len   = pair_data[4 * p + 1];
    int b_off   = pair_data[4 * p + 2];
    int b_len   = pair_data[4 * p + 3];
    int total   = a_len + b_len;
    int d_off   = dst_offsets[p];

    int block_start = (int)blockIdx.x * 2048;
    if (block_start >= total) return;
    int block_len   = std::min(2048, total - block_start);

    int tid       = (int)threadIdx.x;
    int t_start   = block_start + tid * 8;
    int t_end     = std::min(t_start + 8,
                              block_start + block_len);
    if (t_start >= t_end) return;
    int t_len     = t_end - t_start;

    // Merge-path binary search: find partition of A,B for this thread
    int diag = t_start;
    int lo   = std::max(0, diag - b_len);
    int hi   = std::min(diag, a_len);
    while (lo < hi) {
        int m   = (lo + hi) >> 1;
        int mB  = diag - 1 - m;
        if (mB < 0) {
            hi = m;
        } else if (mB >= b_len) {
            lo = m + 1;
        } else if (src[a_off + m] <= src[b_off + mB]) {
            lo = m + 1;
        } else {
            hi = m;
        }
    }

    // lo == position in A after the first 'diag' merged elements.
    // Next element comes from A[lo] or B[diag-lo].
    int pA = a_off + lo;
    int pB = b_off + diag - lo;
    int pA_end = a_off + a_len;
    int pB_end = b_off + b_len;

    float* out = dst + d_off + t_start;
    for (int i = 0; i < t_len; i++) {
        if ((pA < pA_end) && (pB >= pB_end || src[pA] <= src[pB])) {
            out[i] = src[pA++];
        } else {
            out[i] = src[pB++];
        }
    }
}

// ---------------------------------------------------------------------------
// Host: build segment layout
// ---------------------------------------------------------------------------
struct SegLayout { std::vector<int> starts, lengths; };

static SegLayout build_layout(const int* row_offsets, int rows) {
    SegLayout L;
    L.starts.reserve(rows + 1);
    for (int i = 0; i <= rows; i++) L.starts.push_back(row_offsets[i]);
    for (int i = 0; i < rows; i++)
        L.lengths.push_back(row_offsets[i+1] - row_offsets[i]);
    return L;
}

// next_level: produce pair metadata for the next merge level.
// Returns false when 0 or 1 segment remains.
static bool next_level(
    const SegLayout& cur, SegLayout& nxt,
    std::vector<int>& pairs, std::vector<int>& dst_offs,
    int& max_chunks)
{
    int ns = (int)cur.lengths.size();
    if (ns <= 1) return false;
    int np = ns / 2;

    nxt.starts.clear(); nxt.lengths.clear();
    pairs.clear(); dst_offs.clear();
    max_chunks = 0;
    int pos = 0;

    for (int i = 0; i < np; i++) {
        int a0 = cur.starts[2*i];
        int aL = cur.lengths[2*i];
        int b0 = cur.starts[2*i+1];
        int bL = cur.lengths[2*i+1];
        pairs.insert(pairs.end(), {a0, aL, b0, bL});
        dst_offs.push_back(pos);
        int tot = aL + bL;
        nxt.starts.push_back(pos);
        nxt.lengths.push_back(tot);
        pos += tot;
        int ck = (tot + 2048 - 1) / 2048;
        if (ck > max_chunks) max_chunks = ck;
    }
    if (ns % 2) {
        int s0 = cur.starts[ns-1];
        int sL = cur.lengths[ns-1];
        nxt.starts.push_back(pos);
        nxt.lengths.push_back(sL);
        pos += sL;
    }
    nxt.starts.push_back(pos);
    return true;
}

// ---------------------------------------------------------------------------
// Host: upload int vector -> GPU, return pointer (device tensor backed)
// ---------------------------------------------------------------------------
static int* upload(const std::vector<int>& v, std::vector<torch::Tensor>& allocs) {
    auto t = torch::empty({(int64_t)v.size()},
                          torch::dtype(torch::kInt32).device(torch::kCUDA));
    cudaMemcpy(t.data_ptr<int>(), v.data(), v.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    allocs.push_back(t);
    return t.data_ptr<int>();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
torch::Tensor segment_sort(torch::Tensor input, torch::Tensor output) {
    int N = (int)input.numel();
    if (N <= 1) {
        if (N == 1) output[0] = input[0];
        return output;
    }

    int rows = (int)std::sqrt((double)N);
    int cols = (N + rows - 1) / rows;

    // Row offsets (CPU)
    std::vector<int> ro(rows + 1);
    for (int i = 0; i < rows; i++) ro[i] = std::min(i * cols, N);
    ro[rows] = N;

    // Trim trailing empty rows
    int eff = rows;
    while (eff > 0 && ro[eff-1] == ro[eff]) eff--;
    if (eff == 0) return output;

    // --- Phase 1: DeviceSegmentedRadixSort (input -> output, no overlap) ---
    {
        std::vector<torch::Tensor> hold;

        int* d_ro = upload(ro, hold);

        size_t tb = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, tb,
            input.data_ptr<float>(), output.data_ptr<float>(),
            N, eff,
            d_ro, d_ro + 1,
            0, sizeof(float) * 8);

        auto d_temp = torch::empty({(int64_t)tb},
                                   torch::dtype(torch::kUInt8).device(torch::kCUDA));

        auto err = cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp.data_ptr(), tb,
            input.data_ptr<float>(), output.data_ptr<float>(),
            N, eff,
            d_ro, d_ro + 1,
            0, sizeof(float) * 8);
        TORCH_CHECK(err == cudaSuccess, "SegRadixSort failed: ",
                    cudaGetErrorName(err));

        cudaDeviceSynchronize();
    }

    // --- Phase 2: multi-level merge tree ---
    // Copy per-row-sorted data from output -> input so Phase 2 starts from input
    {
        int blocks = (N + 255) / 256;
        copy_kernel<<<blocks, 256>>>(
            output.data_ptr<float>(), input.data_ptr<float>(), N);
        cudaDeviceSynchronize();
    }

    SegLayout cur = build_layout(ro.data(), eff);
    float* bufs[2] = { input.data_ptr<float>(), output.data_ptr<float>() };
    int src_idx = 0;

    while (true) {
        SegLayout nxt;
        std::vector<int> pairs_v, dst_v;
        int max_chunks = 0;

        if (!next_level(cur, nxt, pairs_v, dst_v, max_chunks))
            break;

        int np = (int)dst_v.size();
        if (np == 0) {
            // single segment — copy to dst
            int L = cur.lengths[0], S = cur.starts[0];
            int blk = (L + 255) / 256;
            copy_kernel<<<blk, 256>>>(bufs[src_idx] + S, bufs[1-src_idx], L);
            cudaDeviceSynchronize();
            src_idx = 1 - src_idx;
            cur = nxt;
            continue;
        }

        std::vector<torch::Tensor> hold;
        int* d_pairs = upload(pairs_v, hold);
        int* d_dst    = upload(dst_v, hold);

        dim3 grid(max_chunks, np);
        merge_batched_kernel<<<grid, MRG_THREADS>>>(
            bufs[src_idx], d_pairs, bufs[1-src_idx], d_dst, np);
        cudaDeviceSynchronize();

        // Copy odd leftover segment
        int ns = (int)cur.lengths.size();
        if (ns % 2 && ns > 1) {
            int L = cur.lengths[ns-1], S = cur.starts[ns-1];
            int D = nxt.starts[nxt.lengths.size()-1];
            int blk = (L + 255) / 256;
            copy_kernel<<<blk, 256>>>(bufs[src_idx] + S, bufs[1-src_idx] + D, L);
            cudaDeviceSynchronize();
        }

        src_idx = 1 - src_idx;
        cur = nxt;
    }

    cudaDeviceSynchronize();

    // If final result is in input (src_idx=0 after loop), copy to output
    if (src_idx == 0) {
        int blocks = (N + 255) / 256;
        copy_kernel<<<blocks, 256>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), N);
        cudaDeviceSynchronize();
    }

    return output;
}
"""

segment_module = load_inline(
    name='segment_sort_module',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['segment_sort'],
    extra_cuda_cflags=['--expt-relaxed-constexpr', '-std=c++17'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Segmented sort: Phase 1 = cub::DeviceSegmentedRadixSort (per-row,
    batched), Phase 2 = multi-level batched merge tree.
    Exploits the per-row-normal input: each row clusters around an
    increasing mean, so merges at higher levels have less cross-row overlap.
    """
    input_tensor, output_tensor = data
    segment_module.segment_sort(input_tensor, output_tensor)
    return output_tensor