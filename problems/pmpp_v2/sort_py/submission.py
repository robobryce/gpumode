import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# Incremental approach: Phase 1 (DeviceSegmentedRadixSort per row) +
# Phase 2 (full merge tree, correctness-verified).
# Then add overlap optimization on top.

cpp_source = """
#include <torch/extension.h>
torch::Tensor seg_sort_full(torch::Tensor input, torch::Tensor output);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

// Full merge kernel — always merges entire segments (no overlap optimization).
// 2D grid: (chunks_per_pair, num_pairs). Each block handles 2048 output items.
__global__ void merge_kernel(
    const float* __restrict__ src,
    const int*    __restrict__ pair_data,  // [num_pairs*4]: a_start, a_len, b_start, b_len
    float*        __restrict__ dst,
    const int*    __restrict__ dst_offs,   // [num_pairs]: dst buffer start position
    int num_pairs)
{
    int p = blockIdx.y;
    if (p >= num_pairs) return;
    int a0 = pair_data[4*p], aL = pair_data[4*p+1];
    int b0 = pair_data[4*p+2], bL = pair_data[4*p+3];
    int tot = aL + bL, d0 = dst_offs[p];

    int bs = (int)blockIdx.x * 2048;
    if (bs >= tot) return;
    int bl = min(2048, tot - bs);

    int tid = (int)threadIdx.x;
    int ts = bs + tid * 8;
    int te = min(ts + 8, bs + bl);
    if (ts >= te) return;
    int tl = te - ts;

    // Merge-path binary search for thread's start
    int diag = ts;
    int lo = max(0, diag - bL), hi = min(diag, aL);
    while (lo < hi) {
        int m = (lo + hi) >> 1;
        int mB = diag - 1 - m;
        if (mB < 0) { hi = m; }
        else if (mB >= bL) { lo = m + 1; }
        else if (src[a0 + m] <= src[b0 + mB]) { lo = m + 1; }
        else { hi = m; }
    }
    int pA = a0 + lo, pB = b0 + diag - lo;
    int aE = a0 + aL, bE = b0 + bL;
    float* out = dst + d0 + ts;
    for (int i = 0; i < tl; i++) {
        if (pA < aE && (pB >= bE || src[pA] <= src[pB]))
            out[i] = src[pA++];
        else
            out[i] = src[pB++];
    }
}

__global__ void copy_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
torch::Tensor seg_sort_full(torch::Tensor input, torch::Tensor output) {
    int N = (int)input.numel();
    if (N <= 1) { if (N==1) output[0]=input[0]; return output; }

    int rows = (int)std::sqrt((double)N);
    int cols = (N + rows - 1) / rows;

    std::vector<int> ro(rows + 1);
    for (int i = 0; i < rows; i++) ro[i] = std::min(i * cols, N);
    ro[rows] = N;

    int eff = rows;
    while (eff > 0 && ro[eff-1] == ro[eff]) eff--;
    if (eff == 0) return output;

    // Phase 1: per-row sort
    {
        auto d_ro = torch::empty({eff + 1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
        cudaMemcpy(d_ro.data_ptr<int>(), ro.data(), (eff+1)*sizeof(int), cudaMemcpyHostToDevice);
        size_t tb = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(nullptr, tb,
            input.data_ptr<float>(), output.data_ptr<float>(), N, eff,
            d_ro.data_ptr<int>(), d_ro.data_ptr<int>()+1, 0,sizeof(float)*8);
        auto dtmp = torch::empty({(int64_t)tb}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
        cub::DeviceSegmentedRadixSort::SortKeys(dtmp.data_ptr(), tb,
            input.data_ptr<float>(), output.data_ptr<float>(), N, eff,
            d_ro.data_ptr<int>(), d_ro.data_ptr<int>()+1, 0,sizeof(float)*8);
        cudaDeviceSynchronize();
    }
    // Copy output -> input for Phase 2 ping-pong
    { int blk=(N+255)/256; copy_kernel<<<blk,256>>>(output.data_ptr<float>(),input.data_ptr<float>(),N); cudaDeviceSynchronize(); }

    // Phase 2: full merge tree
    // Build all level metadata on CPU once, upload once
    struct Lvl { std::vector<int> pair_data, dst_offs; int num_pairs, max_chunks; int odd_len, odd_src, odd_dst; };
    std::vector<Lvl> levels;

    {
        std::vector<int> s(ro.begin(), ro.begin()+eff+1);
        std::vector<int> l(eff);
        for (int i=0; i<eff; i++) l[i]=s[i+1]-s[i];
        int ns = eff;
        while (ns > 1) {
            int np = ns/2;
            Lvl lv;
            lv.num_pairs = np;
            lv.max_chunks = 0;
            int pos = 0;
            for (int i=0; i<np; i++) {
                lv.pair_data.insert(lv.pair_data.end(), {s[2*i], l[2*i], s[2*i+1], l[2*i+1]});
                lv.dst_offs.push_back(pos);
                int tot = l[2*i]+l[2*i+1];
                pos += tot;
                int ck = (tot+2047)/2048;
                if (ck>lv.max_chunks) lv.max_chunks=ck;
            }
            lv.odd_len=0; lv.odd_src=0; lv.odd_dst=0;
            if (ns%2 && ns>1) {
                lv.odd_len=l[ns-1]; lv.odd_src=s[ns-1]; lv.odd_dst=pos;
                pos += l[ns-1];
            }
            levels.push_back(lv);

            std::vector<int> ns_s, ns_l;
            pos = 0;
            for (int i=0; i<np; i++) {
                ns_s.push_back(pos); ns_l.push_back(l[2*i]+l[2*i+1]);
                pos += l[2*i]+l[2*i+1];
            }
            if (ns%2) {
                ns_s.push_back(pos); ns_l.push_back(l[ns-1]);
                pos += l[ns-1];
            }
            ns_s.push_back(pos);
            s=ns_s; l=ns_l;
            ns=(int)ns_l.size();
        }
    }

    // Upload all metadata
    int total_pair_elems=0, total_dst_elems=0;
    for (auto& lv : levels) { total_pair_elems += (int)lv.pair_data.size(); total_dst_elems += (int)lv.dst_offs.size(); }
    auto g_pairs = torch::empty({total_pair_elems}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto g_dsts = torch::empty({total_dst_elems}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    int pd_off=0, do_off=0;
    std::vector<int> pd_start, do_start, num_pairs, max_chunks, odd_len, odd_src, odd_dst;
    for (auto& lv : levels) {
        if (!lv.pair_data.empty()) {
            cudaMemcpy(g_pairs.data_ptr<int>()+pd_off, lv.pair_data.data(),
                       lv.pair_data.size()*sizeof(int), cudaMemcpyHostToDevice);
        }
        if (!lv.dst_offs.empty()) {
            cudaMemcpy(g_dsts.data_ptr<int>()+do_off, lv.dst_offs.data(),
                       lv.dst_offs.size()*sizeof(int), cudaMemcpyHostToDevice);
        }
        pd_start.push_back(pd_off); do_start.push_back(do_off);
        num_pairs.push_back(lv.num_pairs); max_chunks.push_back(lv.max_chunks);
        odd_len.push_back(lv.odd_len); odd_src.push_back(lv.odd_src); odd_dst.push_back(lv.odd_dst);
        pd_off += (int)lv.pair_data.size(); do_off += (int)lv.dst_offs.size();
    }

    float* bufs[2] = { input.data_ptr<float>(), output.data_ptr<float>() };
    int src_i = 0, nlev = (int)levels.size();

    for (int lev=0; lev<nlev; lev++) {
        int np = num_pairs[lev];
        if (np > 0) {
            dim3 grid(max_chunks[lev], np);
            merge_kernel<<<grid,256>>>(
                bufs[src_i],
                g_pairs.data_ptr<int>() + pd_start[lev],
                bufs[1-src_i],
                g_dsts.data_ptr<int>() + do_start[lev],
                np);
        }
        if (odd_len[lev] > 0) {
            int blk = (odd_len[lev]+255)/256;
            copy_kernel<<<blk,256>>>(bufs[src_i]+odd_src[lev], bufs[1-src_i]+odd_dst[lev], odd_len[lev]);
        }
        cudaDeviceSynchronize();
        src_i = 1 - src_i;
    }

    cudaDeviceSynchronize();
    if (src_i == 0) {
        int blk=(N+255)/256;
        copy_kernel<<<blk,256>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
        cudaDeviceSynchronize();
    }
    return output;
}
"""

seg_full = load_inline(
    name='seg_sort_full',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['seg_sort_full'],
    extra_cuda_cflags=['--expt-relaxed-constexpr', '-std=c++17'],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Phase 1: DeviceSegmentedRadixSort per row.
    Phase 2: Full merge tree (all levels, full merge).
    """
    input_tensor, output_tensor = data
    seg_full.seg_sort_full(input_tensor, output_tensor)
    return output_tensor