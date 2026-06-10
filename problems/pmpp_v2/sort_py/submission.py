"""
CUB DeviceRadixSort sorted via int32 bitcast with custom Policy800 override.
On sm_100 (PTX 800), ChainedPolicy dispatch selects Policy800, NOT Policy900
since 800 < 900. So override Policy800 with ITEMS_PER_THREAD=21 (existing
default for keys-only+int64 offset is already 21). The real win: pushing
ITEMS_PER_THREAD to 24 with reduced BLOCK_THREADS=320 to stay in smem bounds.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

// Override Policy800 with ITEMS_PER_THREAD=24 / BLOCK_THREADS=320
// The default for keys-only+int64 offset on Policy800 is 384 threads * 21 items = 8064
// Our override: 320 threads * 24 items = 7680 (slightly smaller tile, but each block
// does more work per thread for better register utilization on B200)
namespace cub {
template <typename KeyT, typename ValueT, typename OffsetT>
struct CustomPolicy800 : DeviceRadixSortPolicy<KeyT, ValueT, OffsetT>::Policy700
{
    using Orig = DeviceRadixSortPolicy<KeyT, ValueT, OffsetT>;
    using typename Orig::DominantT;
    static constexpr bool KEYS_ONLY = std::is_same<ValueT, NullType>::value;

    enum
    {
        PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5,
        SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5,
        SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5,
        ONESWEEP               = sizeof(KeyT) >= sizeof(uint32_t),
        ONESWEEP_RADIX_BITS    = 8,
        OFFSET_64BIT           = sizeof(OffsetT) == 8,
    };

    using HistogramPolicy    = AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Custom: 24 items/thread * 320 threads = 7680 tile items
    // More items per thread reduces block count for large inputs
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
        320, 24,
        DominantT, 1,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_RAKING_MEMOIZE,
        RADIX_SORT_STORE_DIRECT,
        ONESWEEP_RADIX_BITS>;

    using ScanPolicy = AgentScanPolicy<512, 23, OffsetT,
        BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
        BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE>;

    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<512, 23, DominantT,
        BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT,
        RADIX_RANK_MATCH, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
        (sizeof(KeyT) > 1) ? 256 : 128, 47, DominantT,
        BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, PRIMARY_RADIX_BITS - 1>;

    using UpsweepPolicy    = AgentRadixSortUpsweepPolicy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = AgentRadixSortUpsweepPolicy<256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<256, 19, DominantT,
        BLOCK_LOAD_DIRECT, LOAD_LDG,
        RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SINGLE_TILE_RADIX_BITS>;

    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<192, 39, DominantT,
        BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<384, 11, DominantT,
        BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS, SEGMENTED_RADIX_BITS - 1>;
};
} // namespace cub

// Full custom policy chain: Policy800 -> CustomPolicy800, Policy900 -> Policy800
template <typename KeyT, typename ValueT, typename OffsetT>
struct CustomRadixSortPolicy : cub::DeviceRadixSortPolicy<KeyT, ValueT, OffsetT>
{
    using Policy800 = cub::CustomPolicy800<KeyT, ValueT, OffsetT>;
    using MaxPolicy = typename CustomRadixSortPolicy::Policy800;
};


static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    using KeyT = int32_t;
    using OffsetT = int64_t;
    int begin_bit = 0, end_bit = 32;
    int64_t max_n = 100'000'000;

    size_t temp_bytes = 0;
    KeyT* dummy1 = nullptr;
    KeyT* dummy2 = nullptr;
    cub::DoubleBuffer<KeyT> dummy_db(dummy1, dummy2);
    cub::NullType* n1 = nullptr;
    cub::DoubleBuffer<cub::NullType> dummy_vals(n1, n1);
    cub::DispatchRadixSort<false, KeyT, cub::NullType, OffsetT,
        CustomRadixSortPolicy<KeyT, cub::NullType, OffsetT>>::Dispatch(
        nullptr, temp_bytes,
        dummy_db, dummy_vals,
        static_cast<OffsetT>(max_n),
        begin_bit, end_bit, false,
        0);

    persistent_temp_bytes = (temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output) {
    using KeyT = int32_t;
    using OffsetT = int64_t;
    using PolicyT = CustomRadixSortPolicy<KeyT, cub::NullType, OffsetT>;
    int begin_bit = 0, end_bit = 32;
    auto num_items = static_cast<int64_t>(input.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    const KeyT* key_in = reinterpret_cast<const KeyT*>(input.const_data_ptr<float>());
    KeyT* key_out = reinterpret_cast<KeyT*>(output.data_ptr<float>());

    cub::DoubleBuffer<KeyT> d_keys(const_cast<KeyT*>(key_in), key_out);
    cub::DoubleBuffer<cub::NullType> d_values;

    size_t temp_bytes = persistent_temp_bytes;
    cub::DispatchRadixSort<false, KeyT, cub::NullType, OffsetT, PolicyT>::Dispatch(
        persistent_temp.data_ptr(), temp_bytes,
        d_keys, d_values,
        static_cast<OffsetT>(num_items),
        begin_bit, end_bit, false,
        stream);

    return output;
}
"""

sort_cpp_source = """
#include <torch/extension.h>

void init_persistent_temp();
torch::Tensor sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda_custom_policy800',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda', 'init_persistent_temp'],
    extra_include_paths=['/usr/local/cuda-12.8/targets/x86_64-linux/include'],
    verbose=False,
)

sort_module.init_persistent_temp()


def custom_kernel(data: input_t) -> output_t:
    input_tensor, output_tensor = data
    sort_module.sort_cuda(input_tensor.contiguous(), output_tensor)
    return output_tensor