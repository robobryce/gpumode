/*
 * C ABI shared library: CUB DeviceRadixSort::SortKeys, persistent temp storage.
 * Compiled directly with nvcc, loaded via ctypes in submission.py.
 * Zero PyTorch dependency at compile/link time — pure CUDA C ABI.
 */
#include <cuda_runtime_api.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

static void* g_temp_storage = nullptr;
static size_t g_temp_bytes = 0;

extern "C" {

int sort_cuda_init() {
    if (g_temp_storage != nullptr) return 0;
    int64_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, g_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int64_t>(max_n),
        0, 32);
    g_temp_bytes = (g_temp_bytes * 11 + 9) / 10;
    cudaMalloc(&g_temp_storage, g_temp_bytes);
    return 0;
}

int sort_cuda_run(int32_t* d_out, const int32_t* d_in, int64_t num_items, cudaStream_t stream) {
    size_t temp_bytes = g_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        g_temp_storage, temp_bytes,
        d_in, d_out, num_items,
        0, 32,
        stream);
    return 0;
}

}  // extern "C"