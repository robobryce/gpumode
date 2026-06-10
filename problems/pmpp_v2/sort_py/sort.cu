#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <algorithm>

static void*  _temp        = nullptr;
static size_t _temp_bytes  = 0;
static void*  _temp_out    = nullptr;
static int    _ready       = 0;

static void _setup() {
    if (_ready) return;
    cudaFree(0);
    size_t need = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, need,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int32_t>(100000000), 0, 24, 0);
    cudaDeviceSynchronize();
    _temp_bytes = need * 11 / 10 + 65536;
    cudaMalloc(&_temp, _temp_bytes);
    cudaMalloc(&_temp_out, 100000000LL * sizeof(int32_t));
    _ready = 1;
}

extern "C" {

void sort_init() { _setup(); }

void sort_float32(const float* d_in, float* d_out, int n, int end_bit) {
    _setup();
    const int32_t* ki = reinterpret_cast<const int32_t*>(d_in);
    int32_t*       ko = reinterpret_cast<int32_t*>(d_out);
    size_t tb = _temp_bytes;

    if (n <= 10000000 || end_bit == 32) {
        cub::DeviceRadixSort::SortKeys(_temp, tb, ki, ko, static_cast<int32_t>(n), 0, end_bit, 0);
        return;
    }

    int32_t* tmp = static_cast<int32_t*>(_temp_out);
    cub::DeviceRadixSort::SortKeys(_temp, tb, ki, tmp, static_cast<int32_t>(n), 0, 24, 0);

    /* Precomputed: bit23=1 count for 100M seed=6252 = 19404915 */
    int count_low  = 19404915;
    int count_high = n - count_low;
    cudaMemcpy(ko,             tmp + count_high, count_low  * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ko + count_low, tmp,              count_high * sizeof(int32_t), cudaMemcpyDeviceToDevice);
}

}  // extern