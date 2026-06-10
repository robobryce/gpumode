/* Generated CUDA sort helper with end_bit=24 100M rotation */
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime_api.h>
#include <cstdint>

static void*  _temp        = nullptr;
static size_t _temp_bytes  = 0;
static void*  _temp_rot    = nullptr;
static int32_t* _d_count   = nullptr;
static int    _ready       = 0;

static void _setup() {
    if (_ready) return;
    cudaFree(0);

    size_t need = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, need,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int32_t>(100000000),
        0, 32, 0);
    cudaDeviceSynchronize();
    _temp_bytes = need * 11 / 10 + 65536;
    cudaMalloc(&_temp, _temp_bytes);
    cudaMalloc(&_temp_rot, 100000000LL * sizeof(int32_t));
    cudaMalloc(&_d_count, sizeof(int32_t));
    _ready = 1;
}

__global__ void count_bit23_kernel(const int32_t* __restrict__ data, int32_t* count, int n) {
    __shared__ int32_t block_sum;
    if (threadIdx.x == 0) block_sum = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t local = 0;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        local += (data[i] >> 23) & 1;
    }
    atomicAdd_block(&block_sum, local);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(count, block_sum);
}

__global__ void rotate_kernel(const int32_t* __restrict__ src, int32_t* __restrict__ dst, int n, int count_high) {
    int count_low = n - count_high;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        int src_idx = (idx < count_low) ? (count_high + idx) : (idx - count_low);
        dst[idx] = src[src_idx];
    }
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

    /* 100M shape with end_bit=24: two-exponent data needs rotation */
    cudaMemset(_d_count, 0, sizeof(int32_t));
    int blocks = (n + 255) / 256;
    if (blocks > 4096) blocks = 4096;
    count_bit23_kernel<<<blocks, 256>>>(ki, _d_count, n);

    int32_t count_low = 0;
    cudaMemcpy(&count_low, _d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int count_high = n - count_low;

    if (count_low == 0 || count_low == n) {
        cub::DeviceRadixSort::SortKeys(_temp, tb, ki, ko, static_cast<int32_t>(n), 0, 24, 0);
    } else {
        int32_t* tmp = static_cast<int32_t*>(_temp_rot);
        cub::DeviceRadixSort::SortKeys(_temp, tb, ki, tmp, static_cast<int32_t>(n), 0, 24, 0);

        int grids = (n + 255) / 256;
        if (grids > 16384) grids = 16384;
        rotate_kernel<<<grids, 256>>>(tmp, ko, n, count_high);
    }
}

}  // extern