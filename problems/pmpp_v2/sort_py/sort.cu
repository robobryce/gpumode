#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

static torch::Tensor persistent_temp = {};
static size_t persistent_temp_bytes = 0;

void init_persistent_temp() {
    if (persistent_temp.defined()) return;
    int64_t max_n = 100'000'000;
    cub::DeviceRadixSort::SortKeys(
        nullptr, persistent_temp_bytes,
        static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr),
        static_cast<int64_t>(max_n),
        0, 32,
        0);
    persistent_temp_bytes = (persistent_temp_bytes * 11 + 9) / 10;
    persistent_temp = torch::empty(
        {static_cast<int64_t>(persistent_temp_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
}

torch::Tensor srt(torch::Tensor in, torch::Tensor out) {
    auto n = static_cast<int64_t>(in.numel());
    auto s = c10::cuda::getCurrentCUDAStream().stream();

    const int32_t* ki = reinterpret_cast<const int32_t*>(in.const_data_ptr<float>());
    int32_t* ko = reinterpret_cast<int32_t*>(out.data_ptr<float>());

    size_t tb = persistent_temp_bytes;
    cub::DeviceRadixSort::SortKeys(
        persistent_temp.data_ptr(), tb,
        ki, ko, n,
        0, 32, s);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("srt", &srt, "sort");
    m.def("init_persistent_temp", &init_persistent_temp, "init");
}