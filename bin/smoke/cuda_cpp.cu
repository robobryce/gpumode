#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

__global__ void add_one(float* x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 256) x[i] += 1.0f;
}

int main() {
  float host[256] = {};
  float* device = nullptr;
  if (cudaMalloc(&device, sizeof(host)) != cudaSuccess) return 1;
  if (cudaMemcpy(device, host, sizeof(host), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
  add_one<<<1, 256>>>(device);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  if (cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
  cudaFree(device);
  for (float value : host) if (std::fabs(value - 1.0f) > 1e-6f) return 5;
  std::puts("CUDA C++ PASS");
  return 0;
}
