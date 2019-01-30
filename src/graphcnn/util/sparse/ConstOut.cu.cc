#if GOOGLE_CUDA


#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void ConstOutKernel(T * data, const int size, T value)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x)
    {
        data[i] = value;
    }
}

template <typename T>
bool ConstOutLauncher(T * data,
                        const int size,
                        const T value,
                        const GPUDevice& d) {
  CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
  ConstOutKernel<<<config.block_count, config.thread_per_block,0,d.stream()>>>(data,size,value);
  return d.ok();
}

template bool ConstOutLauncher<float>(float * data,
                        const int size,
                        const float value,
                        const GPUDevice& d);

template __global__ void ConstOutKernel<float>(float * data, const int size, const float value);

template bool ConstOutLauncher<int64>(int64 * data,
                        const int size,
                        const int64 value,
                        const GPUDevice& d);

template __global__ void ConstOutKernel<int64>(int64 * data, const int size, const int64 value);
#endif