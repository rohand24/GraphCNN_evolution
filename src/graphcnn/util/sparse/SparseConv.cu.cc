#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <math.h>

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

__global__ void SparseConvKernel(const int64 * indices,
                                 const float * h,
                                 const float * values,
                                 const float * vertices,
                                 float * output,
                                 const int num_edges,
                                 const int in_features,
                                 const int out_features,
                                 const int N,
                                 const int batch_size) {

    /*__shared__ int64 currentBatch;
    __shared__ int64 currentN1;
    __shared__ int64 currentN2;
    __shared__ int64 currentL;
    __shared__ float currentVal;
    __shared__ float currentVertex;*/
    const int TILE_SIZE = 200;
    const int TENSOR_RANK = 4;
    int64 currentIndex[TENSOR_RANK];
    float currentVal;
    float currentWeight;
    float currentVertex;
    int f2;
    int eid;
    //for (int f2 = blockIdx.x * blockDim.x + threadIdx.x; f2 < out_features;
     //  f2 += blockDim.x * gridDim.x)
    for (uint64 fid = blockIdx.x * blockDim.x + threadIdx.x; fid < out_features*num_edges;
       fid += blockDim.x * gridDim.x)
    {
        f2 = fid % out_features;
        eid = (fid / out_features) % num_edges;
        #pragma unroll
        for (int i = 0; i < TENSOR_RANK; i++)
        {
            currentIndex[i] = indices[eid*TENSOR_RANK + i];
        }
        currentVal = values[eid];
        float accumulator = 0;
        #pragma unroll 10
        for(int64  f1 = 0; f1 < in_features; f1++)
        {
            currentVertex = vertices[currentIndex[0] * N * in_features + currentIndex[3]*in_features + f1];
            currentWeight = h[currentIndex[2] *in_features * out_features + f1 * out_features + f2];
            accumulator +=
            //output[currentIndex[i*TENSOR_RANK + 0] * N * out_features + currentIndex[i*TENSOR_RANK + 1] * out_features + f2] +=
            currentWeight*currentVal*currentVertex;
        }
        //output[currentIndex[0] * N * out_features + currentIndex[1] * out_features + f2] += accumulator;
        atomicAdd(&(output[currentIndex[0] * N * out_features + currentIndex[1] * out_features + f2]),accumulator);

    }
}

bool SparseConvLauncher(const int64 * indices,
                        const float * h,
                        const float * values,
                        const float * vertices,
                        float * output,
                        const int num_edges,
                        const int in_features,
                        const int out_features,
                        const int N,
                        const int batch_size,
                        const GPUDevice& d) {
  uint64 totalThreads = out_features*num_edges;
  int configThreads = 0;
  if (totalThreads > INT32_MAX)
  {
    configThreads = INT32_MAX;
  }
  else
  {
    configThreads = totalThreads;
  }
  CudaLaunchConfig config = GetCudaLaunchConfig(configThreads, d);
  SparseConvKernel<<<config.block_count, config.thread_per_block,0,d.stream()>>>(indices,
                                                                                 h,
                                                                                 values,
                                                                                 vertices,
                                                                                 output,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 out_features,
                                                                                 N,
                                                                                 batch_size);
  return d.ok();
}

#endif