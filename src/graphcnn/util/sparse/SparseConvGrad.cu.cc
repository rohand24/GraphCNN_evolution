#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

__global__ void SparseConvGradKernel1 (const float * g,
                            const int64 * indices,
                            const float * h,
                            const float * values,
                            const float * vertices,
                            float * grad_a_values_vals,
                            const int num_edges,
                            const int in_features,
                            const int out_features,
                            const int N){
    int64 currentBatch;
    int64 currentN1;
    int64 currentN2;
    int64 currentL;
    float currentWeight;
    float currentVertex;
    float currentGrad;
    int f2;
    int eid;
    for (uint64 fid = blockIdx.x * blockDim.x + threadIdx.x; fid < num_edges*out_features; fid += blockDim.x * gridDim.x)
    {
        f2 = fid % out_features;
        eid = (fid / out_features) % num_edges;
        currentBatch = indices[eid*4 + 0];
        currentN1 = indices[eid*4 + 1];
        currentL = indices[eid*4 + 2];
        currentN2 = indices[eid*4 + 3];

        float accumulator = 0;
        currentGrad = g[currentBatch * N * out_features + currentN1 * out_features + f2];
        #pragma unroll 10
        for(int64  f1 = 0; f1 < in_features; f1++)
        {
            currentWeight = h[currentL *in_features * out_features + f1 * out_features + f2];
            currentVertex = vertices[currentBatch * N * in_features + currentN2*in_features + f1];
            accumulator += currentGrad*currentWeight*currentVertex;
        }
        atomicAdd(&(grad_a_values_vals[eid]),accumulator);
    }
}

__global__ void SparseConvGradKernel2 (const float * g,
                            const int64 * indices,
                            const float * h,
                            const float * values,
                            const float * vertices,
                            float * grad_v_in_vals,
                            const int num_edges,
                            const int in_features,
                            const int out_features,
                            const int N,
                            const int batch_size){
    /*int64 currentBatch;
    int64 currentN1;
    int64 currentN2;
    int64 currentL;*/
    const int TILE_SIZE = 200;
    const int TENSOR_RANK = 4;
    int64 currentIndex[TENSOR_RANK];
    float currentVal;
    float currentWeight;
    float currentGrad;
    int f1;
    int eid;
    //for (int f1 = blockIdx.x * blockDim.x + threadIdx.x; f1 < in_features; f1 += blockDim.x * gridDim.x)
    for (uint64 fid = blockIdx.x * blockDim.x + threadIdx.x; fid < in_features*num_edges;
       fid += blockDim.x * gridDim.x)
    {
        f1 = fid % in_features;
        eid = (fid / in_features) % num_edges;
        #pragma unroll
        for (int i = 0; i < TENSOR_RANK; i++)
        {
            currentIndex[i] = indices[eid*TENSOR_RANK + i];
        }
        currentVal = values[eid];
        float accumulator = 0;
        #pragma unroll 10
        for(int64  f2 = 0; f2 < out_features; f2++)
        {
            currentGrad = g[currentIndex[0] * N * out_features + currentIndex[1] * out_features + f2];
            currentWeight = h[currentIndex[2] *in_features * out_features + f1 * out_features + f2];
            accumulator += currentGrad*currentWeight*currentVal;
        }
        atomicAdd(&(grad_v_in_vals[currentIndex[0] * N * in_features + currentIndex[3]*in_features + f1]),accumulator);
    }
}

__global__ void SparseConvGradKernel3 (const float * g,
                            const int64 * indices,
                            const float * h,
                            const float * values,
                            const float * vertices,
                            float * grad_h_weights_vals,
                            const int num_edges,
                            const int in_features,
                            const int out_features,
                            const int N,
                            const int num_layers){
    const int TILE_SIZE = 200;
    const int TENSOR_RANK = 4;
    __shared__ int64 currentIndex[TILE_SIZE];
    __shared__ float currentVal;
    /*int64 currentBatch;
    int64 currentN1;
    int64 currentN2;
    int64 currentL;
    float currentVal;*/
    float currentGrad;
    float currentVertex;
    int f1;
    int f2;
    int lid;
    for (uint64 fid = blockIdx.x * blockDim.x + threadIdx.x; fid < in_features*out_features; fid += blockDim.x * gridDim.x)
    {
        f2 = fid % out_features;
        f1 = (fid / out_features) % in_features;
        float accumulator = 0;
        #pragma unroll 5
        for (int64 edge = 0; edge < num_edges; edge++)
        {
                if (threadIdx.x == 0)
                {
                    #pragma unroll
                    for (int i = 0; i < TENSOR_RANK; i++)
                    {
                        currentIndex[i] = indices[edge*TENSOR_RANK + i];
                    }
                    currentVal = values[edge];
                }
                __syncthreads();
                //currentN2 = indices[edge*4 + 3];
                //currentVal = values[edge];
                currentVertex = vertices[currentIndex[0] * N * in_features + currentIndex[3]*in_features + f1];
                currentGrad = g[currentIndex[0] * N * out_features + currentIndex[1] * out_features + f2];
                grad_h_weights_vals[currentIndex[2] *in_features * out_features + f1 * out_features + f2] += currentGrad*currentVal*currentVertex;
        }
    }
}


bool SparseConvGradLauncher(const float * g,
                            const int64 * indices,
                            const float * h,
                            const float * values,
                            const float * vertices,
                            float * grad_a_values_vals,
                            float * grad_h_weights_vals,
                            float * grad_v_in_vals,
                            const int num_edges,
                            const int in_features,
                            const int out_features,
                            const int N,
                            const int num_layers,
                            const int batch_size,
                            const GPUDevice& d) {
  uint64 totalThreads1 = num_edges*out_features;
  int configThreads1 = 0;
  if (totalThreads1 > INT32_MAX)
  {
    configThreads1 = INT32_MAX;
  }
  else
  {
    configThreads1 = totalThreads1;
  }
  CudaLaunchConfig config1 = GetCudaLaunchConfig(configThreads1, d);
  SparseConvGradKernel1<<<config1.block_count, config1.thread_per_block,0,d.stream()>>>(g,
                                                                                 indices,
                                                                                 h,
                                                                                 values,
                                                                                 vertices,
                                                                                 grad_a_values_vals,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 out_features,
                                                                                 N);
  uint64 totalThreads2 = num_edges*out_features;
  int configThreads2 = 0;
  if (totalThreads2 > INT32_MAX)
  {
    configThreads2 = INT32_MAX;
  }
  else
  {
    configThreads2 = totalThreads2;
  }
  CudaLaunchConfig config2 = GetCudaLaunchConfig(configThreads2, d);
  SparseConvGradKernel2<<<config2.block_count, config2.thread_per_block,0,d.stream()>>>(g,
                                                                                 indices,
                                                                                 h,
                                                                                 values,
                                                                                 vertices,
                                                                                 grad_v_in_vals,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 out_features,
                                                                                 N,
                                                                                 batch_size);

  uint64 totalThreads3 = in_features*out_features;
  int configThreads3 = 0;
  if (totalThreads3 > INT32_MAX)
  {
    configThreads3 = INT32_MAX;
  }
  else
  {
    configThreads3 = totalThreads3;
  }
  CudaLaunchConfig config3 = GetCudaLaunchConfig(totalThreads3, d);
  SparseConvGradKernel3<<<config3.block_count, config3.thread_per_block,0,d.stream()>>>(g,
                                                                                 indices,
                                                                                 h,
                                                                                 values,
                                                                                 vertices,
                                                                                 grad_h_weights_vals,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 out_features,
                                                                                 N,
                                                                                 num_layers);
  return d.ok();
}

#endif