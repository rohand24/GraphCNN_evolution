#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

//Atomic max taken from https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__global__ void SparseMaxVertexPoolKernel(const int64 * indices,
                                 const float * values,
                                 const float * vertices,
                                 float * output,
                                 const int num_edges,
                                 const int in_features,
                                 const int in_vertex_count,
                                 const int out_vertex_count,
                                 const int batch_size) {
    /*int64 currentBatch;
    int64 currentN1;
    int64 currentN2;
    int64 currentL;
    float currentVal;*/
    const int TILE_SIZE = 200;
    const int TENSOR_RANK = 4;
    int64 currentIndex[TENSOR_RANK];
    float currentVal;
    float proposedMax;
    int eid;
    int f1;
    for (uint64 fid = blockIdx.x * blockDim.x + threadIdx.x; fid < in_features*num_edges;
       fid += blockDim.x * gridDim.x)
    {
        f1 = fid % in_features;
        eid = (fid / in_features) % num_edges;
        for (int i = 0; i < TENSOR_RANK; i++)
        {
            currentIndex[i] = indices[eid*TENSOR_RANK + i];
        }
        currentVal = values[eid];
        proposedMax = currentVal*vertices[currentIndex[0] * in_vertex_count * in_features + currentIndex[3]*in_features + f1];
        atomicMax(&(output[currentIndex[0] * out_vertex_count * in_features + currentIndex[2] * in_features + f1]),proposedMax);
    }
}

bool SparseMaxVertexPoolLauncher(const int64 * indices,
                        const float * values,
                        const float * vertices,
                        float * output,
                        const int num_edges,
                        const int in_features,
                        const int in_vertex_count,
                        const int out_vertex_count,
                        const int batch_size,
                        const GPUDevice& d) {
  uint64 totalThreads = in_features*num_edges;
  int configThreads = 0;
  if (totalThreads > INT32_MAX)
  {
    configThreads = INT32_MAX;
  }
  else
  {
    configThreads = totalThreads;
  }
  CudaLaunchConfig config = GetCudaLaunchConfig(totalThreads, d);
  SparseMaxVertexPoolKernel<<<config.block_count, config.thread_per_block,0,d.stream()>>>(indices,
                                                                                 values,
                                                                                 vertices,
                                                                                 output,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 in_vertex_count,
                                                                                 out_vertex_count,
                                                                                 batch_size);
  return d.ok();
}

#endif