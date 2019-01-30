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

__global__ void SparseMaxVertexPoolForwardKernel(const int64 * indices,
                                 const float * values,
                                 const float * vertices,
                                 float * maxOutput,
                                 int64 * maxEdge,
                                 const int num_edges,
                                 const int in_features,
                                 const int in_vertex_count,
                                 const int out_vertex_count,
                                 const int batch_size) {
    const int TILE_SIZE = 10;
    const int TENSOR_RANK = 4;
    __shared__ int64 currentIndex[TENSOR_RANK];
    __shared__ float currentVal;
    float proposedMax;

    for (uint64 f1 = blockIdx.x * blockDim.x + threadIdx.x; f1 < in_features; f1 += blockDim.x * gridDim.x)
    {
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
            }
            __syncthreads();

            currentVal = values[edge];
            proposedMax = currentVal*vertices[currentIndex[0] * in_vertex_count * in_features + currentIndex[3]*in_features + f1];
            //std::cout << currentBatch << " " << currentN1 << " " << currentN2 << " " << currentL << " " << currentVal << std::endl;
            if (proposedMax > maxOutput[currentIndex[0] * out_vertex_count * in_features + currentIndex[2] * in_features + f1])
            {
                maxEdge[currentIndex[0] * out_vertex_count * in_features + currentIndex[2] * in_features + f1] = edge;
                maxOutput[currentIndex[0] * out_vertex_count * in_features + currentIndex[2] * in_features + f1] = proposedMax;
            }
        }
    }
}


__global__ void SparseMaxVertexPoolGradKernel1(const float * g,
                             const int64 * indices,
                             const float * values,
                             const float * vertices,
                             const int64 * maxEdge,
                             float * grad_p_values_vals,
                             const int num_edges,
                             const int in_features,
                             const int in_vertex_count,
                             const int out_vertex_count)
{
    int64 currentBatch;
    int64 currentN1;
    int64 currentN2;
    int64 currentL;
    float currentVal;
    float currentWeight;
    float currentGrad;
    float currentVertex;
    int64 maxEdgeCurrent;
    for (int64 edge = blockIdx.x * blockDim.x + threadIdx.x; edge < num_edges; edge += blockDim.x * gridDim.x)
    {
        currentBatch = indices[edge*4 + 0];
        currentN1 = indices[edge*4 + 2];
        currentN2 = indices[edge*4 + 3];
        //currentVal = values[edge];
        float accumulator = 0;
        #pragma unroll 10
        for(int64  f1 = 0; f1 < in_features; f1++)
        {
            currentGrad = g[currentBatch * out_vertex_count * in_features + currentN1 * in_features + f1];
            //currentVertex = vertices[currentBatch * in_vertex_count * in_features + currentN2*in_features + f1];
            //P ERROR IS STILL TOO HIGH
            maxEdgeCurrent = maxEdge[currentBatch * out_vertex_count * in_features + currentN1*in_features + f1];
            if (currentN2 == indices[maxEdgeCurrent*4 + 3])
            {
                accumulator += currentGrad*vertices[currentBatch * in_vertex_count * in_features + indices[maxEdgeCurrent*4 + 3]*in_features + f1];
                //grad_p_values_vals[edge] += currentGrad*maxVertex[currentBatch * out_vertex_count * in_features + currentN1*in_features + f1];
            }
        }
        grad_p_values_vals[edge] += accumulator;

    }
}

__global__ void SparseMaxVertexPoolGradKernel2(const float * g,
                             const int64 * indices,
                             const float * values,
                             const float * vertices,
                             const int64 * maxEdge,
                             float * grad_v_in_vals,
                             const int num_edges,
                             const int in_features,
                             const int in_vertex_count,
                             const int out_vertex_count,
                             const int batch_size)
{

    const int SECTION_SIZE = 16;
    const int TENSOR_RANK = 4;
    int64 currentIndex[TENSOR_RANK];
    float currentGrad;
    int f1;
    int eid;
    for (int fid = blockIdx.x * blockDim.x + threadIdx.x; fid < in_features*num_edges;
       fid += blockDim.x * gridDim.x)
    {
        f1 = fid % in_features;
        eid = (fid / in_features) % num_edges;
        #pragma unroll
        for (int i = 0; i < TENSOR_RANK; i++)
        {
            currentIndex[i] = indices[eid*TENSOR_RANK + i];
        }
        currentGrad = g[currentIndex[0] * out_vertex_count * in_features + currentIndex[2] * in_features + f1];

        if ((currentIndex[3] == indices[maxEdge[currentIndex[0] * out_vertex_count * in_features + currentIndex[2]*in_features + f1]*TENSOR_RANK + 3])
        && (currentIndex[2] == indices[maxEdge[currentIndex[0] * out_vertex_count * in_features + currentIndex[2]*in_features + f1]*TENSOR_RANK + 2]))
        {
            atomicAdd(&(grad_v_in_vals[currentIndex[0] * in_vertex_count * in_features + currentIndex[3]*in_features + f1]),
            currentGrad*values[maxEdge[currentIndex[0] * out_vertex_count * in_features + currentIndex[2]*in_features + f1]]);
        }
    }
}

bool SparseMaxVertexPoolGradLauncher(const float * g,
                        const int64 * indices,
                        const float * values,
                        const float * vertices,
                        float * maxOutput,
                        int64 * maxEdge,
                        float * grad_v_in_vals,
                        float * grad_p_values_vals,
                        const int num_edges,
                        const int in_features,
                        const int in_vertex_count,
                        const int out_vertex_count,
                        const int batch_size,
                        const int section_count,
                        const GPUDevice& d) {

  uint64 totalThreadsF = in_features;
  int configThreadsF = 0;
  if (totalThreadsF > INT32_MAX)
  {
    configThreadsF = INT32_MAX;
  }
  else
  {
    configThreadsF = totalThreadsF;
  }
  CudaLaunchConfig config = GetCudaLaunchConfig(configThreadsF, d);
  //CudaLaunchConfig config = GetCudaLaunchConfig(in_features*num_edges, d);
  SparseMaxVertexPoolForwardKernel<<<config.block_count,config.thread_per_block, 0,d.stream()>>>(indices,
                                                                                 values,
                                                                                 vertices,
                                                                                 maxOutput,
                                                                                 maxEdge,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 in_vertex_count,
                                                                                 out_vertex_count,
                                                                                 batch_size);
  uint64 totalThreads1 = in_features*num_edges;
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
  SparseMaxVertexPoolGradKernel1<<<config1.block_count, config1.thread_per_block,0,d.stream()>>>(g,
                                                                                 indices,
                                                                                 values,
                                                                                 vertices,
                                                                                 maxEdge,
                                                                                 grad_p_values_vals,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 in_vertex_count,
                                                                                 out_vertex_count);
  uint64 totalThreads2 = in_features*num_edges;
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
  SparseMaxVertexPoolGradKernel2<<<config2.block_count, config2.thread_per_block,0,d.stream()>>>(g,
                                                                                 indices,
                                                                                 values,
                                                                                 vertices,
                                                                                 maxEdge,
                                                                                 grad_v_in_vals,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 in_vertex_count,
                                                                                 out_vertex_count,
                                                                                 batch_size);
  return d.ok();
}

#endif