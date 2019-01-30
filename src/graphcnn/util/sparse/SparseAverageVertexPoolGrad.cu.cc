#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

__global__ void SparseAverageVertexPoolGradKernel1(const float * g,
                             const int64 * indices,
                             const float * values,
                             const float * vertices,
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
    for (int64 edge = blockIdx.x * blockDim.x + threadIdx.x; edge < num_edges; edge += blockDim.x * gridDim.x)
    {
        currentBatch = indices[edge*4 + 0];
        currentN1 = indices[edge*4 + 2];
        currentN2 = indices[edge*4 + 3];
        currentVal = values[edge];
        for(int64  f1 = 0; f1 < in_features; f1++)
        {
            currentGrad = g[currentBatch * out_vertex_count * in_features + currentN1 * in_features + f1];
            currentVertex = vertices[currentBatch * in_vertex_count * in_features + currentN2*in_features + f1];
            grad_p_values_vals[edge] += currentGrad*currentVertex;
        }
    }
}

__global__ void SparseAverageVertexPoolGradKernel2(const float * g,
                             const int64 * indices,
                             const float * values,
                             const float * vertices,
                             float * grad_v_in_vals,
                             const int num_edges,
                             const int in_features,
                             const int in_vertex_count,
                             const int out_vertex_count,
                             const int batch_size)
{
    int64 currentBatch;
    int64 currentN1;
    int64 currentN2;
    float currentVal;
    float currentGrad;
    int f1;
    int bid;
    int nid;
    for (int fid = blockIdx.x * blockDim.x + threadIdx.x; fid < in_features*batch_size*in_vertex_count;
       fid += blockDim.x * gridDim.x)
    {
        f1 = fid % in_features;
        bid = (fid / in_features) % batch_size;
        nid = fid / (in_features * batch_size);
        for(int64  edge = 0; edge < num_edges; edge++)
        {
            currentBatch = indices[edge*4 + 0];
            currentN1 = indices[edge*4 + 2];
            currentN2 = indices[edge*4 + 3];
            currentVal = values[edge];
            if ((nid == currentN2) && (bid == currentBatch))
            {
                currentGrad = g[currentBatch * out_vertex_count * in_features + currentN1 * in_features + f1];
                //grad_p_values_vals(edge) += currentGrad*currentVertex;
                grad_v_in_vals[currentBatch * in_vertex_count * in_features + currentN2*in_features + f1] += currentGrad*currentVal;
            }
        }
    }
}

bool SparseAverageVertexPoolGradLauncher(const float * g,
                        const int64 * indices,
                        const float * values,
                        const float * vertices,
                        float * grad_v_in_vals,
                        float * grad_p_values_vals,
                        const int num_edges,
                        const int in_features,
                        const int in_vertex_count,
                        const int out_vertex_count,
                        const int batch_size,
                        const GPUDevice& d) {
  CudaLaunchConfig config1 = GetCudaLaunchConfig(num_edges, d);
  SparseAverageVertexPoolGradKernel1<<<config1.block_count, config1.thread_per_block,0,d.stream()>>>(g,
                                                                                 indices,
                                                                                 values,
                                                                                 vertices,
                                                                                 grad_p_values_vals,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 in_vertex_count,
                                                                                 out_vertex_count);

  CudaLaunchConfig config2 = GetCudaLaunchConfig(in_features*batch_size*in_vertex_count, d);
  SparseAverageVertexPoolGradKernel2<<<config2.block_count, config2.thread_per_block,0,d.stream()>>>(g,
                                                                                 indices,
                                                                                 values,
                                                                                 vertices,
                                                                                 grad_v_in_vals,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 in_vertex_count,
                                                                                 out_vertex_count,
                                                                                 batch_size);
  return d.ok();
}

#endif