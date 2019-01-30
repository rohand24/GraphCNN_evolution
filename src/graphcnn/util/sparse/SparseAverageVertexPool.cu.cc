#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

__global__ void SparseAverageVertexPoolKernel(const int64 * indices,
                                 const float * values,
                                 const float * vertices,
                                 float * output,
                                 const int num_edges,
                                 const int in_features,
                                 const int in_vertex_count,
                                 const int out_vertex_count,
                                 const int batch_size) {
    int64 currentBatch;
    int64 currentN1;
    int64 currentN2;
    int64 currentL;
    float currentVal;
    int nid;
    int bid;
    int f1;
    for (int fid = blockIdx.x * blockDim.x + threadIdx.x; fid < in_features*batch_size*out_vertex_count;
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
            if ((nid == currentN1) && (bid == currentBatch))
            {
                //std::cout << currentBatch << " " << currentN1 << " " << currentN2 << " " << currentL << " " << currentVal << std::endl;
                output[currentBatch * out_vertex_count * in_features + currentN1 * in_features + f1] +=
                currentVal*vertices[currentBatch * in_vertex_count * in_features + currentN2*in_features + f1];
            }
        }
    }
}

bool SparseAverageVertexPoolLauncher(const int64 * indices,
                        const float * values,
                        const float * vertices,
                        float * output,
                        const int num_edges,
                        const int in_features,
                        const int in_vertex_count,
                        const int out_vertex_count,
                        const int batch_size,
                        const GPUDevice& d) {
  CudaLaunchConfig config = GetCudaLaunchConfig(in_features*batch_size*out_vertex_count, d);
  SparseAverageVertexPoolKernel<<<config.block_count, config.thread_per_block,0,d.stream()>>>(indices,
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