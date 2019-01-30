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
							float * grad_h_weights_vals,
                            const int num_edges,
                            const int in_features,
                            const int N,
							const int num_filters){
    int64 currentBatch;
    int64 currentN1;
    int64 currentN2;
    int64 currentL;
    float currentWeight;
    float currentVertex;
    float currentGrad;
	float currentVal;
    int filter;
    int edge;
    for (uint64 fid = blockIdx.x * blockDim.x + threadIdx.x; fid < num_filters*num_edges;
       fid += blockDim.x * gridDim.x)
    {
        filter = fid % num_filters;
        edge = (fid / num_filters) % num_edges;
        currentBatch = indices[edge*4 + 0];
        currentN1 = indices[edge*4 + 1];
        currentL = indices[edge*4 + 2];
        currentN2 = indices[edge*4 + 3];

        currentVal = values[edge];
		currentGrad = g[edge];
		currentWeight = h[2*in_features*num_filters + filter];
        atomicAdd(&(grad_a_values_vals[edge]),currentGrad*currentWeight);
		atomicAdd(&(grad_h_weights_vals[2*in_features*num_filters + filter]),currentGrad*currentVal);
    }
}

__global__ void SparseConvGradKernel2 (const float * g,
                            const int64 * indices,
                            const float * h,
                            const float * values,
                            const float * vertices,
							float * grad_h_weights_vals,
                            float * grad_v_in_vals,
                            const int num_edges,
                            const int in_features,
                            const int N,
                            const int batch_size,
							const int num_filters){
    /*int64 currentBatch;
    int64 currentN1;
    int64 currentN2;
    int64 currentL;*/
    const int TILE_SIZE = 200;
    const int TENSOR_RANK = 4;
    int64 currentBatch;
    int64 currentN1;
    int64 currentN2;
    int64 currentL;
    float currentWeightS;
	float currentWeightR;
    float currentVertexS;
	float currentVertexR;
    float currentGrad;
	float currentVal;
    int filter;
    int edge;
	int f1;
    //for (int f1 = blockIdx.x * blockDim.x + threadIdx.x; f1 < in_features; f1 += blockDim.x * gridDim.x)
    for (uint64 fid = blockIdx.x * blockDim.x + threadIdx.x; fid < num_filters*num_edges;
       fid += blockDim.x * gridDim.x)
    {
        filter = fid % num_filters;
        edge = (fid / num_filters) % num_edges;
        currentBatch = indices[edge*4 + 0];
        currentN1 = indices[edge*4 + 1];
        currentL = indices[edge*4 + 2];
        currentN2 = indices[edge*4 + 3];
        currentVal = values[edge];
		currentGrad = g[edge];
		for(int64 f1 = 0; f1 < in_features; f1++)
		{
			currentVertexS = vertices[currentBatch * N * in_features + currentN1*in_features + f1];
			currentVertexR = vertices[currentBatch * N * in_features + currentN2*in_features + f1];
			currentWeightS = h[f1*num_filters + filter];
			currentWeightR = h[(in_features + f1)*num_filters + filter];
			atomicAdd(&(grad_h_weights_vals[f1*num_filters + filter]),currentGrad*currentVertexS);
			atomicAdd(&(grad_h_weights_vals[(in_features + f1)*num_filters + filter]),currentGrad*currentVertexR);
			atomicAdd(&(grad_v_in_vals[currentBatch * N * in_features + currentN1*in_features + f1]),currentGrad*currentWeightS);
			atomicAdd(&(grad_v_in_vals[currentBatch * N * in_features + currentN2*in_features + f1]),currentGrad*currentWeightR);
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
                            const int N,
                            const int num_layers,
                            const int batch_size,
							const int num_filters,
                            const GPUDevice& d) {
  uint64 totalThreads1 = num_edges*num_filters;
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
																				 grad_h_weights_vals,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 N,
																				 num_filters);
  uint64 totalThreads2 = num_edges*num_filters;
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
																				 grad_h_weights_vals,
                                                                                 grad_v_in_vals,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 N,
                                                                                 batch_size,
																				 num_filters);
  return d.ok();
}

#endif