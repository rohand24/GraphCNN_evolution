#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include <math.h>

using namespace tensorflow;

typedef Eigen::GpuDevice GPUDevice;

__global__ void SparseEdgeConvKernel(const int64 * indices,
                        const float * h,
                        const float * values,
                        const float * vertices,
                        float * output,
                        const int num_edges,
                        const int in_features,
                        const int N,
                        const int batch_size,
                        const int num_filters) {

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
    float currentWeightS;
	float currentWeightR;
	float currentWeightE;
    float currentVertexS;
	float currentVertexR;
    int filter;
    int edge;
    //for (int f2 = blockIdx.x * blockDim.x + threadIdx.x; f2 < out_features;
     //  f2 += blockDim.x * gridDim.x)
    for (uint64 fid = blockIdx.x * blockDim.x + threadIdx.x; fid < num_filters*num_edges;
       fid += blockDim.x * gridDim.x)
    {
		const int numWeights = 2*in_features + 1;
        filter = fid % num_filters;
        edge = (fid / num_filters) % num_edges;
        #pragma unroll
        for (int i = 0; i < TENSOR_RANK; i++)
        {
            currentIndex[i] = indices[edge*TENSOR_RANK + i];
        }
        currentVal = values[edge];
		currentWeightE = h[(2*in_features)*num_filters+filter];
		float accumulator = currentWeightE*currentVal;
		#pragma unroll 10
		for (int64 f1 = 0; f1 < in_features; f1++)
		{
			currentWeightS = h[f1*num_filters + filter];
			currentWeightR = h[(in_features + f1)*num_filters + filter];
			currentVertexS = vertices[currentIndex[0] * N * in_features + currentIndex[1]*in_features + f1];
			currentVertexR = vertices[currentIndex[0] * N * in_features + currentIndex[3]*in_features + f1];
			//std::cout << currentBatch << " " << currentN1 << " " << currentN2 << " " << currentL << " " << currentVal << std::endl;
			accumulator += (currentWeightS*currentVertexS + currentWeightR*currentVertexR);
			//temp[f2] += currentWeight*currentVal*vertices(currentBatch,currentN2,f1);
		}
	//output[currentIndex[0] * N * out_features + currentIndex[1] * out_features + f2] += accumulator;
		atomicAdd(&(output[edge]),accumulator);

    }
}

bool SparseEdgeConvLauncher(const int64 * indices,
                        const float * h,
                        const float * values,
                        const float * vertices,
                        float * output,
                        const int num_edges,
                        const int in_features,
                        const int N,
                        const int batch_size,
                        const int num_filters,
                        const GPUDevice& d) {
  uint64 totalThreads = num_edges*num_filters;
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
  SparseEdgeConvKernel<<<config.block_count, config.thread_per_block,0,d.stream()>>>(indices,
                                                                                 h,
                                                                                 values,
                                                                                 vertices,
                                                                                 output,
                                                                                 num_edges,
                                                                                 in_features,
                                                                                 N,
                                                                                 batch_size,
																				 num_filters);
  return d.ok();
}

#endif