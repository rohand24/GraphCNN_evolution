#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include <stdint.h>

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#include <iostream>

using namespace std;

template <typename T>
bool ConstOutLauncher(T * data,
                        const int size,
                        const T value,
                        const GPUDevice& d);

/*bool SparseConvLauncher(const int64 * indices,
                        const float * h,
                        const float * values,
                        const float * vertices,
                        float * output,
                        const int num_edges,
                        const int in_features,
                        const int out_features,
                        const int N,
                        const int batch_size,
                        const GPUDevice& d);*/

//So this just manages the multiplication of the A tensor by V, weights are handled elsewhere
REGISTER_OP("SparseDiagonalMultiply")
    .Input("d: float32")
    .Input("a_indices: int64")
    .Input("a_values: float32")
    .Attr("isLeft: bool")
    .Output("a_values_out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ShapeHandle d_shape;
      ShapeHandle idx_shape;
      ShapeHandle values_shape;

      // Validate shapes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &d_shape)); // V = BxNxF1
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &idx_shape)); // #Edgesx(b,n1,n2,l)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &values_shape)); // Values

      ShapeHandle out = c->MakeShape({c->Dim(values_shape, 0)});

      c->set_output(0, out);
      return Status::OK();
    });

template <typename Device, typename T>
class SparseDiagonalMultiplyOp : public OpKernel {
 public:
  explicit SparseDiagonalMultiplyOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("isLeft", &isLeft));
    OP_REQUIRES(context, isLeft >= 0,
                errors::InvalidArgument("Need num_layers >= 0, got ",
                                        isLeft));
    OP_REQUIRES(context, isLeft <= 1,
                errors::InvalidArgument("Need num_layers <= 1, got ",
                                        isLeft));

  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& Din = context->input(0);
    const Tensor& A_indices = context->input(1);
    const Tensor& A_values = context->input(2);

    auto diagonal = Din.tensor<T, 2>();
    auto indices = A_indices.tensor<int64, 2>();
    auto values = A_values.tensor<T, 1>();


    // Create an output tensor
    TensorShape out_shape;
    out_shape.AddDim(A_values.dim_size(0));
    Tensor* A_out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &A_out));
    auto output = A_out->tensor<T,1>();
    auto output_flat = A_out->flat<T>();

    #if GOOGLE_CUDA

    ConstOutLauncher<float>(output_flat.data(),
                    output_flat.size(),
                    0,
                    context->eigen_gpu_device());
    #else


    const int64 num_unwrapped_features = output_flat.size();
    for (int64 i = 0; i < num_unwrapped_features; i++) {
      output_flat(i) = 0;
    }

    #endif

    const int64 batch_size = Din.dim_size(0);
    const int64 num_vertices = Din.dim_size(1);
    const int num_edges = A_indices.dim_size(0);

    #if GOOGLE_CUDA
    /*SparseConvLauncher(A_indices.flat<int64>().data(),
                       h_weights.flat<T>().data(),
                       A_values.flat<T>().data(),
                       Vin.flat<T>().data(),
                       output_flat.data(),
                       num_edges,
                       in_features,
                       out_features,
                       num_vertices,
                       batch_size,
                       context->eigen_gpu_device());*/
    #else

	const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
	const int num_threads = std::min(thread_pool->num_threads, num_edges);

	if (isLeft)
	{
        auto threadFunc = [&](int thread_id) {
            int64 currentBatch;
            int64 currentN1;
            int64 currentN2;
            int64 currentL;
            T currentVal;
            T currentDiag;
            for(int64 edge = thread_id; edge < num_edges; edge+=num_threads)
            {
                currentBatch = indices(edge,0);
                currentN1 = indices(edge,1);
                currentL = indices(edge,2);
                currentN2 = indices(edge,3);
                currentVal = values(edge);
                currentDiag = diagonal(currentN1);
                output(edge) += currentVal*currentDiag;
            }
        };
        BlockingCounter counter(num_threads-1);
        for (int i = 1; i < num_threads; ++i) {
            thread_pool->workers->Schedule([&, i]() {
                threadFunc(i);
                counter.DecrementCount();
            });
        }
        threadFunc(0);
        counter.Wait();
	}
	else
	{
       auto threadFunc = [&](int thread_id) {
            int64 currentBatch;
            int64 currentN1;
            int64 currentN2;
            int64 currentL;
            T currentVal;
            T currentDiag;
            for(int64 edge = thread_id; edge < num_edges; edge+=num_threads)
            {
                currentBatch = indices(edge,0);
                currentN1 = indices(edge,1);
                currentL = indices(edge,2);
                currentN2 = indices(edge,3);
                currentVal = values(edge);
                currentDiag = diagonal(currentN2);
                output(edge) += currentVal*currentDiag;
            }
        };
        BlockingCounter counter(num_threads-1);
        for (int i = 1; i < num_threads; ++i) {
            thread_pool->workers->Schedule([&, i]() {
                threadFunc(i);
                counter.DecrementCount();
            });
        }
        threadFunc(0);
        counter.Wait();
	}

    #endif


    }
    private:
    bool isLeft;

};

#define REGISTER_CPU_KERNEL(T) \
REGISTER_KERNEL_BUILDER(Name("SparseDiagonalMultiply").Device(DEVICE_CPU), SparseDiagonalMultiplyOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNEL(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("SparseDiagonalMultiply") \
                              .Device(DEVICE_GPU),          \
                          SparseDiagonalMultiplyOp<GPUDevice, T>);

TF_CALL_float(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA
