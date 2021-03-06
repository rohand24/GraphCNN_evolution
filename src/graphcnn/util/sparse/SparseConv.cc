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
                        const GPUDevice& d);

//So this just manages the multiplication of the A tensor by V, weights are handled elsewhere
REGISTER_OP("SparseGraphConvolution")
    .Input("v_in: float32")
    .Input("a_indices: int64")
    .Input("a_values: float32")
	.Input("h: float32")
   // .Attr("num_layers: int")
    .Output("v_out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ShapeHandle v_shape;
      ShapeHandle idx_shape;
      ShapeHandle values_shape;
	  ShapeHandle h_shape;

      // Validate shapes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &v_shape)); // V = BxNxF1
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &idx_shape)); // #Edgesx(b,n1,n2,l)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &values_shape)); // Values
	  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &h_shape)); // h = LxF1xF2

      ShapeHandle out = c->MakeShape({c->Dim(v_shape, 0), c->Dim(v_shape, 1), c->Dim(h_shape, 2)});

      c->set_output(0, out);
      return Status::OK();
    });

template <typename Device, typename T>
class SparseGraphConvolutionOp : public OpKernel {
 public:
  explicit SparseGraphConvolutionOp(OpKernelConstruction* context) : OpKernel(context) {
    /*OP_REQUIRES_OK(context,
                   context->GetAttr("num_layers", &num_layers));
    OP_REQUIRES(context, num_layers >= 0,
                errors::InvalidArgument("Need num_layers >= 0, got ",
                                        num_layers));*/

  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& Vin = context->input(0);
    const Tensor& A_indices = context->input(1);
    const Tensor& A_values = context->input(2);
	const Tensor& h_weights = context->input(3);

    auto vertices = Vin.tensor<T, 3>();
    auto indices = A_indices.tensor<int64, 2>();
    auto values = A_values.tensor<T, 1>();
	auto h = h_weights.tensor<T,3>();


    // Create an output tensor
    TensorShape out_shape;
    out_shape.AddDim(Vin.dim_size(0));
    out_shape.AddDim(Vin.dim_size(1));
    out_shape.AddDim(h_weights.dim_size(2));
    Tensor* V_out = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &V_out));
    auto output = V_out->tensor<T, 3>();
    auto output_flat = V_out->flat<T>();

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

    const int64 batch_size = Vin.dim_size(0);
    const int64 num_vertices = Vin.dim_size(1);
    const int in_features = Vin.dim_size(2);
    const int out_features = V_out->dim_size(2);
    const int num_edges = A_indices.dim_size(0);

    #if GOOGLE_CUDA
    SparseConvLauncher(A_indices.flat<int64>().data(),
                       h_weights.flat<T>().data(),
                       A_values.flat<T>().data(),
                       Vin.flat<T>().data(),
                       output_flat.data(),
                       num_edges,
                       in_features,
                       out_features,
                       num_vertices,
                       batch_size,
                       context->eigen_gpu_device());
    #else

	const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
	const int num_threads = std::min(thread_pool->num_threads, out_features);

	auto threadFunc = [&](int thread_id) {
            int64 currentBatch;
            int64 currentN1;
            int64 currentN2;
            int64 currentL;
            T currentVal;
            T currentWeight;
            for(int64 f2 = thread_id; f2 < out_features; f2+=num_threads)
            {
                for (int64 edge = 0; edge < num_edges; edge++)
                {
                    currentBatch = indices(edge,0);
                    currentN1 = indices(edge,1);
                    currentL = indices(edge,2);
                    currentN2 = indices(edge,3);
                    currentVal = values(edge);
                    for(int64  f1 = 0; f1 < in_features; f1++)
                    {
                        currentWeight = h(currentL,f1,f2);
                        //std::cout << currentBatch << " " << currentN1 << " " << currentN2 << " " << currentL << " " << currentVal << std::endl;
                        output(currentBatch,currentN1,f2) += currentWeight*currentVal*vertices(currentBatch,currentN2,f1);
                        //temp[f2] += currentWeight*currentVal*vertices(currentBatch,currentN2,f1);
                    }
                }
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

    #endif


    }

};

#define REGISTER_CPU_KERNEL(T) \
REGISTER_KERNEL_BUILDER(Name("SparseGraphConvolution").Device(DEVICE_CPU), SparseGraphConvolutionOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNEL(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("SparseGraphConvolution") \
                              .Device(DEVICE_GPU),          \
                          SparseGraphConvolutionOp<GPUDevice, T>);

TF_CALL_float(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA
