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

bool SparseAverageVertexPoolLauncher(const int64 * indices,
                        const float * values,
                        const float * vertices,
                        float * output,
                        const int num_edges,
                        const int in_features,
                        const int in_vertex_count,
                        const int out_vertex_count,
                        const int batch_size,
                        const GPUDevice& d);

//So this just manages the multiplication of the A tensor by V, weights are handled elsewhere
REGISTER_OP("SparseAverageVertexPool")
    .Input("v_in: float32")
    .Input("p_indices: int64")
    .Input("p_values: float32")
	.Input("new_v_shape: int64")
	//.Attr("out_size: int")
    .Output("v_out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ShapeHandle v_shape;
      ShapeHandle idx_shape;
      ShapeHandle values_shape;
      //int out_size = c->attrs().Find("out_size")->i();
	  //const Tensor* p_shape = c->input_tensor(3);

      // Validate shapes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &v_shape)); // V = BxN1xF1
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &idx_shape)); // #Edgesx(B,n1,n2)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &values_shape)); // Values
      //TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &shape_shape)); //dimensions of P, [n1,n2]

      //ShapeHandle out = c->MakeShape({c->Dim(v_shape, 0), out_size, c->Dim(v_shape, 2)});
	  ShapeHandle out;

	  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &out));

      c->set_output(0, out);
      return Status::OK();
    });

template <typename Device, typename T>
class SparseAverageVertexPoolOp : public OpKernel {
 public:
  explicit SparseAverageVertexPoolOp(OpKernelConstruction* context) : OpKernel(context) {
    //OP_REQUIRES_OK(context,
    //               context->GetAttr("out_size", &out_size));
    //OP_REQUIRES(context, out_size >= 0,
    //            errors::InvalidArgument("Need out_size >= 0, got ",
    //                                    out_size));

  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& Vin = context->input(0);
    const Tensor& P_indices = context->input(1);
    const Tensor& P_values = context->input(2);
    const Tensor& new_V_shape = context->input(3);

    auto vertices = Vin.tensor<T, 3>();
    auto indices = P_indices.tensor<int64, 2>();
    auto values = P_values.tensor<T, 1>();
    auto shape = new_V_shape.tensor<int64, 1>();

    // Create an output tensor
    TensorShape out_shape;
    out_shape.AddDim(Vin.dim_size(0));
    //out_shape.AddDim(out_size);
	out_shape.AddDim(shape(1));
    out_shape.AddDim(Vin.dim_size(2));
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
    const int64 out_features = V_out->dim_size(2);
    const int num_edges = P_indices.dim_size(0);
    const int out_num_vertices = shape(1);

    #if GOOGLE_CUDA

    SparseAverageVertexPoolLauncher(P_indices.flat<int64>().data(),
                       P_values.flat<T>().data(),
                       Vin.flat<T>().data(),
                       output_flat.data(),
                       num_edges,
                       in_features,
                       num_vertices,
                       out_num_vertices,
                       batch_size,
                       context->eigen_gpu_device());

    #else

    const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
	const int num_threads = std::min(thread_pool->num_threads, in_features);

	auto threadFunc = [&](int thread_id) {
            int64 currentBatch;
            int64 currentN1;
            int64 currentN2;
            int64 currentL;
            float currentVal;
            for (int64 f1 = thread_id; f1 < in_features; f1+=num_threads)
            {
                for(int64  edge = 0; edge < num_edges; edge++)
                {
                    currentBatch = indices(edge,0);
                    //Transpose
                    currentN1 = indices(edge,2);
                    currentN2 = indices(edge,3);
                    currentVal = values(edge);
                    //std::cout << currentBatch << " " << currentN1 << " " << currentN2 << " " << currentL << " " << currentVal << std::endl;
                    output(currentBatch,currentN1,f1) += currentVal*vertices(currentBatch,currentN2,f1);
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
 //private:
  //int out_size;
};

#define REGISTER_CPU_KERNEL(T) \
REGISTER_KERNEL_BUILDER(Name("SparseAverageVertexPool").Device(DEVICE_CPU), SparseAverageVertexPoolOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNEL(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("SparseAverageVertexPool") \
                              .Device(DEVICE_GPU).HostMemory("new_v_shape"),          \
                          SparseAverageVertexPoolOp<GPUDevice, T>);

TF_CALL_float(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA