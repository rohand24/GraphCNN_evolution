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
                        const GPUDevice& d);

//So this just manages the multiplication of the A tensor by V, weights are handled elsewhere
REGISTER_OP("SparseMaxVertexPoolGrad")
    .Input("grad: float32")
    .Input("v_in: float32")
    .Input("p_indices: int64")
    .Input("p_values: float32")
    .Input("v_new_shape: int64")
   // .Attr("num_layers: int")
    .Output("grad_v_in: float32")
    .Output("grad_p_indices: int64")
    .Output("grad_p_values: float32")
    .Output("grad_v_new_shape: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ShapeHandle grad_shape;
      ShapeHandle v_shape;
      ShapeHandle idx_shape;
      ShapeHandle values_shape;
	  ShapeHandle v_new_shape;

      // Validate shapes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &grad_shape)); // Grad = BxN2xF1
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &v_shape)); // V = BxN1xF1
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &idx_shape)); // #Edgesx(b,n1,n2)
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &values_shape)); // Values
	  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &v_new_shape)); // dimensions of P, [n1,n2]

      // Batch dims match between inputs.


      ShapeHandle grad_v_in = c->MakeShape({c->Dim(v_shape, 0), c->Dim(v_shape, 1), c->Dim(v_shape,2)});
      ShapeHandle grad_p_indices = c->MakeShape({c->Dim(idx_shape, 0), c->Dim(idx_shape, 1)});
      ShapeHandle grad_p_values = c->MakeShape({c->Dim(values_shape, 0)});
      ShapeHandle grad_v_new_shape = c->MakeShape({c->Dim(v_new_shape, 0)});

      c->set_output(0, grad_v_in);
      c->set_output(1, grad_p_indices);
      c->set_output(2, grad_p_values);
      c->set_output(3, grad_v_new_shape);
      return Status::OK();
    });

template <typename Device, typename T>
class SparseMaxVertexPoolGradOp : public OpKernel {
 public:
  explicit SparseMaxVertexPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {
    /*OP_REQUIRES_OK(context,
                   context->GetAttr("num_layers", &num_layers));
    OP_REQUIRES(context, num_layers >= 0,
                errors::InvalidArgument("Need num_layers >= 0, got ",
                                        num_layers));*/

  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& grad = context->input(0);
    const Tensor& Vin = context->input(1);
    const Tensor& P_indices = context->input(2);
    const Tensor& P_values = context->input(3);
	const Tensor& V_new_shape = context->input(4);

    auto g = grad.tensor<float,3>();
    auto vertices = Vin.tensor<float, 3>();
    auto indices = P_indices.tensor<int64, 2>();
    auto values = P_values.tensor<float, 1>();
	auto shape = V_new_shape.tensor<int64,1>();

    // Create an output tensor
    TensorShape grad_v_in_shape;
    grad_v_in_shape.AddDim(Vin.dim_size(0));
    grad_v_in_shape.AddDim(Vin.dim_size(1));
    grad_v_in_shape.AddDim(Vin.dim_size(2));
    Tensor* grad_v_in = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_v_in_shape,
                                                     &grad_v_in));
    auto grad_v_in_vals = grad_v_in->tensor<float, 3>();
    auto grad_v_in_vals_flat = grad_v_in->flat<float>();

    TensorShape grad_p_indices_shape;
    grad_p_indices_shape.AddDim(P_indices.dim_size(0));
    grad_p_indices_shape.AddDim(P_indices.dim_size(1));
    Tensor* grad_p_indices = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_p_indices_shape,
                                                     &grad_p_indices));
    auto grad_p_indices_vals = grad_p_indices->tensor<int64, 2>();
    auto grad_p_indices_vals_flat = grad_p_indices->flat<int64>();

    TensorShape grad_p_values_shape;
    grad_p_values_shape.AddDim(P_values.dim_size(0));
    Tensor* grad_p_values = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_p_values_shape,
                                                     &grad_p_values));
    auto grad_p_values_vals = grad_p_values->tensor<float, 1>();
    auto grad_p_values_vals_flat = grad_p_values->flat<float>();

    TensorShape grad_v_new_shape_shape;
    grad_v_new_shape_shape.AddDim(V_new_shape.dim_size(0));
    Tensor* grad_v_new_shape = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(3, grad_v_new_shape_shape,
                                                     &grad_v_new_shape));
    auto grad_v_new_shape_vals = grad_v_new_shape->tensor<int64, 1>();
    auto grad_v_new_shape_vals_flat = grad_v_new_shape->flat<int64>();

    #if GOOGLE_CUDA

    ConstOutLauncher<T>(grad_v_in_vals_flat.data(),
                    grad_v_in_vals_flat.size(),
                    0,
                    context->eigen_gpu_device());

    ConstOutLauncher<int64>(grad_p_indices_vals_flat.data(),
                    grad_p_indices_vals_flat.size(),
                    0,
                    context->eigen_gpu_device());

    ConstOutLauncher<T>(grad_p_values_vals_flat.data(),
                    grad_p_values_vals_flat.size(),
                    0,
                    context->eigen_gpu_device());

    ConstOutLauncher<int64>(grad_v_new_shape_vals_flat.data(),
                    grad_v_new_shape_vals_flat.size(),
                    0,
                    context->eigen_gpu_device());

    #else

    const int64 num_unwrapped_features_v_in = grad_v_in_vals_flat.size();
    for (int64 i = 0; i < num_unwrapped_features_v_in; i++) {
      grad_v_in_vals_flat(i) = 0;
    }

    //I think this should always have a zero gradient
    const int64 num_unwrapped_features_p_indices = grad_p_indices_vals_flat.size();
    for (int64 i = 0; i < num_unwrapped_features_p_indices; i++) {
      grad_p_indices_vals_flat(i) = 0;
    }

    const int64 num_unwrapped_features_p_values = grad_p_values_vals_flat.size();
    for (int64 i = 0; i < num_unwrapped_features_p_values; i++) {
      grad_p_values_vals_flat(i) = 0;
    }

    const int64 num_unwrapped_features_v_new_shape = grad_v_new_shape_vals_flat.size();
    for (int64 i = 0; i < num_unwrapped_features_v_new_shape; i++) {
      grad_v_new_shape_vals_flat(i) = 0;
    }

    #endif

    const int64 batch_size = Vin.dim_size(0);
    const int64 num_vertices = Vin.dim_size(1);
    const int in_features = Vin.dim_size(2);
    const int num_edges = P_indices.dim_size(0);
    const int out_num_vertices = shape(1);

    //Store state of max
    Tensor maxOutput_t;
    Tensor maxEdge_t;

    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({static_cast<int64>
                (grad.dim_size(0)),static_cast<int64>
                (grad.dim_size(1)),static_cast<int64>
                (grad.dim_size(2))}), &maxOutput_t));
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({static_cast<int64>
                (grad.dim_size(0)),static_cast<int64>
                (grad.dim_size(1)),static_cast<int64>
                (grad.dim_size(2))}), &maxEdge_t));

    auto maxOutput = maxOutput_t.tensor<float,3>();
    auto maxEdge = maxEdge_t.tensor<int64,3>();

    #if GOOGLE_CUDA

    ConstOutLauncher<T>(maxOutput_t.flat<T>().data(),
                    maxOutput_t.flat<T>().size(),
                    INT64_MIN,
                    context->eigen_gpu_device());

    ConstOutLauncher<int64>(maxEdge_t.flat<int64>().data(),
                    maxEdge_t.flat<int64>().size(),
                    -1,
                    context->eigen_gpu_device());

    #else

    for (int current_batch_item = 0; current_batch_item < grad.dim_size(0); current_batch_item++)
    {
        for (int N1 = 0; N1 < grad.dim_size(1); N1++)
        {
            for(int current_feature = 0; current_feature < grad.dim_size(2); current_feature++)
            {
                maxOutput(current_batch_item,N1,current_feature) = INT64_MIN;
                maxEdge(current_batch_item,N1,current_feature) = -1;
            }
        }
    }

    #endif

    #if GOOGLE_CUDA

    SparseMaxVertexPoolGradLauncher(grad.flat<T>().data(),
                        P_indices.flat<int64>().data(),
                        P_values.flat<T>().data(),
                        Vin.flat<T>().data(),
                        maxOutput_t.flat<T>().data(),
                        maxEdge_t.flat<int64>().data(),
                        grad_v_in_vals_flat.data(),
                        grad_p_values_vals_flat.data(),
                        num_edges,
                        in_features,
                        num_vertices,
                        out_num_vertices,
                        batch_size,
                        section_count,
                        context->eigen_gpu_device());

    #else

    //(P indices and shape should always be zero gradient)
    const auto thread_pool = context->device()->tensorflow_cpu_worker_threads();
	const int num_threads = std::min(thread_pool->num_threads, in_features);

    auto threadFunc = [&](int thread_id) {
        int64 currentBatch;
        int64 currentN1;
        int64 currentN2;
        int64 currentL;
        float currentVal;
        float currentWeight;
        float currentGrad;
        float currentVertex;
        float proposedMax;
        for (int64 f1 = thread_id; f1 < in_features; f1+=num_threads)
        {
            for(int64  edge = 0; edge < num_edges; edge++)
            {
                currentBatch = indices(edge,0);
                //Transpose
                currentN1 = indices(edge,2);
                currentN2 = indices(edge,3);
                currentVal = values(edge);
                proposedMax = currentVal*vertices(currentBatch,currentN2,f1);
                //std::cout << currentBatch << " " << currentN1 << " " << currentN2 << " " << currentL << " " << currentVal << std::endl;
                if (proposedMax > maxOutput(currentBatch,currentN1,f1))
                {
                    maxVal(currentBatch,currentN1,f1) = currentVal;
                    maxVertex(currentBatch,currentN1,f1) = vertices(currentBatch,currentN2,f1);
                    maxOutput(currentBatch,currentN1,f1) = proposedMax;
                    maxN2(currentBatch,currentN1,f1) = currentN2;
                    maxN1(currentBatch,currentN1,f1) = currentN1;
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
   // std::mutex mtx;

    const int num_threadsGrad1 = std::min(thread_pool->num_threads, num_edges);
    auto threadFuncGrad = [&](int thread_id) {
        int64 currentBatch;
        int64 currentN1;
        int64 currentN2;
        int64 currentL;
        float currentVal;
        float currentWeight;
        float currentGrad;
        float currentVertex;
        float proposedMax;
        for (int64 edge = thread_id; edge < num_edges; edge+=num_threadsGrad1)
        {
            currentBatch = indices(edge,0);
            currentN1 = indices(edge,2);
            currentN2 = indices(edge,3);
            currentVal = values(edge);
            for(int64  f1 = 0; f1 < in_features; f1++)
            {
                currentGrad = g(currentBatch,currentN1,f1);
                currentVertex = vertices(currentBatch,currentN2,f1);
                if ((currentN2 == maxN2(currentBatch,currentN1,f1)))
                {
                    //mtx.lock();
                    //std::cout << edge << " " << f1 << " " << maxVertex(currentBatch,currentN1,f1) << std::endl;
                    grad_p_values_vals(edge) += currentGrad*maxVertex(currentBatch,currentN1,f1);
                    //mtx.unlock();
                }
            }
        }
    };
    BlockingCounter counterGrad(num_threadsGrad1-1);
    for (int i = 1; i < num_threadsGrad1; ++i) {
        thread_pool->workers->Schedule([&, i]() {
            threadFuncGrad(i);
            counterGrad.DecrementCount();
        });
    }
    threadFuncGrad(0);
    counterGrad.Wait();

    const int num_threadsGrad2 = std::min(thread_pool->num_threads, in_features);
    auto threadFuncGrad2 = [&](int thread_id) {
        int64 currentBatch;
        int64 currentN1;
        int64 currentN2;
        int64 currentL;
        float currentVal;
        float currentWeight;
        float currentGrad;
        float currentVertex;
        float proposedMax;
        for (int64 f1 = thread_id; f1 < in_features; f1+=num_threadsGrad2)
        {
            for(int64  edge = 0; edge < num_edges; edge++)
            {
                    currentBatch = indices(edge,0);
                    currentN1 = indices(edge,2);
                    currentN2 = indices(edge,3);
                    currentVal = values(edge);
                    currentGrad = g(currentBatch,currentN1,f1);
                    currentVertex = vertices(currentBatch,currentN2,f1);
                    if ((currentN2 == maxN2(currentBatch,currentN1,f1)) && (currentN1 == maxN1(currentBatch,currentN1,f1)))
                    {
                        grad_v_in_vals(currentBatch,currentN2,f1) += currentGrad*maxVal(currentBatch,currentN1,f1);
                    }
            }
        }
    };
    BlockingCounter counterGrad2(num_threadsGrad2-1);
    for (int i = 1; i < num_threadsGrad2; ++i) {
        thread_pool->workers->Schedule([&, i]() {
            threadFuncGrad2(i);
            counterGrad2.DecrementCount();
        });
    }
    threadFuncGrad2(0);
    counterGrad2.Wait();

    #endif
  }
 private:
    //int num_filters;
    //int num_layers;
    //# of sections for parallel scan
    const int section_count = 256;

};

#define REGISTER_CPU_KERNEL(T) \
REGISTER_KERNEL_BUILDER(Name("SparseMaxVertexPoolGrad").Device(DEVICE_CPU), SparseMaxVertexPoolGradOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNEL(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("SparseMaxVertexPoolGrad").Device(DEVICE_GPU).HostMemory("v_new_shape"),          \
                          SparseMaxVertexPoolGradOp<GPUDevice, T>);

TF_CALL_float(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA

