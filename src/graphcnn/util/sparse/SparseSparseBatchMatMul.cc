#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include <stdint.h>
#include <vector>
#include "SMMPUtils.h"
#include "SparseSparseMatMulControl.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

#include <iostream>
REGISTER_OP("SparseSparseBatchMatMul")
    .Input("a_indices: int64")
    .Input("a_values: float32")
    .Input("a_shape: int64")
    .Input("b_indices: int64")
    .Input("b_values: float32")
    .Input("b_shape: int64")
    //.Output("a_i: int64")
    //.Output("a_j: int64")
    //.Output("a_reindices: int64")
    //.Output("b_reindices: int64")
    .Output("c_indices: int64")
    .Output("c_values: float32")
    .Output("c_shape: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ShapeHandle a_idx_shape;
      ShapeHandle a_values_shape;
      ShapeHandle a_shape_shape;
      ShapeHandle b_idx_shape;
      ShapeHandle b_values_shape;
      ShapeHandle b_shape_shape;

      // Validate shapes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a_idx_shape)); // V = BxN1xF1
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &a_values_shape)); // Values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &a_shape_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &b_idx_shape)); // V = BxN1xF1
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &b_values_shape)); // Values
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &b_shape_shape));
      //TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &shape_shape)); //dimensions of P, [n1,n2]

      return Status::OK();
    });

//Going to assume that
class SparseSparseBatchMatMulOp : public OpKernel{
    public:
        explicit SparseSparseBatchMatMulOp(OpKernelConstruction* context): OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            //Pass inputs into Tensor objects
            const Tensor& A_indices = context->input(0);
            const Tensor& A_values = context->input(1);
            const Tensor& A_shape = context->input(2);
            const Tensor& B_indices = context->input(3);
            const Tensor& B_values = context->input(4);
            const Tensor& B_shape = context->input(5);

            Tensor * C_idx_ptr;
            Tensor * C_val_ptr;
            Tensor * C_shape_ptr;
            const size_t c_indices_out_idx = 0;
            const size_t c_values_out_idx = 1;
            const size_t c_shape_out_idx = 2;


            auto a_shape_tensor = A_shape.tensor<int64, 1>();
            auto b_shape_tensor = B_shape.tensor<int64, 1>();

            enum MultiplyModeVals {LEFT_MULTIPLY_STACK, RIGHT_MULTIPLY_STACK, STACK_BOTH_SIDES};

            //Figure out how many 2D matrices to keep track of
            const size_t batchSize = a_shape_tensor(SparseSparseMatMulControl::BATCH_INDEX);
            size_t layerCount = 0;
            size_t layerCountA = 0;
            size_t layerCountB = 0;
            MultiplyModeVals multiplyMode = STACK_BOTH_SIDES;
            //Does not check to see if layer counts are the same size!
            //Decide which input is the pool matrix (rank 3) and which is the adjacency tensor (rank 4)
            if (((a_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX) > SparseSparseMatMulControl::SINGLETON) &&
            (b_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX) > SparseSparseMatMulControl::SINGLETON)) ||
            ((a_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX) == SparseSparseMatMulControl::SINGLETON) &&
            (b_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX) == SparseSparseMatMulControl::SINGLETON)))
            {
                layerCount = a_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX);
                layerCountA = a_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX);
                layerCountB = b_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX);
                //std::cout << "MODE: STACK BOTH SIDES" << std::endl;
                if (layerCountA != layerCountB)
                {
                    std::cout << "LAYER COUNTS DON'T MATCH: STACK_BOTH_SIDES MODE" << std::endl;
                    throw "LAYER COUNTS DON'T MATCH: STACK_BOTH_SIDES MODE";
                }
            }
            else if ((a_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX) == SparseSparseMatMulControl::SINGLETON) &&
             (b_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX) > SparseSparseMatMulControl::SINGLETON))
            {
                layerCount = b_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX);
                layerCountA = a_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX);
                layerCountB = b_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX);
                multiplyMode = LEFT_MULTIPLY_STACK;
                //std::cout << "LEFT MULTIPLY STACK" << std::endl;
            }
            else if ((a_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX) > SparseSparseMatMulControl::SINGLETON) &&
            (b_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX) == SparseSparseMatMulControl::SINGLETON))
            {
                layerCount = a_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX);
                multiplyMode = RIGHT_MULTIPLY_STACK;
                layerCountA = a_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX);
                layerCountB = b_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX);
                //std::cout << "RIGHT MULTIPLY STACK" << std::endl;
            }
            else
            {
                std::cout << a_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX) << " " << b_shape_tensor(SparseSparseMatMulControl::LAYER_INDEX) << std::endl;
                std::cout << "NOT A SUPPORTED MODE" << std::endl;
                throw "NOT A SUPPORTED MODE";
            }
            //const size_t batchSize = 1;
            //const size_t layerCount = 1;
            //Separate single index and value tensors into per-slice tensors
            std::vector<std::vector<Tensor>> A_index_slices(batchSize,std::vector<Tensor>(layerCountA,Tensor()));
            std::vector<std::vector<Tensor>> A_value_slices(batchSize,std::vector<Tensor>(layerCountA,Tensor()));
            std::vector<std::vector<Tensor>> B_index_slices(batchSize,std::vector<Tensor>(layerCountB,Tensor()));
            std::vector<std::vector<Tensor>> B_value_slices(batchSize,std::vector<Tensor>(layerCountB,Tensor()));

            //Convert per-slice tensors into CSR format
            std::vector<std::vector<Tensor>> A_is(batchSize,std::vector<Tensor>(layerCountA,Tensor()));
            std::vector<std::vector<Tensor>> A_js(batchSize,std::vector<Tensor>(layerCountA,Tensor()));
            std::vector<std::vector<Tensor>> B_is(batchSize,std::vector<Tensor>(layerCountB,Tensor()));
            std::vector<std::vector<Tensor>> B_js(batchSize,std::vector<Tensor>(layerCountB,Tensor()));

            //Output slices
            std::vector<std::vector<Tensor>> C_is(batchSize,std::vector<Tensor>(layerCount,Tensor()));
            std::vector<std::vector<Tensor>> C_js(batchSize,std::vector<Tensor>(layerCount,Tensor()));
            std::vector<std::vector<Tensor>> C_values(batchSize,std::vector<Tensor>(layerCount,Tensor()));
            //std::cout << "NO SLICES YET" << std::endl;
            //Split tensor into slices
            SparseSparseMatMulControl::ConvertTensorToSlices(A_indices,
                                                             A_values,
                                                             batchSize,
                                                             layerCountA,
                                                             A_index_slices,
                                                             A_value_slices,
                                                             context);
            //std::cout << "A Tensors Sliced" << std::endl;
            SparseSparseMatMulControl::ConvertTensorToSlices(B_indices,
                                                             B_values,
                                                             batchSize,
                                                             layerCountB,
                                                             B_index_slices,
                                                             B_value_slices,
                                                             context);
            //std::cout << "B Tensors Sliced" << std::endl;

            for (size_t currentBatch = 0; currentBatch < batchSize; currentBatch++)
            {
                size_t currentLayerA = 0;
                size_t currentLayerB = 0;
                for (size_t currentLayer = 0; currentLayer < layerCount; currentLayer++)
                {
                    //std::cout << "CURRENT BATCH: " << currentBatch << "LAYER: " << currentLayer << "LAYERA: " << currentLayerA << "LAYERB: " << currentLayerB << std::endl;
                    SparseSparseMatMulControl::ConvertInputsToCSR(A_index_slices[currentBatch][currentLayerA],
                                                                  A_shape,
                                       &(A_is[currentBatch][currentLayerA]),
                                       &(A_js[currentBatch][currentLayerA]),
                                       context);
                    SparseSparseMatMulControl::ConvertInputsToCSR(B_index_slices[currentBatch][currentLayerB],
                                                                  B_shape,
                                       &(B_is[currentBatch][currentLayerB]),
                                       &(B_js[currentBatch][currentLayerB]),
                                       context);
                    SparseSparseMatMulControl::MatMul2D(A_is[currentBatch][currentLayerA],
                             A_js[currentBatch][currentLayerA],
                             A_value_slices[currentBatch][currentLayerA],
                             A_shape,
                             B_is[currentBatch][currentLayerB],
                             B_js[currentBatch][currentLayerB],
                             B_value_slices[currentBatch][currentLayerB],
                             B_shape,
                             &(C_is[currentBatch][currentLayer]),
                             &(C_js[currentBatch][currentLayer]),
                             &(C_values[currentBatch][currentLayer]),
                             context);
                    switch(multiplyMode)
                    {
                        case STACK_BOTH_SIDES:
                            currentLayerA++;
                            currentLayerB++;
                            break;
                        case LEFT_MULTIPLY_STACK:
                            currentLayerB++;
                            break;
                        case RIGHT_MULTIPLY_STACK:
                            currentLayerA++;
                            break;
                    }
                }
            }


            SparseSparseMatMulControl::AggregateForward(C_is,
                      C_js,
                      C_values,
                      batchSize,
                      layerCount,
                      a_shape_tensor(SparseSparseMatMulControl::N1_INDEX),
                      b_shape_tensor(SparseSparseMatMulControl::N2_INDEX),
                      C_idx_ptr,
                      C_val_ptr,
                      C_shape_ptr,
                      c_indices_out_idx,
                      c_values_out_idx,
                      c_shape_out_idx,
                      context);
        }

};

REGISTER_KERNEL_BUILDER(Name("SparseSparseBatchMatMul").Device(DEVICE_CPU), SparseSparseBatchMatMulOp);
