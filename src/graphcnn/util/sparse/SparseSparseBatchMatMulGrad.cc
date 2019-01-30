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
REGISTER_OP("SparseSparseBatchMatMulGrad")
    .Input("grad_values: float32")
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
    .Output("grad_a_indices: int64")
    .Output("grad_a_values: float32")
    .Output("grad_a_shape: int64")
    .Output("grad_b_indices: int64")
    .Output("grad_b_values: float32")
    .Output("grad_b_shape: int64")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ShapeHandle grad_values_shape;
      ShapeHandle a_idx_shape;
      ShapeHandle a_values_shape;
      ShapeHandle a_shape_shape;
      ShapeHandle b_idx_shape;
      ShapeHandle b_values_shape;
      ShapeHandle b_shape_shape;

      // Validate shapes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &grad_values_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &a_idx_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &a_values_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &a_shape_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &b_idx_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 1, &b_values_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 1, &b_shape_shape));

      ShapeHandle grad_a_indices = c->MakeShape({c->Dim(a_idx_shape, 0), c->Dim(a_idx_shape, 1)});
      ShapeHandle grad_a_values = c->MakeShape({c->Dim(a_values_shape, 0)});
      ShapeHandle grad_a_shape = c->MakeShape({c->Dim(a_shape_shape, 0)});
      ShapeHandle grad_b_indices = c->MakeShape({c->Dim(b_idx_shape, 0), c->Dim(b_idx_shape, 1)});
      ShapeHandle grad_b_values = c->MakeShape({c->Dim(b_values_shape, 0)});
      ShapeHandle grad_b_shape = c->MakeShape({c->Dim(b_shape_shape, 0)});

      c->set_output(0, grad_a_indices);
      c->set_output(1, grad_a_values);
      c->set_output(2, grad_a_shape);
      c->set_output(3, grad_b_indices);
      c->set_output(4, grad_b_values);
      c->set_output(5, grad_b_shape);

      return Status::OK();
    });


class SparseSparseBatchMatMulGradOp : public OpKernel{
    public:
        explicit SparseSparseBatchMatMulGradOp(OpKernelConstruction* context): OpKernel(context) {}

        void Compute(OpKernelContext* context) override {
            //Pass inputs into Tensor objects
            const Tensor& G_values = context->input(0);
            const Tensor& A_indices = context->input(1);
            const Tensor& A_values = context->input(2);
            const Tensor& A_shape = context->input(3);
            const Tensor& B_indices = context->input(4);
            const Tensor& B_values = context->input(5);
            const Tensor& B_shape = context->input(6);

            //Make tensor objects indexable
            auto g_values_tensor = G_values.tensor<float, 1>();
            auto a_indices_tensor = A_indices.tensor<int64, 2>();
            auto a_values_tensor = A_values.tensor<float, 1>();
            auto a_shape_tensor = A_shape.tensor<int64, 1>();
            auto b_indices_tensor = B_indices.tensor<int64, 2>();
            auto b_values_tensor = B_values.tensor<float, 1>();
            auto b_shape_tensor = B_shape.tensor<int64, 1>();


            Tensor* Grad_A_indices = NULL;
            Tensor* Grad_A_values = NULL;
            Tensor* Grad_A_shape = NULL;
            Tensor* Grad_B_indices = NULL;
            Tensor* Grad_B_values = NULL;
            Tensor* Grad_B_shape = NULL;

            const size_t grad_a_indices_out_idx = 0;
            const size_t grad_a_values_out_idx = 1;
            const size_t grad_a_shape_out_idx = 2;
            const size_t grad_b_indices_out_idx = 3;
            const size_t grad_b_values_out_idx = 4;
            const size_t grad_b_shape_out_idx = 5;



            //Transposed matrices
            Tensor GTi_tensor;
            Tensor GTj_tensor;
            Tensor ATi_tensor;
            Tensor ATj_tensor;
            Tensor GT_values;
            Tensor AT_values;

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

            std::vector<std::vector<Tensor>> A_is(batchSize,std::vector<Tensor>(layerCountA,Tensor()));
            std::vector<std::vector<Tensor>> A_js(batchSize,std::vector<Tensor>(layerCountA,Tensor()));
            std::vector<std::vector<Tensor>> B_is(batchSize,std::vector<Tensor>(layerCountB,Tensor()));
            std::vector<std::vector<Tensor>> B_js(batchSize,std::vector<Tensor>(layerCountB,Tensor()));
            std::vector<std::vector<Tensor>> G_is(batchSize,std::vector<Tensor>(layerCount,Tensor()));
            std::vector<std::vector<Tensor>> G_js(batchSize,std::vector<Tensor>(layerCount,Tensor()));
            std::vector<std::vector<Tensor>> C_values(batchSize,std::vector<Tensor>(layerCount,Tensor()));
            std::vector<std::vector<Tensor>> dA_values(batchSize,std::vector<Tensor>(layerCountA,Tensor()));
            std::vector<std::vector<Tensor>> dB_values(batchSize,std::vector<Tensor>(layerCountB,Tensor()));

            //Manage grabbing subsets of G_values
            size_t G_values_ptr = 0;
            Tensor G_subset;

            for (size_t currentBatch = 0; currentBatch < batchSize; currentBatch++)
            {
                size_t currentLayerA = 0;
                size_t currentLayerB = 0;

                //Should only reset output on the first try when a flat matrix is multiplied by a stack
                bool resetOutputA = true;
                bool resetOutputB = true;
                for (size_t currentLayer = 0; currentLayer < layerCount; currentLayer++)
                {
                   // std::cout << "CURRENT BATCH: " << currentBatch << "LAYER: " << currentLayer << "LAYERA: " << currentLayerA << "LAYERB: " << currentLayerB << std::endl;
                    if (currentLayerA == currentLayer)
                    {
                        SparseSparseMatMulControl::ConvertInputsToCSR(A_index_slices[currentBatch][currentLayerA],
                                                  A_shape,
                       &(A_is[currentBatch][currentLayer]),
                       &(A_js[currentBatch][currentLayer]),
                       context);
                    }
                    if (currentLayerB == currentLayer)
                    {
                        //std::cout << "LUL0" << std::endl;
                        SparseSparseMatMulControl::ConvertInputsToCSR(B_index_slices[currentBatch][currentLayerB],
                                                  B_shape,
                       &(B_is[currentBatch][currentLayer]),
                       &(B_js[currentBatch][currentLayer]),
                       context);
                    }

                    //std::cout << "LUL1" << std::endl;
                    SparseSparseMatMulControl::MatMul2D(A_is[currentBatch][currentLayerA],
                             A_js[currentBatch][currentLayerA],
                             A_value_slices[currentBatch][currentLayerA],
                             A_shape,
                             B_is[currentBatch][currentLayerB],
                             B_js[currentBatch][currentLayerB],
                             B_value_slices[currentBatch][currentLayerB],
                             B_shape,
                             &(G_is[currentBatch][currentLayer]),
                             &(G_js[currentBatch][currentLayer]),
                             &(C_values[currentBatch][currentLayer]),
                             context);
                    //std::cout << "LUL2" << std::endl;
                    //Sort Gj because we can assume G_Values is sorted by index
                    SMMPUtils::SortIndices(G_is[currentBatch][currentLayer], &G_js[currentBatch][currentLayer]);
                    //Compute dA
                    if (currentLayerA == currentLayer)
                    {
                        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                                       TensorShape({static_cast<int64>(A_value_slices[currentBatch][currentLayerA].dim_size(0))}),
                                       &(dA_values[currentBatch][currentLayer])));
                    }


                    //std::cout << batchSize << " " << layerCount << std::endl;

                    SparseSparseMatMulControl::GetValueSubset(G_values,
                                   G_values_ptr,
                                   G_js[currentBatch][currentLayer].dim_size(0),
                                   &G_subset,
                                   context);
                    //std::cout << "LUL3" << std::endl;
                    G_values_ptr += G_js[currentBatch][currentLayer].dim_size(0);

                    //std::cout << "dA Compute" << std::endl;

                    SMMPUtils::CsrMatmulGradA(a_shape_tensor(SparseSparseMatMulControl::N1_INDEX),
                                              a_shape_tensor(SparseSparseMatMulControl::N2_INDEX),
                                              G_is[currentBatch][currentLayer],
                                              G_js[currentBatch][currentLayer],
                                              G_subset,
                                              B_is[currentBatch][currentLayerB],
                                              B_js[currentBatch][currentLayerB],
                                              B_value_slices[currentBatch][currentLayerB],
                                              A_is[currentBatch][currentLayerA],
                                              A_js[currentBatch][currentLayerA],
                                              resetOutputA,
                                              &(dA_values[currentBatch][currentLayerA]));
                    //std::cout << "LUL4" << std::endl;

                     //std::cout << "LUL2" << std::endl;

                    //Compute transposes to reuse grad function
                    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({static_cast<int64>(b_shape_tensor(SparseSparseMatMulControl::N2_INDEX) + 1)}), &GTi_tensor));
                    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({static_cast<int64>(G_js[currentBatch][currentLayer].flat<int64>().size())}), &GTj_tensor));

                    if(currentLayerA == currentLayer)
                    {
                        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({static_cast<int64>(a_shape_tensor(SparseSparseMatMulControl::N2_INDEX) + 1)}), &ATi_tensor));
                        OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({static_cast<int64>(A_index_slices[currentBatch][currentLayer].dim_size(0))}), &ATj_tensor));
                        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({static_cast<int64>(A_value_slices[currentBatch][currentLayer].dim_size(0))}), &AT_values));
                    }

                    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({static_cast<int64>(G_values.dim_size(0))}), &GT_values));

                    //Transpose G and A Matrices
                    SMMPUtils::CsrTranspose(a_shape_tensor(SparseSparseMatMulControl::N1_INDEX),
                                            b_shape_tensor(SparseSparseMatMulControl::N2_INDEX),
                                            G_is[currentBatch][currentLayer],
                                            G_js[currentBatch][currentLayer],
                                            G_subset,
                                            &GTi_tensor,
                                            &GTj_tensor,
                                            &GT_values);
                    //std::cout << "LUL5" << std::endl;
                    //std::cout << "LUL3" << std::endl;
                    if (currentLayerA == currentLayer)
                    {
                        SMMPUtils::CsrTranspose(a_shape_tensor(SparseSparseMatMulControl::N1_INDEX),
                            a_shape_tensor(SparseSparseMatMulControl::N2_INDEX),
                            A_is[currentBatch][currentLayerA],
                            A_js[currentBatch][currentLayerA],
                            A_value_slices[currentBatch][currentLayerA],
                            &ATi_tensor,
                            &ATj_tensor,
                            &AT_values);
                    }

                    //std::cout << "LUL6" << std::endl;

                    //std::cout << "LUL4" << std::endl;
                    if (currentLayerB == currentLayer)
                    {
                        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                           TensorShape({static_cast<int64>(B_value_slices[currentBatch][currentLayerB].dim_size(0))}),
                           &(dB_values[currentBatch][currentLayerB])));
                    }


                   // std::cout << "dB Compute" << std::endl;

                    //Compute dB
                    SMMPUtils::CsrMatmulGradA(b_shape_tensor(SparseSparseMatMulControl::N1_INDEX),
                                              b_shape_tensor(SparseSparseMatMulControl::N2_INDEX),
                                              ATi_tensor,
                                              ATj_tensor,
                                              AT_values,
                                              GTi_tensor,
                                              GTj_tensor,
                                              GT_values,
                                              B_is[currentBatch][currentLayerB],
                                              B_js[currentBatch][currentLayerB],
                                              resetOutputB,
                                              &(dB_values[currentBatch][currentLayerB]));

                    //std::cout << "LUL7" << std::endl;
                    switch(multiplyMode)
                    {
                        case STACK_BOTH_SIDES:
                            currentLayerA++;
                            currentLayerB++;
                            break;
                        case LEFT_MULTIPLY_STACK:
                            currentLayerB++;
                            resetOutputA = false;
                            break;
                        case RIGHT_MULTIPLY_STACK:
                            currentLayerA++;
                            resetOutputB = false;
                            break;
                    }
                }
            }

            SparseSparseMatMulControl::AggregateBackward(A_is,
                      A_js,
                      dA_values,
                      batchSize,
                      layerCountA,
                      a_shape_tensor(SparseSparseMatMulControl::N1_INDEX),
                      a_shape_tensor(SparseSparseMatMulControl::N2_INDEX),
                      Grad_A_indices,
                      Grad_A_values,
                      Grad_A_shape,
                      grad_a_indices_out_idx,
                      grad_a_values_out_idx,
                      grad_a_shape_out_idx,
                      context);
            //std::cout << "LUL8" << std::endl;

            SparseSparseMatMulControl::AggregateBackward(B_is,
                      B_js,
                      dB_values,
                      batchSize,
                      layerCountB,
                      b_shape_tensor(SparseSparseMatMulControl::N1_INDEX),
                      b_shape_tensor(SparseSparseMatMulControl::N2_INDEX),
                      Grad_B_indices,
                      Grad_B_values,
                      Grad_B_shape,
                      grad_b_indices_out_idx,
                      grad_b_values_out_idx,
                      grad_b_shape_out_idx,
                      context);

           // std::cout << "LUL9" << std::endl;

            //We don't need to do any reverse conversion because the grads of the indices are all zero
        }
    private:

};

REGISTER_KERNEL_BUILDER(Name("SparseSparseBatchMatMulGrad").Device(DEVICE_CPU), SparseSparseBatchMatMulGradOp);
