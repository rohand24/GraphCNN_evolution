#include "SparseSparseMatMulControl.h"
#include "SMMPUtils.h"

void SparseSparseMatMulControl::ConvertTensorToSlices(const Tensor& A_indices,
                                          const Tensor& A_values,
                                          const size_t batchSize,
                                          const size_t layerCount,
                                          std::vector<std::vector<Tensor>> & A_index_slices,
                                          std::vector<std::vector<Tensor>> & A_value_slices,
                                          OpKernelContext * context)
{
    auto a_index_tensor = A_indices.matrix<int64>();
    auto a_values_tensor = A_values.flat<float>();
    std::vector<std::vector<size_t>> countIndices(batchSize,std::vector<size_t>(layerCount,0));

    //Loop through all edges, counting the beginning and end of every individual unit.
    for (size_t i = 0; i < A_values.dim_size(0); i++)
    {
        size_t currentBatch = a_index_tensor(i,SparseSparseMatMulControl::BATCH_INDEX);
        size_t currentLayer = a_index_tensor(i,SparseSparseMatMulControl::LAYER_INDEX);
        countIndices[currentBatch][currentLayer]++;
    }

    //Allocate all the required memory
    for (size_t currentBatch = 0; currentBatch < batchSize; currentBatch++)
    {
        for (size_t currentLayer = 0; currentLayer < layerCount; currentLayer++)
        {
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64,
                                   TensorShape({static_cast<int64>(countIndices[currentBatch][currentLayer]),
                                   static_cast<int64>(2)}),
                                   &(A_index_slices[currentBatch][currentLayer])));
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT,
                                   TensorShape({static_cast<int64>(countIndices[currentBatch][currentLayer])}),
                                   &(A_value_slices[currentBatch][currentLayer])));
            //reset the counter
            countIndices[currentBatch][currentLayer] = 0;
        }
    }
    //Populate the new slices, using the counter as a pointer into the arrays
    for (size_t i = 0; i < A_values.dim_size(0); i++)
    {
        size_t currentBatch = a_index_tensor(i,SparseSparseMatMulControl::BATCH_INDEX);
        size_t currentLayer = a_index_tensor(i,SparseSparseMatMulControl::LAYER_INDEX);
        auto currentIndexSlice = A_index_slices[currentBatch][currentLayer].matrix<int64>();
        auto currentValueSlice = A_value_slices[currentBatch][currentLayer].flat<float>();

        size_t currentSliceIndexPtr = countIndices[currentBatch][currentLayer];
        currentIndexSlice(currentSliceIndexPtr,0) = a_index_tensor(i,SparseSparseMatMulControl::N1_INDEX);
        currentIndexSlice(currentSliceIndexPtr,1) = a_index_tensor(i,SparseSparseMatMulControl::N2_INDEX);

        currentValueSlice(currentSliceIndexPtr) = a_values_tensor(i);

        countIndices[currentBatch][currentLayer]++;
    }

}

void SparseSparseMatMulControl::ConvertInputsToCSR(const Tensor& A_indices,
                                       const Tensor& A_shape,
                                       Tensor * Ai,
                                       Tensor * Aj,
                                       OpKernelContext * context
                                       )
{
    auto a_shape_tensor = A_shape.tensor<int64, 1>();

    //Allocate CSR Format vectors
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({static_cast<int64>(a_shape_tensor(SparseSparseMatMulControl::N1_INDEX) + 1)}), Ai));
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({static_cast<int64>(A_indices.dim_size(0))}), Aj));

    //Convert input to CSR Format
    SMMPUtils::SparseToCSRSparse(A_indices, Ai, Aj);
}

void SparseSparseMatMulControl::MatMul2D(const Tensor& Ai_tensor,
                      const Tensor& Aj_tensor,
                      const Tensor& A_values,
                      const Tensor& A_shape,
                      const Tensor& Bi_tensor,
                      const Tensor& Bj_tensor,
                      const Tensor& B_values,
                      const Tensor& B_shape,
                      Tensor * Ci_tensor,
                      Tensor * Cj_tensor,
                      Tensor * C_values,
                      OpKernelContext * context
                      )
{
    //Make tensor objects indexable
    auto a_values_tensor = A_values.tensor<float, 1>();
    auto a_shape_tensor = A_shape.tensor<int64, 1>();
    auto b_values_tensor = B_values.tensor<float, 1>();
    auto b_shape_tensor = B_shape.tensor<int64, 1>();

    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({a_shape_tensor(SparseSparseMatMulControl::N1_INDEX)+1}), Ci_tensor));
    SMMPUtils::CsrNonzeroRows(a_shape_tensor(SparseSparseMatMulControl::N1_INDEX), b_shape_tensor(SparseSparseMatMulControl::N2_INDEX),Ai_tensor,Aj_tensor,Bi_tensor,Bj_tensor,Ci_tensor);

    //Computed number of edges
    int64 num_nonzero = Ci_tensor->vec<int64>()(a_shape_tensor(SparseSparseMatMulControl::N1_INDEX));

    //Allocate output vals
    TensorShape c_values_shape;
    c_values_shape.AddDim(num_nonzero);
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, c_values_shape,
                                                     C_values));

    //Number of edges is equal to the final entry of Ci, which is an accumulated total of edges
    OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({num_nonzero}), Cj_tensor));

    //Compute the Matrix Multiplication
    SMMPUtils::CsrMatmul(a_shape_tensor(SparseSparseMatMulControl::N1_INDEX), b_shape_tensor(SparseSparseMatMulControl::N2_INDEX),Ai_tensor,Aj_tensor,A_values,Bi_tensor,Bj_tensor,
               B_values,Ci_tensor,Cj_tensor,C_values);

    TensorShape c_indices_shape;
    c_indices_shape.AddDim(num_nonzero);
    //magic numbers noooo
    c_indices_shape.AddDim(SparseSparseMatMulControl::SUBRANK);
    //OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, c_indices_shape,
    //                                                 C_indices));

    //Turn i and j vectors back into indices
    //SMMPUtils::CSRSparseToSparse(Ci_tensor, Cj_tensor, C_indices);
}

void SparseSparseMatMulControl::AggregateForward(const std::vector<std::vector<Tensor>> & C_i_collection,
               const std::vector<std::vector<Tensor>> & C_j_collection,
               const std::vector<std::vector<Tensor>> & C_values_collection,
               const size_t batchSize,
               const size_t layerCount,
               const size_t num_rows,
               const size_t num_cols,
               Tensor * C_indices,
               Tensor * C_values,
               Tensor * C_shape,
               const size_t c_indices_out_idx,
               const size_t c_values_out_idx,
               const size_t c_shape_out_idx,
               OpKernelContext * context)
{
    std::vector<std::vector<Tensor>> C_index_collection(batchSize,std::vector<Tensor>(layerCount,Tensor()));
    int64 totalElements = 0;
    //Figure out how much memory to allocate
    for (size_t currentBatch = 0; currentBatch < batchSize; currentBatch++)
    {
        for (size_t currentLayer = 0; currentLayer < layerCount; currentLayer++)
        {
            totalElements += C_values_collection[currentBatch][currentLayer].dim_size(0);
        }
    }

    //Allocate the memory
    OP_REQUIRES_OK(context, context->allocate_output(c_indices_out_idx, TensorShape({static_cast<int64>(totalElements),static_cast<int64>(SparseSparseMatMulControl::RANK)}),
                                                     &C_indices));
    OP_REQUIRES_OK(context, context->allocate_output(c_values_out_idx, TensorShape({static_cast<int64>(totalElements)}),
                                                     &C_values));

    auto c_indices_tensor = C_indices->matrix<int64>();
    auto c_values_tensor = C_values->flat<float>();

    int64 outputPtr = 0;

    for (size_t currentBatch = 0; currentBatch < batchSize; currentBatch++)
    {
        for (size_t currentLayer = 0; currentLayer < layerCount; currentLayer++)
        {
            //Values tensor
            auto currentValues = C_values_collection[currentBatch][currentLayer].flat<float>();

            //Make indices tensor from Ci and Cj
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT64, TensorShape({static_cast<int64>
            (C_j_collection[currentBatch][currentLayer].dim_size(0)),static_cast<int64>(SparseSparseMatMulControl::RANK)}), &(C_index_collection[currentBatch][currentLayer])));
            SMMPUtils::CSRSparseToSparse(C_i_collection[currentBatch][currentLayer],
                                         C_j_collection[currentBatch][currentLayer],
                                         &(C_index_collection[currentBatch][currentLayer]));

            auto currentIndices = C_index_collection[currentBatch][currentLayer].matrix<int64>();

            for (size_t inputPtr = 0; inputPtr < C_values_collection[currentBatch][currentLayer].dim_size(0); inputPtr++)
            {
                c_indices_tensor(outputPtr,0) = currentBatch;
                c_indices_tensor(outputPtr,1) = currentLayer;
                c_indices_tensor(outputPtr,2) = currentIndices(inputPtr,0);
                c_indices_tensor(outputPtr,3) = currentIndices(inputPtr,1);
                c_values_tensor(outputPtr) = currentValues(inputPtr);
                outputPtr++;
            }
        }
    }

    //Create an output shape tensor, we know that upfront
    TensorShape c_shape_shape;
    c_shape_shape.AddDim(SparseSparseMatMulControl::RANK);
    OP_REQUIRES_OK(context, context->allocate_output(c_shape_out_idx, c_shape_shape,
                                                     &C_shape));

    auto c_shape_vals_flat = C_shape->flat<int64>();

    c_shape_vals_flat(0) = batchSize;
    c_shape_vals_flat(1) = layerCount;
    c_shape_vals_flat(2) = num_rows;
    c_shape_vals_flat(3) = num_cols;

}

void SparseSparseMatMulControl::AggregateBackward(const std::vector<std::vector<Tensor>> & C_i_collection,
               const std::vector<std::vector<Tensor>> & C_j_collection,
               const std::vector<std::vector<Tensor>> & C_values_collection,
               const size_t batchSize,
               const size_t layerCount,
               const size_t num_rows,
               const size_t num_cols,
               Tensor * C_indices,
               Tensor * C_values,
               Tensor * C_shape,
               const size_t c_indices_out_idx,
               const size_t c_values_out_idx,
               const size_t c_shape_out_idx,
               OpKernelContext * context)
{
    std::vector<std::vector<Tensor>> C_index_collection(batchSize,std::vector<Tensor>(layerCount,Tensor()));
    int64 totalElements = 0;
    //Figure out how much memory to allocate
    for (size_t currentBatch = 0; currentBatch < batchSize; currentBatch++)
    {
        for (size_t currentLayer = 0; currentLayer < layerCount; currentLayer++)
        {
            totalElements += C_values_collection[currentBatch][currentLayer].dim_size(0);
        }
    }

    //Allocate the memory
    OP_REQUIRES_OK(context, context->allocate_output(c_indices_out_idx, TensorShape({static_cast<int64>(totalElements),static_cast<int64>(SparseSparseMatMulControl::RANK)}),
                                                     &C_indices));
    OP_REQUIRES_OK(context, context->allocate_output(c_values_out_idx, TensorShape({static_cast<int64>(totalElements)}),
                                                     &C_values));

    auto c_indices_tensor = C_indices->matrix<int64>();
    auto c_values_tensor = C_values->flat<float>();

    int64 outputPtr = 0;

    for (size_t currentBatch = 0; currentBatch < batchSize; currentBatch++)
    {
        for (size_t currentLayer = 0; currentLayer < layerCount; currentLayer++)
        {
            //Values tensor
            auto currentValues = C_values_collection[currentBatch][currentLayer].flat<float>();

            for (size_t inputPtr = 0; inputPtr < C_values_collection[currentBatch][currentLayer].dim_size(0); inputPtr++)
            {
                c_indices_tensor(outputPtr,0) = 0;
                c_indices_tensor(outputPtr,1) = 0;
                c_indices_tensor(outputPtr,2) = 0;
                c_indices_tensor(outputPtr,3) = 0;
                c_values_tensor(outputPtr) = currentValues(inputPtr);
                outputPtr++;
            }
        }
    }

    //Create an output shape tensor, we know that upfront
    TensorShape c_shape_shape;
    c_shape_shape.AddDim(SparseSparseMatMulControl::RANK);
    OP_REQUIRES_OK(context, context->allocate_output(c_shape_out_idx, c_shape_shape,
                                                     &C_shape));

    auto c_shape_vals_flat = C_shape->flat<int64>();

    c_shape_vals_flat(0) = 0;
    c_shape_vals_flat(1) = 0;
    c_shape_vals_flat(2) = 0;
    c_shape_vals_flat(3) = 0;

}

void SparseSparseMatMulControl::GetValueSubset(const Tensor& values,
                                   const size_t startIndex,
                                   const size_t numElements,
                                   Tensor * subset,
                                   OpKernelContext * context)
{
    //std::cout << "SI " << startIndex << "NE " << numElements << std::endl;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({static_cast<int64>
            (numElements)}), subset));

    //std::cout << "ALLOCATED MEM" << std::endl;

    auto values_tensor = values.flat<float>();
    auto subset_tensor = subset->flat<float>();

    for (size_t i = 0; i < numElements; i++)
    {
        //std::cout << i << std::endl;
        subset_tensor(i) = values_tensor(startIndex + i);
    }
}