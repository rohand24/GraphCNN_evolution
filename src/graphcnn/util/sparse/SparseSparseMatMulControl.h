#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include <stdint.h>
#include <vector>
#include <utility>

using namespace tensorflow;

class SparseSparseMatMulControl
{
    public:
        static const size_t BATCH_INDEX = 0;
        static const size_t LAYER_INDEX = 1;
        static const size_t N1_INDEX = 2;
        static const size_t N2_INDEX = 3;
        static const size_t SINGLETON = 1;

        static void ConvertTensorToSlices(const Tensor& A_indices,
                                          const Tensor& A_values,
                                          const size_t batchSize,
                                          const size_t layerCount,
                                          std::vector<std::vector<Tensor>> & A_index_slices,
                                          std::vector<std::vector<Tensor>> & A_value_slices,
                                          OpKernelContext * context);
        static void ConvertInputsToCSR(const Tensor& A_indices,
                                       const Tensor& A_shape,
                                       Tensor * Ai,
                                       Tensor * Aj,
                                       OpKernelContext * context);
        static void MatMul2D(const Tensor& Ai_tensor,
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
                      );
        static void AggregateForward(const std::vector<std::vector<Tensor>> & C_i_collection,
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
                       OpKernelContext * context);
        static void AggregateBackward(const std::vector<std::vector<Tensor>> & C_i_collection,
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
                       OpKernelContext * context);

        static void GetValueSubset(const Tensor& values,
                                   const size_t startIndex,
                                   const size_t numElements,
                                   Tensor * subset,
                                   OpKernelContext * context);
    private:
        //Disallow instances
        const static size_t RANK = 4;
        const static size_t SUBRANK = 2;
        SparseSparseMatMulControl() {}
        static bool sortByVal(const std::pair<int64,int64>& lhs, const std::pair<int64, int64>& rhs);
};