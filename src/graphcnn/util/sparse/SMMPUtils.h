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

class SMMPUtils
{
    public:
        static void SparseToCSRSparse(const Tensor& indices, Tensor* Ai, Tensor* Aj);
        static void CSRSparseToSparse(const Tensor& Ai, const Tensor& Aj, Tensor * indices);
        static void CsrNonzeroRows(const int64 n_row, const int64 n_col,
                    const Tensor& Ap_t,
                    const Tensor& Aj_t,
                    const Tensor& Bp_t,
                    const Tensor& Bj_t,
                    Tensor *Cp_t);
        static void CsrMatmul(const int64 n_row, const int64 n_cols,
               const Tensor& Ap_t,
               const Tensor& Aj_t,
               const Tensor& Ax_t,
               const Tensor& Bp_t,
               const Tensor& Bj_t,
               const Tensor& Bx_t,
               Tensor* Cp_t,
               Tensor* Cj_t,
               Tensor* Cx_t);
        static void CsrMatmulGradA(const int64 n_row, const int64 n_cols,
               const Tensor& Gp_t,
               const Tensor& Gj_t,
               const Tensor& Gx_t,
               const Tensor& Bp_t,
               const Tensor& Bj_t,
               const Tensor& Bx_t,
               const Tensor& dAp_t,
               const Tensor& dAj_t,
               const bool resetOutput,
               Tensor* dAx_t);
        static void CsrTranspose(const int64 n_row, const int64 n_cols,
               const Tensor& Ap_t,
               const Tensor& Aj_t,
               const Tensor& Ax_t,
               Tensor* Bp_t,
               Tensor* Bj_t,
               Tensor* Bx_t);
        static void SortIndices(const Tensor& Ai, Tensor* Aj);
    private:
        //Disallow instances
        SMMPUtils() {}
        static bool sortByVal(const std::pair<int64,int64>& lhs, const std::pair<int64, int64>& rhs);
};