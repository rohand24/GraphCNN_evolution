#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

#include <iostream>

using namespace std;

//So this just manages the multiplication of the A tensor by V, weights are handled elsewhere
REGISTER_OP("SparseGraphPooling")
    .Input("A_indices: int32")
    .Input("A_values: float32")
    .Input("P: float32")
    .Output("A_index_out: int32")
    .Output("A_values_out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ShapeHandle v_shape;
      ShapeHandle idx_shape;
      ShapeHandle values_shape;

      // Validate shapes
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &v_shape)); // V = BxNxF
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &idx_shape)); // Bx#Edgesx#of Dimensions
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &values_shape)); // BxValues

      // Batch dims match between inputs.
      ShapeHandle v_batch_dims;
      ShapeHandle a_batch_dims;
      ShapeHandle batch_dims;

      int numFilters;

      c->GetAttr("Num_filters",&numFilters)
      TF_RETURN_IF_ERROR(c->Subshape(idx_shape, 0, 0, &a_batch_dims));
      TF_RETURN_IF_ERROR(c->Subshape(v_shape, 0, 0, &v_batch_dims));
      TF_RETURN_IF_ERROR(c->Merge(a_batch_dims, v_batch_dims, &batch_dims));

      ShapeHandle out = c->MakeShape({c->Dim(v_shape, 0), c->Dim(v_shape, 1), numFilters});

      c->set_output(0, out);
      return Status::OK();
    });


