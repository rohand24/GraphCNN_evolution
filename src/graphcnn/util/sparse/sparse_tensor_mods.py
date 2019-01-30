from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tensor_shape

def get_shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.

    Returns:
      A `TensorShape` object.
    """
    if self.static_shape is not None:
        return self.static_shape
    else:
        return tensor_util.constant_value_as_shape(self._dense_shape)


def set_shape(self, shape):
    """Updates the shape of this tensor.
    """

    # Update C shape even if _USE_C_SHAPES = False, since we still want
    # set_shape to be reflected in the C API graph for when we run it.
    if not isinstance(shape, tensor_shape.TensorShape):
        shape = tensor_shape.TensorShape(shape)
    self.static_shape = shape