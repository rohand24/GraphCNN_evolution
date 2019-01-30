from graphcnn.helper import *
import tensorflow as tf
import numpy as np
import math
import os
import os.path
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import scipy.sparse
import pdb
tf.NotDifferentiable('Unique')
here = os.path.dirname(__file__) + '/util/sparse/'
if os.path.isfile(os.path.join(here, 'SparseConv.so')):
    _graphcnn_conv_sparse_module = tf.load_op_library(os.path.join(here, 'SparseConv.so'))
    _graphcnn_conv_sparse_grad_module = tf.load_op_library(os.path.join(here, 'SparseConvGrad.so'))
    
if os.path.isfile(os.path.join(here, 'SparseAverageVertexPool.so')):
    _graphcnn_avg_vertex_pool_sparse_module = tf.load_op_library(os.path.join(here, 'SparseAverageVertexPool.so'))
    _graphcnn_avg_vertex_pool_sparse_grad_module = tf.load_op_library(os.path.join(here, 'SparseAverageVertexPoolGrad.so'))

if os.path.isfile(os.path.join(here, 'SparseMaxVertexPool.so')):
    _graphcnn_max_vertex_pool_sparse_module = tf.load_op_library(os.path.join(here, 'SparseMaxVertexPool.so'))
    _graphcnn_max_vertex_pool_sparse_grad_module = tf.load_op_library(os.path.join(here, 'SparseMaxVertexPoolGrad.so'))

if os.path.isfile(os.path.join(here, 'SparseSparseBatchMatMul.so')):
    _graphcnn_sparse_matmul_sparse_module = tf.load_op_library(os.path.join(here, 'SparseSparseBatchMatMul.so'))
    _graphcnn_sparse_matmul_sparse_grad_module = tf.load_op_library(os.path.join(here, 'SparseSparseBatchMatMulGrad.so'))

if os.path.isfile(os.path.join(here, 'SparseMlp.so')):
    _graphcnn_sparse_mlp_sparse_module = tf.load_op_library(os.path.join(here, 'SparseMlp.so'))
    _graphcnn_sparse_mlp_sparse_grad_module = tf.load_op_library(os.path.join(here, 'SparseMlpGrad.so'))

if os.path.isfile(os.path.join(here, 'SparseEdgeConv.so')):
    _graphcnn_sparse_edge_conv_module = tf.load_op_library(
        os.path.join(here, 'SparseEdgeConv.so'))
    _graphcnn_sparse_edge_conv_grad_module = tf.load_op_library(os.path.join(here, 'SparseEdgeConvGrad.so'))

def _histogram_summaries(var, name=None):
    if name is None:
        return tf.summary.histogram(var.name, var)
    else:
        return tf.summary.histogram(name, var)
    
def make_variable(name, shape, initializer=tf.truncated_normal_initializer(), regularizer=None):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, dtype=dtype)
    _histogram_summaries(var)
    return var
    
def make_bias_variable(name, shape):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1), dtype=dtype)
    _histogram_summaries(var)
    return var

def make_variable_with_weight_decay(name, shape, stddev=0.01, wd=0.005):
    dtype = tf.float32
    regularizer = None
    if wd is not None and wd > 1e-7:
        def regularizer(var):
            return tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    var = make_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev), regularizer=regularizer)
    return var
    
def make_bn(input, phase, axis=-1, epsilon=0.001, mask=None, num_updates=None, name=None):
    default_decay = GraphCNNGlobal.BN_DECAY
    with tf.variable_scope(name, default_name='BatchNorm') as scope:
        input_size = input.get_shape()[axis].value
        if axis == -1:
            axis = len(input.get_shape())-1
        axis_arr = [i for i in range(len(input.get_shape())) if i != axis]
        if mask == None:
            batch_mean, batch_var = tf.nn.moments(input, axis_arr)
        else:
            batch_mean, batch_var = tf.nn.weighted_moments(input, axis_arr, mask)
        gamma = make_variable('gamma', input_size, initializer=tf.constant_initializer(1))
        beta = make_bias_variable('bias', input_size)
        ema = tf.train.ExponentialMovingAverage(decay=default_decay, num_updates=num_updates)
        
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        batch_norm = tf.nn.batch_normalization(input, mean, var, beta, gamma, 1e-3)
        _histogram_summaries(batch_norm, 'batch_norm')
        return batch_norm

      
def batch_mat_mult(A, B):
    A_shape = tf.shape(A)
    A_reshape = tf.reshape(A, [-1, A_shape[-1]])
    
    # So the Tensor has known dimensions
    if B.get_shape()[1] == None:
        axis_2 = -1
    else:
        axis_2 = B.get_shape()[1]
    result = tf.matmul(A_reshape, B)
    result = tf.reshape(result, tf.stack([A_shape[0], A_shape[1], axis_2]))
    return result

#def sparse_sparse_batch_mat_mul(P,A):
    #Prep = tf.tile(tf.expand_dims(P, 2), [1, 1, Ashape[2], 1])
    #Ptranspose = tf.transpose(Prep, perm=[0, 2, 3, 1])
    #Abatched = tf.transpose(A, perm=[0, 2, 1, 3])
    #leftMultiply = tf.matmul(Ptranspose, Abatched)
    #rightMultiply = tf.matmul(leftMultiply, Pnottranspose)
    #Aout = tf.transpose(rightMultiply, perm=[0, 2, 1, 3])
    #cIndices, cVals, cShape = _graphcnn_sparse_matmul_sparse_module.sparse_sparse_batch_mat_mul(P.indices,P.values,P.dense_shape,A#.indices,A.values,A.dense_shape)
    #Preindex,Areindex = _graphcnn_sparse_matmul_sparse_module.sparse_sparse_batch_mat_mul(P.indices, P.values,
    #                                                                                            P.dense_shape,
    #                                                                                            A.indices, A.values,
    #                                                                                            A.dense_shape)
    #SMMP above returns indices out of order, need to sort because Tensorflow assumes sorted
#    return tf.SparseTensor(cIndices, cVals, cShape)

#@ops.RegisterGradient("SparseSparseBatchMatMul")
#def _graphcnn_sparse_sparse_batch_mat_mul_grad_func(op, unusedGradIdx, gradVals, unusedGradShape):
#    return _graphcnn_sparse_matmul_sparse_grad_module.sparse_sparse_batch_mat_mul_grad(gradVals,
#                                                                                  op.inputs[0],
#                                                                                  op.inputs[1],
#                                                                                  op.inputs[2],
#                                                                                  op.inputs[3],
#                                                                                  op.inputs[4],
#                                                                                  op.inputs[5])

def make_softmax_layer(V, axis=1, name=None):
    with tf.variable_scope(name, default_name='Softmax') as scope:
        max_value = tf.reduce_max(V, axis=axis, keep_dims=True)
        exp = tf.exp(tf.subtract(V, max_value))
        prob = tf.div(exp, tf.reduce_sum(exp, axis=axis, keep_dims=True))
        _histogram_summaries(prob)
        return prob
    
def make_graphcnn_layer(V, A, no_filters, no_A, stride=1, order=1, name=None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        weightList = []
        #no_A = A.get_shape()[2].value
        no_features = V.get_shape()[2].value
        W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable('bias', [no_filters])
        result = batch_mat_mult(V, W_I) + b
        weightList.append(W_I)
        
        Acurrent = A
        for k in range(1,order + 1):
            if k % stride == 0:
                with tf.variable_scope('Order' + str(k)) as scope:
                    W = make_variable_with_weight_decay('weights', [no_features*no_A, no_filters], stddev=math.sqrt(1.0/(no_features*(no_A+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
                    A_shape = tf.shape(Acurrent)
                    A_reshape = tf.reshape(Acurrent, tf.stack([-1, A_shape[1]*no_A, A_shape[1]]))
                    n = tf.matmul(A_reshape, V)
                    n = tf.reshape(n, [-1, A_shape[1], no_A*no_features])
                    result = batch_mat_mult(n, W)
                    weightList.append(W)
            Acurrent = tf.transpose(tf.matmul(tf.transpose(Acurrent,[0,2,1,3]), tf.transpose(A,[0,2,1,3])),[0,2,1,3])
        result.set_shape([V.get_shape()[0].value,V.get_shape()[1].value,no_filters])
        _histogram_summaries(Acurrent, "Acurrent")
        _histogram_summaries(result, "Result")
        return result, weightList

#For now stride and order are ignored
def make_sparse_graphcnn_layer(V, A, no_filters, no_A, stride=1, order=1, name=None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        weightList = []
        #print(A)
        #print(A.get_shape())
        #no_A = A.get_shape()[2].value
        no_features = V.get_shape()[2].value
        W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(
            GraphCNNGlobal.GRAPHCNN_I_FACTOR / (no_features * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable('bias', [no_filters])
        result_bias = batch_mat_mult(V, W_I) + b
        weightList.append(W_I)
        W = make_variable_with_weight_decay('weights', [no_A, no_features, no_filters], stddev=math.sqrt(
            1.0 / (no_features * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))

        weightList.append(W)

        #no_A = A.dense_shape[2]
        result = _graphcnn_conv_sparse_module.sparse_graph_convolution(V, A.indices, A.values, W)
        result += result_bias
        #result.set_shape([V.get_shape()[0].value,V.get_shape()[1].value])
        #print(result.get_shape().as_list())
        result.set_shape([V.get_shape()[0].value, V.get_shape()[1].value, no_filters])
        return result, weightList

def make_one_by_one_graphcnn_layer(V,no_filters,no_A,name=None):
    with tf.variable_scope(name, default_name='Graph-CNN') as scope:
        weightList = []
        no_features = V.get_shape()[2].value
        W_I = make_variable_with_weight_decay('weights_I', [no_features, no_filters], stddev=math.sqrt(
            GraphCNNGlobal.GRAPHCNN_I_FACTOR / (no_features * (no_A + 1) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        b = make_bias_variable('bias', [no_filters])
        result = batch_mat_mult(V, W_I) + b
        result.set_shape([V.get_shape()[0].value, V.get_shape()[1].value, no_filters])

        return result, weightList
        
def make_sparse_edge_conv_layer(V,A,num_filters,name=None):
    with tf.variable_scope(name,default_name='EdgeConv') as scope:
    
        Vshape = V.get_shape()
        Hs = make_variable_with_weight_decay('Hs', [Vshape[2].value, num_filters],stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR / (V.get_shape()[2].value * (num_filters) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        Hr = make_variable_with_weight_decay('Hr', [Vshape[2].value, num_filters],stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR / (V.get_shape()[2].value * (num_filters) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
        He = make_variable_with_weight_decay('He', [1, num_filters],stddev=math.sqrt(GraphCNNGlobal.GRAPHCNN_I_FACTOR / ((float(num_filters)) * GraphCNNGlobal.GRAPHCNN_INIT_FACTOR)))
            
        #weightList = [H]

        #AvalOut = _graphcnn_sparse_edge_conv_module.sparse_edge_convolution(V, A.indices, A.values, H)
        #AvalOut = tf.div(AvalOut,float(num_filters*V.get_shape()[2].value))
        #Aout = tf.SparseTensor(A.indices, AvalOut, A.dense_shape)
        #Hs = tf.random_uniform([V.shape[2], num_filters], dtype=tf.float32)
        #Hr = tf.random_uniform([V.shape[2], num_filters], dtype=tf.float32)
        #He = tf.random_uniform([1, num_filters], dtype=tf.float32)
        #H = tf.concat((Hs,Hr,He),axis=0)
        weightList = [Hs,Hr,He]

        VsIndices = tf.stack((A.indices[:,0],A.indices[:,1]),axis=1)
        VrIndices = tf.stack((A.indices[:,0],A.indices[:,3]),axis=1)
        #BxNxF #FxF2
        Vflat = tf.reshape(V,[-1,Vshape[2].value]) #BNxF
        VsWeighted = tf.reshape(tf.reduce_mean(tf.matmul(Vflat,Hs),axis=1),[-1,Vshape[1].value])#BNxF2=>BN=>BxN
        VrWeighted = tf.reshape(tf.reduce_mean(tf.matmul(Vflat,Hr),axis=1),[-1,Vshape[1].value])#BNxF2=>BN=>BxN
        Aweighted = tf.reduce_mean(tf.matmul(tf.expand_dims(A.values,axis=1),He),axis=1) #ExF2=>#E
        Vs = tf.gather_nd(VsWeighted,VsIndices) #E
        Vr = tf.gather_nd(VrWeighted,VrIndices) #E
        AoutVals =  Vs + Vr + Aweighted
        #AoutVals = tf.div(AoutVals,float(V.get_shape()[2].value))
        #AoutValsFlat = tf.reduce_mean(AoutVals,axis=1)
        Aout = tf.SparseTensor(A.indices,AoutVals,A.dense_shape)
    return V,Aout,weightList
    
#@ops.RegisterGradient("SparseEdgeConvolution")
#def _graphcnn_edge_conv_sparse_grad_func(op, grad): 
    #print(grad.shape)
    #print(grad.get_shape().as_list())
#    return _graphcnn_sparse_edge_conv_grad_module.sparse_edge_convolution_grad(grad, op.inputs[0], op.inputs[1],op.inputs[2], op.inputs[3])


def make_sparse_graphcnn_layer_test(V, A, W, name=None):
    no_filters = W.get_shape()[1].value
    no_features = V.get_shape()[2].value
    #For now, assume no weights. This just tests A*V
    if isinstance(A, tf.Tensor):
        no_A = A.get_shape()[2].value
        #with tf.variable_scope('Order' + str(k)) as scope:
        A_shape = tf.shape(A)
        A_reshape = tf.reshape(A, tf.stack([-1, A_shape[1]*no_A, A_shape[1]])) #BxNLxN
        n = tf.matmul(A_reshape, V) #BxNLxF1
        n = tf.reshape(n, [-1, A_shape[1], no_A*no_features]) #BxNxLF1
        result = batch_mat_mult(n, W) #BxNxLF2
        _histogram_summaries(result, "Result")
        return result, []
    elif isinstance(A, tf.SparseTensor):
        no_A = A.dense_shape[2]
        #BxNxLxN #BxNxF1
        #NLxBN #BNxF1
        #NLxF1 x F1xF2
        #NxLxF2
        #with tf.variable_scope('Order' + str(k)) as scope:
        A_shape = tf.shape(A)
        A_reshape = tf.reshape(A, tf.stack([A_shape[1]*no_A, A_shape[1]])) #BxNLxN
        n = tf.matmul(A_reshape, V) #BxNLxF1
        n = tf.reshape(n, [-1, A_shape[1], no_A*no_features]) #BxNxLF1
        result = batch_mat_mult(n, W) #BxNxLF2
        _histogram_summaries(result, "Result")
        #W3d = tf.reshape(W,tf.stack([no_A,no_features,no_filters],axis=0))
        #result = _graphcnn_conv_sparse_module.sparse_graph_convolution(V, A.indices, A.values, W3d)
        #result.set_shape([V.get_shape()[0].value,V.get_shape()[1].value])
        print(result.get_shape().as_list())
        return result, []

@ops.RegisterGradient("SparseGraphConvolution")
def _graphcnn_conv_sparse_grad_func(op, grad):
    #print(grad.shape)
    #print(grad.get_shape().as_list())
    return _graphcnn_conv_sparse_grad_module.sparse_graph_convolution_grad(grad, op.inputs[0], op.inputs[1],op.inputs[2], op.inputs[3])
    
def make_graph_embed_pooling(V, A, no_vertices=1, mask=None, name=None):
    #pdb.set_trace
    with tf.variable_scope(name, default_name='GraphEmbedPooling') as scope:
        factors, W = make_embedding_layer(V, no_vertices, name='Factors')

        if mask is not None:
            factors = tf.multiply(factors, mask)
        factors = make_softmax_layer(factors)

        result = tf.matmul(factors, V, transpose_a=True)

        if no_vertices == 1:
            no_features = V.get_shape()[2].value
            return tf.reshape(result, [-1, no_features]), A

        result_A = tf.reshape(A, (tf.shape(A)[0], -1, tf.shape(A)[-1]))
        result_A = tf.matmul(result_A, factors)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], tf.shape(A)[-1], -1))
        result_A = tf.matmul(factors, result_A, transpose_a=True)
        result_A = tf.reshape(result_A, (tf.shape(A)[0], no_vertices, A.get_shape()[2].value, no_vertices))
        _histogram_summaries(result, "result")
        _histogram_summaries(result_A, "result_a")
        return result, result_A, W

def make_embedding_layer(V, no_filters, name=None):
    with tf.variable_scope(name, default_name='Embed') as scope:
        no_features = V.get_shape()[-1].value
        W = make_variable_with_weight_decay('weights', [no_features, no_filters], stddev=1.0/math.sqrt(no_features))
        b = make_bias_variable('bias', [no_filters])
        V_reshape = tf.reshape(V, (-1, no_features))
        s = tf.slice(tf.shape(V), [0], [len(V.get_shape())-1])
        s = tf.concat([s, tf.stack([no_filters])], 0)
        result = tf.reshape(tf.matmul(V_reshape, W) + b, s)
        _histogram_summaries(result, "result")
        return result, W

def make_init_mask_block(A, name=None):
    with tf.name_scope(name, default_name='InitMaskBlock') as scope:
        Ashape = tf.shape(A)
        no_A = Ashape[2]
        I = tf.eye(Ashape[1],batch_shape=[Ashape[0]])
        I = tf.transpose(tf.tile(tf.expand_dims(I,0), tf.stack([no_A,1,1,1])), [1,2,0,3])
        Aacc = tf.minimum(I + A, tf.ones(tf.shape(A)))
        Ak = A
        # Aout = Aacc
        # pdb.set_trace()
        Amask = tf.maximum(A - I, tf.zeros(tf.shape(A)))
        _histogram_summaries(Amask, "init_Amask")
    return Amask, Aacc, Ak, Aacc
        
def make_graph_pooling_layer(V, A, P, name=None):
    with tf.variable_scope(name,default_name='Graph-Pooling') as scope:
        Vout = tf.matmul(tf.transpose(P,perm=[0,2,1]),V)
        Ashape = tf.shape(A)
        Prep = tf.tile(tf.expand_dims(P,2),[1,1,Ashape[2],1])
        Ptranspose = tf.transpose(Prep,perm=[0,2,3,1])
        Pnottranspose = tf.transpose(Prep,perm=[0,2,1,3])
        Abatched = tf.transpose(A,perm=[0,2,1,3])
        leftMultiply = tf.matmul(Ptranspose,Abatched)
        rightMultiply = tf.matmul(leftMultiply,Pnottranspose)
        Aout = tf.transpose(rightMultiply,perm=[0,2,1,3])
        return Vout, Aout
        
def make_sparse_graph_pooling_layer(V,A,P,name=None):
    with tf.variable_scope(name,default_name='Graph-Pooling') as scope:
        #Generate shape array for BN
        #oldVShapeNonTensor = V.get_shape()
        #Generate shape tensor for vertex pooling function I know hacky af
        oldVShape = tf.shape(V, out_type=tf.int64)
        newVShape = tf.stack([oldVShape[0], P.dense_shape[3], oldVShape[2]])
        Ptranspose = tf.sparse_transpose(P, perm=[0, 1, 3, 2])
        #Areordered = tf.sparse_transpose(A, perm=[0, 2, 1, 3])
        #print(P.get_shape())
        #print(Ptranspose.get_shape())
        #print(Areordered.get_shape())

        Vout = _graphcnn_avg_vertex_pool_sparse_module.sparse_average_vertex_pool(V,
                                                                                  Ptranspose.indices,
                                                                                  Ptranspose.values,
                                                                                  newVShape)
        Vout.set_shape([V.get_shape()[0].value,\
                        P.get_shape()[3].value,\
                        V.get_shape()[2].value])
        #Change indices to pooled mapping
        print(A.indices.get_shape())
        print(A.values.get_shape())
        Pcols = P.indices[:,3]
        newRowidx = tf.gather(Pcols,A.indices[:,1])
        newColidx = tf.gather(Pcols,A.indices[:,3])
        newIndices = tf.stack((A.indices[:,0],newRowidx,A.indices[:,2],newColidx),axis=1)
        print('lol2')
        print(newIndices.get_shape())
        newShape =  tf.stack((A.dense_shape[0],P.dense_shape[3],A.dense_shape[2],P.dense_shape[3]),axis=0)
        Adupe = tf.sparse_reorder(tf.SparseTensor(newIndices,A.values,newShape))
        
        
        #Segment sum to merge duplicate indices
        #Worried about this casting, but matmul doesn't support int64 for some dumb reason
        linearized = tf.cast(tf.matmul(tf.cast(Adupe.indices,tf.float64),
                               tf.cast([[newShape[1] * newShape[2] * newShape[3]], [newShape[2] * newShape[3]],
                               [newShape[3]], [1]],tf.float64)),tf.int64)
        print(linearized.get_shape())

        y, idx = tf.unique(tf.squeeze(linearized))

        # Use the positions of the unique values as the segment ids to
        # get the unique values
        print(idx.get_shape())
        print(Adupe.values.get_shape())
        values = tf.segment_sum(Adupe.values, idx)
        delinearized = tf.stack((y // (newShape[1] * newShape[2] * newShape[3]),
                                       y // (newShape[2] * newShape[3]) % newShape[1], y // newShape[3] % newShape[2],
                                       y % newShape[3]), axis=1)
        Aout = tf.SparseTensor(delinearized,values,newShape)
        return Vout, Aout
    
@ops.RegisterGradient("SparseAverageVertexPool")
def _graphcnn_avg_vertex_pool_sparse_grad_func(op, grad):
    #print(grad.shape)
    #print(grad.get_shape().as_list())
    return _graphcnn_avg_vertex_pool_sparse_grad_module.sparse_average_vertex_pool_grad(grad, op.inputs[0], op.inputs[1],op.inputs[2], op.inputs[3])


def make_sparse_max_graph_pooling_layer(V, A, P,name=None):
    with tf.variable_scope(name, default_name='Graph-Max-Pooling') as scope:
        #Generate shape array for BN
        #oldVShapeNonTensor = V.get_shape()
        #Generate shape tensor for vertex pooling function I know hacky af
        oldVShape = tf.shape(V, out_type=tf.int64)
        newVShape = tf.stack([oldVShape[0], P.dense_shape[3], oldVShape[2]])
        Ptranspose = tf.sparse_transpose(P, perm=[0, 1, 3, 2])

        Vout = _graphcnn_max_vertex_pool_sparse_module.sparse_max_vertex_pool(V,
                                                                                  Ptranspose.indices,
                                                                                  Ptranspose.values,
                                                                                  newVShape)
        Vout.set_shape([V.get_shape()[0].value,\
                        P.get_shape()[3].value,\
                        V.get_shape()[2].value])

        #Change indices to pooled mapping
        print(P.indices[:,3].get_shape())
        Pcols = P.indices[:,3]
        newRowidx = tf.gather(Pcols,A.indices[:,1])
        newColidx = tf.gather(Pcols,A.indices[:,3])
        newIndices = tf.stack((A.indices[:,0],newRowidx,A.indices[:,2],newColidx),axis=1)
        newShape =  tf.stack((A.dense_shape[0],P.dense_shape[3],A.dense_shape[2],P.dense_shape[3]),axis=0)
        
        Adupe = tf.sparse_reorder(tf.SparseTensor(newIndices,A.values,newShape))
        
        
        #Segment sum to merge duplicate indices
        #Worried about this casting, but matmul doesn't support int64 for some dumb reason
        linearized = tf.cast(tf.matmul(tf.cast(Adupe.indices,tf.float64),
                               tf.cast([[newShape[1] * newShape[2] * newShape[3]], [newShape[2] * newShape[3]],
                               [newShape[3]], [1]],tf.float64)),tf.int64)
        print(linearized.get_shape())

        y, idx = tf.unique(tf.squeeze(linearized))

        # Use the positions of the unique values as the segment ids to
        # get the unique values
        print(idx.get_shape())
        print(Adupe.values.get_shape())
        values = tf.segment_sum(Adupe.values, idx)
        delinearized = tf.stack((y // (newShape[1] * newShape[2] * newShape[3]),
                                       y // (newShape[2] * newShape[3]) % newShape[1], y // newShape[3] % newShape[2],
                                       y % newShape[3]), axis=1)
        Aout = tf.sparse_reorder(tf.SparseTensor(delinearized,values,newShape))
        return Vout, Aout


@ops.RegisterGradient("SparseMaxVertexPool")
def _graphcnn_max_vertex_pool_sparse_grad_func(op, grad):
    # print(grad.shape)
    # print(grad.get_shape().as_list())
    return _graphcnn_max_vertex_pool_sparse_grad_module.sparse_max_vertex_pool_grad(grad, op.inputs[0],
                                                                                        op.inputs[1], op.inputs[2],
                                                                                        op.inputs[3])


#Basically an implicit ReLU is always included
def make_graph_maxpooling_layer(V, A, P, name=None):
    with tf.variable_scope(name,default_name='Graph-Pooling') as scope:
        Pextend = tf.expand_dims(tf.transpose(P,perm=[0,2,1]),3)
        Vextend = tf.expand_dims(V,1)
        #Use broadcasting tricks to get the maximum vertex of each cluster
        #Each column of P^T is an indicator of whether that vertex is a candidate
        #in a given coarse cluster
        #The number of rows is the number of coarse vertices
        #We want to mutiply each individual vertex feature vector by the scalar indicator
        #Then take the maximum for each coarse vertex
        Vout = tf.reduce_max(tf.multiply(Pextend,Vextend),axis=2)
        #Vout = tf.matmul(tf.transpose(P,perm=[0,2,1]),V)
        Ashape = tf.shape(A)
        Prep = tf.tile(tf.expand_dims(P,2),[1,1,Ashape[2],1])
        Ptranspose = tf.transpose(Prep,perm=[0,2,3,1])
        Pnottranspose = tf.transpose(Prep,perm=[0,2,1,3])
        Abatched = tf.transpose(A,perm=[0,2,1,3])
        leftMultiply = tf.matmul(Ptranspose,Abatched)
        rightMultiply = tf.matmul(leftMultiply,Pnottranspose)
        Aout = tf.transpose(rightMultiply,perm=[0,2,1,3])
        return Vout, Aout

def make_reduce_max(V,name):
    with tf.variable_scope(name,default_name = 'GraclusPool') as scope:
        return tf.reduce_max(V,axis=1)

def make_vertex_attention_1d(V,num_filters,name=None):
    with tf.variable_scope(name, default_name='VertexAttention1D') as scope:
        num_vertices = V.get_shape()[1].value
        num_weights = V.get_shape()[2].value
        W = make_variable_with_weight_decay('weights', [num_weights, num_filters], stddev=1.0/math.sqrt(float(num_weights)))
        V2d = tf.reshape(V,[-1,num_weights]) #BNxF
        logits2d = tf.matmul(V2d,W)#BNxF2
        logits1d = tf.reduce_max(logits2d,axis=1)#BNx1
        logits = tf.reshape(logits1d,[-1,num_vertices]) #BxN
        alpha = tf.nn.softmax(logits)
        Valpha = tf.multiply(V,tf.expand_dims(alpha,2))
        Vout = V + Valpha
    return Vout#, alpha,logits
    
def make_ufc_layer(V,num_filters,num_vertices,name=None):
    with tf.variable_scope(name, default_name='Un-Fully-Connected') as scope:
         #BxC #1xNxF
        W = make_variable_with_weight_decay('weights', [1,num_filters*num_vertices], stddev=1.0/math.sqrt(float(num_filters*num_vertices)))
        Vtranspose = tf.reshape(V,[-1,1]) #BCx1
        logits2d = tf.matmul(Vtranspose,W) #BCxFN
        logits3d = tf.reshape(logits2d,[-1,V.get_shape()[1],num_filters,num_vertices])#BxCxFxN
        flattenFilters = tf.reduce_max(logits3d,axis=2)#BxCxN
        Vout = tf.transpose(flattenFilters,perm=[0,2,1])
    return Vout,W#, alpha,logits

def make_vertex_attention_conv(V,A,num_filters,no_A,name=None):
    with tf.variable_scope(name, default_name='VertexAttention1D') as scope:
        Vlogits,_ = make_sparse_graphcnn_layer(V, A, num_filters, no_A, stride=1, order=1) #BxNxF
        logits = tf.reduce_max(Vlogits,axis=2)#BxN
        alpha = tf.nn.softmax(logits)#BxN
        Valpha = tf.multiply(V,tf.expand_dims(alpha,2))
        Vout = V + Valpha
    return Vout
    
def make_global_feature_concat(V,name=None):
    with tf.variable_scope(name, default_name='Global-Feature-Concat') as scope:
        #V = BxNxF
        num_vertices = V.get_shape()[1].value
        global_features = tf.reduce_max(V,axis=1,keepdims=True) #Bx1xF
        global_features_vertices = tf.tile(global_features,[1,num_vertices,1]) #BxNxF
        return global_features_vertices

def make_sparse_max_graph_unpooling_layer(V, P,name=None):
    with tf.variable_scope(name, default_name='Graph-Max-Pooling') as scope:
        #Generate shape array for BN
        #oldVShapeNonTensor = V.get_shape()
        #Generate shape tensor for vertex pooling function I know hacky af
        oldVShape = tf.shape(V, out_type=tf.int64) #BxN1xF
        #P = Bx1xN1xN2
        newVShape = tf.stack([oldVShape[0], P.dense_shape[2], oldVShape[2]]) #BxN2xF

        Vout = _graphcnn_max_vertex_pool_sparse_module.sparse_max_vertex_pool(V,
                                                                                  P.indices,
                                                                                  P.values,
                                                                                  newVShape)
        Vout.set_shape([V.get_shape()[0].value,\
                        P.get_shape()[2].value,\
                        V.get_shape()[2].value])
    return Vout

def make_sparse_graph_unpooling_layer(V, P,name=None):
    with tf.variable_scope(name, default_name='Graph-Max-Pooling') as scope:
        #Generate shape array for BN
        #oldVShapeNonTensor = V.get_shape()
        #Generate shape tensor for vertex pooling function I know hacky af
        oldVShape = tf.shape(V, out_type=tf.int64) #BxN1xF
        #P = Bx1xN1xN2
        newVShape = tf.stack([oldVShape[0], P.dense_shape[2], oldVShape[2]]) #BxN2xF

        Vout = _graphcnn_avg_vertex_pool_sparse_module.sparse_average_vertex_pool(V,
                                                                                  P.indices,
                                                                                  P.values,
                                                                                  newVShape)
        Vout.set_shape([V.get_shape()[0].value,\
                        P.get_shape()[2].value,\
                        V.get_shape()[2].value])
    return Vout
