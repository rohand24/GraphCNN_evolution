#Having trouble running TF's numeric gradient tool so this separates out the max pool test from the rest
import sys
import os
import os.path
import plyfile
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy
import scipy.spatial
import scipy.sparse
import math
import transforms3d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../preprocessing')))
# from mnist_to_graph_tensor import mnist_adj_mat
from graphcnn.layers import *
import tensorflow as tf
from graphcnn.util.modelnet.GraphData import GraphData
from graphcnn.util.pooling.GeometricAdjacencyCompander import GeometricAdjacencyCompander
# from graphcnn.util.pooling.ImageAdjacencyCompander import ImageAdjacencyCompander
from graphcnn.util.pooling.LloydPoolingPyramid import LloydPoolingPyramid
from graphcnn.util.pooling.SpectralClusteringPoolingPyramid import SpectralClusteringPoolingPyramid
from graphcnn.util.pooling.GraclusPoolingPyramid import GraclusPoolingPyramid
from graphcnn.util.pooling.SphericalAdjacencyCompander import SphericalAdjacencyCompander
from graphcnn.util.pooling.SphericalUtil import *
from sklearn.metrics import mean_squared_error

FILE = '/home/miguel/Documents/data/modelnet10_1000_points/chair/test/chair_0890.ply'
FILE2 = '/home/miguel/Documents/data/modelnet10_1000_points/chair/test/chair_0890.ply'
THETA = 4
PHI = 4
K = 10
RADIUS = 0.5
STRIDE = 2
MAX_SIZE = 1000
POOL_RATIOS = [0.5]
B = 32
T = 78

def ply2graph(plyPath, neighborCount):
    plydata = plyfile.PlyData.read(plyPath)
    V = [plydata['vertex'].data['x'],plydata['vertex'].data['y'],plydata['vertex'].data['z']]
    V = np.array(V).transpose()
    p = 0.05
    keptIndices = np.random.choice(range(V.shape[0]), size=int(math.ceil((1-p)*V.shape[0])),replace=False)
    V = V[keptIndices,:]
    if V.shape[0] > MAX_SIZE:
        V = V[0:MAX_SIZE,:]
    elif V.shape[0] < MAX_SIZE:
        Vscaled = np.zeros((MAX_SIZE,3))
        Vscaled[0:V.shape[0],:] = V
        V = Vscaled
    vertexMean = np.mean(V, axis=0)
    vertexStd = np.std(V, axis=0)
    #Jiggle the model a little bit if it is perfectly aligned with the axes
    #print(input)
    if not vertexStd.all():
        M = np.eye(3)
        angle = np.random.uniform(0.01,0.1,size=3)
        sign = np.random.choice([-1,1],size=3,replace=True)
        M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], sign[0] * angle[0]), M)
        M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], sign[1] * angle[1]), M)
        M = np.dot(transforms3d.axangles.axangle2mat([1,0,0], sign[2] * angle[2]), M)
        V = np.dot(V,M.T)
        vertexMean = np.mean(V, axis=0)
        vertexStd = np.std(V, axis=0)
    V = (V - vertexMean)/vertexStd
    kdtree = scipy.spatial.KDTree(V)
    #First nearest neighbor is always the point itself!
    _,knns = kdtree.query(V,k=K)
    #knns = kdtree.query_ball_tree(kdtree,r=RADIUS)
    A = np.zeros((MAX_SIZE,THETA*PHI,MAX_SIZE))
    numNeighbors = [len(x) for x in knns]
    v1 = np.repeat(np.arange(V.shape[0]),numNeighbors)
    knnsStack = np.concatenate(knns)
    edges = V[v1] - V[knnsStack]
    #zindex = np.dot([4, 2, 1], np.greater((V[v1] - V[knnsStack]).transpose(), np.zeros((3,len(knnsStack)))));
    edgeLen = 1

    #Extra 0.01 due to quantization error border issues
    THETA_KEYS = np.linspace(0, 2*np.pi + 0.01,num=THETA + 1,endpoint=True)
    PHI_KEYS = np.linspace(0,np.pi + 0.01,num=PHI + 1,endpoint=True)

    THETA_VALS = np.arange(THETA)
    PHI_VALS = np.arange(PHI)
    r, theta, phi = toSpherical(edges)
    #for p in sorted(phi):
    #    print(p)
    thetaClassLabels = np.digitize(theta,THETA_KEYS) - 1
    phiClassLabels = np.digitize(phi,PHI_KEYS) - 1
    zindex = classifyPoint(thetaClassLabels,phiClassLabels,PHI)
    print(max(zindex))
    print(max(thetaClassLabels))
    print(max(phiClassLabels))
    #print((V[v1[i]] - V[knnsStack[i]]).shape)
    A[v1,zindex,knnsStack] = edgeLen
    A[knnsStack,zindex,v1] = edgeLen
    #print(np.count_nonzero(A,axis=(0,2)))

    return V,A

def processFile(filename):
    V, A = ply2graph(filename, K)
    V = np.expand_dims(np.array(V), axis=0)
    A = np.expand_dims(np.transpose(np.array(A), axes=(1, 0, 2)), axis=0)  # BxNxLxN?
    pooler = LloydPoolingPyramid(len(POOL_RATIOS), SphericalAdjacencyCompander, POOL_RATIOS)
    P = np.expand_dims(np.expand_dims(
    np.transpose(pooler.makeP(np.transpose(A, axes=(0, 2, 1, 3)).sum(axis=0), V.sum(axis=0), THETA, PHI)[0]),
    axis=0), axis=0)
    return V, A, P


def Main():
    # fig = plt.figure(figsize=(20,4))
    #A = np.expand_dims(np.array([[[0,0,0,0],[0,2,1,3],[1,0,0,1],[0,4,0,0]],[[0,2,1,3],[1,0,0,1],[0,4,0,0],[0,0,0,0]]]),axis=0)
    V1, A1, P1 = processFile(FILE)
    V2, A2, P2 = processFile(FILE2)
    A = np.concatenate((A1,A2),axis=0)
    P = np.concatenate((P1,P2),axis=0)
    #A = sample.flattenA()
    #indices = np.transpose(np.nonzero(A)).astype(np.int64)
    indices = np.transpose(np.nonzero(A)).astype(np.int64)
    values = np.ravel(A[np.nonzero(A)]).astype(np.float32)
    #print(indices)
    #print(values)
    #P = np.expand_dims(np.array([[[0,2,3,0],[1,0,1,0]]]),axis=0)#,[[1,0,1,0],[0,2,3,0]]]),axis=0)#,[0, 4, 0, 0],[0,0,0,2]])
    #P = np.array([[10, 20, 0, 0, 0, 0], [0, 30, 0, 40, 0, 0], [0, 0, 50, 60, 70, 0], [0, 0, 0, 0, 0, 80]])
    #P = scipy.sparse.random(100,100).todense()
    print(A.shape)
    print(P.shape)
    #print(np.dot(P,A))
    dense_shape = np.asarray(A.shape, dtype=np.int64)
    Pindices = np.transpose(np.nonzero(P)).astype(np.int64)
    Pvals = np.ravel(P[np.nonzero(P)]).astype(np.float32)
    #print(Pindices)
    #print(Pvals)
    Pshape = np.asarray(P.shape, dtype=np.int64)
    Avarsparse = tf.sparse_placeholder(tf.float32, shape=dense_shape)
    # Avar = tf.placeholder(tf.float32, shape=(None, None, None, None))
   # Vvar = tf.placeholder(tf.float32, shape=V.shape)

    idxTensor = tf.placeholder(tf.int64, shape=(None, None))
    valTensor = tf.placeholder(tf.float32, shape=(None))

    PidxTensor = tf.placeholder(tf.int64, shape=(None, None))
    PvalTensor = tf.placeholder(tf.float32, shape=(None))

    Asparse = tf.SparseTensor(indices=idxTensor, values=valTensor, dense_shape=dense_shape)
    Adense = tf.cast(tf.sparse_to_dense(Asparse.indices, Asparse.dense_shape, Asparse.values), tf.float32)

    Psparse = tf.SparseTensor(indices=PidxTensor, values=PvalTensor, dense_shape=Pshape)
    Pdense = tf.cast(tf.sparse_to_dense(Psparse.indices, Psparse.dense_shape, Psparse.values), tf.float32)
    Pdenserep = tf.tile(Pdense, [1, tf.shape(Adense)[1], 1, 1])
    # W = make_variable('weights', [V.shape[2]*A.shape[2], 16], initializer=tf.truncated_normal_initializer(stddev=math.sqrt(1.0/(V.shape[2]*(A.shape[2]+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR))))
    # W = tf.ones([V.shape[2]*A.shape[2], 16],dtype=tf.float32)
    # W = tf.fill([V.shape[2]*A.shape[2], 16],2.0)
    #W = tf.random_uniform([V.shape[2] * A.shape[2], NUM_FILTERS], dtype=tf.float32)

    AsparseLeft = sparse_sparse_batch_mat_mul(Psparse,Asparse)
    AsparseRight = sparse_sparse_batch_mat_mul(Asparse,tf.sparse_transpose(Psparse,perm=[0,1,3,2]))
    AdenseLeft = tf.matmul(Pdenserep, Adense)
    AdenseRight = tf.matmul(Adense,tf.transpose(Pdenserep,perm=[0,1,3,2]))
    #AdenseLeft = tf.matmul(Pdense,Adense)
    #AdenseRight = tf.matmul(Adense,tf.transpose(Pdense,perm=[0,1,3,2]))

    #Preindex, Areindex = sparse_sparse_batch_mat_mul(Psparse,Asparse)
    #outIndices = AsparseLeft.indices
    AsparseLeftDense = tf.sparse_to_dense(AsparseLeft.indices,AsparseLeft.dense_shape,AsparseLeft.values,validate_indices=False)
    AsparseRightDense = tf.sparse_to_dense(AsparseRight.indices,AsparseRight.dense_shape,AsparseRight.values,validate_indices=False)

    dAsparseLeftVals = tf.gradients(AsparseLeft.values, valTensor)
    dAsparseLeftIdx = tf.gradients(AsparseLeft.values, idxTensor)
    dPsparseLeftVals = tf.gradients(AsparseLeft.values, PvalTensor)
    dPsparseLeftIdx = tf.gradients(AsparseLeft.values, PidxTensor)
    dAdenseLeft = tf.gradients(AdenseLeft, Adense)
    dPdenseLeft = tf.gradients(AdenseLeft, Pdense)

    dAsparseRightVals = tf.gradients(AsparseRight.values, valTensor)
    dAsparseRightIdx = tf.gradients(AsparseRight.values, idxTensor)
    dPsparseRightVals = tf.gradients(AsparseRight.values, PvalTensor)
    #dPsparseRightIdx = tf.gradients(AsparseRight.values, PidxTensor)
    dAdenseRight = tf.gradients(AdenseRight, Adense)
    dPdenseRight = tf.gradients(AdenseRight, Pdense)

    #AdenseLeft = tf.matmul(Pdense,Adense)
    #dA1= tf.gradients(AsparseLeft, Asparse)
    #dP1 = tf.gradients(AsparseLeft,Psparse)

    print(tf.float32.as_datatype_enum)
    print(tf.int32.as_datatype_enum)
    print(tf.int64.as_datatype_enum)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        outputs = [AsparseLeftDense,
                   AdenseLeft,
                   AsparseRightDense,
                   AdenseRight,
                   dAsparseLeftVals,
                   dAsparseLeftIdx,
                   dPsparseLeftVals,
                   dPsparseLeftIdx,
                   dAdenseLeft,
                   dPdenseLeft,
                   dAsparseRightVals,
                   #dAsparseRightIdx,
                   dPsparseRightVals,
                   #dPsparseRightIdx
                   dAdenseRight,
                   dPdenseRight]
        [AsparseLeftDenseOut,
         AdenseLeftOut,
         AsparseRightDenseOut,
         AdenseRightOut,
         dAsparseLeftValsOut,
         dAsparseLeftIdxOut,
         dPsparseLeftValsOut,
         dPsparseLeftIdxOut,
         dAdenseLeftOut,
         dPdenseLeftOut,
         dAsparseRightValsOut,
         #dAsparseRightIdxOut,
         dPsparseRightValsOut,
         #dPsparseRightIdxOut
         dAdenseRightOut,
         dPdenseRightOut
         ] = sess.run(outputs, \
                                       feed_dict={idxTensor: indices, \
                                                  valTensor: values, \
                                                  PidxTensor: Pindices, \
                                                  PvalTensor: Pvals})
        # dVMaxPoolTheory, dVMaxPoolNumeric = tf.test.compute_gradient(Vvar, \
        #                                                 np.array(V.shape), \
        #                                                 VmaxpoolOutSparse, \
        #                                                 np.array([V.shape[0], int(V.shape[1] * POOL_RATIOS[0]),
        #                                                           V.shape[2]]), extra_feed_dict={idxTensor: indices, \
        #                                                                                          valTensor: values, \
        #                                                                                          PidxTensor: Pindices, \
        #                                                                                          PvalTensor: Pvals,
        #                                                                                          Vvar: V})
        # dPMaxPoolTheory, dPMaxPoolNumeric = tf.test.compute_gradient(PvalTensor, \
        #                                                 np.array(Pvals.shape), \
        #                                                 VmaxpoolOutSparse, \
        #                                                 np.array([V.shape[0],
        #                                                           int(V.shape[1] * POOL_RATIOS[0]),
        #                                                           V.shape[2]]),
        #                                                 extra_feed_dict={idxTensor: indices, \
        #                                                                  valTensor: values, \
        #                                                                  PidxTensor: Pindices, \
        #                                                                  PvalTensor: Pvals, Vvar: V})
        #
        # dVMaxPoolError = tf.test.compute_gradient_error(Vvar, \
        #                                                 np.array(V.shape), \
        #                                                 VmaxpoolOutSparse, \
        #                                                 np.array([V.shape[0], int(V.shape[1] * POOL_RATIOS[0]),
        #                                                           V.shape[2]]),extra_feed_dict={idxTensor: indices, \
        #                                           valTensor: values, \
        #                                           PidxTensor: Pindices, \
        #                                           PvalTensor: Pvals, Vvar: V})
        # dPMaxPoolError = tf.test.compute_gradient_error(PvalTensor, \
        #                                                  np.array(Pvals.shape), \
        #                                                  VmaxpoolOutSparse, \
        #                                                  np.array([V.shape[0],
        #                                                            int(V.shape[1] * POOL_RATIOS[0]),
        #                                                            V.shape[2]]),
        #                                                  extra_feed_dict={idxTensor: indices, \
        #                                                                   valTensor: values, \
        #                                                                   PidxTensor: Pindices, \
        #                                                                   PvalTensor: Pvals, Vvar: V})

        print(AsparseLeftDenseOut.shape)
        print(AdenseLeftOut.shape)
        #print(AsparseLeftDenseOut[0])
        #print(AdenseLeftOut[0])
        mse = mean_squared_error(np.ravel(AsparseLeftDenseOut), np.ravel(AdenseLeftOut))
        print('MSE Aleft: {0}'.format(mse))
        print(np.linalg.norm(AsparseLeftDenseOut))
        print(np.linalg.norm(AdenseLeftOut))

        print(AsparseRightDenseOut.shape)
        print(AdenseRightOut.shape)
        #print(AsparseRightDenseOut[0])
        #print(AdenseRightOut[0])
        mse = mean_squared_error(np.ravel(AsparseRightDenseOut), np.ravel(AdenseRightOut))
        print('MSE Aright: {0}'.format(mse))
        print(np.linalg.norm(AsparseRightDenseOut))
        print(np.linalg.norm(AdenseRightOut))
        print(np.nonzero(A))
        print(dAsparseLeftValsOut)
        dAsparseLeftOut = np.zeros(dAdenseLeftOut[0].shape)
        dAdenseLeftOut2 = np.zeros(dAdenseLeftOut[0].shape)
        dAsparseLeftOut[np.nonzero(A)] = dAsparseLeftValsOut[0]
        dAdenseLeftOut2[np.nonzero(A)] = dAdenseLeftOut[0][np.nonzero(A)]
        #print(dAsparseLeftOut)
        #print(dAdenseLeftOut2)
        #print(dAdenseLeftOut[0])
        mse = mean_squared_error(np.ravel(dAsparseLeftOut), np.ravel(dAdenseLeftOut2))
        print('MSE dA Left: {0}'.format(mse))
        print(np.linalg.norm(dAsparseLeftOut))
        print(np.linalg.norm(dAdenseLeftOut2))

        dPsparseLeftOut = np.zeros(dPdenseLeftOut[0].shape)
        dPdenseLeftOut2 = np.zeros(dPdenseLeftOut[0].shape)
        dPsparseLeftOut[np.nonzero(P)] = dPsparseLeftValsOut[0]
        dPdenseLeftOut2[np.nonzero(P)] = dPdenseLeftOut[0][np.nonzero(P)]
        #print(dPsparseLeftOut)
        #print(dPdenseLeftOut2)
        #print(dPdenseLeftOut[0])
        mse = mean_squared_error(np.ravel(dPsparseLeftOut), np.ravel(dPdenseLeftOut2))
        print('MSE dP Left: {0}'.format(mse))
        print(np.linalg.norm(dPsparseLeftOut))
        print(np.linalg.norm(dPdenseLeftOut2))

        dAsparseRightOut = np.zeros(dAdenseRightOut[0].shape)
        dAdenseRightOut2 = np.zeros(dAdenseRightOut[0].shape)
        dAsparseRightOut[np.nonzero(A)] = dAsparseRightValsOut[0]
        dAdenseRightOut2[np.nonzero(A)] = dAdenseRightOut[0][np.nonzero(A)]
        #print(dAsparseRightOut)
        #print(dAdenseRightOut2)
        #print(dAdenseRightOut[0])
        mse = mean_squared_error(np.ravel(dAsparseRightOut), np.ravel(dAdenseRightOut2))
        print('MSE dA Right: {0}'.format(mse))
        print(np.linalg.norm(dAsparseRightOut))
        print(np.linalg.norm(dAdenseRightOut2))

        dPsparseRightOut = np.zeros(dPdenseRightOut[0].shape)
        dPdenseRightOut2 = np.zeros(dPdenseRightOut[0].shape)
        dPsparseRightOut[np.nonzero(P)] = dPsparseRightValsOut[0]
        dPdenseRightOut2[np.nonzero(P)] = dPdenseRightOut[0][np.nonzero(P)]
        #print(dPsparseRightOut)
        #print(dPdenseRightOut2)
        #print(dPdenseRightOut[0])
        mse = mean_squared_error(np.ravel(dPsparseRightOut), np.ravel(dPdenseRightOut2))
        print('MSE dP Right: {0}'.format(mse))
        print(np.linalg.norm(dPsparseRightOut))
        print(np.linalg.norm(dPdenseRightOut2))


Main()