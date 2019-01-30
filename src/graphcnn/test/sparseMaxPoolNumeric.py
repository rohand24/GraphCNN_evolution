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
NUM_FILTERS = 16

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
    A = np.expand_dims(np.array(A), axis=0)  # BxNxLxN?
    print(A.shape)
    pooler = LloydPoolingPyramid(len(POOL_RATIOS), SphericalAdjacencyCompander, POOL_RATIOS)
    P = np.expand_dims(np.expand_dims(pooler.makeP(A.sum(axis=0), V.sum(axis=0), THETA, PHI)[0],
    axis=0),axis=0)
    return V, A, P

def Main():
    np.set_printoptions(linewidth=180)
    V1, A1, P1 = processFile(FILE)
    V2, A2, P2 = processFile(FILE2)
    A = np.concatenate((A1, A2), axis=0)
    P = np.concatenate((P1, P2), axis=0)
    V = np.concatenate((V1, V2), axis=0)
    print(A.shape)
    indices = np.transpose(np.nonzero(A)).astype(np.int64)
    values = np.ravel(A[np.nonzero(A)]).astype(np.float32)
       # print(values.shape)
    dense_shape = np.asarray(A.shape, dtype=np.int64)
    pooler = LloydPoolingPyramid(len(POOL_RATIOS), GeometricAdjacencyCompander, POOL_RATIOS)
    Pindices = np.transpose(np.nonzero(P)).astype(np.int64)
    Pvals = P[np.nonzero(P)].astype(np.float32)
    Pshape = np.asarray(P.shape, dtype=np.int64)
    #A = np.transpose(np.array([[[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]],[[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]]]),axes=(0,2,1,3))
    #print(A.shape)
    #indices = np.transpose(np.nonzero(A)).astype(np.int64)
    #values = A[np.nonzero(A)].astype(np.float32)
    #V = np.transpose(np.array([[[9,10,11,12],[5,6,7,8],[1,2,3,4]],[[1,2,3,4],[5,6,7,8],[9,10,11,12]]]),axes=(0,2,1))
    #print(V.shape)
    #P = np.transpose(np.array([[[[0,1,0,0.1],[1,0,1,0]]],[[[1,0,1,0],[0,1,0,0.1]]]]),axes=(0,1,3,2))
    #dense_shape = np.asarray(A.shape, dtype=np.int64)
    #Pindices = np.transpose(np.nonzero(P)).astype(np.int64)
    #Pvals = P[np.nonzero(P)].astype(np.float32)
    #Pshape = np.asarray(P.shape, dtype=np.int64)
    print("PSHAPE: {0}".format(Pshape))
    Avarsparse = tf.sparse_placeholder(tf.float32, shape=dense_shape)
    Avar = tf.placeholder(tf.float32, shape=(None, None, None, None))
    Vvar = tf.placeholder(tf.float32, shape=V.shape)

    idxTensor = tf.placeholder(tf.int64, shape=(None, None))
    valTensor = tf.placeholder(tf.float32, shape=(None))

    PidxTensor = tf.placeholder(tf.int64, shape=(None, None))
    PvalTensor = tf.placeholder(tf.float32, shape=(None))

    Asparse = tf.SparseTensor(indices=idxTensor, values=valTensor, dense_shape=dense_shape)
    Adense = tf.cast(tf.sparse_to_dense(Asparse.indices, Asparse.dense_shape, Asparse.values), tf.float32)

    Psparse = tf.SparseTensor(indices=PidxTensor, values=PvalTensor, dense_shape=Pshape)
    Pdense = tf.squeeze(tf.cast(tf.sparse_to_dense(Psparse.indices, Psparse.dense_shape, Psparse.values), tf.float32),axis=1)
    # W = make_variable('weights', [V.shape[2]*A.shape[2], 16], initializer=tf.truncated_normal_initializer(stddev=math.sqrt(1.0/(V.shape[2]*(A.shape[2]+1)*GraphCNNGlobal.GRAPHCNN_INIT_FACTOR))))
    # W = tf.ones([V.shape[2]*A.shape[2], 16],dtype=tf.float32)
    # W = tf.fill([V.shape[2]*A.shape[2], 16],2.0)
    W = tf.random_uniform([V.shape[2] * A.shape[2], NUM_FILTERS], dtype=tf.float32)

    VmaxpoolOutSparse,Aignore = make_sparse_max_graph_pooling_layer(Vvar, Asparse, Psparse)
    dV= tf.gradients(VmaxpoolOutSparse, Vvar)
    dP = tf.gradients(VmaxpoolOutSparse,PvalTensor)


    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        outputs = [VmaxpoolOutSparse,dV,dP]
        [VmaxpoolOutSparseOut,dVOut,dPOut] = sess.run(outputs, \
                                       feed_dict={idxTensor: indices, \
                                                  valTensor: values, \
                                                  PidxTensor: Pindices, \
                                                  PvalTensor: Pvals, Vvar: V})
        dVMaxPoolTheory, dVMaxPoolNumeric = tf.test.compute_gradient(Vvar, \
                                                        np.array(V.shape), \
                                                        VmaxpoolOutSparse, \
                                                        np.array([V.shape[0], int(V.shape[1] * POOL_RATIOS[0]),
                                                                  V.shape[2]]), extra_feed_dict={idxTensor: indices, \
                                                                                                 valTensor: values, \
                                                                                                 PidxTensor: Pindices, \
                                                                                                 PvalTensor: Pvals,
                                                                                                 Vvar: V})
        dPMaxPoolTheory, dPMaxPoolNumeric = tf.test.compute_gradient(PvalTensor, \
                                                        np.array(Pvals.shape), \
                                                        VmaxpoolOutSparse, \
                                                        np.array([V.shape[0],
                                                                  int(V.shape[1] * POOL_RATIOS[0]),
                                                                  V.shape[2]]),
                                                        extra_feed_dict={idxTensor: indices, \
                                                                         valTensor: values, \
                                                                         PidxTensor: Pindices, \
                                                                         PvalTensor: Pvals, Vvar: V})

        dVMaxPoolError = tf.test.compute_gradient_error(Vvar, \
                                                        np.array(V.shape), \
                                                        VmaxpoolOutSparse, \
                                                        np.array([V.shape[0], int(V.shape[1] * POOL_RATIOS[0]),
                                                                  V.shape[2]]),extra_feed_dict={idxTensor: indices, \
                                                  valTensor: values, \
                                                  PidxTensor: Pindices, \
                                                  PvalTensor: Pvals, Vvar: V})
        dPMaxPoolError = tf.test.compute_gradient_error(PvalTensor, \
                                                         np.array(Pvals.shape), \
                                                         VmaxpoolOutSparse, \
                                                         np.array([V.shape[0],
                                                                   int(V.shape[1] * POOL_RATIOS[0]),
                                                                   V.shape[2]]),
                                                         extra_feed_dict={idxTensor: indices, \
                                                                          valTensor: values, \
                                                                          PidxTensor: Pindices, \
                                                                          PvalTensor: Pvals, Vvar: V})
        print(V)
        print(np.transpose(P,axes=(0,1,3,2)))
        print('VMaxPoolResults')
        print(VmaxpoolOutSparseOut)
        #print(dVOut[0].shape)
        print(dVOut[0])
        dPOutArray = np.zeros(P.shape)
        dPOutArray[np.nonzero(P)] = dPOut[0]
        print(dPOutArray)
        print(dVMaxPoolTheory)
        print(dVMaxPoolNumeric)
        print(dPMaxPoolTheory)
        print(dPMaxPoolNumeric)
        mse = mean_squared_error(np.ravel(dVMaxPoolTheory), np.ravel(dVMaxPoolNumeric))
        print('MSE dV: {0}'.format(mse))
        print('MSE dV: {0}'.format(dVMaxPoolError))
        print(np.linalg.norm(dVMaxPoolTheory))
        print(np.linalg.norm(dVMaxPoolNumeric))

        mse = mean_squared_error(np.ravel(dPMaxPoolTheory), np.ravel(dPMaxPoolNumeric))
        print('MSE dP: {0}'.format(mse))
        print('MSE dP: {0}'.format(dPMaxPoolError))
        print(np.linalg.norm(dPMaxPoolTheory))
        print(np.linalg.norm(dPMaxPoolNumeric))
        print(np.linalg.norm(VmaxpoolOutSparseOut))

        #print('MSE Theory dV: {0}'.format(dVMaxPoolError))

Main()