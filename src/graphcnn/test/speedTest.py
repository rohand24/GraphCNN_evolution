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
import time
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

FILE = '/home/miguel/Documents/data/modelnet10_1000_points/chair/test/chair_0891.ply'
FILE2 = '/home/miguel/Documents/data/modelnet10_1000_points/chair/test/chair_0891.ply'
THETA = 4
PHI = 4
K = 20
RADIUS = 0.5
STRIDE = 2
MAX_SIZE = 1000
POOL_RATIOS = [0.5]
B = 32
T = 78
NUM_ITERS = 100

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
    #fig = plt.figure(figsize=(20,4))
    V1, A1, P1 = processFile(FILE)
    V2, A2, P2 = processFile(FILE2)
    A = np.concatenate((A1, A2), axis=0)
    P = np.concatenate((P1, P2), axis=0)
    # A = sample.flattenA()
    # indices = np.transpose(np.nonzero(A)).astype(np.int64)
    indices = np.transpose(np.nonzero(A)).astype(np.int64)
    values = np.ravel(A[np.nonzero(A)]).astype(np.float32)
    # print(indices)
    # print(values)
    # P = np.expand_dims(np.array([[[0,2,3,0],[1,0,1,0]]]),axis=0)#,[[1,0,1,0],[0,2,3,0]]]),axis=0)#,[0, 4, 0, 0],[0,0,0,2]])
    # P = np.array([[10, 20, 0, 0, 0, 0], [0, 30, 0, 40, 0, 0], [0, 0, 50, 60, 70, 0], [0, 0, 0, 0, 0, 80]])
    # P = scipy.sparse.random(100,100).todense()
    print(P.shape)
    # print(np.dot(P,A))
    dense_shape = np.asarray(A.shape, dtype=np.int64)
    Pindices = np.transpose(np.nonzero(P)).astype(np.int64)
    Pvals = np.ravel(P[np.nonzero(P)]).astype(np.float32)
    # print(Pindices)
    # print(Pvals)
    Pshape = np.asarray(P.shape, dtype=np.int64)
    Avarsparse = tf.sparse_placeholder(tf.float32, shape=dense_shape)

    ATensor = tf.placeholder(tf.float32, shape=(None, None,None,None))
    PTensor = tf.placeholder(tf.float32, shape=(None,None,None,None))

    idxTensor = tf.placeholder(tf.int64, shape=(None, None))
    valTensor = tf.placeholder(tf.float32, shape=(None))

    PidxTensor = tf.placeholder(tf.int64, shape=(None, None))
    PvalTensor = tf.placeholder(tf.float32, shape=(None))

    Asparse = tf.SparseTensor(indices=idxTensor, values=valTensor, dense_shape=dense_shape)
    Psparse = tf.SparseTensor(indices=PidxTensor, values=PvalTensor, dense_shape=Pshape)

    AsparseLeft = sparse_sparse_batch_mat_mul(Psparse, Asparse)

    Pdenserep = tf.tile(PTensor, [1, tf.shape(ATensor)[1], 1, 1])

    AdenseLeft = tf.matmul(Pdenserep, ATensor)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        start = time.time()
        for i in range(NUM_ITERS):
            [AdenseLeftOut] = sess.run([AdenseLeft], \
                      feed_dict={ATensor: A, PTensor: P})
        end = time.time()
        print('Dense time: {0} seconds'.format(end - start))
        start = time.time()
        for i in range(NUM_ITERS):
            [AsparseLeftOut] = sess.run([AsparseLeft], \
                          feed_dict={idxTensor: indices, \
                                     valTensor: values, \
                                     PidxTensor: Pindices, \
                                     PvalTensor: Pvals})
        end = time.time()
        print('Sparse time: {0} seconds'.format(end - start))

Main()


