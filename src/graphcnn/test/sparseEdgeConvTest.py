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
import transforms3d
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../preprocessing')))
from graphcnn.util.pooling.SphericalUtil import *

# from mnist_to_graph_tensor import mnist_adj_mat
from graphcnn.layers import *
import tensorflow as tf
from sklearn.metrics import mean_squared_error

FILE = '/shared/kgcoe-research/mil/modelnet/modelnet10_1000_points/chair/test/chair_0890.ply'
FILE2 = '/shared/kgcoe-research/mil/modelnet/modelnet10_1000_points/chair/test/chair_0890.ply'
THETA = 4
PHI = 4
K = 10
RADIUS = 0.5
STRIDE = 2
MAX_SIZE = 1000
IDEAL_POOL_RATIOS = [0.5]
REAL_POOL_RATIOS = [0.4]
B = 32
T = 78
NUM_FILTERS = 16
num_nonzero_vertices = int(REAL_POOL_RATIOS[0]*MAX_SIZE)
num_zero_vertices = int(IDEAL_POOL_RATIOS[0]*MAX_SIZE) - num_nonzero_vertices

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
    return V, A

def Main():
    np.set_printoptions(linewidth=180)
    V1, A1 = processFile(FILE)
    V2, A2 = processFile(FILE2)
    A = np.concatenate((A1, A2), axis=0)
    V = np.concatenate((V1, V2), axis=0)
    #A = np.ones((2,4,2,4))
    #V = np.ones((2,4,3))
    print(A.shape)
    indices = np.transpose(np.nonzero(A)).astype(np.int64)
    values = np.ravel(A[np.nonzero(A)]).astype(np.float32)
       # print(values.shape)
    dense_shape = np.asarray(A.shape, dtype=np.int64)
    #pooler = LloydPoolingPyramid(len(REAL_POOL_RATIOS), GeometricAdjacencyCompander, REAL_POOL_RATIOS)

    Avarsparse = tf.sparse_placeholder(tf.float32, shape=dense_shape)
    Avar = tf.placeholder(tf.float32, shape=(None, None, None, None))
    Vvar = tf.placeholder(tf.float32, shape=V.shape)

    idxTensor = tf.placeholder(tf.int64, shape=(None, None))
    valTensor = tf.placeholder(tf.float32, shape=(None))

    Asparse = tf.SparseTensor(indices=idxTensor, values=valTensor, dense_shape=dense_shape)
    Adense = tf.cast(tf.sparse_to_dense(Asparse.indices, Asparse.dense_shape, Asparse.values), tf.float32)

    Hs = tf.random_uniform([V.shape[2], NUM_FILTERS], dtype=tf.float32)
    Hr = tf.random_uniform([V.shape[2], NUM_FILTERS], dtype=tf.float32)
    He = tf.random_uniform([1, NUM_FILTERS], dtype=tf.float32)
    H = tf.concat((Hs,Hr,He),axis=0)

    VsIndices = tf.stack((Asparse.indices[:,0],Asparse.indices[:,1]),axis=1)
    VrIndices = tf.stack((Asparse.indices[:,0],Asparse.indices[:,3]),axis=1)
    Vs = tf.gather_nd(Vvar,VsIndices)
    Vr = tf.gather_nd(Vvar,VrIndices)
    
    AoutVals = tf.matmul(Vs,Hs) + tf.matmul(Vr,Hr) + tf.matmul(tf.expand_dims(Asparse.values,axis=1),He)
    AoutValsFlat = tf.reduce_mean(AoutVals,axis=1)
    Aout = tf.SparseTensor(Asparse.indices,AoutValsFlat,Asparse.dense_shape)
    AoutDense = tf.sparse_tensor_to_dense(Aout)
    
    VoutSparse,AoutSparse,_ = make_sparse_edge_conv_layer(Vvar,Asparse,H,NUM_FILTERS)
    
    dVDense = tf.gradients(AoutValsFlat,Vvar)
    dADense = tf.gradients(AoutValsFlat,valTensor)
    dHsDense = tf.gradients(AoutValsFlat,Hs)
    dHrDense = tf.gradients(AoutValsFlat,Hr)
    dHeDense = tf.gradients(AoutValsFlat,He)
    
    dVSparse = tf.gradients(AoutSparse.values,Vvar)
    dASparse = tf.gradients(AoutSparse.values,valTensor)
    dHsSparse = tf.gradients(AoutSparse.values,Hs)
    dHrSparse = tf.gradients(AoutSparse.values,Hr)
    dHeSparse = tf.gradients(AoutSparse.values,He)


    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        outputs = [AoutValsFlat,dVDense,dADense,dHsDense,dHrDense,dHeDense,AoutSparse.values,dVSparse,dASparse,dHsSparse,dHrSparse,dHeSparse]
        [AoutDenseOut,dVDenseOut,dADenseOut,dHsDenseOut,dHrDenseOut,dHeDenseOut,AoutSparseOut,dVSparseOut,dASparseOut,dHsSparseOut,dHrSparseOut,dHeSparseOut] = sess.run(outputs, \
                                       feed_dict={idxTensor: indices, \
                                                  valTensor: values,Vvar: V})
                                                  
        print(dHsDenseOut[0].shape)
        print(dHrDenseOut[0].shape)
        print(dHeDenseOut[0].shape)
        
        print(dHsSparseOut[0].shape)
        print(dHrSparseOut[0].shape)
        print(dHeSparseOut[0].shape)
        
        print('MSE Answer')
        #print(AoutDenseOut.shape)
        #print(AoutSparseOut.shape)
        print(mean_squared_error(np.ravel(AoutDenseOut),np.ravel(AoutSparseOut)))
        #print(AoutDenseOut[1:10])
        #print(AoutSparseOut[1:10])
        print('dV MSE')
        #print(dVDenseOut[0].shape)
        #print(dVSparseOut[0].shape)
        #print(dVDenseOut)
        #print(dVSparseOut)
        print(mean_squared_error(np.ravel(dVDenseOut),np.ravel(dVSparseOut)))
        
        print('dA MSE')
        print(dADenseOut[0].shape)
        print(dASparseOut[0].shape)
        print(dADenseOut)
        print(dASparseOut)
        print(mean_squared_error(np.ravel(dADenseOut),np.ravel(dASparseOut)))
        
        print('dHs MSE')
        #print(dHsDenseOut[0].shape)
        #print(dHsSparseOut[0].shape)
        #print(dHsDenseOut)
        #print(dHsSparseOut)
        print(mean_squared_error(np.ravel(dHsDenseOut),np.ravel(dHsSparseOut)))
        
        print('dHr MSE')
        #print(dHrDenseOut[0].shape)
        #print(dHrSparseOut[0].shape)
        #print(dHrDenseOut)
        #print(dHrSparseOut)
        print(mean_squared_error(np.ravel(dHrDenseOut),np.ravel(dHrSparseOut)))
        
        print('dHe MSE')
        #print(dHeDenseOut[0].shape)
       # print(dHeSparseOut[0].shape)
        #print(dHeDenseOut)
        #print(dHeSparseOut)
        print(mean_squared_error(np.ravel(dHeDenseOut),np.ravel(dHeSparseOut)))
        
        
        #print('MSE Theory dV: {0}'.format(dVMaxPoolError))

Main()
