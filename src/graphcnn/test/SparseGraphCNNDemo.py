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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../../preprocessing')))
#from mnist_to_graph_tensor import mnist_adj_mat
from graphcnn.layers import *
import tensorflow as tf
from graphcnn.util.modelnet.GraphData import GraphData
from graphcnn.util.pooling.GeometricAdjacencyCompander import GeometricAdjacencyCompander
#from graphcnn.util.pooling.ImageAdjacencyCompander import ImageAdjacencyCompander
from graphcnn.util.pooling.LloydPoolingPyramid import LloydPoolingPyramid
from graphcnn.util.pooling.SpectralClusteringPoolingPyramid import SpectralClusteringPoolingPyramid
from graphcnn.util.pooling.GraclusPoolingPyramid import GraclusPoolingPyramid
from graphcnn.util.pooling.SphericalAdjacencyCompander import SphericalAdjacencyCompander
from graphcnn.util.pooling.SphericalUtil import *
from sklearn.metrics import mean_squared_error
from tensorflow.python.client import timeline

FILE = '/home/miguel/Documents/data/modelnet10_1000_points/chair/test/chair_0890.ply'
FILE2 = '/home/miguel/Documents/data/modelnet10_1000_points/chair/test/chair_0890.ply'
THETA = 4
PHI = 4
K = 10
RADIUS = 0.5
STRIDE = 2
MAX_SIZE = 1000
POOL_RATIOS = [0.5,0.5,0.5,0.5]
B = 32
T = 78
NUM_FILTERS = 16
GLOBAL_STEP = tf.Variable(0,dtype=tf.int64)

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
    Plist = pooler.makeP(A.sum(axis=0), V.sum(axis=0), THETA, PHI)
    for i in range(len(Plist)):
        Plist[i] = np.expand_dims(np.expand_dims(Plist[i],axis=0),axis=0)
    return V, A, Plist

def graphConv(V,A,noFilters,is_training,name):
    V, _ = make_sparse_graphcnn_layer(V, A, noFilters, name=name)
    V = make_bn(V, is_training, mask=None, num_updates = GLOBAL_STEP)
    V = tf.nn.relu(V)
    return V

def graphPool(V,A,P,is_training,name):
    V, A = make_sparse_max_graph_pooling_layer(V, A, P, name=name)
    V = make_bn(V, is_training, mask=None, num_updates = GLOBAL_STEP)
    V = tf.nn.relu(V)
    return V, A

def fc(V, no_filters, is_training,name):
    if len(V.get_shape()) > 2:
        no_input_features = int(np.prod(V.get_shape()[1:]))
        V = tf.reshape(V, [-1, no_input_features])
    V, _ = make_embedding_layer(V, no_filters)
    V = make_bn(V, is_training, mask=None, num_updates=GLOBAL_STEP)
    V = tf.nn.relu(V)
    return V

def lossFunc(V,labels):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=V, labels=labels))
    correct_prediction = tf.cast(tf.equal(tf.argmax(V, 1), tf.argmax(labels, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    return loss, accuracy

def Main():
    V1, A1, Plist1 = processFile(FILE)
    V2, A2, Plist2 = processFile(FILE2)
    A = np.concatenate((A1, A2), axis=0)
    Plist = []
    Pindices = []
    Pvals = []
    Pshape = []
    for i in range(len(Plist1)):
        Plist.append(np.concatenate((Plist1[i], Plist2[i]), axis=0))
        Pindices.append(np.transpose(np.nonzero(Plist[i])).astype(np.int64))
        Pvals.append(Plist[i][np.nonzero(Plist[i])].astype(np.float32))
        Pshape.append(np.asarray(Plist[i].shape, dtype=np.int64))
        #print("PSHAPE: {0}".format(Pshape))
    Vval = np.concatenate((V1, V2), axis=0)
    print(Vval.shape)
    print(A.shape)
    print("LUL")
    print(Pindices[0].shape)
    indices = np.transpose(np.nonzero(A)).astype(np.int64)
    values = np.ravel(A[np.nonzero(A)]).astype(np.float32)
    dense_shape = np.asarray(A.shape, dtype=np.int64)
    print(indices.shape)
    print(Pshape)
    labelArray = np.zeros((2,10))
    labelArray[0,0] = 1
    labelArray[1,0] = 1

    isTraining = tf.constant(True,dtype=tf.bool)


    #print(A.shape)
    #print(P.shape)

    #Pshape = np.asarray(P.shape, dtype=np.int64)
    Avarsparse = tf.sparse_placeholder(tf.float32, shape=dense_shape)

    idxTensor = tf.placeholder(tf.int64, shape=(None, None))
    valTensor = tf.placeholder(tf.float32, shape=(None))

    PidxTensor1 = tf.placeholder(tf.int64, shape=(None, None))
    PvalTensor1 = tf.placeholder(tf.float32, shape=(None))
    P1 = tf.SparseTensor(indices=PidxTensor1, values=PvalTensor1, dense_shape=Pshape[0])

    PidxTensor2 = tf.placeholder(tf.int64, shape=(None, None))
    PvalTensor2 = tf.placeholder(tf.float32, shape=(None))
    P2 = tf.SparseTensor(indices=PidxTensor2, values=PvalTensor2, dense_shape=Pshape[1])

    PidxTensor3 = tf.placeholder(tf.int64, shape=(None, None))
    PvalTensor3 = tf.placeholder(tf.float32, shape=(None))
    P3 = tf.SparseTensor(indices=PidxTensor3, values=PvalTensor3, dense_shape=Pshape[2])

    PidxTensor4 = tf.placeholder(tf.int64, shape=(None, None))
    PvalTensor4 = tf.placeholder(tf.float32, shape=(None))
    P4 = tf.SparseTensor(indices=PidxTensor4, values=PvalTensor4, dense_shape=Pshape[3])

    Vin = tf.placeholder(tf.float32, shape=(2,1000,3))

    A = tf.SparseTensor(indices=idxTensor, values=valTensor, dense_shape=dense_shape)

    labels = tf.constant(labelArray,dtype=tf.float32)

    #Hardcode a deep network to benchmark the sparse code
    V = graphConv(Vin, A, 16,isTraining,name='Conv1')
    V = graphConv(V, A, 32,isTraining,name='Conv2')
    V = graphConv(V, A, 32, isTraining, name='Conv3')
    V = graphConv(V, A, 32, isTraining, name='Conv4')
    V,A = graphPool(V, A, P1,isTraining,name='Pool1')
    V = graphConv(V, A, 64, isTraining, name='Conv5')
    V = graphConv(V, A, 128, isTraining, name='Conv6')
    V = graphConv(V, A, 128, isTraining, name='Conv7')
    V = graphConv(V, A, 128, isTraining, name='Conv8')
    V, A = graphPool(V, A, P2, isTraining, name='Pool2')
    V = graphConv(V, A, 256, isTraining, name='Conv9')
    V = graphConv(V, A, 256, isTraining, name='Conv10')
    V = graphConv(V, A, 256, isTraining, name='Conv11')
    V = graphConv(V, A, 256, isTraining, name='Conv12')
    V, A = graphPool(V, A, P3, isTraining, name='Pool3')
    V = graphConv(V, A, 512, isTraining, name='Conv13')
    V = graphConv(V, A, 512, isTraining, name='Conv14')
    V = graphConv(V, A, 512, isTraining, name='Conv15')
    V = graphConv(V, A, 512, isTraining, name='Conv16')
    V, A = graphPool(V, A, P4, isTraining, name='Pool4')
    #V = graphConv(V, A, 1024, isTraining, name='Conv17')
    #V = graphConv(V, A, 1024, isTraining, name='Conv18')
    V = fc(V, 2048, isTraining, name='FC1')
    V = fc(V, 1024, isTraining, name='FC2')
    V = fc(V, 10, isTraining, name='FC3')
    loss,accuracy = lossFunc(V,labels)

    #loss = tf.reduce_mean(V)
    train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss, global_step=GLOBAL_STEP)


    #AoutDense = tf.sparse_to_dense(A.indices,A.dense_shape,A.values)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        outputs = [loss,accuracy,train_step]
        for i in range(100000):
            start = time.time()
            [lossOut,accuracyOut,trainStepOut] = sess.run(outputs, \
                          feed_dict={idxTensor: indices, \
                                     valTensor: values, \
                                     PidxTensor1: Pindices[0], \
                                     PvalTensor1: Pvals[0], \
                                     PidxTensor2: Pindices[1], \
                                     PvalTensor2: Pvals[1],
                                     PidxTensor3: Pindices[2], \
                                     PvalTensor3: Pvals[2],
                                     PidxTensor4: Pindices[3], \
                                     PvalTensor4: Pvals[3], \
                                     Vin: Vval})#, options=options, run_metadata=run_metadata)
            end = time.time()
            #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            #chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #with open('timeline_01.json', 'w') as f:
            #    f.write(chrome_trace)
            print('Sparse time Iteration {0}: {1} seconds Loss: {2} Accuracy: {3}'.format(i,end - start,lossOut,accuracyOut))

Main()