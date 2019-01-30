import sys
import os
import os.path
import plyfile
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.spatial
import scipy.sparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../../preprocessing')))
from graphcnn.layers import *
import tensorflow as tf
from graphcnn.util.modelnet.GraphData import GraphData
from graphcnn.util.pooling.LloydPoolingPyramid import LloydPoolingPyramid
from graphcnn.util.pooling.SphericalAdjacencyCompander import SphericalAdjacencyCompander
from graphcnn.util.pooling.SphericalUtil import *
#import matlab.engine
from PIL import Image
import pyamg
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy.random
import math
import transforms3d
THETA = 4
PHI = 4
K = 10
RADIUS = 0.5
STRIDE = 2
MAX_SIZE = 1000
POOL_RATIOS = [0.5, 0.5, 0.5, 0.5, 0.5]
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

def draw3DGraph(V,A,vertexCount,subplotId,fig):
    ax = fig.add_subplot(subplotId, projection='3d')
    #flatA = graph.flattenA()
    for i in range(vertexCount):
        for j in range(vertexCount):
            #print(A[i,j])
            if A[i,j] > 0:
                ax.plot([V[i,0],V[j,0]],[V[i,1],V[j,1]],[V[i,2],V[j,2]])
    ax.scatter(V[:,0],V[:,1],V[:,2])
    #print(V.shape)
    #print(V[:,0].shape)
    #print(np.squeeze(V[:,0]).shape)
    #ax.scatter([V[:,0]],[V[:,1]],[V[:,2]], c='r', marker='o')

    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')

def Main():
    #matplotlib.use('Agg')
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        extension = os.path.splitext(sys.argv[1])[1]
        fig = plt.figure()
        if extension == '.ply':

            #A = np.expand_dims(A,axis=0)
            V, A = ply2graph(sys.argv[1],K)
            V = np.expand_dims(np.array(V),axis=0)
            print(V.shape)
            #A = np.expand_dims(np.transpose(np.array(A),axes=(1,0,2)),axis=0)
            A = np.expand_dims(np.array(A),axis=0)
            print(A.shape)
            #eng = matlab.engine.start_matlab()
            #eng.addpath(os.path.abspath(os.path.join(os.path.dirname(__file__),'../util/pooling/graclus1.2/matlab/')))
            #pooler = GraclusPoolingPyramid(len(POOL_RATIOS), GeometricAdjacencyCompander, POOL_RATIOS)#,eng)
            #pooler = SpectralClusteringPoolingPyramid(len(POOL_RATIOS), MemoryAdjacencyCompander, POOL_RATIOS)
            pooler = LloydPoolingPyramid(len(POOL_RATIOS), SphericalAdjacencyCompander, POOL_RATIOS)
            #pooler = SBFSStrideAdjacencyPyramid(len(POOL_RATIOS), GeometricAdjacencyCompander, STRIDE, POOL_RATIOS)
            #draw3DGraph(V.sum(axis=0), A.sum(axis=(0, 2)), V.shape[1], '141', fig)

        #plt.show()
        Plist = pooler.makeP(A.sum(axis=0),V.sum(axis=0),THETA,PHI)
        #print(Plist[0].shape)
        Avar = tf.placeholder(tf.float32, shape=(None, None, None, None))
        Vvar = tf.placeholder(tf.float32, shape=(None, None, None))
        Pvar = tf.placeholder(tf.float32, shape=(None, None, None))
        Vout,Aout = make_graph_pooling_layer(Vvar,Avar,Pvar)
        #vshape = V.shape
        #V = tf.constant(V,dtype=tf.float32,shape=vshape)
        #A = tf.constant(A,dtype=tf.float32,shape=A.shape)
        #Vout,Aout,W = make_graph_embed_pooling(V, A, no_vertices=111)
        #Vrev,Arev = make_graph_unpooling_layer(Vvar,Avar,Pvar)

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(2,2+len(POOL_RATIOS)):
                subplotId = '16' + str(i)
                [V,A] = sess.run([Vout,Aout],feed_dict={Vvar: V, Avar: A, Pvar: np.expand_dims(Plist[i-2],axis=0)})
                print(V.shape)
                if extension == '.ply':
                    draw3DGraph(V.sum(axis=0),A.sum(axis=(0,2)),Plist[i - 2].shape[1],subplotId,fig)

            #fig2 = plt.figure(2)
            #subplotId = '421'
            #draw3DGraph(V.sum(axis=0), A.sum(axis=(0, 2)), A.shape[1], subplotId, fig2)
            #for i in range(2,2 + len(POOL_RATIOS)):
            #    subplotId = '42' + str(i)
            #    print(V.shape)
            #    [V, A] = sess.run([Vrev, Arev], feed_dict={Vvar: V, Avar: A, Pvar: np.expand_dims(Plist[len(POOL_RATIOS) - (i - 2) - 1], axis=0)})
            #    draw3DGraph(V.sum(axis=0), A.sum(axis=(0, 2)), Plist[len(POOL_RATIOS) - (i - 2) - 1].shape[1], subplotId, fig2)
        plt.show()

    else:
        print('Could not find path LOL!')

Main()
