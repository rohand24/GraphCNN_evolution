import numpy as np
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
from graphcnn.util.pooling.sparse.SphericalSparseAdjacencyCompander import SphericalSparseAdjacencyCompander
from graphcnn.util.pooling.sparse.GeometricSparseAdjacencyCompander import GeometricSparseAdjacencyCompander
from graphcnn.util.pooling.sparse.SpectralClusteringSparsePoolingPyramid import SpectralClusteringSparsePoolingPyramid
from graphcnn.util.pooling.sparse.GraclusSparsePoolingPyramid import GraclusSparsePoolingPyramid
from graphcnn.util.pooling.sparse.LloydSparsePoolingPyramid import LloydSparsePoolingPyramid
from graphcnn.util.pooling.sparse.SparsePoolingFactory import SparsePoolingFactory
from graphcnn.layers import *
import tensorflow as tf
from graphcnn.util.pooling.SphericalUtil import *
from graphcnn.util.pooling.SplitUtils import *
#import matlab.engine
from PIL import Image
import pyamg
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy.random
import math
import transforms3d
THETA = 8
PHI = 8
#THETA = 0
#PHI = 0
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

    #Extra 0.01 due to quantization error border issues
    THETA_KEYS = np.linspace(0, 2*np.pi + 0.01,num=THETA + 1,endpoint=True)
    PHI_KEYS = np.linspace(0,np.pi + 0.01,num=PHI + 1,endpoint=True)

    THETA_VALS = np.arange(THETA)
    PHI_VALS = np.arange(PHI)
    #Aindices, Avalues, Adenseshape = sphericalSplit(V, v1, knnsStack, MAX_SIZE, THETA, PHI, THETA_KEYS,
    #                                                PHI_KEYS)
    Aindices, Avalues, Adenseshape = geometricSplit(V, v1, knnsStack, MAX_SIZE)


    return V,Aindices, Avalues, Adenseshape

def draw3DGraph(V,Arows,Acols,subplotId,fig):
    ax = fig.add_subplot(subplotId, projection='3d')
    temp = np.stack((Arows,Acols),axis=1)
    #print(temp.shape)
    _, uniqueIdx = np.unique(temp,return_index=True,axis=0)
    #print(Arows[uniqueIdx])
    #print("LOL")
    #print(V.shape)
    for i in uniqueIdx:
        ax.plot([V[Arows[i],0],V[Acols[i],0]],[V[Arows[i],1],V[Acols[i],1]],[V[Arows[i],2],V[Acols[i],2]])
    #print(Aindices.shape)
    ax.scatter(V[:,0],V[:,1],V[:,2])

    #ax.set_xlabel('X')
    #ax.set_ylabel('Y')
    #ax.set_zlabel('Z')

#def PCsrToSparseTensor(P):
#    Pcurrent = P.tocoo()
#    singleton = np.zeros(Pcurrent.row.shape[0])
#    Pindices = np.stack((singleton, singleton, Pcurrent.row, Pcurrent.col), axis=1)
#    Pvalues = Pcurrent.data
#    Pshape = np.array([1, 1, Pcurrent.shape[0], Pcurrent.shape[1]])
#    return tf.sparse_reorder(tf.SparseTensor(Pindices,Pvalues,Pshape))

def Main():
    #matplotlib.use('Agg')
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        extension = os.path.splitext(sys.argv[1])[1]
        fig = plt.figure()
        if extension == '.ply':

            #A = np.expand_dims(A,axis=0)
            V, Aindices, Avalues, Ashape = ply2graph(sys.argv[1],K)
            draw3DGraph(V, Aindices[:,0],Aindices[:,2], '161', fig)

            V = np.expand_dims(np.array(V),axis=0)
            print(V.shape)
            #A = np.expand_dims(np.transpose(np.array(A),axes=(1,0,2)),axis=0)
            #A = np.expand_dims(np.array(A),axis=0)
            #print(A.shape)
            #eng = matlab.engine.start_matlab()
            #eng.addpath(os.path.abspath(os.path.join(os.path.dirname(__file__),'../util/pooling/graclus1.2/matlab/')))
            #pooler = GraclusSparsePoolingPyramid(len(POOL_RATIOS), SphericalSparseAdjacencyCompander,
                                                            #POOL_RATIOS)
            #pooler = LloydSparsePoolingPyramid(len(POOL_RATIOS), GeometricSparseAdjacencyCompander,
                                                           # POOL_RATIOS)
            pooler = SparsePoolingFactory().CreatePoolingPyramid(len(POOL_RATIOS), SphericalSparseAdjacencyCompander, POOL_RATIOS, 'Lloyd')

        #plt.show()
        Pidxlist,Pvallist, Pshapelist = pooler.makeP(Aindices,Avalues,Ashape,V.sum(axis=0),THETA,PHI)
        #print(Plist[0].shape)
        #Avar = tf.placeholder(tf.float32, shape=(None, None, None, None))
        AindicesVar = tf.placeholder(tf.int64, shape=(None, None))
        AvaluesVar = tf.placeholder(tf.float32, shape=(None))
        #AshapeVar = tf.placeholder(tf.int64, shape=(None))
        Vvar = tf.placeholder(tf.float32, shape=(None, None, None))
        P1 = tf.sparse_reorder(tf.SparseTensor(Pidxlist[0], Pvallist[0], Pshapelist[0]))
        P2 = tf.sparse_reorder(tf.SparseTensor(Pidxlist[1], Pvallist[1], Pshapelist[1]))
        P3 = tf.sparse_reorder(tf.SparseTensor(Pidxlist[2], Pvallist[2], Pshapelist[2]))
        P4 = tf.sparse_reorder(tf.SparseTensor(Pidxlist[3], Pvallist[3], Pshapelist[3]))
        P5 = tf.sparse_reorder(tf.SparseTensor(Pidxlist[4], Pvallist[4], Pshapelist[4]))
        #P1 = PCsrToSparseTensor(Plist[0])
        #P2 = PCsrToSparseTensor(Plist[1])
        #P3 = PCsrToSparseTensor(Plist[2])
        #P4 = PCsrToSparseTensor(Plist[3])
        #P5 = PCsrToSparseTensor(Plist[4])
        Aindices = np.concatenate((np.zeros((Aindices.shape[0], 1)), Aindices), axis=1).astype(np.int64)
        Ashape = np.concatenate(([1],Ashape),axis=0)
        print(Ashape)
        print(Aindices.shape)
        print(Avalues.shape)
        Asparse = tf.sparse_reorder(tf.SparseTensor(AindicesVar,AvaluesVar,Ashape))
        #Pvar = tf.placeholder(tf.float32, shape=(None, None, None))
        Vout1, Aout1 = make_sparse_graph_pooling_layer(Vvar,Asparse,P1)
        Vout2, Aout2 = make_sparse_graph_pooling_layer(Vout1, Aout1, P2)
        Vout3, Aout3 = make_sparse_graph_pooling_layer(Vout2, Aout2, P3)
        Vout4, Aout4 = make_sparse_graph_pooling_layer(Vout3, Aout3, P4)
        Vout5, Aout5 = make_sparse_graph_pooling_layer(Vout4, Aout4, P5)

        #vshape = V.shape
        #V = tf.constant(V,dtype=tf.float32,shape=vshape)
        #A = tf.constant(A,dtype=tf.float32,shape=A.shape)
        #Vout,Aout,W = make_graph_embed_pooling(V, A, no_vertices=111)
        #Vrev,Arev = make_graph_unpooling_layer(Vvar,Avar,Pvar)

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
                #subplotId = '16' + str(i)

            print("LOL")
            outputs = [Vout1,
                       Vout2,
                       Vout3,
                       Vout4,
                       Vout5,
                       Aout1.indices,
                       Aout2.indices,
                       Aout3.indices,
                       Aout4.indices,
                       Aout5.indices]
            [Vout1Out,
             Vout2Out,
             Vout3Out,
             Vout4Out,
             Vout5Out,
             Aout1Out,
             Aout2Out,
             Aout3Out,
             Aout4Out,
             Aout5Out,
             ] = sess.run(outputs,feed_dict={Vvar: V,\
                                                    AindicesVar: Aindices,\
                                                    AvaluesVar: Avalues,\
                                                    })
            draw3DGraph(Vout1Out.sum(axis=0),Aout1Out[:,1],Aout1Out[:,3],'162',fig)
            draw3DGraph(Vout2Out.sum(axis=0), Aout2Out[:,1],Aout2Out[:,3], '163', fig)
            draw3DGraph(Vout3Out.sum(axis=0),Aout3Out[:,1],Aout3Out[:,3],'164',fig)
            draw3DGraph(Vout4Out.sum(axis=0),Aout4Out[:,1],Aout4Out[:,3],'165',fig)
            draw3DGraph(Vout5Out.sum(axis=0),Aout5Out[:,1],Aout5Out[:,3],'166',fig)

        plt.show()

    else:
        print('Could not find path LOL!')

Main()
