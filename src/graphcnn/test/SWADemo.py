import sys
import os
import os.path
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../../preprocessing')))
#from mnist_to_graph_tensor import mnist_adj_mat
from graphcnn.layers import *
from graphcnn.network import *
import numpy as np

CHKPT1 = '/home/mad6384/G3DNet3/G3DNet/snapshots/CVPR2019Large3/eigenfeatures3_test/model-10000'
CHKPT2 = '/home/mad6384/G3DNet3/G3DNet/snapshots/CVPR2019Large3/eigenfeatures3_test/model-10200'
ARCH = 'OC,coo_32_1_1,rc0_32-32_1-1_1-1_0-0_1-1_1-1,gmp_0,c_64_1_1,rc0_64-64_1-1_1-1_0-0_1-1_1-1,gmp_1,c_128_1_1,rc0_32-32-128_1-1-1_1-1-1_1-0-1_1-1-1_1-1-1,gmp_2,c_256_1_1,rc0_64-64-256_1-1-1_1-1-1_1-0-1_1-1-1_1-1-1,gmp_3,rc0_64-64-256_1-1-1_1-1-1_1-0-1_1-1-1_1-1-1,gmp_4,c_512_1_1,rc0_128-128-512_1-1-1_1-1-1_1-0-1_1-1-1_1-1-1,coo_1024_1_1,rm,fc_2048_1_1_1,fc_1024_1_1_1,fc_10_0_0_0'
SPARSE = 1
NUM_SLICES = 36
SIGMA = 1
GRACLUS_NUM_ITERS = 5
POOL_RATIOS = [0.5, 0.5, 0.5, 0.5, 0.5]
N = 4000
NUM_FEATURES = 13
NUM_CLASSES = 10
g1 = tf.Graph()
g2 = tf.Graph()
inputV = tf.placeholder(dtype=tf.float32,shape=[None,N,NUM_FEATURES])
inputA = tf.sparse_placeholder(dtype=tf.float32,shape=[None,N,NUM_SLICES,N])
inputLabel = tf.placeholder(dtype=tf.float32,shape=[None,NUM_CLASSES])
inputs = [inputV,inputA,inputLabel]
prevSize = N
for pidx in range(len(POOL_RATIOS)):
    currentSize = np.floor(prevSize * POOL_RATIOS[pidx]).astype(np.int64)
    prevSize = currentSize
    inputs.append(tf.sparse_placeholder(dtype=tf.float32,shape=[None,1,prevSize,currentSize]))
print(inputs)
with g1.as_default():
    net1 = GraphCNNNetwork()
    saver1 = tf.train.import_meta_graph(CHKPT1 + '.meta')
    session1 = tf.Session(graph=g1)
    with session1 as sess:
        saver1.restore(sess,CHKPT1)
        valsDict1 = dict()
        for var in g1.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            print(var.name)
            curVal = tf.get_default_graph().get_tensor_by_name(var.name)
            valsDict1[var.name] = sess.run(curVal)
    with g2.as_default():
        net2 = GraphCNNNetwork()
        saver2 = tf.train.import_meta_graph(CHKPT2 + '.meta')
        session2 = tf.Session(graph=g2)
        with session2 as sess:
            saver2.restore(sess,CHKPT2)
            valsDict2 = dict()
            for var in g2.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                print(var.name)
                curVal = tf.get_default_graph().get_tensor_by_name(var.name)
                valsDict2[var.name] = sess.run(curVal)
                valsDict2[var.name] = sess.run(curVal)

#valsDictAvg = dict()
    session3 = tf.Session(graph=g1)
    with session3 as sess:
        for curVar in tf.trainable_variables():
            if curVar.name in valsDict1.keys():
                key = curVar.name
                curVal = (valsDict1[key] + valsDict2[key]) / 2
                assign_op = curVar.assign(curVal)
                sess.run(assign_op)
                curTensorVal = sess.run(curVar)
                print('{0} {1} {2}'.format(np.mean(curTensorVal),np.mean(valsDict1[key]),np.mean(valsDict2[key])))

