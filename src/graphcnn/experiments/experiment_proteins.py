from graphcnn.helper import *
from graphcnn.network import *
from graphcnn.layers import *
import tensorflow as tf
from graphcnn.experiments.experiment import GraphCNNExperiment
from tensorflow.python.training import queue_runner
import sys
import os.path
import multiprocessing
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../../preprocessing')))

def _make_batch_queue(input, capacity, num_threads=1):
    queue = tf.PaddingFIFOQueue(capacity=capacity, dtypes=[s.dtype for s in input],shapes=[s.get_shape() for s in input])
    tf.summary.scalar("fraction_of_%d_full" % capacity,
           tf.cast(queue.size(), tf.float32) *
           (1. / capacity))
    enqueue_ops = [queue.enqueue(input)]*num_threads
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))
    return queue

# This class is responsible for setting up and running experiments
# Also provides helper functions related to experiments (e.g. get accuracy)
class GraphCNNProteinExperiment(GraphCNNExperiment):
    def __init__(self, dataset_name, model_name, net_constructor, numClasses, dataset, isTrain, l2=0, l1=0, poolRatios=None,sparse=False,path_pretrained_weights=None, current_arch=None, change_index=None, mutation_choice = None):
        GraphCNNExperiment.__init__(self, dataset_name, model_name, net_constructor, l2, l1,path_pretrained_weights,current_arch, change_index, mutation_choice)
        self.isTrain = isTrain
        self.poolRatios = poolRatios
        self.graph_vertices = dataset[0]
        self.graph_indices = dataset[1]
        self.graph_values = dataset[2]
        self.graph_dense_shape = dataset[3]
        self.graph_labels = dataset[4]
        self.Plist = dataset[5]
        self.N = self.graph_vertices.shape[1]
        self.C = self.graph_vertices.shape[2]
        self.L = self.graph_dense_shape[2]

        self.no_samples = self.graph_labels.shape[0]
        self.numClasses = numClasses
        self.sparse = sparse

    def calculate_features_wrap(self,indices):
        pyOutput = tf.py_func(self.calculate_features, [indices], [tf.float32, tf.int64, tf.float32, tf.int64] +\
                              [tf.int64]*len(self.poolRatios)+[tf.float32]*len(self.poolRatios)+\
                              [tf.int64]*len(self.poolRatios))
        pyOutput[0].set_shape((self.N,self.C))
        pyOutput[0] = tf.reshape(pyOutput[0], [self.N, self.C])
        V = pyOutput[0]
        A = tf.sparse_reorder(tf.SparseTensor(pyOutput[1],pyOutput[2],[self.N, self.L, self.N]))
        if not self.sparse:
            A = tf.sparse_tensor_to_dense(A)

        pyOutput[3] = tf.one_hot(pyOutput[3],(self.numClasses))
        pyOutput[3].set_shape([self.numClasses])
        label = pyOutput[3]
        Pouts = pyOutput[4:]
        NUM_POOLS = len(self.poolRatios)
        Pidxlist = Pouts[0:NUM_POOLS]
        Pvallist = Pouts[NUM_POOLS:(2*NUM_POOLS)]
        #Pshapelist = Pouts[2*NUM_POOLS:(3*NUM_POOLS)]
        Plist = []
        prevSize = self.N
        for pidx in range(NUM_POOLS):
            currentSize = np.floor(prevSize * self.poolRatios[pidx]).astype(np.int64)
            currentPSparse = tf.sparse_reorder(tf.SparseTensor(Pidxlist[pidx],Pvallist[pidx],[1,prevSize,currentSize]))
            #currentPSparse.set_shape([1,prevSize,currentSize])
            if not self.sparse:
                currentPDense = tf.squeeze(tf.sparse_tensor_to_dense(currentPSparse))
                currentPDense.set_shape([prevSize,currentSize])
                Plist.append(currentPDense)
            else:
                Plist.append(currentPSparse)
            prevSize = currentSize
        return [V,A,label] + Plist

    def create_data(self):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                dtypes = []
                shapes = []
                classes = []
                shapes.append(tf.TensorShape([None,self.N, self.C])) #V Shape
                shapes.append(tf.TensorShape([None,self.N,self.L,self.N])) #A Shape
                shapes.append(tf.TensorShape([None,self.numClasses])) # labels shape
                dtypes.append(tf.float32)
                dtypes.append(tf.float32)
                dtypes.append(tf.float32)
                classes.append(tf.Tensor)
                if self.sparse:
                    classes.append(tf.SparseTensor)
                    classes.append(tf.Tensor)
                    prevSize = self.N
                    for pidx in range(len(self.poolRatios)):
                        currentSize = np.floor(prevSize * self.poolRatios[pidx]).astype(np.int64)
                        shapes.append([None,1,prevSize,currentSize])
                        dtypes.append(tf.float32)
                        classes.append(tf.SparseTensor)
                        prevSize = currentSize
                else:
                    classes.append(tf.Tensor)
                    classes.append(tf.Tensor)
                    prevSize = self.N
                    for pidx in range(len(self.poolRatios)):
                        currentSize = np.floor(prevSize * self.poolRatios[pidx]).astype(np.int64)
                        shapes.append([None,prevSize, currentSize])
                        dtypes.append(tf.float32)
                        classes.append(tf.Tensor)
                        prevSize = currentSize

                shapes = tuple(shapes)
                dtypes = tuple(dtypes)
                classes = tuple(classes)
                # Create the training queue
                with tf.variable_scope('train_data') as scope:
                    self.print_ext('Creating training Tensorflow Tensors')
                    print(self.train_idx.shape)
                    #trainQueue_Idx = tf.train.slice_input_producer([self.train_idx], shuffle=True, seed=1000)

                    #single_sample_train = self.calculate_features_wrap(trainQueue_Idx)
                    #train_queue = _make_batch_queue(single_sample_train, capacity=self.train_batch_size*2, num_threads=1)
                    trainDataset = tf.data.Dataset.from_tensor_slices(self.train_idx).repeat().shuffle(buffer_size=50000)
                    #Not sure what to do with buffersize except to make it really big
                    trainDataset = trainDataset.map(lambda x: self.calculate_features_wrap(x),num_parallel_calls=multiprocessing.cpu_count())
                    trainDataset = trainDataset.batch(self.train_batch_size)
                    trainIterator = tf.data.Iterator.from_structure(dtypes, shapes,output_classes=classes)
                    trainInitializer = trainIterator.make_initializer(trainDataset)
                    nextTrainBatch = trainIterator.get_next()
                # Create the test queue
                with tf.variable_scope('test_data') as scope:
                    self.print_ext('Creating test Tensorflow Tensors')

                    testDataset = tf.data.Dataset.from_tensor_slices(self.test_idx).shuffle(buffer_size=50000).repeat()
                    # Not sure what to do with buffersize except to make it really big
                    testDataset = testDataset.map(lambda x: self.calculate_features_wrap(x),num_parallel_calls=multiprocessing.cpu_count())
                    testDataset = testDataset.batch(self.test_batch_size)
                    testIterator = tf.data.Iterator.from_structure(dtypes, shapes,output_classes=classes)
                    testInitializer = testIterator.make_initializer(testDataset)
                    nextTestBatch = testIterator.get_next()
                return tf.cond(self.net.is_training, lambda: nextTrainBatch, lambda: nextTestBatch),\
                       trainInitializer, testInitializer, shapes
                # Create the training queue

    def create_data_test(self):
        dtypes = []
        shapes = []
        classes = []
        shapes.append(tf.TensorShape([None, self.N, self.C]))  # V Shape
        shapes.append(tf.TensorShape([None, self.N, self.L, self.N]))  # A Shape
        shapes.append(tf.TensorShape([None, self.numClasses]))  # labels shape
        dtypes.append(tf.float32)
        dtypes.append(tf.float32)
        dtypes.append(tf.float32)
        classes.append(tf.Tensor)
        if self.sparse:
            classes.append(tf.SparseTensor)
            classes.append(tf.Tensor)
            prevSize = self.N
            for pidx in range(len(self.poolRatios)):
                currentSize = np.floor(prevSize * self.poolRatios[pidx]).astype(np.int64)
                shapes.append([None, 1, prevSize, currentSize])
                dtypes.append(tf.float32)
                classes.append(tf.SparseTensor)
                prevSize = currentSize
        else:
            classes.append(tf.Tensor)
            classes.append(tf.Tensor)
            prevSize = self.N
            for pidx in range(len(self.poolRatios)):
                currentSize = np.floor(prevSize * self.poolRatios[pidx]).astype(np.int64)
                shapes.append([None, prevSize, currentSize])
                dtypes.append(tf.float32)
                classes.append(tf.Tensor)
                prevSize = currentSize

        shapes = tuple(shapes)
        dtypes = tuple(dtypes)
        classes = tuple(classes)

        with tf.device("/cpu:0"):
             with tf.variable_scope('test_data') as scope:
                self.print_ext('Creating test Tensorflow Tensors')
                testDataset = tf.data.Dataset.from_tensor_slices(self.test_idx)
                # Not sure what to do with buffersize except to make it really big
                testDataset = testDataset.map(lambda x: self.calculate_features_wrap(x),num_parallel_calls=multiprocessing.cpu_count())
                testDataset = testDataset.batch(self.test_batch_size)
                testIterator = tf.data.Iterator.from_structure(dtypes, shapes, output_classes=classes)
                testInitializer = testIterator.make_initializer(testDataset)
                nextTestBatch = testIterator.get_next()

                return nextTestBatch, testInitializer, shapes

    def calculate_features(self, index):  ##input tensor = NxNx3
        #print(type(index))
        V = self.graph_vertices[index,:,:]
        Aindices = self.graph_indices[np.where(self.graph_indices[:,0] == index)]
        Avalues = self.graph_values[np.where(self.graph_indices[:,0] == index)]
        label = self.graph_labels[index].squeeze()
        Plist = self.Plist[index]
        #print(Aindices[:,1:].shape)
        #print(Avalues.shape)
        return [V, Aindices[:,1:], Avalues, label] + Plist
