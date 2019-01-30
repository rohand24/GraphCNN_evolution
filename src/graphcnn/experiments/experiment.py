from graphcnn.helper import *
from graphcnn.network import *
from graphcnn.layers import *
import numpy as np
import tensorflow as tf
import time
from tensorflow.python.training import queue_runner
import pdb, transforms3d, random
import datetime
import sys
from collections import defaultdict
import os
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
from graphcnn.util.sparse.sparse_tensor_mods import get_shape, set_shape
#from tensorflow.python.client import timeline
#from sklearn.model_selection import KFold
# This function is used to create tf.cond compatible tf.train.batch alternative
def _make_batch_queue(input, capacity, num_threads=1):
    queue = tf.PaddingFIFOQueue(capacity=capacity, dtypes=[s.dtype for s in input], shapes=[s.get_shape() for s in input])
    tf.summary.scalar("fraction_of_%d_full" % capacity,
           tf.cast(queue.size(), tf.float32) *
           (1. / capacity))
    enqueue_ops = [queue.enqueue(input)]*num_threads
    queue_runner.add_queue_runner(queue_runner.QueueRunner(queue, enqueue_ops))
    return queue

# This class is responsible for setting up and running experiments
# Also provides helper functions related to experiments (e.g. get accuracy)
class GraphCNNExperiment(object):
    def __init__(self, dataset_name, model_name, net_constructor,l2=0,l1=0,path_pretrained_weights=None, current_arch=None, change_index=None, mutation_choice = None):
        # Initialize all defaults
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_iterations = 200
        self.iterations_per_test = 5
        self.display_iter = 5
        self.snapshot_iter = 1000000
        self.train_batch_size = 0
        self.test_batch_size = 0
        self.crop_if_possible = True
        self.debug = False
        self.loading_weights_flag = False
        self.loading_weights = {}
        self.arch_loading=None
        
        #for loading previous weights
        self.path_pretrained_weights = path_pretrained_weights
        self.current_arch = current_arch
        self.change_index = change_index
        self.mutation_choice = mutation_choice
        
        self.starter_learning_rate = 0.1
        self.learning_rate_exp = 0.1
        self.learning_rate_step = 1000
        self.l2reg = l2
        self.l1reg = l1
        self.reports = {}
        self.silent = False
        self.optimizer = 'momentum'
        
        self.net_constructor = net_constructor
        self.net = GraphCNNNetwork()
        self.net_desc = GraphCNNNetworkDescription()
        tf.reset_default_graph()
        
    def get_loading_layers(self,mutation, loading_name_layers,current_name_layers,change_layer_index, shift):
        
        mutation_split = mutation.split('_')
        mut_type_dict = {'add':1, 'remove': 0, 'replace':0, 'pool' : 2,'pool-gep' : 2, 'LR' : 3, 'reg':3}
        mut_layer_dict = {'skip':1,'conv':0,'filter':0,'one-to-one':0,'attention':0, 'edgeconv':0, 'fc':0, 'change':2, 'mutation':2}
        mut_code = [mut_type_dict[mutation_split[0]],mut_layer_dict[mutation_split[1]]]
        loading_layers_dict={}
        layers_to_load = []
        previous_layers = loading_name_layers[:change_layer_index]
        
        if mut_code[0] < 2 : 
            #pdb.set_trace()
            if mut_code == [1,1]:
                next_layers = loading_name_layers[change_layer_index+3:]
            elif mut_code == [0,0]:
                next_layers = loading_name_layers[change_layer_index+2:]
            else:
                next_layers = loading_name_layers[change_layer_index+1:]
            if len(next_layers)>0 and next_layers[0] == 'fc1':
                next_layers = next_layers[1:]
            layers_to_load = previous_layers +next_layers
            for i in range(len(loading_name_layers)):
                if loading_name_layers[i] in layers_to_load:
                    if i >= change_layer_index:
                        loading_layers_dict[loading_name_layers[i]] =  current_name_layers[i+shift]
                    else:
                        loading_layers_dict[loading_name_layers[i]] =  current_name_layers[i]
                        
        elif mut_code[0] == 2 :
            layers_to_load = previous_layers #+next_layers        
            for i in range(len(loading_name_layers)):
                if loading_name_layers[i] in layers_to_load:
                    loading_layers_dict[loading_name_layers[i]] =  current_name_layers[i]

        else :
            layers_to_load = loading_name_layers
            for i in range(len(loading_name_layers)):
                loading_layers_dict[loading_name_layers[i]] =  current_name_layers[i]
    

        return layers_to_load, loading_layers_dict
    
    #0:Mutator.learning_rate_mutation,1:Mutator.add_fc, \
                         # 2:Mutator.remove_fc , 3:Mutator.add_conv_layer, 4:Mutator.remove_conv,\
                         # 5:Mutator.add_skip, 6:Mutator.remove_skip, 7:Mutator.add_edgeconv_layer, 8:Mutator.remove_edgeconv,9:Mutator.add_one_to_one,\
                         # 10:Mutator.remove_one_to_one ,11: Mutator.add_attention_layer,12:Mutator.remove_attention, 13:Mutator.regularization_mutation, \
                         # 14:Mutator.replace_mutation,15:Mutator.pool_mutation, 16: Mutator.pool_gep_mutation
    def handle_weights(self, arch_loading,current_arch,name_layers,change_index, mutation_choice):
        
        #pdb.set_trace()
        mutation_dict = defaultdict(lambda: 'default',  {0:'LR_change',1:'add_fc', \
                         2:'remove_fc' , 3:'add_conv', 4:'remove_conv',\
                         5:'add_skip', 6:'remove_skip', 7:'add_edgeconv', 8:'remove_edgeconv',9:'add_one-to-one', 10:'remove_one-to-one' ,\
                         11:'add_attention',12:'remove_attention', 13:'reg_mutation',14:'replace_filter', 15:'pool_mutation', 16:'pool-gep_mutation'}  )
        mutation = mutation_dict[mutation_choice]
        
        #one_change = ['conv','one-to-one','attention', 'fc']
        current_name_layers = self.get_layers_name_for_loading_weights(current_arch)
        loading_name_layers = self.get_layers_name_for_loading_weights(arch_loading)
        shift = len(current_name_layers) - len(loading_name_layers)
        
        if change_index != None:
            change_layer_index = change_index - 1
        else:
            change_layer_index = 0
        layers_to_load, loading_layers_dict = self.get_loading_layers(mutation, loading_name_layers,current_name_layers,change_layer_index, shift)

        return layers_to_load, loading_layers_dict
    
    # print_ext can be disabled through the silent flag
    def print_ext(self, *args):
        if self.silent == False:
            print_ext(*args)
            
    # Will retrieve the value stored as the maximum test accuracy on a trained network
    # SHOULD ONLY BE USED IF test_batch_size == ALL TEST SAMPLES
    def get_max_accuracy(self):
        tf.reset_default_graph()
        with tf.variable_scope('loss') as scope:
            max_acc_test = tf.Variable(tf.zeros([]), name="max_acc_test")
        saver = tf.train.Saver()
        with tf.Session() as sess:
            max_it = self.load_model(sess, saver)
            return sess.run(max_acc_test), max_it
        		# load_model if any checkpoint exist
    
        
    # Run all folds in a CV and calculate mean/std
    def run_kfold_experiments(self, no_folds=10):
        acc = []
        self.net.is_training = tf.placeholder(tf.bool, shape=())
        inputData = self.create_data()
        self.net_constructor.create_network(self.net_desc, [])
        desc = self.net_desc.get_description()
        self.print_ext('Running CV for:', desc)
        start_time = time.time()
        for i in range(no_folds):
            tf.reset_default_graph()
            self.set_kfold(no_folds=no_folds, fold_id=i)
            cur_max, max_it = self.run()
            self.print_ext('Fold %d max accuracy: %g at %d' % (i, cur_max, max_it))
            acc.append(cur_max)
        acc = np.array(acc)
        mean_acc= np.mean(acc)*100
        std_acc = np.std(acc)*100
        self.print_ext('Result is: %.2f (+- %.2f)' % (mean_acc, std_acc))
        
        verify_dir_exists('./results/')
        with open('./results/%s.txt' % self.dataset_name, 'a+') as file:
            file.write('%s\t%s\t%d-fold\t%d seconds\t%.2f (+- %.2f)\n' % (str(datetime.now()), desc, no_folds, time.time()-start_time, mean_acc, std_acc))
        return mean_acc, std_acc
        
    # Create CV information
    def set_kfold(self, no_folds = 10, fold_id = 0):
        inst = KFold(n_splits = no_folds, shuffle=True, random_state=125)
        self.fold_id = fold_id
        
        self.KFolds = list(inst.split(np.arange(self.no_samples)))
        self.train_idx, self.test_idx = self.KFolds[fold_id]
        self.no_samples_train = self.train_idx.shape[0]
        self.no_samples_test = self.test_idx.shape[0]
        self.print_ext('Data ready. no_samples_train:', self.no_samples_train, 'no_samples_test:', self.no_samples_test)
        
        if self.train_batch_size == 0:
            self.train_batch_size = self.no_samples_train
        if self.test_batch_size == 0:
            self.test_batch_size = self.no_samples_test
        self.train_batch_size = min(self.train_batch_size, self.no_samples_train)
        self.test_batch_size = min(self.test_batch_size, self.no_samples_test)
        
    # This function is cropped before batch
    # Slice each sample to improve performance
    def crop_single_sample(self, single_sample):
        vertices = tf.slice(single_sample[0], np.array([0, 0], dtype=np.int64), tf.cast(tf.stack([single_sample[3], -1]), tf.int64))
        vertices.set_shape([None, self.graph_vertices.shape[2]])
        adjacency = tf.slice(single_sample[1], np.array([0, 0, 0], dtype=np.int64), tf.cast(tf.stack([single_sample[3], -1, single_sample[3]]), tf.int64))
        adjacency.set_shape([None, self.graph_adjacency.shape[2], None])
        
        # V, A, labels, mask
        return [vertices, adjacency, single_sample[2], tf.expand_dims(tf.ones(tf.slice(tf.shape(vertices), [0], [1])), axis=-1)]
        
    def create_input_variable(self, input):
        for i in range(len(input)):
            placeholder = tf.placeholder(tf.as_dtype(input[i].dtype), shape=input[i].shape)
            var = tf.Variable(placeholder, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
            self.variable_initialization[placeholder] = input[i]
            input[i] = var
        return input
    # Create input_producers and batch queues
    def create_data(self):
        with tf.device("/cpu:0"):
            with tf.variable_scope('input') as scope:
                # Create the training queue
                with tf.variable_scope('train_data') as scope:
                    self.print_ext('Creating training Tensorflow Tensors')
                    
                    # Create tensor with all training samples
                    training_samples = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]
                    training_samples = [s[self.train_idx, ...] for s in training_samples]
                    
                    if self.crop_if_possible == False:
                        training_samples[3] = get_node_mask(training_samples[3], max_size=self.graph_vertices.shape[1])
                        
                    # Create tf.constants
                    training_samples = self.create_input_variable(training_samples)
                    
                    # Slice first dimension to obtain samples
                    single_sample = tf.train.slice_input_producer(training_samples, shuffle=True, capacity=self.train_batch_size)
                    
                    # Cropping samples improves performance but is not required
                    if self.crop_if_possible:
                        self.print_ext('Cropping smaller graphs')
                        single_sample = self.crop_single_sample(single_sample)
                    
                    # creates training batch queue
                    train_queue = _make_batch_queue(single_sample, capacity=self.train_batch_size*2, num_threads=6)

                # Create the test queue
                with tf.variable_scope('test_data') as scope:
                    self.print_ext('Creating test Tensorflow Tensors')
                    
                    # Create tensor with all test samples
                    test_samples = [self.graph_vertices, self.graph_adjacency, self.graph_labels, self.graph_size]
                    test_samples = [s[self.test_idx, ...] for s in test_samples]
                    
                    # If using mini-batch we will need a queue 
                    if self.test_batch_size != self.no_samples_test:
                        if self.crop_if_possible == False:
                            test_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[1])
                        test_samples = self.create_input_variable(test_samples)
                        
                        single_sample = tf.train.slice_input_producer(test_samples, shuffle=True, capacity=self.test_batch_size)
                        if self.crop_if_possible:
                            single_sample = self.crop_single_sample(single_sample)
                            
                        test_queue = _make_batch_queue(single_sample, capacity=self.test_batch_size*2, num_threads=1)
                        
                    # If using full-batch no need for queues
                    else:
                        test_samples[3] = get_node_mask(test_samples[3], max_size=self.graph_vertices.shape[1])
                        test_samples = self.create_input_variable(test_samples)
                        
                # obtain batch depending on is_training and if test is a queue
                if self.test_batch_size == self.no_samples_test:
                    return tf.cond(self.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_samples)
                return tf.cond(self.net.is_training, lambda: train_queue.dequeue_many(self.train_batch_size), lambda: test_queue.dequeue_many(self.test_batch_size))
    
    
    def zoom(self, x):
        return transforms3d.zooms.zfdir2mat(x)
    def augmentation_data(self, P):
        # pdb.set_trace()
        diameter = tf.reduce_max(tf.reduce_max(P,axis=0) - tf.reduce_min(P,axis=0))
        # M = transforms3d.zooms.zfdir2mat(32/diameter)
        M = tf.py_func(self.zoom, [32/diameter], tf.float32)
        s = random.uniform(1/1.1, 1.1)
        M = tf.tensordot(tf.py_func(self.zoom, [s], tf.float32), M, 1)
        P = tf.tensordot(P, tf.transpose(M), 1)
        return P

    def create_loss_function(self):
        with tf.variable_scope('loss') as scope:
            self.print_ext('Creating loss function and summaries')
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.net.current_V, labels=self.net.labels))
            currentBatchSize = tf.cast(tf.shape(self.net.current_V)[0],tf.float32)
            softmax = tf.nn.softmax(logits=self.net.current_V)
            for weight in self.net.weightList:
                cross_entropy += self.l2reg*tf.nn.l2_loss(weight) + (self.l1reg/currentBatchSize)*tf.reduce_sum(tf.abs(weight))

            #Metric loss
            #cross_entropy += tf.contrib.losses.metric_learning.triplet_semihard_loss(tf.squeeze(tf.argmax(self.net.labels,1)),self.net.embedding)


            correct_prediction = tf.cast(tf.equal(tf.argmax(self.net.current_V, 1), tf.argmax(self.net.labels, 1)), tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
            
            ##Bring back max accuracy monitor for every training batch
            self.max_acc_train = tf.Variable(tf.zeros([]), name="max_acc_train")
            max_acc = tf.assign(self.max_acc_train, tf.maximum(self.max_acc_train, accuracy))
            
            tf.add_to_collection('losses', cross_entropy)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('max_accuracy', max_acc)
            tf.summary.scalar('cross_entropy', cross_entropy)

            #fc1 = tf.Graph.get_tensor_by_name('fc1')
            
            
            self.reports['losses'] = cross_entropy
            self.reports['softmax'] = softmax
            #self.reports['fc1'] = fc1
            self.reports['max acc training'] = max_acc
            self.reports['accuracy'] = accuracy
            self.reports['pred']=correct_prediction
            
    # check if the model has a saved iteration and return the latest iteration step
    def check_model_iteration(self):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None:
            return 0
        return int(latest[len(self.snapshot_path + 'model-'):])
        
    # load_model if any checkpoint exist
    def load_model(self, sess, saver, ):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None:
            return 0
        saver.restore(sess, latest)
        i = int(latest[len(self.snapshot_path + 'model-'):])
        self.print_ext("Model restored at %d." % i)
        return i
        
    def save_model(self, sess, saver, i):
        latest = tf.train.latest_checkpoint(self.snapshot_path)
        if latest == None or i != int(latest[len(self.snapshot_path + 'model-'):]):
            self.print_ext('Saving model at %d' % i)
            verify_dir_exists(self.snapshot_path)
            result = saver.save(sess, self.snapshot_path + 'model', global_step=i)
            self.print_ext('Model saved to %s' % result)


    def load_best_model(self, sess, saver, ):
    
        latest = tf.train.latest_checkpoint(self.best_model_path)
        if latest == None:
            return 0
        saver.restore(sess, latest)
        i = int(latest[len(self.best_model_path + 'model-'):])
        self.print_ext(" Using Best Model. \n Best model restored at %d." % i)
        
        return i
            

    def save_best_model(self, sess, saver, i):
        latest = tf.train.latest_checkpoint(self.best_model_path)
        if latest == None or i != int(latest[len(self.best_model_path + 'model-'):]):
            self.print_ext('Saving best model at %d' % i)
            verify_dir_exists(self.best_model_path)
            result = saver.save(sess, self.best_model_path + 'model', global_step=i)
            self.print_ext('Best model saved to %s' % result)


    def get_layers_name_for_loading_weights(self, arch):
        layers = arch.split(',')
        num_layers = len(layers)
        name_layers = []
        conv_no = 1
        resnet_conv_no = 1
        pool_no = 1
        fc = 1
        rm=1
        gp = 1
        gmp = 1
        identity = 1
        coo_no = 1
        a1d = 1
        ec = 1
        p = 1
        for i in xrange(1, num_layers):
            config = layers[i].split('_')
            if config[0]=='c':
                name_layers.append('conv'+str(conv_no))
                conv_no+=1
            elif config[0]=='fc':
                name_layers.append('fc'+str(fc))
                fc+=1
            elif 'rc' in config[0]:
                if config[0][-1] == '0' :
                    name_layer = 'resnet_conv_block'
                elif config[0][-1] == '1' :
                    name_layer = 'densenet_conv_block'
                name_layers.append(name_layer+str(resnet_conv_no))
                resnet_conv_no+=1
            elif config[0]=='coo':
                name_layers.append('coo'+str(coo_no))
                coo_no+=1
            elif config[0]=='a1d':
                name_layers.append('a1d'+str(a1d))
                a1d+=1
            elif config[0]=='rm':
                name_layers.append('rm'+str(rm))
                rm+=1
            elif config[0]=='gp':
                name_layers.append('gp'+str(gp))
                gp+=1
            elif config[0]=='gmp':
                name_layers.append('gmp'+str(gmp))
                gmp+=1
            elif config[0]=='i':
                name_layers.append('identity'+str(identity))
                identity+=1
            elif config[0]=='ec':
                name_layers.append('ec'+str(ec))
                ec+=1
            elif config[0]=='p':
                name_layers.append('p'+str(p))
                p+=1
        return name_layers
    # Create graph (input, network, loss)
    # Handle checkpoints
    # Report summaries if silent == false
    # start/end threads

    @staticmethod
    def modifySparseTensor():
        setattr(tf.SparseTensor,'static_shape',None)
        setattr(tf.SparseTensor,'set_shape',set_shape)
        setattr(tf.SparseTensor,'get_shape',get_shape)
        
    def run(self):
        self.variable_initialization = {}

        self.modifySparseTensor()
        #pdb.set_trace()
        train_error = 2.0
#        train_loss_dict = {}
        self.print_ext('Training model "%s"!' % self.model_name)
        if hasattr(self, 'fold_id') and self.fold_id:
            self.snapshot_path = '/shared/kgcoe-research/mil/rohand24/snapshots/%s/%s/' % (self.dataset_name, self.model_name + '_fold%d' % self.fold_id)
            self.best_model_path = '/shared/kgcoe-research/mil/rohand24/best_model/%s/%s/' % (self.dataset_name, self.model_name + '_fold%d' % self.fold_id)
            self.test_summary_path = '/shared/kgcoe-research/mil/rohand24/summary/%s/test/%s_fold%d' %(self.dataset_name, self.model_name, self.fold_id)
            self.train_summary_path = '/shared/kgcoe-research/mil/rohand24/summary/%s/train/%s_fold%d' %(self.dataset_name, self.model_name, self.fold_id)
        else:
            self.snapshot_path = '/shared/kgcoe-research/mil/rohand24/snapshots/%s/%s/' % (self.dataset_name, self.model_name)
            self.best_model_path = '/shared/kgcoe-research/mil/rohand24/best_model/%s/%s/' % (self.dataset_name, self.model_name)            
            self.test_summary_path = '/shared/kgcoe-research/mil/rohand24/summary/%s/test/%s' %(self.dataset_name, self.model_name)
            self.train_summary_path = '/shared/kgcoe-research/mil/rohand24/summary/%s/train/%s' %(self.dataset_name, self.model_name)
        if self.debug:
            i = 0
        else:
            i = self.check_model_iteration()
        if self.isTrain:
            if i < self.num_iterations:
                config = tf.ConfigProto(log_device_placement=False)
                config.gpu_options.allow_growth = True

                if self.loading_weights_flag:
                    self.print_ext('Loading pretrained weights from '+self.path_pretrained_weights)
                    if self.path_pretrained_weights:
                        with tf.Session(config=config) as sess:
                            #pdb.set_trace()
                            name_weights_to_load = []
                            saver = tf.train.import_meta_graph(self.path_pretrained_weights+'.meta')
                            saver.restore(sess, self.path_pretrained_weights)
                            name_layers = self.get_layers_name_for_loading_weights(self.arch_loading)
                            new_name_layers, layer_map = self.handle_weights(self.arch_loading, self.current_arch, name_layers, self.change_index, self.mutation_choice)
                            weights_to_load = [v for v in tf.trainable_variables() if any(l in v.name for l in new_name_layers)]
                            weights_map = {}
                            for w in weights_to_load:
                                self.loading_weights[w.name] = sess.run(w)
                                #pdb.set_trace()
                                weights_map[w.name] = w.name.replace(w.name.split('/')[0].encode('ascii'), layer_map[w.name.split('/')[0].encode('ascii')])

                            weights_map = {v: k for k, v in weights_map.iteritems()}
                #Reset graph just loaded
                tf.reset_default_graph()

                                
                self.print_ext('Creating training network')
                self.net.is_training = tf.placeholder(tf.bool, shape=())
                self.net.global_step = tf.Variable(0,name='global_step',trainable=False)
                input, trainDataInitializer, testDataInitializer,shapes = self.create_data()
                counter = 0
                for tensorVal in input:
                    tensorVal.set_shape(shapes[counter])
                    counter += 1
                    print(tensorVal.get_shape())
                self.net_constructor.create_network(self.net, input)
                self.create_loss_function()
                
                self.print_ext('Preparing training')
                loss = tf.add_n(tf.get_collection('losses'))
                if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
                    loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                    
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)          
                
                with tf.control_dependencies(update_ops):
                    if self.optimizer == 'adam':
                        train_step = tf.train.AdamOptimizer().minimize(loss, global_step=self.net.global_step)
                    else:
                        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.net.global_step, self.learning_rate_step, self.learning_rate_exp, staircase=True)
                        train_step = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(loss, global_step=self.net.global_step)
                        self.reports['lr'] = self.learning_rate
                        tf.summary.scalar('learning_rate', self.learning_rate)

                
                
                with tf.Session(config=config) as sess:

                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer(), self.variable_initialization)
                    sess.run(trainDataInitializer)
                    sess.run(testDataInitializer)
                    
                    if self.debug == False:
                        saver = tf.train.Saver(max_to_keep=2)
                        self.load_model(sess, saver)
                                    
                        self.print_ext('Starting summaries')
                        test_writer = tf.summary.FileWriter(self.test_summary_path, sess.graph)
                        train_writer = tf.summary.FileWriter(self.train_summary_path, sess.graph)
                   
                    summary_merged = tf.summary.merge_all()
                
                    self.print_ext('Starting threads')
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                    self.print_ext('Starting training. train_batch_size:', self.train_batch_size, 'test_batch_size:', self.test_batch_size)
                    wasKeyboardInterrupt = False
                    training = True
                    
                    if self.loading_weights_flag:
                        # all_trainable_variables = tf.trainable_variables()
                        #pdb.set_trace()
                        for v in tf.trainable_variables():
                            if (v.name in weights_map.keys()) and (weights_map[v.name] in self.loading_weights):
                                try :
                                    assign_op = v.assign(self.loading_weights[weights_map[v.name]])
                                    sess.run(assign_op)
                                except ValueError:
                                    pass
                                except IndexError:
                                    pass
                                    
                    try:
                        total_training = 0.0
                        total_testing = 0.0
                        start_at = time.time()
                        last_summary = time.time()
                        while i < self.num_iterations:
                            if i % self.snapshot_iter == 0 and self.debug == False:
                                self.save_model(sess, saver, i)

                            if (i>0) and (i % self.iterations_per_test == 0):
                                ##Start validation test
                                collect_pred = np.array([])
                                start_temp = time.time()
                                reports = sess.run(self.reports, feed_dict={self.net.is_training:0})
                                collect_pred = np.append(collect_pred, reports['pred'], axis=0)
                                total_testing += time.time() - start_temp
                                # pdb.set_trace()
                                accuracy = sum(collect_pred)/len(collect_pred)
                                self.print_ext('Test Step %d Finished' % i)
                                self.print_ext('Random batch from validation set accuracy:%.4f' % accuracy)
                                summary = sess.run(summary_merged, feed_dict={self.net.is_training: 0})
                                if self.debug == False:
                                    test_writer.add_summary(summary, i)
                            start_temp = time.time()
                            summary, _, reports = sess.run([summary_merged, train_step, self.reports], feed_dict={self.net.is_training:1})
                            if (i>0) and (i % self.display_iter == 0):
                                if reports['losses'] <= train_error:
                                    self.save_best_model(sess, saver, i)                                 
                                    #if ((i-1)in train_loss_dict) and(reports['losses']<train_loss_dict[i-1]):
                                    train_error = reports['losses']
                            total_training += time.time() - start_temp
                            #train_loss_dict[i]=reports['losses']
                            i += 1
                            if ((i-1) % self.display_iter) == 0:
                                if self.debug == False:
                                    train_writer.add_summary(summary, i-1)
                                total = time.time() - start_at
                                self.print_ext('Training Step %d Finished Timing (Training: %g, Test: %g) after %g seconds' % (i-1, total_training/total, total_testing/total, time.time()-last_summary)) 
                                for key, value in reports.items():
                                    if key != 'pred' and key != 'softmax':
                                        self.print_ext('Training Step %d "%s" = ' % (i-1, key), value)
                                last_summary = time.time()            
                            if (i-1) % 100 == 0:
                                total_training = 0.0
                                total_testing = 0.0
                                start_at = time.time()
                        if i % self.iterations_per_test == 0:
                            summary = sess.run(summary_merged, feed_dict={self.net.is_training:0})
                            if self.debug == False:
                                test_writer.add_summary(summary, i)
                            self.print_ext('Test Step %d Finished' % i)
                    except KeyboardInterrupt as err:
                        self.print_ext('Training interrupted at %d' % i)
                        wasKeyboardInterrupt = True
                        raisedEx = err
                    finally:
                        if i > 0 and self.debug == False:
                            self.save_model(sess, saver, i)
                        self.print_ext('Training completed, starting cleanup!')
                        coord.request_stop()
                        coord.join(threads)
                        self.print_ext('Cleanup completed!')
                        if wasKeyboardInterrupt:
                            raise raisedEx
                    self.print_ext('Max accuracy among all training batches, global_step')
                    return sess.run([self.max_acc_train, self.net.global_step])

            else:
                return self.run_test()
        else:
             return self.run_test()    

    def run_test(self):
        self.modifySparseTensor()
        self.print_ext('Creating testing network')
        tf.reset_default_graph()
        self.net.is_training = tf.placeholder(tf.bool, shape=())
        self.net.global_step = tf.Variable(0,name='global_step',trainable=False)
        
        input, dataInitializer, shapes = self.create_data_test()
        counter = 0
        for tensorVal in input:
            tensorVal.set_shape(shapes[counter])
            counter += 1
            print(tensorVal.get_shape())
        self.net_constructor.create_network(self.net, input)
        self.create_loss_function()
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        
        collect_pred = np.array([])
        softmaxes = []
        fc1Features = []
        labels = []
        with tf.Session(config=config) as sess:    
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer(), self.variable_initialization)
            sess.run(dataInitializer)

            self.print_ext('Starting threads')   
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)   
            saver = tf.train.Saver()
            max_it = self.load_best_model(sess, saver)
            #fc1FeaturesTensor = sess.graph.get_tensor_by_name('fc1/Relu:0')
            try:
                for step in range(1000):
                    if coord.should_stop():
                        break
                    # reports, input_V, input_A, label = sess.run([self.reports]+input, feed_dict={self.net.is_training:0})
                    #[reports,fc1] = sess.run([self.reports, fc1FeaturesTensor], feed_dict={self.net.is_training: 0})
                    [reports] = sess.run([self.reports], feed_dict={self.net.is_training: 0})
                    #reports = outputs[0]
                    #fc1 = outputs[1]
                    #input_V = outputs[2]
                    #input_A = outputs[3]
                    #label = outputs[4]
                    
                    #labels.append(label)
                    collect_pred = np.append(collect_pred, reports['pred'], axis=0)
                    print('Step {0}'.format(step))
                    #print(fc1.shape)
                    #softmaxes.append(reports['softmax'])
                   # fc1Features.append(fc1)
                    
                    
            except Exception as e:
                # Report exceptions to the coordinator.
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
            self.print_ext('Number of test samples: %d' % collect_pred.shape[0])
            self.print_ext('Testing accuracy.')
            #softmaxes = np.concatenate(softmaxes,axis=0)
            #fc1Features = np.concatenate(fc1Features,axis=0)
            #np.save(datetime.datetime.now().strftime('predictions-%Y%m%d-%H%M%S'),softmaxes)
            #np.save(datetime.datetime.now().strftime(self.dataset_name + '_' + self.model_name + '_fc1Features-%Y%m%d-%H%M%S'),fc1Features)#.datetime.now()#.strftime('predictions-%Y%m%d-%H%M%S'),fc1Features)
            return sum(collect_pred)/len(collect_pred), max_it
