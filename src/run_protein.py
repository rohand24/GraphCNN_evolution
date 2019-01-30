import sys
#sys.path.insert(0, './')
import graphcnn.util.proteins.chemical as sc
from graphcnn.experiments.experiment_proteins import *
import argparse
from sklearn.model_selection import KFold
import numpy as np
from protein_to_tensor import load_protein_dataset,preprocess_data
import shlex
from joblib import Parallel, delayed
import multiprocessing as mp

import glob2



def set_kfold(exp, fold_id=0, no_folds=10, separate_batches_flag=0):
    inst = KFold(n_splits=no_folds, shuffle=True, random_state=125)
    exp.fold_id = fold_id

    exp.KFolds = list(inst.split(np.arange(exp.no_samples)))
    exp.train_idx, exp.test_idx = exp.KFolds[fold_id]
    exp.no_samples_train = exp.train_idx.shape[0]
    exp.no_samples_test = exp.test_idx.shape[0]
    exp.print_ext('Data ready. no_samples_train:', exp.no_samples_train, 'no_samples_test:', exp.no_samples_test)

    if exp.train_batch_size == 0:
        exp.train_batch_size = exp.no_samples_train
    if exp.test_batch_size == 0:
        exp.test_batch_size = exp.no_samples_test
    if separate_batches_flag==0:
        exp.train_batch_size = min(exp.train_batch_size, exp.no_samples_train)
        exp.test_batch_size = min(exp.test_batch_size, exp.no_samples_test)
        
def set_single_fold(exp, fold_id=0, no_folds=10, separate_batches_flag=0):
    inst = KFold(n_splits=no_folds, shuffle=True, random_state=None)
    exp.fold_id = fold_id

    exp.KFolds = list(inst.split(np.arange(exp.no_samples)))
    exp.train_idx, exp.test_idx = exp.KFolds[fold_id]
    exp.no_samples_train = exp.train_idx.shape[0]
    exp.no_samples_test = exp.test_idx.shape[0]
    exp.print_ext('Data ready. no_samples_train:', exp.no_samples_train, 'no_samples_test:', exp.no_samples_test)

    if exp.train_batch_size == 0:
        exp.train_batch_size = exp.no_samples_train
    if exp.test_batch_size == 0:
        exp.test_batch_size = exp.no_samples_test
    if separate_batches_flag==0:
        exp.train_batch_size = min(exp.train_batch_size, exp.no_samples_train)
        exp.test_batch_size = min(exp.test_batch_size, exp.no_samples_test)

        
class ProteinExperiment(object):
    def __init__(self, architecture,sparse,numSlices):
        self.arch = architecture.split(',')
        self.num_layers = len(self.arch)
        self.sparse = sparse
        self.numSlices = numSlices

    def create_network(self, net, input):
        net.create_network(input)
        conv_no = 1
        resnet_conv_no = 1
        pool_no = 1
        fc = 1
        gp = 1
        gmp = 1
        ggmp = 1
        identity = 1
        coo_no = 1
        a1d = 1
        aconv = 1
        dn = 1
        dno = 1
        rnq = 1
        p=1
        ec = 1
        convolution_type = self.arch[0]
        if not self.sparse:
            make_conv = net.make_graphcnn_layer
            make_conv_resnet_or_densenet = net.make_graphcnn_resnet_or_densenet_layer
            make_graph_pool = net.make_graph_pooling_layer
            make_graph_maxpool = net.make_graph_maxpooling_layer
        else:
            make_conv = net.make_sparse_graphcnn_layer
            make_conv_resnet_or_densenet = net.make_sparse_graphcnn_resnet_layer
            make_graph_pool = net.make_sparse_graph_pooling_layer
            make_graph_maxpool = net.make_sparse_graph_maxpooling_layer

        
        for i in xrange(1, self.num_layers):
            config = self.arch[i].split('_')
            if config[0]=='c':
                print(conv_no)
                make_conv(int(config[1]),\
                          self.numSlices,\
                          stride=int(config[2]), \
                          order=int(config[3]),\
                          name='conv'+str(conv_no), \
                          with_bn=True, \
                          with_act_func=True)
                conv_no+=1
            if config[0]=='coo':
                net.make_one_by_one_graphcnn_layer(int(config[1]),\
                                                   self.numSlices,\
                                                   stride=int(config[2]),\
                                                   order=int(config[3]),\
                                                   name='coo' + str(coo_no),\
                                                   with_bn=True,
                                                   with_act_func=True)
                coo_no += 1
            elif config[0]=='fc':
                net.make_fc_layer(int(config[1]),name='fc'+str(fc), with_bn=int(config[2]), with_act_func=int(config[3]),is_embedding=config[4])
                fc+=1
            elif 'rc' in config[0]:
                if config[0][-1] == '0' :
                    name_layer = 'resnet_conv_block'
                elif config[0][-1] == '1' :
                    name_layer = 'densenet_conv_block'
                make_conv_resnet_or_densenet(self.numSlices,\
                                             no_filters=[int(j) for j in config[1].split('-')],\
                                             stride=[int(j) for j in config[2].split('-')],\
                                             order=[int(j) for j in config[3].split('-')],\
                                             isOneByOne=[int(j) for j in config[4].split('-')],\
                                             name=name_layer+str(resnet_conv_no),\
                                             with_bn=[bool(j) for j in config[5].split('-')],\
                                             with_act_func=[bool(j) for j in config[6].split('-')])
                resnet_conv_no+=1
            elif config[0] == 'p':
                net.make_graph_embed_pooling(int(config[1]), name='p' + str(p))
                p += 1
            elif config[0]=='gp':
                make_graph_pool(int(config[1]), name='gp' + str(gp))
                gp += 1
            elif config[0]=='gmp':
                make_graph_maxpool(int(config[1]), name='gmp' + str(gp))
                gmp += 1
            elif config[0] == 'i':
                net.make_identity_layer(name='i' + str(identity),with_bn=config[1],with_act_func=config[2])
                identity += 1
            elif config[0] == 'rm':
                net.make_reduce_max_layer(name="REDUCE_MAX")
            elif config[0] == 'a1d':
                net.make_vertex_attention_1d(config[1],name='a1d' + str(a1d))
                a1d += 1
            elif config[0] == 'aconv':
                net.make_vertex_attention_conv(config[1],self.numSlices,name='aconv' + str(aconv))
                aconv += 1
            elif config[0] == 'dn':
                net.make_sparse_densenet_layer(self.numSlices,int(config[1]),int(config[2]),float(config[3]),int(config[4]),name='dn' + str(dn))
                dn += 1
            elif config[0] == 'dno':
                net.make_densenet_one_by_one_layer(self.numSlices,int(config[1]),int(config[2]),name='dno' + str(dno))
                dno += 1
            elif config[0] == 'rnq':
                net.make_sparse_resnet_quick_layer(self.numSlices,int(config[1]),int(config[2]),int(config[3]),name='rnq' + str(rnq))
                rnq += 1
            elif config[0] == 'ec':
                net.make_edge_conv_layer(int(config[1]), name='ec' + str(ec))
                ec += 1

def get_args(parser):

        parser.add_argument('--arch', default='OC,c_16_1_1,ec_128,c_16_1_1,c_16_1_1,fc_6_0_0_0', help='Defines the model')
        parser.add_argument('--date', default='test_load_weights', help='Data run model')
        parser.add_argument('--dataset_name', default='Modelnet10', help='Dataset name')
        parser.add_argument('--train_flag', default=1, type=int,help='training flag')
        parser.add_argument('--debug_flag', default=0, type=int,help='debugging flag, if set as true will not save anything to summary writer')
        parser.add_argument('--num_iter', default=800, type=int,help='Number of iterations')
        parser.add_argument('--num_classes', default=2, type=int,help='Number of classes')
        #Add loading pretrained weights option
        parser.add_argument('--loading_weights_flag', default=0, type=int,help='loading weights flag')
        parser.add_argument('--path_pretrained_weights', default='/shared/kgcoe-research/mil/rohand24/snapshots/', help='Path to pretrained weights')
        parser.add_argument('--arch_loading', default='OC,c_16_1_1,c_16_1_1,c_16_1_1,c_16_1_1', help='Specific architecture to load weights from')
        
        parser.add_argument('--train_batch_size', default=60, type=int,help='Batch size for training')
        parser.add_argument('--test_batch_size', default=50, type=int,help='Batch size for testing')
        parser.add_argument('--snapshot_iter', default=300, type=int,help='Take snapshot each number of iterations')
        parser.add_argument('--starter_learning_rate', default=0.01, type=float,help='Started learning rate')
        parser.add_argument('--learning_rate_step', default=1000, type=int,help='Learning rate step decay')
        parser.add_argument('--learning_rate_exp', default=0.1, type=float,help='Learning rate exponential')
        parser.add_argument('--optimizer', default='adam', help='Choose optimizer type')
        parser.add_argument('--iterations_per_test', default=100, type=int,help='Test model by validation set each number of iterations')
        parser.add_argument('--display_iter', default=5, type=int,help='Display training info each number of iterations')
        parser.add_argument('--l2',default=0.0,type=float,help="L2 Regularization parameter")
        parser.add_argument('--l1',default=0.0,type=float,help="L1 Regularization parameter")
        parser.add_argument('--pool_ratios',default='1.0_1.0_1.0',help="Ratio of vertex reductions for each pooling")
        parser.add_argument('--separate_batches_flag', default=0, type=int,help='Separate samples to mini batches for protein dataset')
        parser.add_argument('--cluster_alg',default='Lloyd',help='How should pooling cluster vertices?')
        parser.add_argument('--group_name',default='WACV2018',help='Experiment Directory Name')
        parser.add_argument('--trial_name',default='test_network',help='Experiment Directory Name')
        parser.add_argument('--sparse',type=int,default=0,help='Use Sparse Tensors')

        args = parser.parse_args()
        
        return args

    
    
    
    
    
def get_exp(args, poolRatiosList, dataset, noSlices,fold_id,change_index, mutation_choice,path_pretrained_weights_dict,no_folds=10):

        exp = GraphCNNProteinExperiment(args.group_name, args.trial_name, ProteinExperiment(args.arch,args.sparse,noSlices), args.num_classes, dataset, args.train_flag, args.l2,
                                        args.l1, \
                                        poolRatiosList,args.sparse, path_pretrained_weights_dict[fold_id],args.arch, change_index, mutation_choice)
        exp.num_iterations = args.num_iter
        exp.optimizer = args.optimizer
        exp.debug = bool(args.debug_flag)

        exp.crop_if_possible = False
        exp.snapshot_iter = args.snapshot_iter
        exp.learning_rate_step = args.learning_rate_step
        exp.starter_learning_rate = args.starter_learning_rate
        exp.learning_rate_exp = args.learning_rate_exp
        exp.loading_weights_flag = bool(args.loading_weights_flag)
        exp.arch_loading=args.arch_loading
        exp.iterations_per_test = args.iterations_per_test
        exp.display_iter = args.display_iter
        exp.train_batch_size = args.train_batch_size
        exp.test_batch_size = args.test_batch_size
        set_kfold(exp, fold_id, no_folds=no_folds,  separate_batches_flag=args.separate_batches_flag)
        
        return exp



def run_fold(args, poolRatiosList, dataset, noSlices,fold_id,no_folds,change_index, mutation_choice, path_pretrained_weights_dict):

    
    exp = get_exp(args, poolRatiosList, dataset, noSlices,fold_id,change_index, mutation_choice,path_pretrained_weights_dict,no_folds)
        
    if args.train_flag:
        exp.run()
        exp = GraphCNNProteinExperiment(args.group_name, args.trial_name, ProteinExperiment(args.arch,args.sparse,noSlices), args.num_classes, dataset, False,
                                            args.l2,
                                            args.l1, \
                                            poolRatiosList,args.sparse, path_pretrained_weights_dict[fold_id],args.arch,change_index, mutation_choice)

        set_kfold(exp, fold_id, no_folds, separate_batches_flag=args.separate_batches_flag)
    cur_max, max_it = exp.run()
    exp.print_ext('Fold %d max accuracy: %g at %d' % (fold_id, cur_max, max_it))
    
    return cur_max
    

def get_dataset(args):

    dataset = load_protein_dataset(args.dataset_name)
    poolRatiosList = [float(x) for x in args.pool_ratios.split('_')]
    dataset = preprocess_data(dataset,poolRatiosList,args.cluster_alg)
    
    return dataset
 
def run_protein_expt(args, dataset, change_index, mutation_choice ):

    no_folds = args.folds
    acc = []
     
    poolRatiosList = [float(x) for x in args.pool_ratios.split('_')]
    noSlices = dataset[3][2]
    path_pretrained_weights_dict = {}
    for i in range(no_folds):
        fold_id = i
        if args.loading_weights_flag ==1:
            if fold_id != 0:
                
                path = args.path_pretrained_weights
                best_model_ckpts = glob2.glob(path+'_fold' + str(fold_id)+'/*.meta')
                best_model_iter = max([int(x.split('/')[-1].split('.')[0].split('-')[-1]) for x in best_model_ckpts])
                path_pretrained_weights_dict[fold_id] = args.path_pretrained_weights+ '_fold' + str(fold_id) + '/model-' + str(best_model_iter)
            else:
                
                path = args.path_pretrained_weights
                best_model_ckpts = glob2.glob(path+'/*.meta')
                best_model_iter = max([int(x.split('/')[-1].split('.')[0].split('-')[-1]) for x in best_model_ckpts])
                path_pretrained_weights_dict[fold_id] = args.path_pretrained_weights+ '/model-' + str(best_model_iter)
        else:
            path_pretrained_weights_dict[fold_id] = args.path_pretrained_weights
        #pdb.set_trace()
        #cur_max = run_fold(args, poolRatiosList, dataset, noSlices,fold_id,no_folds,change_index, mutation_choice,path_pretrained_weights_dict)
        #acc.append(cur_max)
    #pdb.set_trace()
    pool = mp.Pool(args.processes)
    results = [pool.apply_async(run_fold, (args, poolRatiosList, dataset, noSlices,i,no_folds,change_index, mutation_choice, path_pretrained_weights_dict)) for i in range(no_folds)]
    acc = [result.get() for result in results]
    pool.close()
    pool.join()
    
    acc = np.array(acc)
    mean_acc = np.mean(acc) * 100
    std_acc = np.std(acc) * 100
    args_list = ['--'+k+' '+str(v) for k,v in args.__dict__.iteritems()]
    command_string = ' '.join(args_list)
    
    return mean_acc, std_acc, command_string

def main():
    parser = argparse.ArgumentParser(description='Process input architecture')
    args = get_args(parser)
    dataset = get_dataset(args)
    acc1, std1, command_string = run_protein_expt(args, dataset, 2, 4)
    print('Result is: %.4f (+- %.4f)' % (acc1, std1))
    print command_string
    
    return acc1, std1, command_string
    
    
if __name__ == "__main__":
    main()
