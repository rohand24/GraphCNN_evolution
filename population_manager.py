import sys
sys.path.insert(0, './src/')


from run_protein_test import run_protein_expt, get_dataset
from fitness_checker import *
from population_manager_helper import *

import os
import pdb
import timeit
import argparse
start = timeit.default_timer()


def get_args():
        
        parser = argparse.ArgumentParser(description='Process input architecture')
        parser.add_argument('--cycles', default='10', help='Specifies the total mutation cycles to undergo in current experiment.')
        parser.add_argument('--num_models', default= 5, type=int, help='Specifies the total number of models to be generated for current experiment.')
        parser.add_argument('--dataset_name', default='ENZYMES', help='Specifies the dataset for current experiment.')
        parser.add_argument('--log_path', default= './logs/', help='Specifies the file path to write individual cycle logs.' )
        parser.add_argument('--result_path', default= './results/', help='Specifies the file path to write final results of experiment.' )
        parser.add_argument('--load_lastcycle',default=1, type=int, help='Start experiment from last cycle.')
        parser.add_argument('--folds',default=10, type=int, help='Number of folds of Cross-validation.')
        parser.add_argument('--prob_cycle',default=1, type=int, help='Cycle at which mutation probability is changed.')
        parser.add_argument('--processes',default=5, type=int, help='Number of parallel processes to open.')
        
        
        #Add loading pretrained weights option
        parser.add_argument('--loading_weights_flag', default=0, type=int,help='loading weights flag')
        parser.add_argument('--path_pretrained_weights', default='/shared/kgcoe-research/mil/rohand24/best_model/', help='Path to pretrained weights')
        parser.add_argument('--arch_loading', default='OC,c_16_1_1,c_16_1_1,c_16_1_1,c_16_1_1', help='Specific architecture to load weights from')
        
        parser.add_argument('--arch', default='OC,c_16_1_1,c_16_1_1,c_16_1_1,c_16_1_1,p_16,fc_10_0_0_0', help='Defines the model')
        parser.add_argument('--date', default='Sept02', help='Data run model')
        parser.add_argument('--train_flag', default=1, type=int,help='training flag')
        parser.add_argument('--debug_flag', default=0, type=int,help='debugging flag, if set as true will not save anything to summary writer')
        parser.add_argument('--num_iter', default=10, type=int,help='Number of iterations')
        parser.add_argument('--num_classes', default=10, type=int,help='Number of classes')

        parser.add_argument('--train_batch_size', default=60, type=int,help='Batch size for training')
        parser.add_argument('--test_batch_size', default=50, type=int,help='Batch size for testing')
        parser.add_argument('--snapshot_iter', default=200, type=int,help='Take snapshot each number of iterations')
        parser.add_argument('--starter_learning_rate', default=0.01, type=float,help='Started learning rate')
        parser.add_argument('--learning_rate_step', default=1000, type=int,help='Learning rate step decay')
        parser.add_argument('--learning_rate_exp', default=0.1, type=float,help='Learning rate exponential')
        parser.add_argument('--optimizer', default='adam', help='Choose optimizer type')
        parser.add_argument('--iterations_per_test', default=4000, type=int,help='Test model by validation set each number of iterations')
        parser.add_argument('--display_iter', default=5, type=int,help='Display training info each number of iterations')
        parser.add_argument('--l2',default=0.0,type=float,help="L2 Regularization parameter")
        parser.add_argument('--l1',default=0.0,type=float,help="L1 Regularization parameter")
        parser.add_argument('--pool_ratios',default='1.0_1.0_1.0',help="Ratio of vertex reductions for each pooling")
        parser.add_argument('--separate_batches_flag', default=0, type=int,help='Separate samples to mini batches for protein dataset')
        parser.add_argument('--cluster_alg',default='Lloyd',help='How should pooling cluster vertices?')
        parser.add_argument('--group_name',default='WACV2018',help='Experiment Directory Name')
        parser.add_argument('--trial_name',default='G3DNet18',help='Experiment Directory Name')
        parser.add_argument('--sparse',type=int,default=0,help='Use Sparse Tensors')
        parser.add_argument('--prob_reward', type=float, default = 0.1, help = 'reward multipler for probability. Between [0,1)')
        parser.add_argument('--load_cycle', type=int, default = 50, help = 'Cycle to load pretrained weights from. Ususally the second to last cycle.')
        parser.add_argument('--reset_cycle', type=int, default = 300, help = 'Cycle to reset weights and mutation probabilities')

        args = parser.parse_args()
        
        return args, parser

    
def train_models(args, dataset, trial_name,group_name, parser , reset = 0, old_models = None):    
    
    if reset==0:
        mnames = ['model'+str(i) for i in range(args.num_models)]
        models = {}

        for i in range(args.num_models):
                
            arch_string, output_classes = generate_arch(args.dataset_name)
            args.arch = arch_string
            args.group_name = group_name
            args.trial_name = trial_name+'/'+mnames[i]
            args.num_classes = output_classes
            acc, std, command_string = run_protein_expt(args, dataset, None, None)
            models[mnames[i]] = [str(np.round(acc, decimals=4)),str(np.round(std, decimals=4)),command_string]
        
    else:
        models = {}
        for key in list(old_models.keys()):
            m_string = old_models[key][2]
            m_split = m_string.split(' ')
            new_args, unknown = parser.parse_known_args(args= m_split)
            new_args.loading_weights_flag = 0
            new_args.group_name = group_name
            new_args.processes = args.processes
            new_args.trial_name = trial_name+ '/' + key
            acc, std, command_string = run_protein_expt(new_args, dataset, None, None)
            models[key] = [str(np.round(acc, decimals=4)),str(np.round(std, decimals=4)),command_string]
    
    return models
            
    
    
def main():
    

    args, parser = get_args()
    
    group_name = args.group_name
    trial_name = args.trial_name
    prob_cycle = args.prob_cycle
    reset_cycle = args.reset_cycle
    dataset= get_dataset(args)
    
    if args.load_lastcycle:
        models, last_cycle, mut_prob= load_lastcycle(args)
        print('Loading from cycle %d' %last_cycle)
        total_cycles = range(last_cycle+1,int(args.cycles))#len(mutation_sequence)

    else:    
        total_cycles = range(0,int(args.cycles))#len(mutation_sequence))#int(args.cycles))
        mut_prob = get_mutation_prob(None, 1, None, args.prob_reward)
    mut_prob_changed = list(mut_prob)
    for cycle in total_cycles:
        print('Strating Cycle %d' %cycle)
        args.group_name = group_name + '/cycle_'+ str(cycle)
        
        if (cycle % reset_cycle) == 0:
            if cycle == 0:
                models = train_models(args, dataset, trial_name, args.group_name,parser,reset=0)
                save_cycle(models, args, None, mut_prob)
            else:

                models = train_models(args, dataset, trial_name, args.group_name,parser,reset=1, old_models = models)
                mut_prob = get_mutation_prob(None, 1, None, args.prob_reward)
                save_cycle(models, args, None, mut_prob)
                delete_old_cycle(group_name,cycle)
        else:

            load_cycle = int(cycle-1)
            if (cycle%prob_cycle) == 0:
                mut_prob_changed = list(mut_prob)
            models, changed_model, mutation = mutate_probability(models, dataset,parser, mut_prob_changed ,trial_name, args.group_name, load_cycle, args.path_pretrained_weights, args.processes)#, mutation_sequence[cycle])
            if changed_model is not None:
                mut_prob = get_mutation_prob(mutation, 1, mut_prob, args.prob_reward)
            else:
                mut_prob = get_mutation_prob(mutation, 0, mut_prob, args.prob_reward)
            save_cycle(models,args,mutation, mut_prob)
            copy_checkpoints(models, changed_model, group_name, cycle, trial_name)
            
    metrics = get_metrics(models)
    
    avg_acc,avg_std, max_acc, best_model, best_model_string = metrics

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    result_file = args.result_path + group_name+'-'+trial_name+'-'+'result.txt'
    with open(result_file,'w+')as f:
        f.write('Average Accuracy = %0.4f+-%0.4f\n' % (avg_acc,avg_std))
        f.write('Max Accuracy = %0.4f\n' % (max_acc))
        f.write('Model with Max. Accuracy  is %s \n%s' % (best_model, best_model_string))
    print('Average Accuracy = %0.4f+-%0.4f\n' % (avg_acc,avg_std))
    print('Max Accuracy = %0.4f\n' % (max_acc))
    print('Model with Max. Accuracy  is %s \n%s' % (best_model, best_model_string))
    
 
    
if __name__ == "__main__":
    main()
