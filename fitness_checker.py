import sys
sys.path.insert(0, './src/')

from run_protein import run_protein_expt, get_dataset
from GraphMutator import get_mutation
from GraphMutator_helper import load_weights, get_layer_indices, get_mutation_dict
from population_manager_helper import get_arch_string, check_pool_change

import os
import numpy as np
import random
import pdb
import argparse


def print_stats(models, mutate_model, replace_model, final_string, acc,mutation, check = True):

        b_model_arch = get_arch_string(models[mutate_model][2])
        w_model_arch = get_arch_string(models[replace_model][2])
        m_model_arch = get_arch_string(final_string)
        mut_str= get_mutation_dict()[mutation]
        print('From Randomly selected 2 models...')
        print('Accuracy of \'better model\' with archictecture %s  = %s' % (b_model_arch,models[mutate_model][0]))
        print('Accuracy of \'worse model\' with archictecture %s  = %s' % (w_model_arch,models[replace_model][0]))
        print('Accuracy of \'better model\'  with archictecture %s after mutation  = %0.4f' % (m_model_arch,acc))
        print('Mutation is: %s'%(mut_str))

        if check:
            print('Replaced model is: %s' % replace_model)
            print('Final String of mutated model is:')
            print(final_string)
            print('######################################################################')
            
        else:
            print('No replacement')
            print('Mutation did not increase accuracy')

def mutate_probability(models,dataset,parser, mut_prob ,trial_name, group_name, load_cycle, path_pretrained_weights, processes):

        selected_models = random.sample(list(models), 2)
        if float(models[selected_models[0]][0])>float(models[selected_models[1]][0]):
            mutate_model = selected_models[0]
            replace_model = selected_models[1]
        else:
            mutate_model = selected_models[1]
            replace_model = selected_models[0]


        b_model_arch = get_arch_string(models[mutate_model][2])
        w_model_arch = get_arch_string(models[replace_model][2])
        print('Model to mutate is: %s \narchitecture : %s' % (mutate_model, b_model_arch))
        print('Model to replace is: %s \narchitecture : %s' % (replace_model, w_model_arch))        
        
        model_string = models[mutate_model][2]
        success_flag = 0
        while not success_flag :
            mutation_choice = np.random.choice(list(range(0,17)), p = mut_prob)
            old_string,final_string,success_flag, change_index = get_mutation(model_string,mutation_choice)
                    
        new_args, dataset = check_pool_change(old_string, final_string, parser, dataset, group_name, trial_name, replace_model)
        
        path_pretrained_weights = path_pretrained_weights +  new_args.group_name.split('/')[0] +'/' + 'cycle_'+str(load_cycle) +'/' + new_args.trial_name.split('/')[0] +'/' + mutate_model
        new_args.path_pretrained_weights = path_pretrained_weights
        new_args.arch_loading = b_model_arch
        new_args.loading_weights_flag = 1
        new_args.processes = processes
        arch_list = new_args.arch.split(',')
        gep_idx = get_layer_indices(arch_list, 'p')
        if len(gep_idx)>0:
            new_args.sparse = 0
        else:
            new_args.sparse = 1
        
        acc, std, final_string = run_protein_expt(new_args, dataset, change_index, mutation_choice)
        
        if acc> float(models[replace_model][0]):
            print_stats(models, mutate_model, replace_model, final_string, acc, mutation_choice, check = True)
            models[replace_model] = [str(np.round(acc, decimals=4)),str(np.round(std, decimals=4)),final_string]

            return models, replace_model, mutation_choice
        else:
            
            print_stats(models, mutate_model, replace_model, final_string, acc, mutation_choice,check = False)
            return models, None, mutation_choice


def test_and_mutate(models,dataset,parser, trial_name, group_name):#,choice):
    
    	#pdb.set_trace()
        selected_models = random.sample(list(models), 2)
        if models[selected_models[0]][0]>models[selected_models[1]][0]:
            mutate_model = selected_models[0]
            replace_model = selected_models[1]
        else:
            mutate_model = selected_models[1]
            replace_model = selected_models[0]
        
        b_model_arch = get_arch_string(models[mutate_model][2])
        w_model_arch = get_arch_string(models[replace_model][2])
        print('Model to mutate is: %s \narchitecture : %s' % (mutate_model, b_model_arch))
        print('Model to replace is: %s \narchitecture : %s' % (replace_model, w_model_arch)) 
        
        model_string = models[mutate_model][2]
        success_flag = 0
        while not success_flag :
            mutation_choice = np.random.randint(0,high =12)
            old_string,final_string,success_flag, index = get_mutation(model_string,mutation_choice)
        
        old_split = old_string.split(' ')
        final_split = final_string.split(' ')
        new_args, unknown = parser.parse_known_args(args= final_split)
        
        new_args.group_name = group_name
        new_args.trial_name = trial_name+ '/' + replace_model
        if ('--pool_ratios' not in old_split) and ('--pool_ratios' in final_split):
            dataset = get_dataset(new_args)
        if '--pool_ratios' in old_split:
            if not (old_split[old_split.index('--pool_ratios')+1] == final_split[final_split.index('--pool_ratios')+1]):
                dataset = get_dataset(new_args)    
        acc, std, final_string = run_protein_expt(new_args, dataset)
        
        models[replace_model] = [str(np.round(acc, decimals=4)),str(np.round(std, decimals=4)),final_string]

        print_stats(models, mutate_model, replace_model, final_string, acc, check = True)
        
        return models, replace_model
        
def test_mutate_check(models,dataset,parser, trial_name, group_name):#,choice):
    
    	#pdb.set_trace()
        selected_models = random.sample(list(models), 2)
        if float(models[selected_models[0]][0])>float(models[selected_models[1]][0]):
            mutate_model = selected_models[0]
            replace_model = selected_models[1]
        else:
            mutate_model = selected_models[1]
            replace_model = selected_models[0]
        
        b_model_arch = get_arch_string(models[mutate_model][2])
        w_model_arch = get_arch_string(models[replace_model][2])
        print('Model to mutate is: %s \narchitecture : %s' % (mutate_model, b_model_arch))
        print('Model to replace is: %s \narchitecture : %s' % (replace_model, w_model_arch)) 
        
        model_string = models[mutate_model][2]
        success_flag = 0
        while not success_flag :
            mutation_choice = np.random.randint(0,high =11)
            old_string,final_string,success_flag, index = get_mutation(model_string,mutation_choice)
        
        new_args, dataset = check_pool_change(old_string, final_string, parser, dataset, group_name, trial_name, replace_model)
        acc, std, final_string = run_protein_expt(new_args, dataset)
        
        if acc> float(models[replace_model][0]):
            print_stats(models, mutate_model, replace_model, final_string, acc, check = True)
            models[replace_model] = [str(np.round(acc, decimals=4)),str(np.round(std, decimals=4)),final_string]

            return models, replace_model , mutation_choice
        else:
            
            print_stats(models, mutate_model, replace_model, final_string, acc, check = False)
            return models, None , mutation_choice