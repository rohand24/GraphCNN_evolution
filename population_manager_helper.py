import sys
sys.path.insert(0, './src/')


from run_protein import run_protein_expt, get_dataset
from GraphMutator_helper import make_layer_string, get_output_classes

import os
import numpy as np
import glob2
import shutil
from distutils.dir_util import copy_tree
from collections import defaultdict
import pdb
import json
import csv
import timeit
import argparse
start = timeit.default_timer()

def get_arch_string(command_string):
            
        cmd_split = command_string.split(' ')
        arch_string = cmd_split[cmd_split.index('--arch')+1]
        
        return arch_string

def generate_arch(dataset_name):
    
    output_classes = get_output_classes(dataset_name)
                    
    num_nodes = np.random.randint(10,500)
    layer1 = make_layer_string('fc',num_nodes)
    layer2 = 'fc_'+str(output_classes) + '_0_0_0' #make_layer_string('fc',output_classes)
    arch_string = ','.join(['OC', layer1, layer2])    #
    
    return arch_string, output_classes

def load_lastcycle(args):
    
    group_name = args.group_name.split('/')
    log_path = args.log_path + group_name[0]
    log_file = (sorted(glob2.glob(log_path+'/*.csv'), key=os.path.getmtime))[-1]
    mut_file = log_path +'/mutation_log.txt'
    last_cycle = int(log_file.split('/')[-1].split('_')[-1].split('.')[0])
    ascii_models={}
    with open(log_file, 'r+') as f:
        models = (json.load(f))
    for k,v in models.iteritems():
        k = k.encode('ascii')
        v = [i.encode('ascii') for i in v]
        ascii_models[k]=v
    with open(mut_file,'r+') as f:
        lines = f.readlines()
        mut_prob_str = lines[-1].strip().split('\t')
        mut_prob = [float(i) for i in mut_prob_str]
    
    return ascii_models, last_cycle, mut_prob
        
def save_cycle(models, args, mutation, mut_prob):

    
    group_name = args.group_name.split('/')
    log_path = args.log_path + group_name[0]
    log_file = log_path+'/'+group_name[1]+'.csv'
    mut_file = log_path +'/mutation_log.txt'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(log_file, 'w+') as f:
        json.dump(models,f,indent=0)
    with open(mut_file,'a+') as f:
        if mutation is None:
            f.write('%s is only Training. No Mutation.\n' % group_name[1])
            for prob in mut_prob:
                f.write('%s\t'%str(prob))
            f.write('\n')
        else:    
            f.write('Mutation for %s is %d\n' % (group_name[1], mutation))
            for prob in mut_prob:
                f.write('%s\t'%str(prob))
            f.write('\n')
        
        
def get_metrics(models):
    
    
    acc= [float(v[0]) for v in models.values()]
    std = [float(v[1]) for v in models.values()]
    avg_acc = np.mean(acc)
    avg_std = np.mean(std)
    max_acc = np.max(acc)
    for key, value in models.iteritems():
        if value[0] == str(max_acc):
            best_model = key
            best_model_string = value[2]
    
    metrics = (avg_acc,avg_std, max_acc, best_model, best_model_string)
    
    return metrics

def delete_old_cycle(group_name, cycle):

    old_cycle = '/shared/kgcoe-research/mil/rohand24/best_model/%s/cycle_%d' % (group_name, cycle-2)
    old_snapshot = '/shared/kgcoe-research/mil/rohand24/snapshots/%s/cycle_%d' % (group_name, cycle-2)
    if os.path.exists(old_cycle):
        shutil.rmtree(old_cycle)
    if os.path.exists(old_snapshot):
        shutil.rmtree(old_snapshot)
    print('Cycle %d deleted!!!' %(cycle-2)) 
 
def copy_checkpoints(models, changed_model, group_name, cycle, trial_name):

            
            unchanged_models = list(models)
            delete_old_cycle(group_name,cycle)
            if changed_model is not None:
                unchanged_models.remove(changed_model)
            previous_ckpts = glob2.glob('/shared/kgcoe-research/mil/rohand24/best_model/'+group_name+'/cycle_'+str(cycle-1)+'/' +trial_name+'/*')
            new_ckpt_path = '/shared/kgcoe-research/mil/rohand24/best_model/'+group_name+'/cycle_'+str(cycle)+'/' +trial_name+'/'
            if changed_model == None:
                shutil.rmtree(new_ckpt_path)
                if not os.path.exists(new_ckpt_path):
                    os.makedirs(new_ckpt_path)
            for ckpt in previous_ckpts:
                for umodel in unchanged_models:
                    if umodel in ckpt:
                        mname = ckpt.split('/')[-1]
                        new_ckpt = new_ckpt_path+mname+'/'
                        copy_tree(ckpt,new_ckpt)
            print('Copied checkpoints from cycle %d to cycle %d' %(cycle-1,cycle))
            
                
def get_mutation_prob(mutation, isHelpful, mut_prob = None, reward = 0.1):

    mut_list = list(range(0,17))
    m_prob = mut_prob
    if mutation == None:
        m_prob = [1.0/len(mut_list)]*len(mut_list)
    else:
        if isHelpful:
            m_prob[mutation] = m_prob[mutation]*(1+ reward)
        else:
            m_prob[mutation] = m_prob[mutation]*(1-reward)
        m_prob = [float(i)/sum(m_prob) for i in m_prob]
    
    return m_prob
                
def check_pool_change(old_string, final_string, parser, dataset, group_name, trial_name,replace_model):
    
    old_split = old_string.split(' ')
    final_split = final_string.split(' ')
    arch_loading = old_split[old_split.index('--arch')+1]
    new_args, unknown = parser.parse_known_args(args= final_split)
        
    new_args.group_name = group_name
    new_args.trial_name = trial_name+ '/' + replace_model
    if ('--pool_ratios' not in old_split) and ('--pool_ratios' in final_split):
        dataset = get_dataset(new_args)
    if '--pool_ratios' in old_split:
        if not (old_split[old_split.index('--pool_ratios')+1] == final_split[final_split.index('--pool_ratios')+1]):
            dataset = get_dataset(new_args) 
                
                
    return new_args, dataset
    
def get_model_history_parameters(command_string):
    
    evolution_params = {}
    cmd_split = command_string.split(' ')
    m_arch = cmd_split[cmd_split.index('--arch')+1]
    evolution_params['arch'] = m_arch
    lr_params = {}
    lr_params['--starter_learning_rate']= cmd_split[cmd_split.index('--starter_learning_rate')+1]
    lr_params['--learning_rate_step']= cmd_split[cmd_split.index('--learning_rate_step')+1]
    lr_params['--learning_rate_exp'] = cmd_split[cmd_split.index('--learning_rate_exp')+1]
    reg_params = {}
    reg_params['l1']= cmd_split[cmd_split.index('--l1')+1]
    reg_params['l2']= cmd_split[cmd_split.index('--l2')+1]
    
    evolution_params['lr'] = lr_params
    evolution_params['reg']=reg_params
    evolution_params['num_iter'] = cmd_split[cmd_split.index('--num_iter')+1]
    
    return evolution_params
    
def model_history_init(args, models):

    
    group_name = args.group_name.split('/')
    history_file = args.history_path+group_name[0]+'_model_history.csv'
    

    model_history_dict = {}
    for mname in list(models):
        evolution_params = get_model_history_parameters(models[mname][2])
        model_history_dict[mname] = [evolution_params]
    with open(history_file, 'w+') as f:
        json.dump(model_history_dict,f,indent=0)
            
    
def model_history_update(args, models, mutated_model, replaced_model):    
    
    
    group_name = args.group_name.split('/')
    history_file = args.history_path+group_name[0]+'_model_history.csv'
    model_history_dict = {}
           
    with open(history_file, 'r+') as f:
        model_hist = (json.load(f))
    for k,v in model_hist.iteritems():
        k = k.encode('ascii')
        v = [i.encode('ascii') for i in v]
        model_history_dict[k]=v
    replaced_model_params = get_model_history_parameters(models[replaced_model][2])
    mutated_model_params = model_history_dict[mutated_model]
    mutated_model_params.append(replaced_model_params)
    model_history_dict[replaced_model] =  mutated_model_params
    with open(history_file, 'w+') as f:
        json.dump(model_history_dict,f,indent=0)