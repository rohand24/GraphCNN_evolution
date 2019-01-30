import sys
sys.path.insert(0, './src/')
import numpy as np
import os
import random
import tensorflow as tf
import time
import pdb

from collections import defaultdict

def get_layer_indices(arch_list, key):
    indices= []
    for idx,n in enumerate(arch_list):
        layer_split = n.split('_')
        if layer_split[0]==key:
            indices.append(idx)
            
    return indices
    

def mutation_name_dict():

    mutation_dict = defaultdict(lambda: 'defalut',  {0:'LR',1:'add_fc', \
                         2:'remove_fc' , 3:'add_conv', 4:'remove_conv',\
                         5:'add_skip', 6:'remove_skip', 7:'pool_mutation', 8:'add_one-to-one', 9:'remove_one-to-one' ,\
                         10:'add_attention',11:'remove_attention'}  )
                           
    return mutation_dict
    
def modify_cmd_list(cmd_split, modify_arg, modify_value):

    if modify_arg in cmd_split:
        cmd_split[cmd_split.index(modify_arg)+1] = modify_value
    else:
        cmd_split.append(modify_arg)
        cmd_split.insert((cmd_split.index(modify_arg)+1),modify_value)
        
    return cmd_split
    
def order_pooling(insert_index,gp_idx,arch_list,pr_dict):
    
    pool_ratio_value = round(np.random.uniform(0,1),2)
                      
    gp_idx.append(insert_index)
    gp_idx.sort()
    gp_idx = [i+1 if i>insert_index else i for i in gp_idx]
    for k,v in pr_dict.items():
        if k>insert_index:
            pr_dict[k+1]=pr_dict[k]
            del pr_dict[k]                         
    pr_dict[insert_index] = pool_ratio_value

    for i in range(len(gp_idx)):
        arch_list[gp_idx[i]] = 'gmp_'+str(i)
                        
    pr_string = []
    for key in sorted(list(pr_dict.keys())):
        pr_string.append(str(pr_dict[key]))
                                    
    pr_string = '_'.join(pr_string)
    
    return arch_list, pr_string


def get_output_classes(dataset_name):

    if dataset_name == 'modelnet10':
        output_classes = 10
    elif dataset_name == 'modelnet40':
        output_classes = 40
    elif dataset_name == 'modelnetfull':
        output_classes = 421
    elif dataset_name == 'shapenetcore':
        output_classes = 55
    elif dataset_name == 'sydney': #in ['sydney0','sydney1','sydney2','sydney3']:
        output_classes = 14
    elif dataset_name in ['MUTAG','NCI1']:
        output_classes = 2
    elif dataset_name == 'ENZYMES':
        output_classes = 6

    return output_classes
    
def make_layer_string(type, units, rcunits = None):
        
        bn = np.random.randint(2)#batch_norm
        act = np.random.randint(2)#act_function
        emb = np.random.randint(2)# embedding
        stride = np.random.randint(2)#1,high=6)
        order = np.random.randint(2)#1,high=6)
        onebyone = np.random.randint(2)
        rctype = np.random.randint(2) # 0=resnet/1=densenet
        
        def get_rc_group(rcsize=2):
            
            return '-'.join(str(e) for e in (np.random.randint(2, size=rcsize)))
        
        if units !=None:
            if type=='fc':
                lstring = '_'.join([type,str(units), str(np.random.randint(2)),str(np.random.randint(2)),str(np.random.randint(2)) ])   # 'fc' + str(units) + '_' + str(bn) + '_' + str(act) + '_' + str(emb)
            elif type=='c':
                lstring = '_'.join([type,str(units), str(1), str(1) ])#str(np.random.randint(2)),str(np.random.randint(2))])   #'c_'+ str(units) + '_' + str(stride) + '_' + str(order)
            elif type=='coo':
                lstring = '_'.join([type,str(units), str(1), str(1) ])#str(np.random.randint(2)),str(np.random.randint(2))]) #'coo_'+ str(units) + '_' + str(stride) + '_' + str(order)
            elif type=='a1d':
                lstring = '_'.join([type,str(units)])#str(np.random.randint(2)),str(np.random.randint(2))]) #'a1d_'+ str(units)
            elif type=='ec':
                lstring = '_'.join([type,str(units)])
        else:
            if type=='rc':

                onebyone = np.random.randint(2)
                rc_unit_string = '-'.join(str(e) for e in (rcunits))
                lstring = '_'.join([type+str(np.random.randint(2)), rc_unit_string, str('1-1'), str('1-1'),str(onebyone)+'-'+str(1-onebyone), str('1-1'), str('1-1')])  #get_rc_group(len(rcunits)), get_rc_group(len(rcunits)) ]) 
            
        return lstring


def replace_filter(arch_list, idx_list ,filter_list):
    
    
    replace_index = np.random.choice(idx_list)
    replace_layer = arch_list[replace_index]
    rp_layer_split = replace_layer.split('_')
    modified_filter_list = list(filter_list)
    modified_filter_list.remove(int(rp_layer_split[1]))
    replace_value = np.random.choice(modified_filter_list)
    print('Replacing Conv layer with filter %d with %d' % (int(rp_layer_split[1]), int(replace_value)))
    del rp_layer_split[1]
    rp_layer_split.insert(1,str(replace_value))
    arch_list[replace_index] = ('_').join(rp_layer_split)

    return arch_list , replace_index

def get_mutation_dict():
    mut_dict = {0:'LR_change',1:'add_fc', \
                         2:'remove_fc' , 3:'add_conv', 4:'remove_conv',\
                         5:'add_skip', 6:'remove_skip', 7:'add_edgeconv', 8:'remove_edgeconv',9:'add_one-to-one', 10:'remove_one-to-one' ,\
                         11:'add_attention',12:'remove_attention', 13:'reg_mutation',14:'replace_filter', 15:'pool_mutation', 16:'pool-gep_mutation'}
                         
    return mut_dict
    
    
    
