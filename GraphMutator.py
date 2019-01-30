import sys
sys.path.insert(0, './src/')

from run_protein import run_protein_expt, get_dataset
from GraphMutator_helper import *

import numpy as np
import os
import random
import tensorflow as tf
import time
import pdb
import argparse

from collections import defaultdict

success_flag = 0

class Mutator(object):

    def __init__(self, command_string):
        pass
    #########################################################################################
    # This function performs the mutation of changing the direction by randomly choosing    #
    # a value for theta and phi. The range of the values are defined by the ones from the   #
    # slide.                                                                                #
    #########################################################################################
    
    def n_directions(self, old_string):
        print("Changing directions")
        command_string_cpy = old_string
        cmd_split = old_string.split(' ')
        
        if '--feature_type' in cmd_split:
            ft_ix = cmd_split.index('--feature_type')
            cmd_split[ft_ix+1] = str(0)
            command_string = ' '.join(cmd_split)
        else:
            command_string= old_string + " --feature_type " + str(0)

        if '--theta' in cmd_split:
            theta_ix = cmd_split.index('--theta')
            cmd_split[theta_ix+1] = str(int(np.random.uniform(1,2*np.pi+1)))
            command_string = ' '.join(cmd_split)
        else:
            theta = int(np.random.uniform(1,2*np.pi+1))
            command_string= old_string + " --theta " + str(theta)
        
        if '--phi' in cmd_split:
            phi_ix = cmd_split.index('--phi')
            cmd_split[phi_ix+1] = str(int(np.random.uniform(1,np.pi+1)))
            command_string = ' '.join(cmd_split)
        else:
            phi = int (np.random.uniform(1,np.pi+1))
            command_string= old_string + " --phi " + str(phi)

        success_flag = 1
        return old_string, command_string, success_flag, None

    #########################################################################################
    # This function performs the mutation of changing the value of K by randomly choosing   #
    # a value from the array.                                                               #
    #########################################################################################

    def n_neighbours(self, old_string):
        print("Changing K value for knn graph.")
        command_string_cpy = old_string
        cmd_split = old_string.split(' ')
        k = [2,3,6,8] #Example of different values of K.
        
        if '--K' in cmd_split:
            k_ix = cmd_split.index('--K')
            cmd_split[k_ix+1] = str(k[np.random.randint(4)])
            command_string = ' '.join(cmd_split)
        else:
            k_mutated =  k[np.random.randint(4)]
            command_string= old_string + " --K " + str(k_mutated)
        success_flag = 1
        return old_string, command_string, success_flag, None

    #########################################################################################
    # This function performs the mutation of randomly performing an augmentation operation. #
    # This can be either dropout, flip, rotate, any 2 or all 3.                             #
    #########################################################################################

    def agg_augment(self, old_string):
        print("Performing Augmentation")
        augment_selector = np.random.choice(2,3)
        cmd_split = old_string.split(' ')
        
        if np.array_equal(augment_selector,[0,0,0]):
            print ("No Augmentations")
            success_flag = 1
             
        if augment_selector[0]:
            #add dropout 
            print ("Adding dropout")
            dropout_amnt = round(np.random.uniform(0,1),2)
            cmd_split = modify_cmd_list(cmd_split, '--dropout', str(dropout_amnt))
            success_flag = 1
        
        if augment_selector[1]:
            #add rotate 
            print ("Adding Rotation")
            rotate_intesity = np.random.randint(-15,15)
            cmd_split = modify_cmd_list(cmd_split, '--rotate', str(rotate_intesity))
            success_flag = 1
            
        if augment_selector[2]:
            #add flip
            print ("Adding flip")
            flip_flag = np.random.randint(2)
            cmd_split = modify_cmd_list(cmd_split, '--flip', str(flip_flag))
            success_flag = 1
        command_string = ' '.join(cmd_split)
        
        return old_string, command_string, success_flag, None

    def regularization_mutation(self, old_string):
    
        print("Adding Regularization.")
        reg_selector = np.random.choice(2,2)
        cmd_split = old_string.split(' ')
        
        if np.array_equal(reg_selector,[0,0]):
            print ("No Regularization added.")
            success_flag = 0
        
        if reg_selector[0]:
            print("Changing L1 regularization")
            if '--l1' in cmd_split:
                ix = cmd_split.index('--l1')
                old_l1 = cmd_split[ix+1]
                old_l1 = float(old_l1)
                if old_l1 == 0.0:
                    new_l1 = np.random.uniform(0,0.001)
                else:
                    new_l1 = np.random.uniform(old_l1*0.5,2*old_l1)
                print('Old L1 = %0.6f'%old_l1)
                print('New L1 = %0.6f'%new_l1)
                cmd_split[ix+1] = str(new_l1)
            else:
                cmd_split.append('--l1')
                cmd_split.append(str(round(np.random.uniform(0,0.001),4)))
            success_flag = 1
            
        if reg_selector[1]:
            print("Changing L2 regularization")
            if '--l2' in cmd_split:
                ix = cmd_split.index('--l2')
                old_l2 = cmd_split[ix+1]
                old_l2 = float(old_l2)
                if old_l2 == 0.0:
                    new_l2 = np.random.uniform(0,0.001)
                else:
                    new_l2 = np.random.uniform(old_l2*0.5,2*old_l2)
                print('Old L2 = %0.6f'%old_l2)
                print('New L2 = %0.6f'%new_l2)
                cmd_split[ix+1] = str(new_l2)
            else:
                cmd_split.append('--l2')
                cmd_split.append(str(round(np.random.uniform(0,0.001),4)))
            success_flag = 1
        command_string = ' '.join(cmd_split)
        
        return old_string, command_string, success_flag, None
        
    ###########################################################################################
    # This mutation alters the LR using a random uniform distribution.                        #
    ###########################################################################################
    def learning_rate_mutation(self, old_string):
        
        
        print("Performing Learning rate mutation")
        augment_selector = np.random.choice(2,3)
        cmd_split = old_string.split(' ')
        
        if np.array_equal(augment_selector,[0,0,0]):
            print ("No learning rate mutation.")
            success_flag = 0
             
        if augment_selector[0]:
            #Change LR 
            print ("Changing Learning rate")
            if '--starter_learning_rate' in cmd_split:
                ix = cmd_split.index('--starter_learning_rate')
                old_lr = cmd_split[ix+1]
                old_lr = float (old_lr)
                new_lr = np.random.uniform(old_lr*0.5,2*old_lr)
                print('Old LR = %0.6f'%old_lr)
                print('New LR = %0.6f'%new_lr)
                cmd_split[ix+1] = str(new_lr)
            else:
                cmd_split.append('--starter_learning_rate')
                cmd_split.append(str(round(np.random.uniform(0,1),4)))
            success_flag = 1
        
        if augment_selector[1]:
            #Change LR step
            print ("Changing Learning rate step")
            iter = int(cmd_split[cmd_split.index('--num_iter')+1])
            if '--learning_rate_step' in cmd_split:
                ix = cmd_split.index('--learning_rate_step')                
                old_lrs = cmd_split[ix+1]
                old_lrs = int(old_lrs)
                new_lrs = int(np.random.uniform(old_lrs*0.5,2*old_lrs))
                if new_lrs>(0.9*iter):
                    new_lrs = int(0.5*iter)
                print('Old LR step = %d'%old_lrs)
                print('New LR step = %d'%new_lrs)
                cmd_split[ix+1] = str(new_lrs)
            else:
                cmd_split.append('--learning_rate_step')
                cmd_split.append(str(round(np.random.randint(int(num_iter/2),num_iter))))
            success_flag = 1
            
        if augment_selector[2]:
            print ("Changing Learning rate exponent")
            if '--learning_rate_exp' in cmd_split:
                ix = cmd_split.index('--learning_rate_exp')
                old_lre = cmd_split[ix+1]
                old_lre = float(old_lre)
                new_lre = np.random.uniform(old_lre*0.5,1.0)
                print('Old LR exp = %0.4f'%old_lre)
                print('New LR exp = %0.4f'%new_lre)
                cmd_split[ix+1] = str(new_lre)
            else:
                cmd_split.append('--learning_rate_exp')
                cmd_split.append(str(round(np.random.uniform(0,1),4)))
            success_flag = 1
            
        command_string = ' '.join(cmd_split)

       
        return old_string, command_string, success_flag, None

    ###########################################################################################
    # This mutation alters stride, actual functioning TBD.                                    #
    ###########################################################################################
    def stride_mutator(self, old_string):
        power = np.random.randint(5)
        #stride = 2**power
        command_string_cpy = old_string
        cmd_split = old_string.split(' ')
        if '--stride' in cmd_split:
            s_ix = cmd_split.index('--stride')
            cmd_split[s_ix+1] = str(2**power)
            command_string = ' '.join(cmd_split)
        else:
            command_string= old_string +  '--stride ' + str(2**power)
        success_flag = 1
        return old_string, command_string, success_flag, None

    ###########################################################################################
    # Identity mutation runs the same model for 26500 iterations.                             #
    ###########################################################################################
    def identity_mutation(self, old_string):

        cmd_split = old_string.split(' ')
        ix = cmd_split.index('--num_iter')
        dataset_idx = cmd_split.index('--dataset_name')
        if 'modelnet' in cmd_split[dataset_idx+1] :
            new_iter = np.random.randint(5000,15000)
        else:
            new_iter = np.random.randint(500,2000)
        cmd_split[ix+1] = str(new_iter)
        command_string = ' '.join(cmd_split)
        success_flag = 1
        print("Extending previous model for %d iterations" % new_iter )
        
        return old_string, command_string, success_flag, None

    ######################################################################################################################
    # This functions mutates filter size to any odd value. It is done so by adding an additional filter size             #
    # flag to the command string. Changes have been made to accomodate this in run.py, experiments.py, experiments_pcd.py#
    # and network.py.                                                                                                    #
    ######################################################################################################################
    def filter_size_mutation(self, old_string):
        print("Changing Filter Size")
        command_string_cpy = old_string
        cmd_split = old_string.split(' ')
        
        if '--filter_size' in cmd_split:
            f_ix = cmd_split.index('--filter_size')
            cmd_split[f_ix+1] = str(random.randrange(1,8,2))
            command_string = ' '.join(cmd_split)
        else:
            command_string= old_string + ' --filter_size ' + str(random.randrange(1,11,2))

        success_flag = 1
        return old_string, command_string, success_flag, None

    #########################################################################################
    # This function performs the mutation of changing the direction by randomly choosing    #
    # a value for theta and phi. The range of the values are defined by the ones from the   #
    # slide.                                                                                #
    #########################################################################################

    def pool_mutation(self, old_string):
        
        print("Pooling mutation")
        command_string_cpy = old_string
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        pr_dict = {}	
        
        fc_idx = get_layer_indices(arch_list, 'fc')
        gp_idx = get_layer_indices(arch_list, 'gmp')
        c_idx = get_layer_indices(arch_list, 'c')    
        ec_idx = get_layer_indices(arch_list,'ec')
        coo_idx = get_layer_indices(arch_list,'coo')
        c_idx = c_idx + coo_idx + ec_idx
        rm_idx = get_layer_indices(arch_list,'rm')
        
        if (len(gp_idx)>0) and ('--pool_ratios' in cmd_split):
                pr_idx = cmd_split.index('--pool_ratios')
                pratios = cmd_split[pr_idx+1].split('_')
                n_pools = len(pratios)
                pr_dict = {key: val for key, val in zip(gp_idx,pratios)}
        else:
            n_pools = 0

        
        if len(rm_idx)==0:
            if len(fc_idx)>0:
            
                insert_index= fc_idx[0]
                rm_string = 'rm'
                #pool_string = 'gmp_'+ str(n_pools)
                arch_list.insert(insert_index, rm_string)
                cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
                command_string = ' '.join(cmd_split)
                success_flag = 1
            else:
                print ("Cannot Add Pooling(RM) Yet")
                success_flag = 0
                command_string = old_string
                insert_index=None
        else:
            if len(c_idx) > 0:
                conv_check_idx = np.random.choice(c_idx)
                next_layer = arch_list[conv_check_idx+1].split('_')
                next_layer2 = arch_list[conv_check_idx+2].split('_')
                if next_layer[0] in ['rc0','rc1']:
                    if next_layer2[0] in ['rm', 'gmp', 'p']:
                        insert_index = None
                        print ("Pooling(/RM) Layer Already There")
                    else:
                        insert_index = conv_check_idx+2
                        print ("Adding Pooling after RC")
                else:
                    if next_layer[0] in ['rm', 'gmp', 'p']:
                        insert_index = None
                        print ("Pooling(/RM/ GEP) Layer Already There")
                    else:
                        insert_index = conv_check_idx+1
                        print ("Adding Pooling after C")
                    
                if insert_index != None:
                    if len(gp_idx)==0:
                        gp_layer = 'gmp_'+ str(n_pools)
                        arch_list.insert(insert_index,gp_layer)
                        pool_ratio_value = round(np.random.uniform(0,1),2)
                        pr_dict[insert_index] = pool_ratio_value
                        cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
                        cmd_split = modify_cmd_list(cmd_split, '--pool_ratios', str(pool_ratio_value))
                            
                    else:
                        gp_layer = 'gmp_'+ str(n_pools)
                        arch_list.insert(insert_index,gp_layer)
                        arch_list, pr_string = order_pooling(insert_index,gp_idx,arch_list,pr_dict)
                        cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
                        cmd_split = modify_cmd_list(cmd_split, '--pool_ratios', pr_string)
                            
                    
                    command_string = ' '.join(cmd_split)
                    success_flag = 1
                    
                else:
                    success_flag = 0
                    command_string = old_string
                        
            else:
                print('Cannot add more pooling as no convolutions present.')
                success_flag = 0
                command_string = old_string
                insert_index=None

        
        return old_string,command_string,success_flag, insert_index
        

    def pool_gep_mutation(self, old_string):
    
        print("Pooling (GEP) mutation")
        command_string_cpy = old_string
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        
        fc_idx = get_layer_indices(arch_list, 'fc')
        gp_idx = get_layer_indices(arch_list, 'gmp')
        gep_idx = get_layer_indices(arch_list, 'p')
        c_idx = get_layer_indices(arch_list, 'c')
        ec_idx = get_layer_indices(arch_list,'ec')
        coo_idx = get_layer_indices(arch_list,'coo')
        rm_idx = get_layer_indices(arch_list,'rm')
        
        c_idx = c_idx + ec_idx + coo_idx
        
        gep_layer = 'p_' + str(np.random.choice([4,8,16,32,64]))
        
        if len(c_idx)>0:
            
            conv_check_idx = np.random.choice(c_idx)
            next_layer = arch_list[conv_check_idx+1].split('_')

            
            if next_layer[0] in ['rc0','rc1']:
                if (conv_check_idx+2)<len(arch_list):
                    next_layer2 = arch_list[conv_check_idx+2].split('_')
                    if next_layer2[0] in ['rm', 'gmp', 'p']:
                        insert_index = None
                        print ("Pooling(/RM/GEP) Layer Already There")
                    else:
                        insert_index = conv_check_idx+2
                        print ("Adding Pooling after RC")
                        arch_list.insert(insert_index,gep_layer)
                        cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
                        command_string = ' '.join(cmd_split)
                        success_flag = 1
            elif next_layer[0] in ['rm', 'gmp', 'p']:
                print("Pooling(/RM/GEP) Layer Already There")
                success_flag = 0
                command_string = old_string
                insert_index=None
            else:
                insert_index = conv_check_idx+1
                print ("Adding Graph Embed Pooling after C")
                arch_list.insert(insert_index,gep_layer)
                cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
                command_string = ' '.join(cmd_split)
                success_flag = 1
            
        elif (len(c_idx)==0) and (len(fc_idx))>0:
            
            fc1 = fc_idx[0]
            current_layer = arch_list[fc1-1].split('_')
            if current_layer[0] == 'OC':
                print('Inserting GEP at index 1')
                insert_index = fc1
                arch_list.insert(insert_index,gep_layer)
                cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
                command_string = ' '.join(cmd_split)
                success_flag = 1
            elif current_layer[0] in ['rm', 'gmp', 'p']:
                print("Pooling(/RM/GEP) Layer Already There")
                success_flag = 0
                command_string = old_string
                insert_index=None
            else:
                print('Inserting GEP at before fc1')
                insert_index = fc1
                arch_list.insert(insert_index,gep_layer)
                cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
                command_string = ' '.join(cmd_split)
                success_flag = 1            
        
        else:
            print('Cannot add more pooling as no convolutions present.')
            success_flag = 0
            command_string = old_string
            insert_index=None
          
        
        return old_string,command_string,success_flag, insert_index
        
    ##############################################################################################
    # This function performs the mutation of adding an FC layer at the end of the architecture.  #
    # A variable specifying the desired output classes will be required and maybe even variables #
    # specifying stride and order. Right now the layer added would be of the type fc_400_1_2.    #
    ##############################################################################################

    def add_fc(self, old_string):
        print("Add FC layer")

        cmd_split = old_string.split(' ')
        dataset_name = cmd_split[cmd_split.index('--dataset_name')+1]
        output_classes = get_output_classes(dataset_name)
        
        if '--arch' not in cmd_split: #len(fc_idx)==0:
            arch_list = ['OC']
            fc_layer = make_layer_string('fc',output_classes)
            arch_list.append(fc_layer)
            insert_index = 1
        else:
            arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
            fc_idx= get_layer_indices(arch_list,'fc')
            insert_index = np.random.choice(fc_idx)
            fc_size = np.random.randint(output_classes+1,500)
            fc_layer = make_layer_string('fc', fc_size)
            arch_list.insert(insert_index,fc_layer)            
            
        cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
        command_string = ' '.join(cmd_split)
        success_flag = 1
        
        return old_string,command_string,success_flag,insert_index

    ###########################################################################################
    #Mutation to remove a convolution layer.                                                  #
    ###########################################################################################
    def remove_fc(self, old_string):
        print("Removing FC layer")
        #command_string_cpy = old_string
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        fc_idx = get_layer_indices(arch_list,'fc')
        fc_idx = fc_idx[:-1]

        if len(fc_idx)>0:	
            remove_index = np.random.choice(fc_idx)
            print ("Removing FC layer at " + str(remove_index))
            del arch_list[remove_index]
            cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))

            command_string = ' '.join(cmd_split)
            success_flag = 1
        else:
            print ("No Removable FC Layer present")
            command_string = old_string
            success_flag = 0
            remove_index = 0
        
        return old_string,command_string,success_flag, remove_index
    ########################################################################################################
    # This mutation adds a conv layer in the architecture string. It works at all positions except         #  
    # for 0,12,13,14. At the exception position it does not add a convolution layer. The size of 		   #
    # the conv layer is randomly chosen except for when it is added in between a conv and skip layer (rc0).#
    # Added lists that store layers where weights are kept the same and the layer where weights need to be #
    # initialized.                                                                                         #
    ########################################################################################################
    def add_conv_layer(self, old_string):                                                                              
        print("Adding Conv layer")
        #f_size = [128,256,512]
        
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        fc_idx = get_layer_indices(arch_list, 'fc')
        rm_idx = get_layer_indices(arch_list, 'rm')
        fc_idx = sorted(fc_idx+rm_idx)
        
        if len(fc_idx)>0:
            insert_index = np.random.choice(range(1,fc_idx[0]+1))
        else:
            insert_ix = 1
        print ("Index chosen is " + str(insert_index))
        
        layer_string = arch_list[insert_index].split('_')
        if 'rc' in layer_string[0]:
            rc_split = layer_string[1].split('-')
            conv_filters = str(rc_split[0])
        else: 
            conv_filters = np.random.choice([4,8,16,32,64,128,256,512])
        
        conv_layer= make_layer_string('c', conv_filters)          
        arch_list.insert(insert_index,conv_layer)
        
        same_weights = []
        reset_weights = arch_list[insert_index]
        same_weights[1:insert_index] = arch_list[1:insert_index]
        same_weights[insert_index+1:len(same_weights)]=arch_list[insert_index+1:len(arch_list)]
        cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))

        command_string = ' '.join(cmd_split)
        success_flag = 1

        return old_string,command_string,success_flag,insert_index


    ###########################################################################################
    #Mutation to remove a convolution layer.                                                  #
    ###########################################################################################
    def remove_conv(self, old_string):
        print("Removing Conv layer")
        #command_string_cpy = old_string
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        c_idx = get_layer_indices(arch_list,'c')

        if len(c_idx)>0:	
            remove_index = np.random.choice(c_idx)
            print ("Removing Conv layer at " + str(remove_index))
            del arch_list[remove_index]
            cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))

            command_string = ' '.join(cmd_split)
            success_flag = 1
        else:
            print ("No Convolution Layer to Remove")
            command_string = old_string
            success_flag = 0
            remove_index = 0
        
        return old_string,command_string,success_flag, remove_index

    ###########################################################################################
    #Mutation to remove a skip connection.                                                    #
    #rc0_128-128_1-1_1-1_1-1_1-1 ------>>>> c_128_1_1,c_128_1_1                               #
    ###########################################################################################
    def remove_skip(self, old_string):
        print("Removing Skip connection")
        
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        rc_idx= sorted(get_layer_indices(arch_list,'rc0')+ get_layer_indices(arch_list,'rc1'))
        remove_index = []

        if len(rc_idx)>0:
            remove_index = np.random.choice(rc_idx)
            print ("Removing rc at " + str(remove_index))
            rc_split = arch_list[remove_index].split('_')
            conv_num = len(rc_split[1].split('-'))
            conv_list = []

            for i in range(0,conv_num):
                c_size = rc_split[1].split('-')
                c_string = make_layer_string('c', c_size[i])
                conv_list.append(c_string)

            conv_string = ','.join(conv_list)
            arch_list[remove_index] = conv_string
            cmd_list = modify_cmd_list(cmd_split, '--arch', ','.join(arch_list))

            command_string = ' '.join(cmd_split) 
            success_flag = 1
            
        else:
            print ("No Skip Connection to Remove")
            command_string = old_string
            success_flag = 0
            remove_index=0
            
        return old_string,command_string, success_flag, remove_index

    ###########################################################################################
    #Mutation to add a skip connection.                                                       #
    #c_128_1_1,c_128_1_1  ------>>>> c_128_1-1, rc0_128-128_1-1_1-1_1-1_1-1                   #
    ###########################################################################################

    def add_skip(self, old_string):
        print("Adding skip connection")

        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        add_idx = []
        skip_string = ""
        c_idx = get_layer_indices(arch_list, 'c')

        if len(c_idx)>1:	
            
            insert_index = np.random.choice(c_idx)
            present_layer = arch_list[insert_index].split('_')
            next_layer = arch_list[insert_index+1].split('_')
            
 
            if next_layer[0]== 'c':

                skip_con = make_layer_string('rc', None ,[present_layer[1],next_layer[1]])
                arch_list.insert(insert_index+1,skip_con)
                del arch_list[insert_index+2]
                del arch_list[insert_index]
                print ("Deleting at pos " + str(insert_index+2))
                success_flag = 1
                
            elif next_layer[0] in ['rc0','rc1']:
                print ("Skip Connection already present.")
                success_flag = 0
                insert_index = 0
            else:
                
                print ("Two consecutive convolutions not present.")
                success_flag = 0
                insert_index = 0
                        
            same_weights = []
            reset_weights = [arch_list[insert_index],arch_list[insert_index+1]]
            same_weights[1:insert_index] = arch_list[1:insert_index]
            same_weights[insert_index+2:len(arch_list)] = arch_list[insert_index+2:len(arch_list)]
            cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
            
            command_string = ' '.join(cmd_split)
            
            return old_string,command_string, success_flag, insert_index
        
        else:
        
            print ("Cannot Add Skip Connection due to no convolutions.")
            success_flag = 0
            insert_index = 0
        
            return old_string,old_string, success_flag, insert_index

    ###########################################################################################
    # This function adds a one-to-one (batchnorm and relu) identity function.                 #
    #                                                                                         #
    ###########################################################################################
    def add_one_to_one(self, old_string):                                                      
        print("Add one-to-one layer")
        reset_weights = []
        same_weights = []
        
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        fc_idx = get_layer_indices(arch_list,'fc')
        rm_idx = get_layer_indices(arch_list, 'rm')
        fc_idx = sorted(fc_idx+rm_idx)
        c_idx=get_layer_indices(arch_list,'c')
        
        if (len(fc_idx)>0) and (len(c_idx)>0):
            insert_index = np.random.randint(1,np.max(c_idx)+1)
        else:
            insert_index = 1

        coo_filters = np.random.choice([4,8,16,32,64,128,256,512])
        coo_layer = make_layer_string('coo',coo_filters)
        arch_list.insert(insert_index,coo_layer)
        reset_weights = arch_list[insert_index]
        cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
        
        command_string = ' '.join(cmd_split)
        print ("Adding One-by-One layer at " + str(insert_index))
        success_flag = 1
        
        return old_string,command_string, success_flag, insert_index
        
    ###########################################################################################
    # This function removes one-by-one layer                                                   #
    #                                                                                         #
    ###########################################################################################    

    def remove_one_to_one(self, old_string):
        print("Removing one-by-one layer")
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        coo_idx = get_layer_indices(arch_list,'coo')

        if len(coo_idx)>0:	
            remove_index = np.random.choice(coo_idx)
            print ("Removing one-by-one layer at " + str(remove_index))
            del arch_list[remove_index]
            cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))

            command_string = ' '.join(cmd_split)
            success_flag = 1
        else:
            print ("No one-by-one Layer to Remove")
            command_string = old_string
            success_flag = 0
            remove_index = 0
        
        return old_string,command_string,success_flag, remove_index
    ###########################################################################################
    # This function adds an attention layer                                                   #
    #                                                                                         #
    ###########################################################################################
    def add_attention_layer(self, old_string):                                
        print("Add attention layer")
        reset_weights = []
        same_weights = []
        
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        fc_idx = get_layer_indices(arch_list,'fc')
        rm_idx = get_layer_indices(arch_list, 'rm')
        fc_idx = sorted(fc_idx+rm_idx)
        c_idx = get_layer_indices(arch_list,'c')
        
        if (len(fc_idx)>0) and (len(c_idx)>0):
            insert_index = np.random.randint(1,np.max(c_idx)+1)
        else:
            insert_index = 1

        a_filters = np.random.choice([4,8,16,32,64,128,256,512])
        a_layer = make_layer_string('a1d',a_filters)         
        arch_list.insert(insert_index,a_layer)
        reset_weights = arch_list[insert_index]
        cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
        
        command_string = ' '.join(cmd_split)
        print ("Adding Attention layer at " + str(insert_index))
        success_flag = 1
        
        return old_string,command_string, success_flag, insert_index
        
    ###########################################################################################
    # This function removes attention layer                                                   #
    #                                                                                         #
    ###########################################################################################    

    def remove_attention(self, old_string):
        print("Removing attention layer")

        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        a1d_idx = get_layer_indices(arch_list,'a1d')

        if len(a1d_idx)>0:	
            remove_index = np.random.choice(a1d_idx)
            print ("Removing attention layer at " + str(remove_index))
            del arch_list[remove_index]
            cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))

            command_string = ' '.join(cmd_split)
            success_flag = 1
        else:
            print ("No attention Layer to Remove")
            command_string = old_string
            success_flag = 0
            remove_index = 0
        
        return old_string,command_string,success_flag, remove_index


    def add_edgeconv_layer(self, old_string):                                                   
        print("Adding Edge Conv layer")
        #f_size = [128,256,512]
        
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        fc_idx = get_layer_indices(arch_list, 'fc')
        rm_idx = get_layer_indices(arch_list, 'rm')
        fc_idx = sorted(fc_idx+rm_idx)
        
        if len(fc_idx)>0:
            insert_index = np.random.choice(range(1,fc_idx[0]+1))
        else:
            insert_ix = 1
        print ("Index chosen is " + str(insert_index))
        
        layer_string = arch_list[insert_index].split('_')
        edgeconv_filters = np.random.choice([4,8,16,32,64,128,256,512])
        
        edgeconv_layer= make_layer_string('ec', edgeconv_filters)  
        arch_list.insert(insert_index,edgeconv_layer)
        
        same_weights = []
        reset_weights = arch_list[insert_index]
        same_weights[1:insert_index] = arch_list[1:insert_index]
        same_weights[insert_index+1:len(same_weights)]=arch_list[insert_index+1:len(arch_list)]
        cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))

        command_string = ' '.join(cmd_split)
        success_flag = 1

        return old_string,command_string,success_flag,insert_index
    
    ###########################################################################################
    # This function removes attention layer                                                   #
    #                                                                                         #
    ###########################################################################################    

    def remove_edgeconv(self, old_string):
        print("Removing Edge Conv layer")
        
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        ec_idx = get_layer_indices(arch_list,'ec')

        if len(ec_idx)>0:	
            remove_index = np.random.choice(ec_idx)
            print ("Removing edge conv layer at " + str(remove_index))
            del arch_list[remove_index]
            cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))

            command_string = ' '.join(cmd_split)
            success_flag = 1
        else:
            print ("No Edge Conv Layer to Remove")
            command_string = old_string
            success_flag = 0
            remove_index = 0
        
        return old_string,command_string,success_flag, remove_index

        
        
    def replace_mutation(self, old_string):
    
        print("Replace Mutation")
        cmd_split = old_string.split(' ')
        arch_list = cmd_split[cmd_split.index('--arch')+1].split(',')
        fc_idx = get_layer_indices(arch_list,'fc')
        c_idx = get_layer_indices(arch_list,'c')
        ec_idx = get_layer_indices(arch_list,'ec')
        coo_idx = get_layer_indices(arch_list,'coo')
        filter_list = [4,8,16,32,64,128,256,512]
        replace_flag = 0
        
        if (len(c_idx)>0) or (len(ec_idx)>0) or (len(coo_idx)>0): 
            while not replace_flag:
                type = np.random.choice(['c','ec','coo'])
                if (type == 'c') and len(c_idx)>0 :
                    replace_flag = 1
                elif (type == 'ec') and len(ec_idx)>0 :
                    replace_flag = 1
                elif (type == 'coo') and len(coo_idx)>0 :
                    replace_flag = 1
                else:
                    replace_flag = 0
            
            if (type == 'c') :
                arch_list, replace_index = replace_filter(arch_list, c_idx ,filter_list)
                print ("Replacing Convolution layer at " + str(replace_index))
            elif (type == 'ec') :
                arch_list, replace_index = replace_filter(arch_list, ec_idx ,filter_list)
                print ("Replacing Edge Convolution layer at " + str(replace_index))
            elif (type == 'coo') :
                arch_list, replace_index = replace_filter(arch_list, coo_idx ,filter_list)
                print ("Replacing One-by-One Convolution layer at " + str(replace_index))
            
            cmd_split = modify_cmd_list(cmd_split,'--arch',','.join(arch_list))
            command_string = ' '.join(cmd_split)
            success_flag = 1
                      
        else:
            print("No type of Convolutions present for replacement")
            command_string = old_string
            success_flag = 0
            replace_index = 0
        
        return old_string,command_string,success_flag, replace_index


def get_mutation(path, mutation_choice):
        
    Mutator_test = Mutator(path)
    options = defaultdict(lambda: 'defalut',  {0:Mutator.learning_rate_mutation,1:Mutator.add_fc, \
                         2:Mutator.remove_fc , 3:Mutator.add_conv_layer, 4:Mutator.remove_conv,\
                         5:Mutator.add_skip, 6:Mutator.remove_skip, 7:Mutator.add_edgeconv_layer, 8:Mutator.remove_edgeconv,9:Mutator.add_one_to_one,\
                         10:Mutator.remove_one_to_one ,11: Mutator.add_attention_layer,12:Mutator.remove_attention, 13:Mutator.regularization_mutation, \
                         14:Mutator.replace_mutation,15:Mutator.pool_mutation, 16: Mutator.pool_gep_mutation}  )

        
    old_string,final_string,success_flag,index = options[mutation_choice](Mutator_test,path)
        
    return old_string,final_string, success_flag, index

def test_mutation(path, dataset):
    
    # Test mutation sequence
    mutation_sequence = [6,6, 20, 18,16,18,18,7,7,11,11,18,20,12,12,12,12,12]
    Mutator_test = Mutator(path)
    
    options = defaultdict(lambda: 'defalut', {0:Mutator.n_directions, 1:Mutator.n_neighbours,2:Mutator.agg_augment, 3:Mutator.learning_rate_mutation,\
                           4:Mutator.identity_mutation,5:Mutator.filter_size_mutation, 6: Mutator.add_fc, 7:Mutator.add_conv_layer,\
                           8:Mutator.remove_skip, 9:Mutator.add_skip, 10: Mutator.remove_conv, 11:Mutator.add_one_to_one,\
                           12:Mutator.pool_mutation, 13:Mutator.add_attention_layer, 14:Mutator.remove_one_to_one, 15:Mutator.remove_attention ,\
                           16: Mutator.pool_gep_mutation, 17:Mutator.regularization_mutation, 18:Mutator.add_edgeconv_layer, 19:Mutator.remove_edgeconv, 20:Mutator.replace_mutation} ) 
    
    acc_list = []
    for mut_idx,choice in enumerate(mutation_sequence):
        pdb.set_trace()
        path = path + ' --trial_name mut'+ str(mut_idx)
        old_string,final_string,success_flag,index = options[choice](Mutator_test,path)
        print(final_string)
        old_split = old_string.split(' ')
        final_split = final_string.split(' ')

        if success_flag:
            acc = 0
            std = 0
            
            acc_list.append((acc,std))
        new_path = final_string.split(' ')
        del new_path[new_path.index('--trial_name')+1]
        del new_path[new_path.index('--trial_name')]
        new_path = ' '.join(new_path)
        path = new_path
        
    return acc_list, final_string

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


        args = parser.parse_args()
        
        return args, parser
    
def main():
    
    command_string = 'src/run_protein_test.py --dataset_name ENZYMES --num_iter 10 --num_classes 6'
    cmd_split = command_string.split(' ')
 
    args, parser = get_args()
    args_list =cmd_split[1:]
    new_args = parser.parse_args(args= args_list)
  
    acc_list, final_string = test_mutation(command_string, dataset)
    print(acc_list)
    print(final_string)
    
if __name__ == "__main__":
    main()
    
