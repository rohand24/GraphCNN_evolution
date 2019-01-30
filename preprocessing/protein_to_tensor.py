from __future__ import print_function
import scipy.io
import numpy as np
import time
import os
from datetime import datetime
import pdb
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../src/')))
from graphcnn.util.pooling.sparse.SparseIdentityCompander import SparseIdentityCompander
from graphcnn.util.pooling.sparse.SparsePoolingFactory import SparsePoolingFactory
###########################################################################
#HELPER functions begin

def print_ext(*args):
    print(str(datetime.now()), *args)

def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))

def get_node_mask(graph_size, max_size=None):
    if max_size == None:
        max_size = np.max(graph_size)
    return np.array([np.pad(np.ones([s, 1]), ((0, max_size-s), (0, 0)), 'constant', constant_values=(0)) for s in graph_size], dtype=np.float32)

def _tf_print(*args):
    for i in range(len(args)):
        print(args[i].shape)
        print(args[i])
    return args

#def make_print(*args):
#    import tensorflow as tf
#
#    result = tf.py_func(_tf_print, args, [ s.dtype for s in args])
#    for i in range(len(args)):
#        result[i].set_shape(args[i].get_shape())
#    return result

_default_location = 'datasets/'

def get_file_location(file):
    return _default_location + file

def locate_or_download_file(file, url):
    if os.path.isfile(_default_location + file) == False:
        print_ext('Downloading "%s", this might take a few minutes' % url)
        verify_dir_exists(os.path.dirname(_default_location + file) + '/')
        try:
            from urllib import urlretrieve
        except ImportError:
            # Support python 3
            from urllib.request import urlretrieve
        urlretrieve (url, _default_location + file)
        return file

def locate_or_extract_file(file, folder):
    if os.path.isdir(_default_location + folder) == False and os.path.isfile(_default_location + folder) == False:
        print_ext('Extracting "%s", this might take a few minutes' % file)
        if os.path.splitext(_default_location + file)[1] in ['.tar', '.tgz']:
            import tarfile
            tar = tarfile.open(_default_location + file)
            tar.extractall(os.path.dirname(_default_location + file))
            tar.close()
        elif os.path.splitext(_default_location + file)[1] == '.zip':
            import zipfile
            zip_ref = zipfile.ZipFile(_default_location + file, 'r')
            zip_ref.extractall(os.path.dirname(_default_location + file))
            zip_ref.close()
        else:
            print_ext('Cannot extract: Invalid extension name')

###############################################################################
#HELPER functions end
###############################################################################



chemical_datasets_list = ['DD', 'ENZYMES', 'MUTAG', 'NCI1', 'NCI109']

def load_protein_dataset(dataset_name, weights_as_features = True):
    '''
    load the .mat data
    '''
    if dataset_name not in chemical_datasets_list:
        print_ext('Dataset doesn\'t exist. Options:', chemical_datasets_list)
        return
    locate_or_download_file('proteins/proteins_data.zip', 'http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/Graphkernels/data.zip')
    locate_or_extract_file('proteins/proteins_data.zip', 'proteins/data')
    mat = scipy.io.loadmat(get_file_location('proteins/data/%s.mat' % dataset_name))

    #Count unique node values
    input = mat[dataset_name]


    #print (len(input[0,0]['am'].indices))
    #print(np.array(input[0,0]['am'].nonzero()).T[:,0])
      

    #print (input[0,0]['am'])
    #print (input.dtype.names)


    labels = mat['l' + dataset_name.lower()]
    labels = labels - min(labels)

    node_vals = set()

    #Note
    node_labels = input['nl']

                

    #print (node_labels[:, 0][0])
    #print (node_labels[0, 0]['values'][0, 0].shape)


   

    #********  VERTEX  ****************
    #Finds number of unique weights, not necessary for now
    '''
    v_labels = 0

    #For each sample
    for i in range(node_labels.shape[1]):
        v_labels = max(v_labels, max(node_labels[0, i]['values'][0, 0])[0]) 
    '''

    #************************




    #***********  ADJACENCY  *************
    #This gets e_labels
    #Get number of features/weights

    unique_vals = set()

    e_labels = 1

    if dataset_name == 'NCI1' or dataset_name == 'NCI109' or dataset_name == 'MUTAG':

       #Here, maybe get the dense_shape

       #For each sample
        for i in range(input.shape[1]):
            #For each edge
            for j in range(input[0, i]['el']['values'][0, 0].shape[0]):
                #pdb.set_trace()
                if dataset_name == 'NCI1' or dataset_name == 'NCI109':
                    if input[0, i]['el']['values'][0, 0][j, 0].shape[0] > 0:
                        #Iterate through lables, to find all unique values
                        for k in (input[0, i]['el']['values'][0, 0][j, 0][0]):
                            unique_vals.add(k)
                            

                        #print (e_labels)
                        #pdb.set_trace()
                else: #if MUTAG
                    #pdb.set_trace()
                    for k in (input[0, i]['el']['values'][0, 0]):
                        unique_vals.add(k[2])
    else:
        unique_vals.add(1)
                      
    #************************

    
    e_labels = len(unique_vals)


    weights_to_indices = {}


    #Get dictionary to map weights to indices
    for count, unique_num in enumerate(unique_vals):
        weights_to_indices[unique_num] = count



    # For each sample
    #print (e_labels)
    samples_V = []
    samples_A = []
    max_no_nodes = 0

    indices = []
    values = []
    dense_shape = []



    #For each sample
    for i in range(input.shape[1]):

        
        #************************ VERTEX ************************
        #Number of nodes
        no_nodes = node_labels[0, i]['values'][0, 0].shape[0]
        #Get maximum num of nodes
        max_no_nodes = max(max_no_nodes, no_nodes)

        #This assumes there is only one vertex feature. Needs to be modified
        #to account for multiple vertex features
        V = np.ones([no_nodes, 1])
        
        
        for node_count in range(node_labels[0,i]['values'][0,0].shape[0]):
            V[node_count,0] = node_labels[0,i]['values'][0,0][node_count][0]
            
        samples_V.append(V)


        #If we treat weights as features
        ''' 
        V = np.ones([no_nodes, v_labels])
        for l in range(v_labels):
            V[..., l] = np.equal(node_labels[0, i]['values'][0, 0][..., 0], l+1).astype(np.float32)
        samples_V.append(V)
        #************************************************
        '''


        #************************ EDGE  ***********************
        A = np.zeros([no_nodes, e_labels, no_nodes])
        #Iterate through edge labels, and fill in adjacency matrix.


        if dataset_name == 'NCI1' or dataset_name == 'NCI109':


            #print (input[0,0]['al'].shape[0])
            for j in range(input[0,i]['al'].shape[0]):
                if len(input[0,i]['al'][j,0]) > 0:
                    for k_count, k in enumerate(input[0,i]['al'][j,0][0]):
                        #print(input[0,i]['el']['values'][0,0][j,0][0])

                        if weights_as_features:
                            indices.append([i, j, weights_to_indices[input[0,i]['el']['values'][0,0][j,0][0][k_count]], k-1])
                        else:
                            indices.append([i, j, k-1])
                            values.append(input[0,i]['el']['values'][0,0][j,0][0][k_count])

                        A[j, input[0,i]['el']['values'][0,0][j,0][0][k_count] - 1, k-1] = 1


  
        elif dataset_name == 'MUTAG': #if MUTAG
            for z in (np.array(input[0, i]['el']['values'][0,0])):
                if weights_as_features:
                    indices.append([i, z[0] - 1, weights_to_indices[z[2]], z[1] - 1])
                else:
                    indices.append([i, z[0] - 1, z[1] - 1])
                    values.append(z[2]) 




        else: #DD and ENZYME

            #1 row per index
            for j in np.array(input[0,i]['am'].nonzero()).T:
                if weights_as_features:
                    indices.append([i, j[0], 0, j[1]])
                else:
                    indices.append([i, j[0], j[1]])
                    values.append(1.)
                  
        
        #************************************************


    #print('--------------------')

    indices = np.array(indices)
    values = np.array(values)

    if weights_as_features:
        values = np.ones(shape = (indices.shape[0]))
        dense_shape  = np.array([input.shape[1], max_no_nodes, len(unique_vals), max_no_nodes])

    else:
        dense_shape  = np.array([input.shape[1], max_no_nodes, max_no_nodes])
 
    dense_shape = np.array(dense_shape)

    samples_V = np.array(samples_V)



    graph_vertices = []
    for i in range(len(samples_V)):
    # pad all vertices to match size
        graph_vertices.append(np.pad(samples_V[i].astype(np.float32), ((0, max_no_nodes - samples_V[i].shape[0]), (0, 0)),
                   'constant', constant_values=(0)))

    graph_vertices = np.array(graph_vertices)

    
    '''  
    print (indices.shape)
    print (values.shape)
    print (dense_shape)
    print (graph_vertices.shape)
    '''
    
    

    return graph_vertices, (indices, values, dense_shape), np.reshape(labels, [-1])
    



def dense_to_sparse_tensor(dense_tensor, ndims_4 = True):

    #---------------------4D Tensor----------------------
    indices_4D = np.array(np.nonzero(dense_tensor)).T
    values_4D = np.ones(shape = (indices_4D.shape[0]))
    dense_shape_4D = np.array(dense_tensor.shape)
    #------------------------------------------------------


    #---------------------3D Tensor----------------------
    indices_3D = np.empty(shape = (indices_4D.shape[0], 3)) 
    values_3D = np.ones(shape = (indices_4D.shape[0]))
    dense_shape_3D = np.array([dense_tensor.shape[0], dense_tensor.shape[1], dense_tensor.shape[3]])
    #-----------------------------------------------------


    for count, i in enumerate(indices_4D):
        #3D_slice = np.array()
        indices_3D[count] = np.array([int(i[0]), int(i[1]), int(i[3])])
        
        values_3D[count] = i[2] + 1

    indices_3D = indices_3D.astype(int)
    values_3D = values_3D.astype(int)

    if ndims_4:
        return (indices_4D, values_4D, dense_shape_4D)
    else:
        return (indices_3D, values_3D, dense_shape_3D)



def preprocess_data(dataset, poolRatios, poolId, ndims_4 = True):
    #pdb.set_trace()
    graph_labels = dataset[2].astype(np.int64)
    graph_vertices = dataset[0]
    indices, values, dense_shape = dataset[1]
    Plist = []



    if poolRatios is not None:
        poolFactory = SparsePoolingFactory()

        #For each sample
        for i in range(dense_shape[0]):
            pooler = poolFactory.CreatePoolingPyramid(len(poolRatios), SparseIdentityCompander, poolRatios, poolId)
            #Plistcurrent = pooler.makeP(graph_adjacency[i, :, :, :].sum(axis=1), graph_vertices[i, :, :])
            Plistcurrent = pooler.makeP(indices[np.where(indices[:,0] == i)][:,1:],values[np.where(indices[:,0] == i)],dense_shape[1:], graph_vertices[i, :, :])
            Plist.append(Plistcurrent)
        #stackedPlist = [[] for x in
        #                range(len(dataset[0]))]
        #for sampleIndex in range(len(dataset[0])):
        #    for poolIndex in range(len(poolRatios)):
        #        stackedPlist[poolIndex][sampleIndex, :, :] = Plist[sampleIndex][poolIndex]

                # print('V shape: {0}'.format(self.graph_vertices.shape))
                # print('A shape: {0}'.format(self.graph_adjacency.shape))
                # print('Label shape: {0}'.format(self.graph_labels.shape))
    # for P in stackedPlist:
    #    print('P shape: {0}'.format(P.shape))

    #Plist = stackedPlist



  


    




    #------------------------------------------------------

    '''

    print ('---------------')

    print (indices_4D.shape)
    print (values_3D.shape)
    print (dense_shape_4D)

    print ('*************')

    print (indices_3D.shape)
    print (values_3D.shape)
    print (dense_shape_3D)



   
    print ('----------------')
    
    '''

    
    return graph_vertices, indices.astype(np.int64), values.astype(np.float32), dense_shape, graph_labels.astype(np.int64), Plist


    #print(stackedPlist[0].shape)
    #print(stackedPlist[1].shape)


#################################### BASIC RUN ####################################
#['DD', 'ENZYMES', 'MUTAG', 'NCI1', 'NCI109']
#dataset = load_protein_dataset('NCI1')

'''
dataset = load_protein_dataset('ENZYMES')

preprocess_data(dataset, [0.5,0.5], 'Lloyd', True)

print('lol')
'''

#MUTAG, NCI1,  NCI109


#The above code as of now works only for 'NCI1', 'NCI109', 'ENZYMES'
###################################################################################



#Test dense_to_sparse:
'''
dense = np.zeros(shape = (100,100,100,100))

dense[10,10,10,10] = 1
dense[20,20,20,20] = 1
dense[30,30,30,30] = 1

indices, values, dense_shape = dense_to_sparse_tensor(dense, False)

print (indices)
print (values)
print (dense_shape)
'''







#################ERROR LIST######################################################
##MUTAG ERROR#########
#   File "protein_to_tensor.py", line 161, in <module>
#     dataset = load_protein_dataset('MUTAG')
#   File "protein_to_tensor.py", line 106, in load_protein_dataset
#     if input[0, i]['el']['values'][0, 0][j, 0].shape[0] > 0:
# IndexError: tuple index out of range
#########################


##ENZYMES ERROR######
#   File "protein_to_tensor.py", line 161, in <module>
#     dataset = load_protein_dataset('ENZYMES')
#   File "protein_to_tensor.py", line 101, in load_protein_dataset
#     edge_labels = input[0, 0]['el']
# ValueError: no field of name el
############################


##DD ERROR#############
# File "protein_to_tensor.py", line 161, in <module>
#     dataset = load_protein_dataset('DD')
#   File "protein_to_tensor.py", line 101, in load_protein_dataset
#     edge_labels = input[0, 0]['el']
# ValueError: no field of name el
##DD ERROR#############


#NOTE: 1
#For NCI1,NCI109 and MUTAG we have =>
#('nl', 'O'), ('am', 'O'), ('al', 'O'), ('el', 'O')]

#For DD,ENZYMES we do not have => el


#NOTE:2
#for MUTAG => if input[0, i]['el']['values'][0, 0][j, 0].shape[0] > 0:
#ERROR -> IndexError: tuple index out of range



#NCI1
# input.shape
# (1, 4110)
# ==> 4110 graphs
#
# input[0,1]['el'].shape
# (1, 1)
#
#
# input[0,1]['el']['values'].shape
# (1, 1)
# ==> shape of edge values for 1st graph
#
# input[0,1]['el']['values'][0,0].shape
# (24, 1)
# ==>


#
#
# input[0, i]['el']['values'][0, 0][j, 0]
#
#
# #2nd graph adjacency matrix
# input[0]['am'][1,].shape
