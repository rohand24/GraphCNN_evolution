import numpy as np
import pcl
import os
import os.path
import sys
import math
import numpy.random as random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../../preprocessing')))
from graphcnn.util.pooling.sparse.SphericalSparseAdjacencyCompander import SphericalSparseAdjacencyCompander
from graphcnn.util.pooling.sparse.SparsePoolingFactory import SparsePoolingFactory
import transforms3d
from graphcnn.util.pooling.SphericalUtil import *
from graphcnn.util.pooling.SplitUtils import *

POOL_RATIOS = [0.5, 0.5, 0.5, 0.5, 0.5]
POOLING_ID = 'Lloyd'
FILE_LIST_PATH = '../../../preprocessing/modelnet40_auto_aligned_trainvaltest.csv'
PREFIX = '/shared/kgcoe-research/mil/modelnet/modelnet40_auto_aligned_10000_pcd/'
K=16
MAX_SIZE = 4000
THETA = 4
PHI = 4
THETA_KEYS = np.linspace(0, 2*np.pi + 0.01,num=THETA + 1,endpoint=True)
PHI_KEYS = np.linspace(0,np.pi + 0.01,num=PHI + 1,endpoint=True)


def calculate_features(input, K=3, MAX_SIZE=500, aug=False,feature_type=1):##input tensor = NxNx3

    #inputData = prefix + '/' + input
   # label = self.labels[input.split('/')[-3]]
    cloud = pcl.load(input)
    basename,ext = os.path.splitext(input)
    featureData = np.load(basename + '.npy')
    #stolen from ECC code, drops out random points
    #if aug:
        #Probability a point is dropped
    p = 0.1

    cloudArray = cloud.to_array()
    if cloudArray.shape[0] > MAX_SIZE:
        keptIndices = np.random.choice(range(cloudArray.shape[0]), size=int(MAX_SIZE), replace=False)
        cloudArray = cloudArray[keptIndices, :]
        featureData = featureData[keptIndices, :]
    #print(cloudArray.shape)
    keptIndices = random.choice(range(cloudArray.shape[0]), size=int(math.ceil((1-p)*cloudArray.shape[0])),replace=False)
    cloudArray = cloudArray[keptIndices,:]
    featureData = featureData[keptIndices,:]
    featureData = np.concatenate((featureData, np.zeros((MAX_SIZE - featureData.shape[0],10))))
    cloud.from_array(cloudArray)
    cloud.resize(MAX_SIZE)

    xyz = cloud.to_array()[:,:3]
    kd = pcl.KdTreeFLANN(cloud)
    indices, sqr_distances = kd.nearest_k_search_for_cloud(cloud,
                                                           K)  # K = 2 gives itself and other point from cloud which is closest
    sqr_distances[:, 0] += 1  # includes self-loops
    valid = np.logical_or(indices > 0, sqr_distances > 1e-10)
    rowi, coli = np.nonzero(valid)
    idx = indices[(rowi, coli)]
    #Stolen from ECC code
    if aug:
        angle = np.random.uniform(0,2*np.pi,size=3)
        scale = np.random.uniform(1,1.2,size=3)
        #M = np.eye(3)
        #M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], angle[0]), M)
        #M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle[1]), M)
        #M = np.dot(transforms3d.axangles.axangle2mat([1,0,0], angle[2]), M)
        #M = np.dot(transforms3d.zooms.zfdir2mat(scale[0], [1,0,0]), M)
        #M = np.dot(transforms3d.zooms.zfdir2mat(scale[1], [0,1,0]), M)
        #M = np.dot(transforms3d.zooms.zfdir2mat(scale[2], [0,0,2]), M)

        #xyz = np.dot(xyz,M.T)
        M = np.eye(3)
        s = random.uniform(1/1.1, 1.1)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
        #angle = random.uniform(0, 2*math.pi)
        #M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], angle), M) # z=upright assumption
        if random.random() < 0.5/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < 0.5/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,1,0]), M)
        xyz = np.dot(xyz,M.T)
    if feature_type == 1:

        vertexMean = np.mean(xyz, axis=0)
        vertexStd = np.std(xyz, axis=0)
        #Jiggle the model a little bit if it is perfectly aligned with the axes
        #print(input)
        if not vertexStd.all():
            M = np.eye(3)
            angle = random.uniform(0.01,0.1,size=3)
            sign = random.choice([-1,1],size=3,replace=True)
            M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], sign[0] * angle[0]), M)
            M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], sign[1] * angle[1]), M)
            M = np.dot(transforms3d.axangles.axangle2mat([1,0,0], sign[2] * angle[2]), M)
            xyz = np.dot(xyz,M.T)
            vertexMean = np.mean(xyz, axis=0)
            vertexStd = np.std(xyz, axis=0)
        xyz = (xyz - vertexMean)/vertexStd

        num_nodes = xyz.shape[0]

        Aindices, Avalues, Adenseshape = geometricSplit(xyz, rowi, idx, MAX_SIZE)

    elif feature_type == 0:
        RADIUS = 0.5
        vertexMean = np.mean(xyz, axis=0)
        vertexStd = np.std(xyz, axis=0)
        #Jiggle the model a little bit if it is perfectly aligned with the axes
        #print(input)
        if not vertexStd.all():
            M = np.eye(3)
            angle = np.random.uniform(0.01,0.1,size=3)
            sign = np.random.choice([-1,1],size=3,replace=True)
            M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], sign[0] * angle[0]), M)
            M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], sign[1] * angle[1]), M)
            M = np.dot(transforms3d.axangles.axangle2mat([1,0,0], sign[2] * angle[2]), M)
            xyz = np.dot(xyz,M.T)
            vertexMean = np.mean(xyz, axis=0)
            vertexStd = np.std(xyz, axis=0)
        xyz = (xyz - vertexMean)/vertexStd
        #V = np.pad(V,pad_width=((0,MAX_SIZE - V.shape[0]),(0,0)),mode='constant')
        #kdtree = scipy.spatial.KDTree(xyz)
        #knns = kdtree.query_ball_tree(kdtree,r=RADIUS)
        #A = np.zeros(shape=(MAX_SIZE,self.theta*self.phi, MAX_SIZE))
        #numNeighbors = [len(x) for x in knns]
        #v1 = np.repeat(np.arange(xyz.shape[0]),numNeighbors)
        #knnsStack = np.concatenate(knns)

        Aindices, Avalues, Adenseshape = sphericalSplit(xyz,rowi, idx,MAX_SIZE,THETA,PHI,THETA_KEYS,PHI_KEYS)
    #Final step, add eigenvectors and curvature features. Smallest eigenvector feature is surface normal
    #features = np.zeros((xyz.shape[0], 10))
    #for i in range(xyz.shape[0]):
    #    X = xyz[indices[i, :], :]
    #    pca = sklearn.decomposition.PCA()
    #    pca.fit(X)
    #    features[i, 0:9] = np.ravel(pca.components_)
    #    curvature = np.amin(pca.explained_variance_) / np.sum(pca.explained_variance_)
    #    if np.isnan(curvature):
    #        curvature = 0
    #    features[i, 9] = curvature
    #features
    #print(featureData.shape)
    #print(xyz.shape)
    xyz = np.concatenate((xyz,featureData),axis=1)

    return xyz.astype(np.float32), Aindices.astype(np.int64), Avalues.astype(np.float32)

poolFactory = SparsePoolingFactory()
pooler = poolFactory.CreatePoolingPyramid(len(POOL_RATIOS), SphericalSparseAdjacencyCompander,\
                                           POOL_RATIOS,POOLING_ID)

with open(FILE_LIST_PATH) as f:
    fileList = f.readlines()
counter = 0
for file in fileList:
    file = file.strip()
    V,Aindices,Avalues = calculate_features(PREFIX + '/' + file, K=K, MAX_SIZE=MAX_SIZE, aug=True,feature_type=0)
    print(file)
    counter += 1
    print(counter)
    print(np.amax(Aindices[:,1]))
    print(np.amin(Aindices[:,1]))
    if np.amax(Aindices[:,1]) >= THETA*PHI or np.amin(Aindices[:,1]) < 0:
        print(np.amax(Aindices[:,1]))
        print(np.amin(Aindices[:,1]))
        print('FAIL')
        sys.exit(0)
    pooler.makeP(Aindices,Avalues,[MAX_SIZE,THETA*PHI,MAX_SIZE],V, theta=THETA,phi=PHI)
