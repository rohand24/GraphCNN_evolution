import numpy as np
import sklearn.decomposition.pca
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plyfile
import scipy.spatial
import transforms3d
import time
FILE = "C:/data/modelnet10_10000_points/modelnet10_10000_points/chair/test/chair_0890.ply"
K = 13
def ply2array(plyPath):
    plydata = plyfile.PlyData.read(plyPath)
    vertices = []
    #vertices = numpy.array(plydata['vertex'][0])
    for i in range(plydata['vertex'].count):
        vertices.append(list(plydata['vertex'][i]))
    vertices = np.array(vertices)
    return vertices

V = ply2array(FILE)
vertexMean = np.mean(V, axis=0)
vertexStd = np.std(V, axis=0)
if not vertexStd.all():
    M = np.eye(3)
    angle = np.random.uniform(0.01, 0.1, size=3)
    sign = np.random.choice([-1, 1], size=3, replace=True)
    M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], sign[0] * angle[0]), M)
    M = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], sign[1] * angle[1]), M)
    M = np.dot(transforms3d.axangles.axangle2mat([1, 0, 0], sign[2] * angle[2]), M)
    V = np.dot(V, M.T)
    vertexMean = np.mean(V, axis=0)
    vertexStd = np.std(V, axis=0)
V = (V - vertexMean) / vertexStd
kdtree = scipy.spatial.KDTree(V)
# First nearest neighbor is always the point itself!
_, knns = kdtree.query(V, k=K)
#print(knns)
#X = np.random.rand(6,3)
#3x3 eigenvector parameters and 1 curvature parameters
features = np.zeros((V.shape[0],10))
tic = time.time()
for i in range(V.shape[0]):
    X = V[knns[i,:],:]
    pca = sklearn.decomposition.PCA()
    pca.fit(X)
    features[i,0:9] = np.ravel(pca.components_)
    curvature = np.amin(pca.explained_variance_) / np.sum(pca.explained_variance_)
    if np.isnan(curvature):
        curvature = 0
    features[i,9] = curvature

toc = time.time() - tic
print(toc)
print(features)

#rint(pca.components_)
#print(pca.explained_variance_ratio_)
#fig = plt.figure()
#ax = fig.add_subplot('111', projection='3d')
# flatA = graph.flattenA()
#eigenvalues = pca.explained_variance_
#curvature = np.amin(eigenvalues)/np.sum(eigenvalues)
#print(curvature)
#for i in range(pca.components_.shape[0]):
#    ax.plot([pca.components_[i, 0],0], [pca.components_[i, 1], 0], [pca.components_[i, 2], 0])
#ax.scatter(X[:, 0], X[:, 1], X[:, 2])
#plt.show()