from .AbstractPoolingPyramid import AbstractPoolingPyramid
import scipy.sparse
import pyamg
import numpy as np
#from graphcnn.util.modelnet.pointCloud2Graph import ply2graph
import tensorflow as tf

class LloydPoolingPyramid(AbstractPoolingPyramid):

    def __init__(self,numRepresentations,companderConstructor, ratios):
        super(LloydPoolingPyramid, self).__init__(numRepresentations,companderConstructor)
        self.ratios = ratios

    def makeP(self,A,V=None, theta=0,phi=0):
        Plist = []
        #print(V.shape)
        if theta > 0 and phi > 0:
            companderInstance = self.companderConstructor(V,A,theta,phi)
        else:
            companderInstance = self.companderConstructor(V,A)

        for pIndex in range(self.numRepresentations):
            #print(pIndex)
            Vcollapse = np.sum(V,axis=1)
            Vnonzero = np.count_nonzero(Vcollapse)
            if Vnonzero > (V.shape[0]* self.ratios[pIndex]):
                P = pyamg.aggregation.aggregate.lloyd_aggregation(\
                scipy.sparse.csr_matrix(companderInstance.contractA()),ratio=self.ratios[pIndex],distance='same',maxiter=10)[0]
                P = P.todense()
                Pcolsum = np.tile(np.count_nonzero(P,axis=0),(P.shape[0],1))
                Pcolsum[Pcolsum == 0] = 1
                P = np.divide(P,Pcolsum.astype(np.float64))
            else:
                companderInstance.contractA()
                P = np.eye(V.shape[0],np.floor(V.shape[0]*self.ratios[pIndex]).astype(np.int64))
            Plist.append(P.astype(np.float32))
            #print(P.shape)
            companderInstance.update(P)
            A = companderInstance.expandA()
            V = companderInstance.V
        return Plist

    def write(self,Ps,As):
        AsparseList = []
        for A in As:
            currentA = A.tolist()
            pass
