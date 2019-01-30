from .AbstractSparsePoolingPyramid import AbstractSparsePoolingPyramid
import scipy.sparse
import pyamg
import numpy as np
#from graphcnn.util.modelnet.pointCloud2Graph import ply2graph

class LloydSparsePoolingPyramid(AbstractSparsePoolingPyramid):

    def __init__(self,numRepresentations,companderConstructor, ratios):
        super(LloydSparsePoolingPyramid, self).__init__(numRepresentations,companderConstructor)
        self.ratios = ratios

    def makeP(self,Aindices,Avalues,Adenseshape,V=None, theta=0,phi=0):
        Pidxlist = []
        Pvallist = []
        Pshapelist = []
        if theta > 0 and phi > 0:
            companderInstance = self.companderConstructor(V,Aindices,Avalues,Adenseshape,theta,phi)
        else:
            companderInstance = self.companderConstructor(V,Aindices,Avalues,Adenseshape)

        for pIndex in range(self.numRepresentations):
            Vcollapse = np.sum(V,axis=1)
            Vnonzero = np.count_nonzero(Vcollapse)
            if Vnonzero > (V.shape[0]* self.ratios[pIndex]):
                #print(pIndex)
                #print(Aindices.shape)
                #print(Adenseshape)
                P = pyamg.aggregation.aggregate.lloyd_aggregation(\
                scipy.sparse.csr_matrix(companderInstance.contractA()),ratio=self.ratios[pIndex],distance='same',maxiter=10)[0]
                P = P.astype(np.float32)
                Pcolsum = P.sum(axis=0)
                #print(Pcolsum)
                Pcolsum[Pcolsum == 0] = 1
                Pcolsum = np.reciprocal(Pcolsum)
                #print(Pcolsum)
                D = scipy.sparse.diags(np.ravel(Pcolsum), format='csr')
                P = P.tocsr()
                P = scipy.sparse.csr_matrix.dot(P, D)
            else:
                companderInstance.contractA()
                P = scipy.sparse.eye(V.shape[0],np.floor(V.shape[0]*self.ratios[pIndex]).astype(np.int64))
                P = P.tocsr()
                #print('lol')
            Pcurrent = P.tocoo()
            singleton = np.zeros(Pcurrent.row.shape[0])
            Pindices = np.stack((singleton, Pcurrent.row, Pcurrent.col), axis=1)
            Pvalues = Pcurrent.data
            Pshape = np.array([1, Pcurrent.shape[0], Pcurrent.shape[1]])
            Pidxlist.append(Pindices.astype(np.int64))
            Pvallist.append(Pvalues.astype(np.float32))
            Pshapelist.append(Pshape.astype(np.int64))
            companderInstance.update(P)
            Aindices, Avalues, Adenseshape = companderInstance.expandA()
            V = companderInstance.V

        return Pidxlist + Pvallist + Pshapelist
