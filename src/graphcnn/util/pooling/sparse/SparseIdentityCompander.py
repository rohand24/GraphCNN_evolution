from .AbstractSparseAdjacencyCompander import AbstractSparseAdjacencyCompander
import numpy as np
import sys
import os
import os.path
import scipy.sparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))

class SparseIdentityCompander(AbstractSparseAdjacencyCompander):

    def __init__(self,V,Aindices,Avalues,Adenseshape):
        super(SparseIdentityCompander, self).__init__(V,Aindices,Avalues,Adenseshape)
        self.numDirs = 8

    def expandA(self):
        return self.Aindices, self.Avalues, self.Adenseshape

    def update(self,P):
        Arows = []
        Acols = []
        Alayers = []
        Avals = []
        for i in range(self.Adenseshape[1]):
            indexSlice = self.Aindices[np.where(self.Aindices[:,1]==i)]
            valuesSlice = self.Avalues[np.where(self.Aindices[:,1]==i)]
            Aslice = scipy.sparse.coo_matrix((valuesSlice,(indexSlice[:,0],indexSlice[:,2])),shape=(self.Adenseshape[0],self.Adenseshape[2]))
            Aslice = Aslice.tocsr()
            Aslice = scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(P.transpose().tocsr(),Aslice),P)
            Aslice = Aslice.tocoo()
            Arows.append(Aslice.row)
            Acols.append(Aslice.col)
            Avals.append(Aslice.data)
            Alayers.append(i*np.ones(len(Aslice.row)))
        Arows = np.concatenate(Arows)
        Acols = np.concatenate(Acols)
        Alayers = np.concatenate(Alayers)
        self.Avalues = np.concatenate(Avals)
        self.Aindices =  np.stack((Arows, Alayers, Acols), axis=1)
        self.Adenseshape = [P.shape[1], self.Adenseshape[1], P.shape[1]]


        self.V = scipy.sparse.csr_matrix.dot(P.transpose().tocsr(),self.V)
