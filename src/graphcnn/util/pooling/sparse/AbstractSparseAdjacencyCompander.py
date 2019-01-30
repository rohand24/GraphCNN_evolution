from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.sparse


class AbstractSparseAdjacencyCompander(object):
    __metaclass__ = ABCMeta

    def __init__(self,V,Aindices,Avalues,Adenseshape):
        self.V = V
        self.Aindices = Aindices.astype(np.int64) #Ex3 representing an NxLxN structure
        self.Avalues = Avalues #Ex1
        self.Adenseshape = Adenseshape
        self.N = V.shape[0]
        self.flatA = 0

    def contractA(self):
        #print(self.Adenseshape)
        #print(self.Aindices.shape)
        self.flatA = scipy.sparse.coo_matrix((self.Avalues,(self.Aindices[:,0],self.Aindices[:,2])),shape=(self.Adenseshape[0],self.Adenseshape[2]))
        self.flatA.sum_duplicates()
        self.flatA = self.flatA.tocsr()
        #print(self.flatA.shape)
        #print(self.flatA)
        return self.flatA
    @abstractmethod
    def expandA(self):
        pass

    #Assum
    def update(self,P):
        self.flatA = scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(P.transpose().tocsr(),self.flatA),P)
        self.V = scipy.sparse.csr_matrix.dot(P.transpose().tocsr(),self.V)
        self.N = self.V.shape[0]
