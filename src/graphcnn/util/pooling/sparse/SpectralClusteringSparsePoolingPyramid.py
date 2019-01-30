from .AbstractSparsePoolingPyramid import AbstractSparsePoolingPyramid
import scipy.sparse
import numpy as np
import sklearn.cluster
import scipy.sparse

class SpectralClusteringSparsePoolingPyramid(AbstractSparsePoolingPyramid):

    def __init__(self,numRepresentations,companderConstructor, ratios):
        super(SpectralClusteringSparsePoolingPyramid, self).__init__(numRepresentations,companderConstructor)
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
            #print(pIndex)
            outSize = int(np.floor(self.ratios[pIndex]*Adenseshape[0]))
            #print(outSize)
            #t = time.time()
            numComponents = int(np.maximum(np.floor(outSize/4),1))
            labels = sklearn.cluster.spectral_clustering(scipy.sparse.csr_matrix(companderInstance.contractA()),\
                                            n_clusters=outSize,eigen_solver='arpack',n_init=1,n_components=numComponents)
            #print(labels.shape)
            #print(Adenseshape[0])
            Pindices = np.stack((np.arange(Adenseshape[0]),labels),axis=1).astype(np.int64)
            Pvalues = np.ones(Adenseshape[0])
            P = scipy.sparse.coo_matrix((Pvalues,(Pindices[:,0],Pindices[:,1])))
            #print(P)
            #print('Nonzero P: {0}'.format(np.count_nonzero(P)))
            Pcolsum = P.sum(axis=0)
            Pcolsum[Pcolsum == 0] = 1
            Pcolsum = np.reciprocal(Pcolsum)
            #print("LOL")
            #print(Pcolsum.shape)
            D = scipy.sparse.diags(np.ravel(Pcolsum),format='csr')
            P = P.tocsr()
            #print(D.shape)
            #print(P.shape)
            P = scipy.sparse.csr_matrix.dot(P,D)
            #print(P.shape)
            companderInstance.update(P)
            Aindices,Avalues,Adenseshape = companderInstance.expandA()
            Pcurrent = P.tocoo()
            singleton = np.zeros(Pcurrent.row.shape[0])
            Pindices = np.stack((singleton, Pcurrent.row, Pcurrent.col), axis=1)
            Pvalues = Pcurrent.data
            Pshape = np.array([1, Pcurrent.shape[0], Pcurrent.shape[1]])
            Pidxlist.append(Pindices.astype(np.int64))
            Pvallist.append(Pvalues.astype(np.float32))
            Pshapelist.append(Pshape.astype(np.int64))
        return Pidxlist + Pvallist + Pshapelist
