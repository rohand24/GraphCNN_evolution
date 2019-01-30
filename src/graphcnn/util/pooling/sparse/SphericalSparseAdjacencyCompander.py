from .AbstractSparseAdjacencyCompander import AbstractSparseAdjacencyCompander
import numpy as np
import sys
import os
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from SphericalUtil import *
from SplitUtils import *

class SphericalSparseAdjacencyCompander(AbstractSparseAdjacencyCompander):

    def __init__(self,V,Aindices,Avalues,Adenseshape,numTheta,numPhi):
        super(SphericalSparseAdjacencyCompander, self).__init__(V,Aindices,Avalues,Adenseshape)
        self.numTheta = numTheta
        self.numPhi = numPhi

        self.thetaKeys = np.linspace(0, 2 * np.pi + 0.01, num=self.numTheta + 1, endpoint=True)
        self.phiKeys = np.linspace(0, np.pi + 0.01, num=self.numPhi + 1, endpoint=True)

        self.thetaVals = np.arange(self.numTheta)
        self.phiVals = np.arange(self.numPhi)

    def expandA(self):
        flatACoo = self.flatA.tocoo()
        iVals = flatACoo.row
        jVals = flatACoo.col
        self.Aindices, self.Avalues, self.Adenseshape = sphericalSplit(self.V,iVals,jVals,self.N,self.numTheta,self.numPhi,self.thetaKeys,self.phiKeys)
        return self.Aindices, self.Avalues, self.Adenseshape
