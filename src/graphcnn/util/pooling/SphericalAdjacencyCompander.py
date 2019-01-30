from .AbstractAdjacencyCompander import AbstractAdjacencyCompander
import numpy as np
from .SphericalUtil import *
from .SplitUtils import *

class SphericalAdjacencyCompander(AbstractAdjacencyCompander):

    def __init__(self,V,A,numTheta,numPhi):
        super(SphericalAdjacencyCompander, self).__init__(V, A)
        self.numPhi = numPhi
        self.numTheta = numTheta
        #Extra 0.01 due to quantization error border issues
        self.thetaKeys = np.linspace(0, 2*np.pi + 0.01,num=self.numTheta + 1,endpoint=True)
        self.phiKeys = np.linspace(0,np.pi + 0.01,num=self.numPhi + 1,endpoint=True)

        self.thetaVals = np.arange(self.numTheta)
        self.phiVals = np.arange(self.numPhi)

    def contractA(self):
        self.flatA = self.A.sum(axis=1)
        return self.flatA

    def expandA(self):
        (iVals,jVals) = np.nonzero(self.flatA)
        Aindices, Avalues, Adenseshape = sphericalSplit(self.V, iVals, jVals, self.N, self.numTheta, self.numPhi, self.thetaKeys, self.phiKeys)
        expandedA = np.zeros(Adenseshape)
        expandedA[Aindices[:,0],Aindices[:,1],Aindices[:,2]] = Avalues
        #expandedA[Aindices[:, 2], Aindices[:, 1], Aindices[:, 0]] = Avalues
        self.A = expandedA

        return expandedA
