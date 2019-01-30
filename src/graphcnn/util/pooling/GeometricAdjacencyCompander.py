from .AbstractAdjacencyCompander import AbstractAdjacencyCompander
import numpy as np
from .SplitUtils import *

class GeometricAdjacencyCompander(AbstractAdjacencyCompander):

    def __init__(self,V,A):
        super(GeometricAdjacencyCompander, self).__init__(V, A)
        self.numDirs = 8

    def contractA(self):
        self.flatA = self.A.sum(axis=1)
        return self.flatA

    def expandA(self):
        (iVals, jVals) = np.nonzero(self.flatA)
        Aindices, Avalues, Adenseshape = geometricSplit(self.V, iVals, jVals, self.N)
        expandedA = np.zeros(Adenseshape)
        expandedA[Aindices[:, 0], Aindices[:, 1], Aindices[:, 2]] = Avalues
        #expandedA[Aindices[:, 2], Aindices[:, 1], Aindices[:, 0]] = Avalues
        self.A = expandedA

        return expandedA
