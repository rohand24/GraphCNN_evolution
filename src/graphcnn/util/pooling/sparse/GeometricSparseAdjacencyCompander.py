from .AbstractSparseAdjacencyCompander import AbstractSparseAdjacencyCompander
import numpy as np
import sys
import os
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from SplitUtils import *

class GeometricSparseAdjacencyCompander(AbstractSparseAdjacencyCompander):

    def __init__(self,V,Aindices,Avalues,Adenseshape):
        super(GeometricSparseAdjacencyCompander, self).__init__(V,Aindices,Avalues,Adenseshape)
        self.numDirs = 8

    def expandA(self):
        flatACoo = self.flatA.tocoo()
        iVals = flatACoo.row
        jVals = flatACoo.col
        self.Aindices, self.Avalues, self.Adenseshape = geometricSplit(self.V,iVals,jVals,self.N)
        return self.Aindices, self.Avalues, self.Adenseshape
