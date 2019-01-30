import sys
import os
import os.path
from .AbstractSparsePoolingPyramid import AbstractSparsePoolingPyramid
#from .GraclusPoolingPyramid import GraclusPoolingPyramid
from .LloydSparsePoolingPyramid import LloydSparsePoolingPyramid
from .SpectralClusteringSparsePoolingPyramid import SpectralClusteringSparsePoolingPyramid
#import matlab.engine

class SparsePoolingFactory():

    #def __init__(self):
     #   pass
        #self.eng = matlab.engine.start_matlab()
        #self.eng.addpath(os.path.abspath(os.path.join(os.path.dirname(__file__),'./graclus1.2/matlab/')))

    def CreatePoolingPyramid(self,numRepresentations, companderConstructor, ratios, id='Spectral'):
        if id == 'Lloyd':
            return LloydSparsePoolingPyramid(numRepresentations,companderConstructor,ratios)
        elif id == 'Spectral':
            return SpectralClusteringSparsePoolingPyramid(numRepresentations,companderConstructor,ratios)
        elif id == 'Graclus':
            raise('I do not yet support Sparse Graclus SO TIME TO CRASH')
            #return GraclusPoolingPyramid(numRepresentations,companderConstructor,ratios)#,self.eng)
