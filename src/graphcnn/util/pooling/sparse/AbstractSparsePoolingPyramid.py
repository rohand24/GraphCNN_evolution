from abc import ABCMeta, abstractmethod

class AbstractSparsePoolingPyramid:
    __metaclass__ = ABCMeta

    def __init__(self,numRepresentations,companderConstructor):
        self.numRepresentations = numRepresentations
        self.companderConstructor = companderConstructor

    @abstractmethod
    def makeP(self,Aindices,Avalues,Adenseshape,V=None,theta=0,phi=0):
        pass