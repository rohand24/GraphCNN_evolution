from abc import ABCMeta, abstractmethod

class AbstractPoolingPyramid:
    __metaclass__ = ABCMeta

    def __init__(self,numRepresentations,companderConstructor):
        self.numRepresentations = numRepresentations
        self.companderConstructor = companderConstructor

    @abstractmethod
    def makeP(self,A,V=None,theta=0,phi=0):

        pass

    @abstractmethod
    def write(self,Ps):
        pass
