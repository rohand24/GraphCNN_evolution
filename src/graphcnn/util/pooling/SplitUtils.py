from SphericalUtil import *

def sphericalSplit(V,iVals,jVals,N,numTheta,numPhi,thetaKeys,phiKeys):
    edges = V[iVals,:] - V[jVals,:]
    Avalues = np.linalg.norm(V[iVals,:] - V[jVals,:],axis=1)
    Avalues = np.concatenate((Avalues, Avalues))
    r, theta, phi = toSpherical(np.array(edges))
    thetaClassLabels = np.clip(np.digitize(theta,thetaKeys) - 1,0,numTheta - 1)
    phiClassLabels = np.clip(np.digitize(phi,phiKeys) - 1,0,numPhi - 1)
    zindex = classifyPoint(thetaClassLabels,phiClassLabels,numPhi)
    #print('{0} {1} {2}'.format(np.amax(thetaClassLabels),np.amax(phiClassLabels),np.amax(zindex)))
    Aindices = np.stack((np.ravel(iVals), np.ravel(zindex), np.ravel(jVals)), axis=1)
    AindicesT = np.stack((np.ravel(jVals), np.ravel(zindex), np.ravel(iVals)), axis=1)
    Aindices = np.concatenate((Aindices,AindicesT))
    Aindices,uniqueIndices = np.unique(Aindices,return_index=True,axis=0)
    Avalues = Avalues[uniqueIndices]
    Adenseshape = np.array([N, numTheta*numPhi, N])
    return Aindices.astype(np.int64), Avalues,Adenseshape

def geometricSplit(V,iVals,jVals,N):
    V = V[:,0:3]
    numDirs = 8
    zindex = np.dot([4, 2, 1],
                    np.greater((V[iVals, :] - V[jVals, :]).transpose(), np.zeros((3, iVals.shape[0]))));
    Avalues = np.linalg.norm(V[iVals, :] - V[jVals, :], axis=1)
    Avalues = np.concatenate((Avalues,Avalues))
    Aindices = np.stack((np.ravel(iVals), np.ravel(zindex), np.ravel(jVals)), axis=1)
    AindicesT = np.stack((np.ravel(jVals), np.ravel(zindex), np.ravel(iVals)), axis=1)
    Aindices = np.concatenate((Aindices,AindicesT))
    Aindices,uniqueIndices = np.unique(Aindices,return_index=True, axis=0)
    Avalues = Avalues[uniqueIndices]
    Adenseshape = np.array([N, numDirs, N])

    return Aindices.astype(np.int64), Avalues, Adenseshape
