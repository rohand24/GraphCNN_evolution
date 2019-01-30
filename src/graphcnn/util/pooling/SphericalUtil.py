import numpy as np

def binarySearch(keys, vals, point):
    L = 0
    R = len(keys) - 1
    M = (L + R) // 2
    while L < R:
        M = (L + R) // 2
        if (keys[M] <= point):
            L = M + 1
        else:
            R = M
    return vals[L - 1]

def classifyPoint(thetaLabel,phiLabel,numPhi):
    return (thetaLabel)*numPhi + phiLabel

def fromSpherical(theta, phi, radius):
    x = radius*np.cos(theta)*np.sin(phi)
    y = radius*np.sin(theta)*np.sin(phi)
    z = radius*np.cos(phi)
    return (x,y,z)

def toSpherical(V):
    r = np.sqrt(np.sum(V*V,axis=1))
    theta = np.arctan2(V[:,1],V[:,0]) + np.pi #Not sure if adding pi to go from -pi to pi to 0 to 2pi is correct
    #for p in sorted(V[:,2]/(r + np.finfo(np.float32).eps)):
    #    print(p)
    phi = np.arccos(V[:,2]/(r + np.finfo(np.float32).eps))

    return (r,theta,phi)
