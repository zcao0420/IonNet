import numpy as np
import math
from scipy.constants import N_A

def bin_water(zmax, zmin, idx, xyz, binSize = 0.01):
    mid = (zmax+zmin)/2
    n_bins = math.ceil((zmax-mid)/binSize)
    left = [0 for i in range(n_bins)]
    right = left.copy()

    for oz in xyz[idx, 2]:
        whichBin = math.floor(abs((oz-mid))//binSize)
        if oz>=mid:
            right[whichBin] += 1
        else:
            left[whichBin] += 1

    return np.array(left[::-1] + right)

def bin_ion(zmax, zmin, idx, xyz, binSize = 0.01, avg = False):
    mid = (zmax+zmin)/2
    n_bins = math.ceil((zmax-mid)/binSize)
    left = [0 for i in range(n_bins)]
    right = left.copy()

    for oz in xyz[idx, 2]:
        whichBin = math.floor(abs((oz-mid))//binSize)
        if oz>=mid:
            right[whichBin] += 1
        else:
            left[whichBin] += 1
    if avg:
        right = left = np.mean(np.vstack([right, left]), axis = 0)
        return np.hstack([left[::-1], right])
    return np.array(left[::-1] + right)

def binDensity(xdim, ydim, zdim, count):
    mass = count/N_A * 18.01528 #mass in gram
    vol = xdim*ydim*zdim * 1e-21
    return mass/vol

def binMolDen(xdim, ydim, zdim, count):
    mol = count/N_A
    vol = xdim*ydim*zdim * 1e-24 # volume in Liter
    return mol/vol
