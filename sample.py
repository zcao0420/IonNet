import mdtraj as md
import numpy as np
import os, math, glob
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
from solv_utils import *
LJ = {
    'NA': ['0.3526', '2.1600'],
    'CL': ['0.0128', '4.8305'],
    'MG': ['0.8750', '2.1200'],
    'LI': ['0.3367', '1.4094'],
    'K':  ['0.4297', '2.8384']
}

def loadTraj(fname):
    topPath = fname.replace('simulation/traj', 'pdb/ionized/').replace('lammpstrj', 'pdb')
    traj = md.load(fname, top=topPath)
    return traj

def sample_single(trajName, ionType, gap, conc):
    traj = loadTraj(trajName)
    top = traj.topology
    ions = top.select('resname %s'%ionType)
    ion_coords = traj.xyz[:, ions, -1]
    ion_coords = np.hstack(ion_coords)
    gra = top.select('resname GRA')
    center = np.mean(traj.xyz[0, gra, -1])
    ion_coords = np.abs(ion_coords-center)

    if ionType == 'CL':
        charge = -1
    elif ionType == 'MG':
        charge = 2
    else:
        charge = 1
    eps, sig = LJ[ionType]

    dist1 = np.random.uniform(0, center+0.1, size=800)
    dist2 = np.random.uniform(max(0, center-0.5), center, size=800)
    dist = np.hstack((dist1, dist2))
    output = []
    for r in dist:
        p = np.sum(ion_coords<r)/len(ion_coords)
        # [dist, gap, concentration, sigma, epsilon, charge, label(prob)]
        output.append([r, gap, conc, float(sig), float(eps), charge, p])
    output=np.array(output)
    return output

if __name__ == "__main__":
    test_gap = set(['1.6', '2.0', '2.4', '2.8'])
    test_conc = set(['1.4', '2.2', '3'])

    ions = list(LJ.keys())
    gap = [0.8+i*0.1 for i in range(23)]
    conc = [0.8+i*0.2 for i in range(15)]
    comb = []
    for i in ions:
        for g in gap:
            for c in conc:
                comb.append((i, g, c))

    train_data = []
    test_data = []
    for item in tqdm(comb):
        ion, g, c = item
        trajName = 'simulation/traj/%s_%s_%s.lammpstrj'%(ion, f2str(g), f2str(c))
        data = sample_single(trajName, ion, g, c)
        if f2str(g) in test_gap or f2str(c) in test_conc:
            test_data.append(data)
        else:
            train_data.append(data)
    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)
    np.savez('data/train.npz', data=train_data)
    np.savez('data/test.npz', data=test_data)
    # pd.DataFrame(train_data).to_csv("data/train.csv", header=None, index=None)
    # pd.DataFrame(test_data).to_csv("data/test.csv", header=None, index=None)
