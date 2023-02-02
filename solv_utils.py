import numpy as np
import mdtraj as md
from scipy.constants import N_A
from mdtraj.core.element import Element
from pathlib import Path
import os, subprocess, math, shutil
HEADER="""LAMMPS data file. CGCMM style. Author: Zhonglin Cao
 #NATOMS atoms
 #NBONDS bonds
 #NANGLES angles
 0 dihedrals
 0 impropers
 #NTYPES atom types
 1 bond types
 1 angle types
 0 dihedral types
 0 improper types
 #XMIN #XMAX  xlo xhi
 #YMIN #YMAX  ylo yhi
 #ZMIN #ZMAX  zlo zhi
 
# Pair Coeffs
#
# 1  C
# 2  H
# 3  O
# 4  #ION


 Masses

"""
atom_dict = {
    'C': [1, 12.01078, '0.000000'],
    'H': (2, 1.007947, '0.423800'),
    'O': (3, 15.99943, '-0.847600'),
    'Na': (4, 22.98977, '1.000000'),
    'Li': (4, 6.941, '1.000000'),
    'K': (4, 39.0983, '1.000000'),
    'Mg': (4, 24.305, '2.000000'),
    'Cl': (4, 35.453, '-1.000000'),
    
}

PARAMS = {
    'C_sigma': 0.339,
    'xdim': [0-0.0614012, 3.0700600+0.0614012],
    'ydim': [0-0.0709, 3.2613998+0.0709]
}

def NWater(x, y, z):
    """
    Given x, y and z dimension in nm,
    return number of water molecules in bulk.
    Results rounded to int
    """
    vol = x*y*z * 1e-21 # volume in cm^3
    mass = vol # mass in gram
    mol = mass / 18.01528 # mole of water equivalent to the mass
    n = mol * N_A
    return int(n)
    
def solvateWrapper(z, x=3, y=3, solvPath='pdb/waterbox/'):
    dim = [f2str(3.1928624), f2str(3.4031998), f2str(z)]
    pdbName = os.path.join(solvPath, 'wb_%s.pdb'%(dim[2]))
    topName = os.path.join(solvPath, 'wb_%s.top'%(dim[2]))
    if os.path.exists(pdbName):
        delExist(pdbName)
    if os.path.exists(topName):
        delExist(topName)
    nwater = NWater(x=3.1928624, y=3.4031998, z=z)
    
    command_pdb = [' '.join([
        '/usr/local/gromacs/bin/gmx',
        'solvate', '-box %s %s %s'%(dim[0],dim[1],dim[2]),
        '-o %s'%pdbName, '-scale 0.001', '-maxsol %d'%nwater
        ])
    ]

    tempPath = 'pdb/temp'

    command_top = [' '.join([
        'printf "8" |'
        '/usr/local/gromacs/bin/gmx',
        'pdb2gmx', '-f %s'%pdbName,
        '-o %s'%(os.path.join(tempPath, 'temp.pdb')),
        '-p %s'%topName, '-water spce'
        ])
    ]
    stdout = subprocess.call(
        command_pdb, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    stdout = subprocess.call(
        command_top, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    delExist(os.path.join(tempPath, 'temp.pdb'))
    return

def delExist(fname):
    if os.path.exists(fname):
        os.remove(fname)
    return

def mod_top(topology, mode):
    ion_id = topology.select('resname NA CL')
    if mode == 'pos':
        ion = 'Na'
    else:
        ion = 'Cl'
    for idx in ion_id:
        topology.atom(idx).element=Element.getBySymbol(ion)
    return topology


def increaseZ(pdbName):
    # with is like your try .. finally block in this case
    with open(pdbName, 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    # now change the 2nd line, note that you have to add a newline
    data[2] = 'CRYST1   40.000   40.000   30.000  90.00  90.00  90.00 P 1           1\n'
    # and write everything back
    with open(pdbName, 'w') as file:
        file.writelines(data)

def NIons(x, y, z, conc):
    actual = (conc * N_A)/(1e24) * (x*y*z)
    if actual < 0.5:
        raise Exception('Actual number of ions is less than 0.5, large mismatch of concentration.')
    return int(round(actual))

def f2str(num):
    s = str(num).rstrip('0').rstrip('.')
    if len(s)>16:
        s = s[:-1].rstrip('0')
    return s

def shrink_patty(xdim, ydim, patty):
    xc, yc = np.mean(xdim), np.mean(ydim)
    top = patty.topology
    xyz = patty.xyz[0]
    o_id = top.select('resname HOH and element O')
    ion_id = top.select('resname NA CL LI K MG')
    # Align patty to the center of the simulation box
    non_gra_id = top.select('resname HOH NA CL LI K MG')
    xmax, xmin = max(xyz[non_gra_id, 0]), min(xyz[non_gra_id, 0])
    ymax, ymin = max(xyz[non_gra_id, 1]), min(xyz[non_gra_id, 1])
    dx, dy = (xmax+xmin)/2-xc, (ymax+ymin)/2-yc
    # print(dx, dy, xc, yc)
    patty.xyz[0, non_gra_id, 0] -= dx
    patty.xyz[0, non_gra_id, 1] -= dy
    # Shrink the patty
    xratio = (xdim[1]-xdim[0])/(max(patty.xyz[0, non_gra_id, 0])-min(patty.xyz[0, non_gra_id, 0]))
    yratio = (ydim[1]-ydim[0])/(max(patty.xyz[0, non_gra_id, 1])-min(patty.xyz[0, non_gra_id, 1]))
    xratio -= 0.02
    yratio -= 0.02
    # print(xratio, yratio)
    for idx in o_id:
        x, y = xyz[idx, 0], xyz[idx, 1]
        for i in range(3):
            patty.xyz[0, idx+i, 0] -= (x-xc)*(1-xratio)
            patty.xyz[0, idx+i, 1] -= (y-yc)*(1-yratio)
    for idx in ion_id:
        x, y = xyz[idx, 0], xyz[idx, 1]
        patty.xyz[0, idx, 0] = xc+((x-xc)*xratio)
        patty.xyz[0, idx, 1] = yc+((y-yc)*yratio)
    return patty

def processCoord(c):
    c = np.round(c, 3)
    return '{:.6f}'.format(c)

def write2lammpsdata(Trajectory, outputName, params=PARAMS):
    topo = Trajectory.topology
    xyz = Trajectory.xyz[0] * 10
    df, bondList = topo.to_dataframe()
    atom_types = df['element'].to_list()
    revised_atom_types = []
    resname_list = df['resName'].to_list()
    n_ions = 0
    n_carbons = 0
    for i in range(len(resname_list)):
        if resname_list[i] == 'HOH':
            if 'H' in atom_types[i]:
                revised_atom_types.append('H')
            else:
                revised_atom_types.append('O')
        elif resname_list[i] == 'GRA':
            n_carbons +=1
            revised_atom_types.append('C')
        else:
            if resname_list[i] == 'MG':
                n_ions-=2
            elif resname_list[i] == 'CL':
                n_ions+=1
            else:
                n_ions-=1
            revised_atom_types.append(resname_list[i].capitalize())
            ion = resname_list[i].capitalize()
    all_atom_types = set(revised_atom_types)
    
    n_atoms = topo.n_atoms
    n_bonds = topo.n_bonds
    n_angles = n_bonds//2
    n_types = len(all_atom_types)

    xmin, xmax = params['xdim'][0]*10, params['xdim'][1]*10
    ymin, ymax = params['ydim'][0]*10, params['ydim'][1]*10

    zmin, zmax = np.min(xyz[:, 2])-40, np.max(xyz[:, 2])+40
    
    carbon_charge = n_ions/n_carbons

    header = HEADER.replace('#NATOMS', str(n_atoms)).replace('#NBONDS', str(n_bonds))\
                    .replace('#NANGLES', str(n_angles)).replace('#NTYPES', str(n_types))\
                    .replace('#ION', ion)\
                    .replace('#XMIN', '{:.6f}'.format(xmin)).replace('#XMAX', '{:.6f}'.format(xmax))\
                    .replace('#YMIN', '{:.6f}'.format(ymin)).replace('#YMAX', '{:.6f}'.format(ymax))\
                    .replace('#ZMIN', '{:.6f}'.format(zmin)).replace('#ZMAX', '{:.6f}'.format(zmax))
    
    f = open(outputName, "w")
    f.write(header)
    for ele in atom_dict.keys():
        if ele in all_atom_types:
            info = atom_dict[ele]
            f.write(' %d %s # %s\n'%(info[0], str(info[1]), ele))
    
    f.write('\n Atoms\n \n')
    for i in range(n_atoms):
        atom = topo.atom(i)
        ele = revised_atom_types[i]
        index = str(i+1)
        type_index, mass, charge = atom_dict[ele]
        if ele == 'C':
            charge = "{:.6f}".format(carbon_charge)
                    
        x, y, z = xyz[i]
        x, y, z = processCoord(x), processCoord(y), processCoord(z)
        f.write(' '.join([index, '0', str(type_index), charge, x, y, z, '#', ele, '\n']))
    
    f.write(' \n Bonds\n \n')
    
    for i, b in enumerate(topo.bonds):
        a1, a2 = b.atom1, b.atom2
        f.write(' '.join([str(i+1),'1', str(a1.index+1), str(a2.index+1)])+'\n')
    
    f.write(' \n Angles\n \n')
    
    for i, b in enumerate(bondList):
        if (i+1)%2 == 0:
            center, atom2 = str(int(b[0]+1)), str(int(b[1]+1))
            atom1 = str(int(bondList[i-1][1]+1))
            angleIndex = str((i+1)//2)
            f.write(' '.join([angleIndex,'1', atom1, center, atom2])+'\n')
    f.close()
    return

def ionize(z, conc, pname='NA', pq=1, pos=True):
    """
    Input argument:
        1. waterbox_pdb: path to waterbox pdb
        2. conc: concentration in mol/liter
        3. pname: cation name (sodium in default)
        4. pq: cation charge (1 in default)
        5. filename: output file name
    """ 
    # fname = waterbox_pdb
    # dim = [f2str(x), f2str(y), f2str(z)]
    dim = [f2str(3.1928624), f2str(3.4031998), f2str(z)]
    fname = 'wb_%s'%(dim[2])
    wbPath = 'pdb/waterbox'
    ionPath = 'pdb/ionized'
    path = os.path.join('pdb/waterbox', fname)

    mdpName = os.path.join(ionPath, '%s_%s_%s.mdp'%(pname, dim[2], f2str(conc)))
    tprName = os.path.join(ionPath, '%s_%s_%s.tpr'%(pname, dim[2], f2str(conc)))
    ionizedPDB = os.path.join(ionPath, '%s_%s_%s.pdb'%(pname, dim[2], f2str(conc)))
    # ionizedTOP = os.path.join(ionPath, '%s_%s_%s.top'%(pname, dim[2], f2str(conc)))
    delExist(mdpName)
    delExist(tprName)
    delExist(ionizedPDB)
    # delExist(ionizedTOP)

    # Path(os.path.join(path, 'ions.mdp')).touch()
    Path(mdpName).touch()

    command_grompp = [' '.join([
        '/usr/local/gromacs/bin/gmx',
        'grompp', '-f %s'%mdpName,
        '-p %s'%os.path.join(wbPath, fname+'.top'),
        '-c %s'%os.path.join(wbPath, fname+'.pdb'),
        '-o %s'%tprName
        ])
    ]
    nIons = NIons(3.1928624, 3.4031998, z, conc)
    if z<3:
        increaseZ(os.path.join(wbPath, fname+'.pdb'))
    if pos:
        ionLine = '-pname %s -np %d'%(pname, nIons)
    else:
        assert(pname=='CL')
        ionLine = '-nname %s -nn %d'%(pname, nIons)
    command_ion = [' '.join([
        'printf "SOL" |',
        '/usr/local/gromacs/bin/gmx',
        'genion', '-s %s'%tprName,
        ionLine,
        '-rmin 0.3',
        '-o %s'%ionizedPDB,
        ])
    ]
    stdout = subprocess.call(
        command_grompp, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    stdout = subprocess.call(
        command_ion, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    delExist(mdpName)
    delExist(tprName)
    # stdout = subprocess.call(
    #     command_grompp, shell=True
    # )
    # stdout = subprocess.call(
    #     command_ion, shell=True
    # )
    return

def addWall(z, conc, x=3, y=3, mode='pos', ion='NA'):
    fname = 'wb_%s'%f2str(z)
    folder = os.path.join('waterbox', fname)
    patty = md.load_pdb('pdb/ionized/%s_%s_%s.pdb'%(ion, f2str(z), f2str(conc)))
    idx = patty.top.select('resname HOH %s'%ion)
    # print(sum(patty.xyz[0][idx, 0]>params['xdim'][1]))
    patty = shrink_patty(PARAMS['xdim'], PARAMS['ydim'], patty)
    # print(sum(patty.xyz[0][idx, 0]>PARAMS['xdim'][1]))
    # print(max(patty.xyz[0, idx, 0]), PARAMS['xdim'][1])
    top = md.load_pdb('pdb/wall/gra3x3.pdb')
    bot = md.load('pdb/wall/gra3x3.pdb')
    top.xyz[:, :, 2] += (PARAMS['C_sigma']/2+z)
    bot.xyz[:, :, 2] -= PARAMS['C_sigma']/2

    xdim = PARAMS['xdim'][1] - PARAMS['xdim'][0]
    ydim = PARAMS['ydim'][1] - PARAMS['ydim'][0]
    final_unitcell_length = np.array([[xdim, ydim, z+6]])

    comb_topology = top.topology.join(bot.topology).join(patty.topology)
    comb_xyz =  np.array([np.vstack((top.xyz[0], bot.xyz[0], patty.xyz[0]))])
    # comb_topology = mod_top(comb_topology, mode=mode)
    combined_traj = md.Trajectory(comb_xyz, comb_topology, 
                                unitcell_lengths = final_unitcell_length,
                                unitcell_angles = patty.unitcell_angles
                                )
    outName = 'pdb/ionized/%s_%s_%s.pdb'%(ion, f2str(z), f2str(conc))
    delExist(outName)
    combined_traj.save_pdb(outName)
    return combined_traj