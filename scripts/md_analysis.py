'''
md_analysis.py
Date: Oct 28, 2026
Author: Emily Wang
Analysis of MD trajectory data from OpenMM
'''

import mdtraj as md
import os
import matplotlib.pyplot as plt
import numpy as np

traj = md.load("/n/home00/ewang03/scratch/GroEL/runs/DHFR_apo_100ns_combined_54960039+55672625/DHFR_apo_100ns_stride100_dry.xtc",
               top = "/n/home00/ewang03/lab/GroEL/results/openMM/DHFR_apo_protein.pdb") # load trajectory (xtc) and topology (pdb)

traj.superpose(traj) # normalize structure to first frame
rmsf = md.rmsf(traj, None, mode='residue') # compute RMSF

# plot RMSF vs. residue index
res_ind = np.arange(traj.n_residues)
plt.plot(res_ind, rmsf)
plt.xlabel('residue index')
plt.ylabel('RMSF')
plt.show()