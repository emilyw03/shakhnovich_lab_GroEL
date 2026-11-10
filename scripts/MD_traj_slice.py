'''
MD_traj_slice.py
Date: Oct 28, 2026
Author: Emily Wang
Slice trajectory to sparsify frames saved and reduce file size
'''

import MDAnalysis as mda
import os

top  = "/n/home00/ewang03/scratch/GroEL/runs/2026-11-04_57868420/wt_nadph_tmp_start.pdb"
traj = "/n/home00/ewang03/scratch/GroEL/runs/2026-11-04_57868420/wt_with_ligands.xtc"

u = mda.Universe(top, traj)   # topology first, then trajectory

print(u)                      # <Universe with 27716 atoms>
print(len(u.trajectory))      # number of frames
print(u.trajectory.dt)        # time between frames (ps)

protein = u.select_atoms("protein or resname TMP NDP")   # protein + TMP + NADPH; drops water (HOH) and ions (NA, CL)
protein.write("/n/home00/ewang03/scratch/GroEL/runs/2026-11-04_57868420/wt_nadph_tmp_dry.pdb")  # matching topology

with mda.Writer("/n/home00/ewang03/scratch/GroEL/runs/DHFR_apo_100ns_combined_54960039+55672625/DHFR_apo_100ns_stride100_dry.xtc", protein.n_atoms) as W:
    for ts in u.trajectory[::100]: # save every 100th trajectory
        W.write(protein)

