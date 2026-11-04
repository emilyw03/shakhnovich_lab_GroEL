# import packages
import urllib

#import nglview
import numpy as np
from openff.toolkit import ForceField, Molecule, Topology
from openff.units import unit

from openff.interchange import Interchange
from openff.interchange.drivers import get_openmm_energies
from openff.interchange.drivers.all import get_summary_data

from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule

from collections import Counter
from openff.packmol import RHOMBIC_DODECAHEDRON, solvate_topology, pack_box
from scipy.spatial.distance import pdist

import openmm
import openmm.app
import openmm.unit as omm_unit

import argparse
import os

# Output folder (the sbatch passes the run directory as the first argument)
parser = argparse.ArgumentParser()
parser.add_argument("outdir", type=str)
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

# step 1: prepare protein component
protein = Topology.from_pdb("/n/home00/ewang03/lab/GroEL/openMM_inputs/WT/WT_apo.pdb") # import protein topology (pdb)
ff14sb = ForceField("ff14sb_off_impropers_0.0.3.offxml") # force field to parametrize protein
protein = protein.molecule(0)

# combine protein and force field into interchange 
protein_intrcg = Interchange.from_smirnoff(
    force_field=ff14sb,
    topology=protein.to_topology(),
)

# step 2: prepare ligand component
def load_with_template(sdf, smiles, name):
    '''
    Use coordinates from sdf exported from ligated PDB
    Use topology from SMILES
    Use SMILES topology to fix incorrect topology of sdf
    '''
    pdb_mol = Chem.MolFromMolFile(sdf, removeHs=False, sanitize=False)
    heavy = Chem.RemoveHs(pdb_mol, sanitize=False)          # drop the file's H atoms
    template = Chem.MolFromSmiles(smiles)                   # heavy-atom template, no AddHs
    fixed = AllChem.AssignBondOrdersFromTemplate(template, heavy)
    fixed = Chem.AddHs(fixed, addCoords=True)               # H's to match the SMILES' protonation
    Chem.AssignStereochemistryFrom3D(fixed)
    mol = Molecule.from_rdkit(fixed)
    mol.name = name
    return mol


tmp = load_with_template("/n/home00/ewang03/lab/GroEL/openMM_inputs/WT/tmp.sdf",
    "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC", "TMP")
nadph = load_with_template("/n/home00/ewang03/lab/GroEL/openMM_inputs/WT/nadph.sdf", 
                           "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)([O-])[O-])[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1", 
                           "NADPH")

# Sage 2.3.0 for the ligands. Charges come from NAGL (openff-gnn-am1bcc-1.0.0), so AmberTools/sqm is not needed
sage = ForceField("openff_unconstrained-2.3.0.offxml")

# Residue names so the ligands can be selected later (otherwise they are written out as "UNK")
for mol, resname in [(tmp, "TMP"), (nadph, "NDP")]:
    for atom in mol.atoms:
        atom.metadata["residue_name"] = resname

# One Interchange per ligand; positions are the SDF (crystal) coordinates
tmp_intrcg = Interchange.from_smirnoff(force_field=sage, topology=[tmp])
nadph_intrcg = Interchange.from_smirnoff(force_field=sage, topology=[nadph])

# Combine protein + TMP + NADPH. Atoms are appended in this order and coordinates are kept as-is,
# so the ligands stay in their crystal poses in the binding site (the apo PDB and the SDFs share a frame).
# Expect InterchangeCombinationWarnings: a generic "validate results" one, and one about 1-4 electrostatics
# scaling 0.833333 (ff14SB) vs 0.8333333333 (Sage) -- harmless, both are 1/1.2
complex_intrcg = protein_intrcg.combine(tmp_intrcg).combine(nadph_intrcg)

print(complex_intrcg.topology.n_atoms, "atoms")  # 2491 protein + 39 TMP + 74 NADPH = 2604
print("net charge:", round(sum(q.m for q in complex_intrcg["Electrostatics"].charges.values()), 4))  # -14 (protein -10, NADPH -4)

# Sanity checks: single-point energy (vacuum, no box yet) and a PDB to inspect the poses in VMD/PyMOL
complex_intrcg.to_pdb(os.path.join(args.outdir, "WT_TMP_NADPH.pdb"))
get_openmm_energies(complex_intrcg, combine_nonbonded_forces=False)

# step 3: solvate 
# 1. Center the complex in a rhombic dodecahedron box and fill it with water + 0.15 M NaCl (neutralizing the -14 charge) with packmol
# 2. Parametrize the water and ions with Sage (TIP3P, Na+/Cl-) in their own Interchange
# 3. Combine with the protein-ligand Interchange

# 1.2 nm of water on every side, so periodic images of the complex are >= 2.4 nm apart (> 2 x 0.9 nm cutoff)
padding = 1.2 * unit.nanometer

# Rhombic dodecahedron: ~29% less water than a cube with the same padding. This is the box solvate_topology
# would build itself; it is computed here so the complex can be centered in it first
solute_xyz = complex_intrcg.positions
box = RHOMBIC_DODECAHEDRON * (pdist(solute_xyz.m_as("nanometer")).max() * unit.nanometer + 2 * padding)

# Center the complex in the region packmol fills (otherwise it keeps its crystal coordinates and sits off-center)
complex_top = Topology(complex_intrcg.topology)
complex_top.set_positions(solute_xyz - solute_xyz.mean(axis=0) + np.diagonal(box.m) * box.u / 2)
complex_top.box_vectors = box

# Add water + 0.15 M NaCl, with extra Na+ to neutralize the complex's -14 charge (~15 s)
solvated_top = solvate_topology(complex_top, nacl_conc=0.15 * unit.molar, padding=None)

print(Counter(m.atom(0).symbol for m in solvated_top.molecules if m.n_atoms < 5))  # ~7281 O (water), 28 Na, 14 Cl
print("total atoms:", solvated_top.n_atoms, "| net charge:", sum(m.total_charge for m in solvated_top.molecules))

# Water + ions are every molecule after the solute (protein, TMP, NADPH)
solvent_top = Topology.from_molecules(list(solvated_top.molecules)[complex_top.n_molecules:])
solvent_top.box_vectors = solvated_top.box_vectors

# Residue/atom names so water and ions can be selected in VMD/PyMOL/MDTraj (otherwise they are written as "UNK")
for i, mol in enumerate(solvent_top.molecules, start=1):
    if mol.n_atoms == 3:
        resname, atom_names = "HOH", ["O", "H1", "H2"]
    else:
        resname = {11: "NA", 17: "CL"}[mol.atom(0).atomic_number]
        atom_names = [resname]
    for atom, name in zip(mol.atoms, atom_names):
        atom.name = name
        atom.metadata["residue_name"] = resname
        atom.metadata["residue_number"] = i

# Sage includes rigid TIP3P water and Na+/Cl- parameters
solvent_intrcg = Interchange.from_smirnoff(force_field=sage, topology=solvent_top)

# Move the complex into the solvated frame (centered positions + box), then append water and ions (~15 s)
complex_intrcg.positions = complex_top.get_positions()
complex_intrcg.box = solvated_top.box_vectors
solvated_intrcg = complex_intrcg.combine(solvent_intrcg)

print(solvated_intrcg.topology.n_atoms, "atoms | net charge:",
      round(sum(q.m for q in solvated_intrcg["Electrostatics"].charges.values()), 4))

# Starting structure for visualization / as the topology file for trajectories
solvated_intrcg.to_pdb(os.path.join(args.outdir, "WT_TMP_NADPH_solvated.pdb"))

# Single-point energy with PME. vdW starts large and positive (~ +3e5 kJ/mol): packmol leaves waters closer
# than their LJ radius. It comes from water-water contacts (not the protein) and goes away on minimization
print(get_openmm_energies(solvated_intrcg, combine_nonbonded_forces=False, platform="CPU"))

# step five: export to OpenMM
openmm_system = solvated_intrcg.to_openmm()
openmm_topology = solvated_intrcg.topology.to_openmm(ensure_unique_atom_names=False)
solvated_intrcg.to_pdb(os.path.join(args.outdir, "wt_nadph_tmp_start.pdb"))
print(type(openmm_system), type(openmm_topology))

# step six: run simulation
# Langevin dynamics at 300 K with 2 fs timestep (H-bonds and water are constrained by Sage)
integrator = openmm.LangevinMiddleIntegrator(300 * omm_unit.kelvin, 1 / omm_unit.picosecond, 2 * omm_unit.femtosecond)

# Barostat starts switched off (frequency 0) for NVT equilibration, turned on later for NPT
barostat = openmm.MonteCarloBarostat(1 * omm_unit.bar, 300 * omm_unit.kelvin, 0)

system = solvated_intrcg.to_openmm_system()
top = solvated_intrcg.topology
bonds = next(f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce))
for i in range(bonds.getNumBonds()):   # water bonds are already constraints and aren't in this force
    a, b, r0, k = bonds.getBondParameters(i)
    if top.atom(a).atomic_number == 1 or top.atom(b).atomic_number == 1:
        system.addConstraint(a, b, r0)
system.addForce(barostat)
simulation = openmm.app.Simulation(solvated_intrcg.to_openmm_topology(), system, integrator)
simulation.context.setPositions(solvated_intrcg.positions.to_openmm())

print("platform:", simulation.context.getPlatform().getName())

# Energy minimization, then NVT equilibration: 100 ps at 300 K
simulation.minimizeEnergy()
print("minimized energy:", simulation.context.getState(getEnergy=True).getPotentialEnergy())

simulation.context.setVelocitiesToTemperature(300 * omm_unit.kelvin)
simulation.step(50_000)

# NPT equilibration: 500 ps at 1 bar so the box relaxes to the right density
barostat.setFrequency(25)
simulation.context.reinitialize(preserveState=True)
simulation.step(250_000)
state = simulation.context.getState()
print("box volume after NPT:", state.getPeriodicBoxVolume())

# Production MD (NPT). Saves a frame every 10 ps; adjust n_steps for length (500,000 steps x 2 fs = 1 ns)
simulation.reporters.append(openmm.app.DCDReporter(os.path.join(args.outdir, "wt_with_ligands.dcd"), 5000))
simulation.reporters.append(openmm.app.StateDataReporter(os.path.join(args.outdir, "wt_with_ligands_prod.csv"), 5000, step=True, time=True, potentialEnergy=True,
    temperature=True, volume=True, density=True, speed=True,
))
simulation.step(500_000)

# Save final state so the run can be extended later
simulation.saveState(os.path.join(args.outdir, "wt_with_ligands.xml"))
