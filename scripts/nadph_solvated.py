'''
nadph_solvated.py
Date: Nov 3, 2026
Author: Emily Wang
Simulate NADPH solvated. Use OpenFF to parametrize the small molecule, then simulate with OpenMM
'''
from openff.toolkit import Molecule, ForceField
from openff.units import unit
from collections import Counter
from openff.interchange.components._packmol import solvate_topology, RHOMBIC_DODECAHEDRON
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

# smiles string representation of molecule
nadph_smiles = "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)([O-])[O-])[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1" # NADPH (4- at pH = 7.3). From ChEBI

# generate molecule from SMILES
nadph = Molecule.from_smiles(nadph_smiles, allow_undefined_stereo=True)

# packmol needs 3D coordinates for the solute
nadph.generate_conformers(n_conformers=1)

# Solvate with water + 0.15 M NaCl; extra Na+ are added to neutralize NADPH's -4 charge
solvated_topology = solvate_topology(
    nadph.to_topology(),
    nacl_conc=0.15 * unit.molar,
    padding=1.2 * unit.nanometer,
    box_shape=RHOMBIC_DODECAHEDRON,
)

print(Counter(m.to_smiles(explicit_hydrogens=False) for m in solvated_topology.molecules if m.n_atoms < 5))
print("total atoms:", solvated_topology.n_atoms)
print("net charge:", sum(m.total_charge for m in solvated_topology.molecules))

# Label residues so NADPH, water and ions can be selected in VMD/PyMOL/MDTraj (otherwise everything is "UNK")
residue_names = {"[H][O][H]": "HOH", "[Na+]": "NA", "[Cl-]": "CL"}
for i, molecule in enumerate(solvated_topology.molecules):
    resname = residue_names.get(molecule.to_smiles(), "NDP")
    for atom in molecule.atoms:
        atom.metadata["residue_name"] = resname
        atom.metadata["residue_number"] = i + 1
    if resname == "HOH":
        for atom, name in zip(molecule.atoms, ["O", "H1", "H2"]):
            atom.name = name
    elif resname in ("NA", "CL"):
        molecule.atom(0).name = resname
    # Rebuild residues from the new metadata; without this the PDB still says "UNK"
    molecule.update_hierarchy_schemes()

# Parametrize the full system (Sage includes TIP3P water and Na+/Cl- parameters)
sage = ForceField("openff-2.3.0.offxml")
interchange = sage.create_interchange(solvated_topology)

# Save starting structure for visualization / as the topology file for the trajectory
interchange.to_pdb(os.path.join(args.outdir, "nadph_water_start.pdb"))

# Langevin dynamics at 300 K with 2 fs timestep (H-bonds and water are constrained by Sage)
integrator = openmm.LangevinMiddleIntegrator(300 * omm_unit.kelvin, 1 / omm_unit.picosecond, 2 * omm_unit.femtosecond)

# Barostat starts switched off (frequency 0) for NVT equilibration, turned on later for NPT
barostat = openmm.MonteCarloBarostat(1 * omm_unit.bar, 300 * omm_unit.kelvin, 0)

simulation = interchange.to_openmm_simulation(integrator, additional_forces=[barostat])
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
simulation.reporters.append(openmm.app.DCDReporter(os.path.join(args.outdir, "nadph_water_prod.dcd"), 5000))
simulation.reporters.append(openmm.app.StateDataReporter(
    os.path.join(args.outdir, "nadph_water_prod.csv"), 5000, step=True, time=True, potentialEnergy=True,
    temperature=True, volume=True, density=True, speed=True,
))
simulation.step(500_000)

# Save final state so the run can be extended later
simulation.saveState(os.path.join(args.outdir, "nadph_water_prod_final.xml"))

