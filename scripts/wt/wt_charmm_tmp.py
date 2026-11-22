# wt_charmm_tmp.py
# Simulate WT DHFR with TMP (neutral) and NADPH bound (CHARMM36m protein, CGenFF v5.0 for small molecules, CHARMM TIP3P for water and ions) with OpenMM.
#
# New run:     python wt_charmm_tmp.py <outdir>
#              builds and solvates the system, minimizes, equilibrates (NVT then NPT), then runs production
# Resume:      python wt_charmm_tmp.py <outdir> --checkpoint <old run dir>/checkpoint.xml
#              skips building and equilibration, loads system.xml + wt_nadph_tmp_start.pdb from the checkpoint's
#              run folder, and continues production from the checkpoint's step until TOTAL_STEPS is reached

# import packages
import openmm
import openmm.app
from openmm.app.element import Element
import openmm.unit as omm_unit

import argparse
import os
import shutil

# Output folder (the sbatch passes the run directory as the first argument)
parser = argparse.ArgumentParser()
parser.add_argument("outdir", type=str)
parser.add_argument("--checkpoint", type=str) # resume from this checkpoint.xml (or an older binary .chk); its run folder must also hold system.xml and wt_nadph_tmp_start.pdb
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

TOTAL_STEPS = 500_000_000        # production length across all runs (500,000,000 steps x 2 fs = 1000 ns = 1 us)
CHECKPOINT_INTERVAL = 50_000    # 100 ps
REPORT_INTERVAL = 5_000       # one frame / log line every 10 ps

# Files a resumed run needs from the run it continues. Kept in every run folder so resumes can be chained
SYSTEM_XML = "system.xml"                # OpenMM System: force field parameters, constraints, barostat
START_PDB = "wt_nadph_tmp_start.pdb"     # topology (atom/residue names) for the Simulation and trajectories
INPUTS = "/n/home00/ewang03/lab/GroEL/openMM_inputs"

def load_mol2(path, resname):
    '''OpenMM Topology + positions for one ligand from a mol2 file (atom names, elements, bonds, coordinates)'''
    sections = open(path).read().split("@<TRIPOS>")
    atom_lines = next(s for s in sections if s.startswith("ATOM")).splitlines()[1:]
    bond_lines = next(s for s in sections if s.startswith("BOND")).splitlines()[1:]
    top = openmm.app.Topology()
    res = top.addResidue(resname, top.addChain())
    atoms, pos = {}, []
    for l in atom_lines:
        f = l.split()
        if len(f) >= 6:
            atoms[f[0]] = top.addAtom(f[1], Element.getBySymbol(f[5].split(".")[0]), res)
            pos.append(openmm.Vec3(*map(float, f[2:5])) * 0.1)      # Angstrom -> nm
    for l in bond_lines:
        f = l.split()
        if len(f) >= 3:
            top.addBond(atoms[f[1]], atoms[f[2]])
    return top, pos * omm_unit.nanometer


# Langevin dynamics at 300 K with 2 fs timestep (rigid CHARMM TIP3P water; X-H bonds constrained via constraints=HBonds in step 4)
integrator = openmm.LangevinMiddleIntegrator(300 * omm_unit.kelvin, 1 / omm_unit.picosecond, 2 * omm_unit.femtosecond)

if args.checkpoint is None:
    # step 1: protein + ligands in their crystal poses (apo PDB and ligand mol2 files share a frame).
    # The mol2 files were built from the M20I crystal poses, which match the WT poses to 0.03 A (TMP) / 0.05 A (NADPH) RMSD
    protein = openmm.app.PDBFile(f"{INPUTS}/WT/WT_apo.pdb")
    modeller = openmm.app.Modeller(protein.topology, protein.positions)
    for path, resname in [(f"{INPUTS}/tmp/tmp.mol2", "TMP"), (f"{INPUTS}/nadph/nadph.mol2", "NDP")]:
        modeller.add(*load_mol2(path, resname))

    # step 2: CHARMM36m + CHARMM TIP3P/ions, CGenFF 5.0 ligands (scripts/cgenff_to_openmm.py)
    ff = openmm.app.ForceField("charmm36_2024.xml", "charmm36_2024/water.xml",
                               f"{INPUTS}/cgenff_xml/tmp_cgenff5.xml", f"{INPUTS}/cgenff_xml/nadph_cgenff5.xml")

    # step 3: rhombic dodecahedron, 1.2 nm padding, 0.15 M NaCl + neutralizing Na+ (~6630 waters, 32 Na, 18 Cl)
    modeller.addSolvent(ff, model="tip3p", padding=1.2 * omm_unit.nanometer, boxShape="dodecahedron",
                        ionicStrength=0.15 * omm_unit.molar, positiveIon="Na+", negativeIon="Cl-", neutralize=True)
    with open(os.path.join(args.outdir, START_PDB), "w") as f:
        openmm.app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

    # step 4: CHARMM36m nonbonded settings; HBonds constraints for the 2 fs timestep
    system = ff.createSystem(modeller.topology, nonbondedMethod=openmm.app.PME,
                             nonbondedCutoff=1.2 * omm_unit.nanometer, switchDistance=1.0 * omm_unit.nanometer,
                             constraints=openmm.app.HBonds, rigidWater=True, useDispersionCorrection=False)
    barostat = openmm.MonteCarloBarostat(1 * omm_unit.bar, 300 * omm_unit.kelvin, 0)
    system.addForce(barostat)
    simulation = openmm.app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    print("platform:", simulation.context.getPlatform().getName())

    # step 5: energy minimization, then NVT equilibration: 100 ps at 300 K
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

    # Re-zero the step counter and simulation time so production starts at step 0 / 0 ps
    simulation.currentStep = 0
    simulation.context.setTime(0)

    # Save the System (with the barostat now on) so a resumed run uses exactly these parameters
    with open(os.path.join(args.outdir, SYSTEM_XML), "w") as f:
        f.write(openmm.XmlSerializer.serialize(system))

else:
    # Resume: reuse the system and topology from the checkpoint's run folder instead of rebuilding
    # (Modeller places water and ions afresh, so rebuilding would not match the checkpoint's coordinates)
    prev_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    for name in (SYSTEM_XML, START_PDB):
        shutil.copy(os.path.join(prev_dir, name), os.path.join(args.outdir, name))

    with open(os.path.join(args.outdir, SYSTEM_XML)) as f:
        system = openmm.XmlSerializer.deserialize(f.read())
    pdb = openmm.app.PDBFile(os.path.join(args.outdir, START_PDB))
    simulation = openmm.app.Simulation(pdb.topology, system, integrator)
    print("platform:", simulation.context.getPlatform().getName())

    # Restores positions, velocities, box vectors, time and the step counter.
    # A .xml State works on any GPU type; a binary .chk only loads on the same platform and GPU model
    if args.checkpoint.endswith(".xml"):
        simulation.loadState(args.checkpoint)
    else:
        simulation.loadCheckpoint(args.checkpoint)

# step 6: production MD (NPT). Step numbers continue from the checkpoint when resuming
remaining_steps = TOTAL_STEPS - simulation.currentStep
print(f"Production from step {simulation.currentStep}: {remaining_steps} steps remaining")

simulation.reporters.append(openmm.app.XTCReporter(os.path.join(args.outdir, "wt_with_ligands.xtc"), REPORT_INTERVAL))
simulation.reporters.append(openmm.app.StateDataReporter(os.path.join(args.outdir, "wt_with_ligands_prod.csv"), REPORT_INTERVAL,
    step=True, time=True, potentialEnergy=True, temperature=True, volume=True, density=True, speed=True))
# Checkpoint every 100 ps; the file is overwritten each time so a killed run can resume with --checkpoint.
# writeState=True saves a portable XML State instead of a binary checkpoint, because the partitions this runs on
# have different GPU models (A100, MIG slices, H200) and a requeued job can land on any of them
simulation.reporters.append(openmm.app.CheckpointReporter(os.path.join(args.outdir, "checkpoint.xml"), CHECKPOINT_INTERVAL, writeState=True))
simulation.step(remaining_steps)

# Save final state so the run can be extended later
simulation.saveState(os.path.join(args.outdir, "wt_with_ligands.xml"))
simulation.saveState(os.path.join(args.outdir, "checkpoint.xml"))
