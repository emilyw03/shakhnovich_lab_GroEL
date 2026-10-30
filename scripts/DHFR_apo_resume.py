# Continue a DHFR apo run that stopped before finishing (e.g. hit the Slurm time limit).
# Same system and integrator settings as DHFR_apo.py, but starts from a saved structure
# (restart PDB) or an OpenMM checkpoint instead of re-solvating and re-minimizing.

from openmm import * # import OpenMM
from openmm.app import *
from openmm.unit import *

# command line argument parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("outdir", type=str)
parser.add_argument("--topology", type=str, required=True) # solvated PDB the original run used (must be the same atoms in the same order)
parser.add_argument("--restart-pdb", type=str) # positions + box from the last trajectory frame (no velocities)
parser.add_argument("--checkpoint", type=str) # OpenMM checkpoint (exact restart: positions, velocities, box, step)
parser.add_argument("--start-step", type=int) # step the restart PDB corresponds to (not needed with --checkpoint)
args = parser.parse_args()
if (args.restart_pdb is None) == (args.checkpoint is None):
    parser.error("give exactly one of --restart-pdb or --checkpoint")
if args.restart_pdb is not None and args.start_step is None:
    parser.error("--restart-pdb needs --start-step")

# Input Files

pdb = PDBFile(args.topology)
forcefield = ForceField('amber19-all.xml', 'amber19/tip3pfb.xml')

# System Configuration (keep identical to DHFR_apo.py)

nonbondedMethod = PME
nonbondedCutoff = 1.0*nanometers
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001
ewaldErrorTolerance = 0.0005

# Integration Options

dt = 0.002*picoseconds
temperature = 300*kelvin
friction = 1.0/picosecond
pressure = 1.0*atmospheres
barostatInterval = 25

# Simulation Options

steps = 50_000_000 # total length of the full run, including what was already simulated
reequilibrationSteps = 5000 # 10 ps to let freshly assigned velocities relax (only used with --restart-pdb, not saved)
checkpointInterval = 50_000 # save a checkpoint every 100 ps so the next interruption can resume exactly

# Prepare the Simulation

print('Building system...')
topology = pdb.topology
system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
    constraints=constraints, rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance)
system.addForce(MonteCarloBarostat(pressure, temperature, barostatInterval))
integrator = LangevinMiddleIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)
simulation = Simulation(topology, system, integrator)

if args.checkpoint is not None:
    print('Loading checkpoint...')
    simulation.loadCheckpoint(args.checkpoint) # restores positions, velocities, box, and currentStep
else:
    print('Loading restart structure and re-equilibrating...')
    restart = PDBFile(args.restart_pdb)
    simulation.context.setPeriodicBoxVectors(*restart.topology.getPeriodicBoxVectors())
    simulation.context.setPositions(restart.positions)
    simulation.context.applyConstraints(constraintTolerance) # trajectory coordinates are rounded to 0.001 nm
    simulation.context.setVelocitiesToTemperature(temperature) # velocities aren't stored in the trajectory
    simulation.step(reequilibrationSteps)
    simulation.currentStep = args.start_step

startStep = simulation.currentStep
remainingSteps = steps - startStep
print(f'Resuming at step {startStep}, {remainingSteps} steps remaining')

# Reporters write new files in outdir; the earlier traj.xtc / log.txt are left untouched.
# Step numbers continue from startStep so the pieces can be concatenated for analysis.
dataReporter = StateDataReporter(args.outdir + '/log.txt', 100, totalSteps=steps,
    step=True, speed=True, progress=True, potentialEnergy=True, temperature=True, density=True, separator='\t')
xtcReporter = XTCReporter(args.outdir + '/traj.xtc', 100)
checkpointReporter = CheckpointReporter(args.outdir + '/checkpoint.chk', checkpointInterval)

# Simulate

print('Simulating...')
simulation.reporters.append(dataReporter)
simulation.reporters.append(xtcReporter)
simulation.reporters.append(checkpointReporter)
simulation.step(remainingSteps)
simulation.saveCheckpoint(args.outdir + '/checkpoint.chk')
