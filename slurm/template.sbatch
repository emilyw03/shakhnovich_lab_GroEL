#!/bin/bash
# Starter Slurm job. Copy this file per job type, edit the settings below, then:
#   sbatch slurm/template.sbatch            (run from the repo root)
#
# Each job gets its own folder: ~/scratch/GroEL/runs/<date>_<jobid>/
# Copy results worth keeping to ~/lab/GroEL/results/ before scratch purges them (90 days).

#SBATCH --job-name=groel
#SBATCH --account=shakhnovich_lab
#SBATCH --partition=test             # test (<=12h, quick checks) | shared | sapphire | shakhnovich | gpu | gpu_test
#SBATCH --time=0-01:00               # D-HH:MM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
##SBATCH --gres=gpu:1                # uncomment (and use a gpu partition) for GPU OpenMM runs
#SBATCH --output=/n/home00/ewang03/scratch/GroEL/logs/%x_%j.out
#SBATCH --error=/n/home00/ewang03/scratch/GroEL/logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=emilywang@g.harvard.edu

set -euo pipefail

# ---- What to run ----
SCRIPT=scripts/your_script.py        # path relative to the repo root
ARGS=""                              # extra arguments for the script

# ---- Paths ----
REPO=/n/home00/ewang03/shakhnovich_lab_GroEL
RUN_DIR=/n/home00/ewang03/scratch/GroEL/runs/$(date +%F)_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# ---- Environment ----
module load python
eval "$(conda shell.bash hook)"
conda activate openMM
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ---- Record what produced this run ----
cd "$REPO"
{
    echo "job_id:   $SLURM_JOB_ID"
    echo "job_name: $SLURM_JOB_NAME"
    echo "node:     $(hostname)"
    echo "started:  $(date)"
    echo "script:   $SCRIPT $ARGS"
    echo "commit:   $(git rev-parse HEAD)"
    git diff --quiet || echo "WARNING: uncommitted changes in repo"
} > "$RUN_DIR/run_info.txt"
cp "$0" "$RUN_DIR/job.sbatch"

# ---- Run ----
# The script should write its outputs to the folder passed as --outdir
python "$SCRIPT" --outdir "$RUN_DIR" $ARGS

echo "finished: $(date)" >> "$RUN_DIR/run_info.txt"
