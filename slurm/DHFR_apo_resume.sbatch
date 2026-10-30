#!/bin/bash
# Resume DHFR_apo 100 ns run 54960039 (stopped at step 4439400 by the 1 h time limit).
#   sbatch slurm/DHFR_apo_resume.sbatch     (run from the repo root)
#
# Each job gets its own folder: ~/scratch/GroEL/runs/<date>_<jobid>/
# Copy results worth keeping to ~/lab/GroEL/results/ before scratch purges them (90 days).

#SBATCH --job-name=DHFR_apo_100ns_resume
#SBATCH --account=shakhnovich_lab
#SBATCH --partition=gpu_test                  # test (<=12h, quick checks) | shared | sapphire | shakhnovich | gpu | gpu_test | gpu_requeue, # can only request one gpu partition at once
                                         # only run MD on gpu, analysis on CPU
#SBATCH --time=0-12:00               # D-HH:MM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1                # uncomment (and use a gpu partition) for GPU OpenMM runs
#SBATCH --output=/n/home00/ewang03/scratch/GroEL/logs/%x_%j.out
#SBATCH --error=/n/home00/ewang03/scratch/GroEL/logs/%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=emilywang@g.harvard.edu

set -euo pipefail

# ---- What to run ----
SCRIPT=scripts/DHFR_apo_resume.py       # path relative to the repo root
PREV_RUN=/n/home00/ewang03/scratch/GroEL/runs/2026-10-23_54960039   # run being continued
ARGS="--topology $PREV_RUN/DHFR_WT_apo_solvated.pdb --restart-pdb $PREV_RUN/restart_step4439400.pdb --start-step 4439400"
# after a cancelled resume, use instead: ARGS="--topology $PREV_RUN/DHFR_WT_apo_solvated.pdb --checkpoint <that run dir>/checkpoint.chk"

# ---- Paths ----
REPO=/n/home00/ewang03/lab/shakhnovich_lab_GroEL/
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
python "$SCRIPT" "$RUN_DIR" $ARGS

echo "finished: $(date)" >> "$RUN_DIR/run_info.txt"
