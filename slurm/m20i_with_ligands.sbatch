#!/bin/bash
# Starter Slurm job. Copy this file per job type, edit the settings below, then:
#   sbatch slurm/template.sbatch            (run from the repo root)
#
# Each job gets its own folder: ~/scratch/GroEL/runs/<date>_<jobid>/
# Copy results worth keeping to ~/lab/GroEL/results/ before scratch purges them (90 days).

#SBATCH --job-name=m20i_with_ligands_1us
#SBATCH --account=shakhnovich_lab
#SBATCH --partition=gpu,gpu_requeue,gpu_h200   # comma-separated, NO spaces; Slurm starts the job on whichever has a GPU free first
                                         # test (<=12h, quick checks) | shared | sapphire | shakhnovich | gpu | gpu_test | gpu_requeue | gpu_h200
                                         # only run MD on gpu, analysis on CPU
#SBATCH --time=3-00:00               # D-HH:MM (3 days is the max on all three partitions)
#SBATCH --requeue                    # gpu_requeue can preempt the job; it is then requeued and resumes from its last checkpoint (see Paths)
#SBATCH --open-mode=append           # keep the .out/.err from earlier attempts when the job is requeued
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
SCRIPT=scripts/m20i_with_ligands.py        # path relative to the repo root
ARGS=""                              # use this line if starting a new sim
#ARGS="--checkpoint /n/home00/ewang03/scratch/GroEL/runs/<date>_<jobid>/checkpoint.xml"     # Uncomment if resuming a run from a checkpoint

# ---- Paths ----
REPO=/n/home00/ewang03/lab/shakhnovich_lab_GroEL/
RUNS=/n/home00/ewang03/scratch/GroEL/runs
ATTEMPT=${SLURM_RESTART_COUNT:-0}    # 0 on first start, 1, 2, ... each time Slurm requeues this job

if [ "$ATTEMPT" -eq 0 ]; then
    RUN_DIR=$RUNS/$(date +%F)_${SLURM_JOB_ID}
else
    # Requeued (preempted): each attempt gets its own folder so earlier output is never overwritten,
    # and resumes from the newest checkpoint written by an earlier attempt of this job
    RUN_DIR=$RUNS/$(date +%F)_${SLURM_JOB_ID}_requeue${ATTEMPT}
    PREV_CHK=$(ls -t "$RUNS"/*_"${SLURM_JOB_ID}"*/checkpoint.xml 2>/dev/null | head -n 1 || true)
    if [ -n "$PREV_CHK" ]; then
        ARGS="--checkpoint $PREV_CHK"
    fi                               # no checkpoint yet (preempted during setup/equilibration): start over with the original ARGS
fi
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
python -u "$SCRIPT" "$RUN_DIR" $ARGS   # -u: print to the .out log immediately instead of buffering

echo "finished: $(date)" >> "$RUN_DIR/run_info.txt"
