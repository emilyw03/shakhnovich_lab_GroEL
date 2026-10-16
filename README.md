File lay out
~/shakhnovich_lab_GroEL/        # git repo (code only)
  src/ or scripts/              # analysis and simulation code
  slurm/                        # sbatch job scripts
  notebooks/
  config.py (or paths.yaml)     # data locations defined in one place
  environment.yml               # conda env spec, so the env can be rebuilt
  .gitignore                    # ignores data/, results/, *.h5, *.npy, etc.

~/lab/GroEL/
  raw/                          # original inputs, never modified
  results/                      # final processed outputs you want to keep
  figures/

~/scratch/GroEL/
  runs/<date_or_jobid>/         # per-job outputs
