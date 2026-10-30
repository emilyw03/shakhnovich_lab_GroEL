'''
join_runs.py
Date: Oct 30, 2026
Author: Emily Wang
Join log.txt and traj.xtc from a run and the runs that resumed it from checkpoints into one
continuous log and trajectory.

Usage:
    python join_runs.py <outdir> <run folder 1> <run folder 2> ...

When a job hits the time limit, its trajectory and log keep going past the last checkpoint,
and the resumed run repeats those steps. Each earlier run is therefore cut just before the
first step saved by the run that follows it, so every step appears once. Runs are put in
order by their first saved step, so the folders can be given in any order.

The trajectory is streamed in chunks, so memory use stays small however long the run is,
but the joined traj.xtc takes as much disk space as the pieces combined.
'''

import mdtraj as md
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("outdir", type=str) # folder for the joined log.txt and traj.xtc
parser.add_argument("runs", type=str, nargs='+') # run folders to join
parser.add_argument("--log-name", type=str, default='log.txt') # log file name inside each run folder
parser.add_argument("--traj-name", type=str, default='traj.xtc') # trajectory file name inside each run folder
parser.add_argument("--chunk", type=int, default=1000) # frames read at a time
args = parser.parse_args()


def first_traj_step(path):
    '''step number of the first frame in an xtc file (reads one frame, no index needed)'''
    with md.formats.XTCTrajectoryFile(path, 'r') as f:
        xyz, time, step, box = f.read(n_frames=1)
    if len(step) == 0:
        raise ValueError(f'{path} has no frames')
    return int(step[0])


def read_chunks(path, chunk):
    '''yield (xyz, time, step, box) chunks of an xtc file. A job killed mid-write can leave a truncated
    last frame; if a chunk fails to read, re-read it one frame at a time to keep the whole frames before it.'''
    nRead = 0
    with md.formats.XTCTrajectoryFile(path, 'r') as f:
        while True:
            try:
                frames = f.read(n_frames=chunk)
            except Exception:
                break
            if len(frames[2]) == 0:
                return
            nRead += len(frames[2])
            yield frames
    with md.formats.XTCTrajectoryFile(path, 'r') as f:
        f.seek(nRead)
        while True:
            try:
                frames = f.read(n_frames=1)
            except Exception as e:
                print(f'WARNING: {path} ends in a truncated frame after {nRead} whole frames ({e})')
                return
            if len(frames[2]) == 0:
                return
            nRead += 1
            yield frames


def read_log(path):
    '''header line and data rows of a StateDataReporter log; drops a partial last line from a killed job'''
    with open(path) as f:
        lines = f.read().splitlines()
    header = lines[0]
    columns = header.lstrip('#').split('\t')
    stepCol = columns.index('"Step"')
    rows = []
    for line in lines[1:]:
        fields = line.split('\t')
        if len(fields) != len(columns):
            continue
        rows.append((int(fields[stepCol]), line))
    return header, rows


# Put runs in order and find where each one is cut

runs = []
for runDir in args.runs:
    traj = os.path.join(runDir, args.traj_name)
    log = os.path.join(runDir, args.log_name)
    for path in (traj, log):
        if not os.path.exists(path):
            parser.error(f'{path} not found')
    runs.append({'dir': runDir, 'traj': traj, 'log': log, 'start': first_traj_step(traj)})
runs.sort(key=lambda r: r['start'])
for i, run in enumerate(runs):
    run['cutoff'] = runs[i + 1]['start'] if i + 1 < len(runs) else None # keep steps < cutoff (None = keep all)
    print(f"{run['dir']}: starts at step {run['start']}, keeping steps < {run['cutoff'] if run['cutoff'] is not None else 'end'}")

os.makedirs(args.outdir, exist_ok=True)

# Join logs

print('Joining logs...')
nRows = 0
lastStep = -1
with open(os.path.join(args.outdir, 'log.txt'), 'w') as out:
    for i, run in enumerate(runs):
        header, rows = read_log(run['log'])
        if i == 0:
            out.write(header + '\n')
        for step, line in rows:
            if run['cutoff'] is not None and step >= run['cutoff']:
                break
            if step <= lastStep:
                print(f"WARNING: {run['log']} step {step} is not after previous step {lastStep}")
            out.write(line + '\n')
            lastStep = step
            nRows += 1
print(f'Wrote {nRows} log rows')

# Join trajectories

print('Joining trajectories...')
nFrames = 0
lastStep = -1
with md.formats.XTCTrajectoryFile(os.path.join(args.outdir, 'traj.xtc'), 'w') as out:
    for run in runs:
        nRun = 0
        for xyz, time, step, box in read_chunks(run['traj'], args.chunk):
            keep = np.ones(len(step), dtype=bool) if run['cutoff'] is None else step < run['cutoff']
            if keep.any():
                if np.any(np.diff(np.concatenate(([lastStep], step[keep]))) <= 0):
                    print(f"WARNING: {run['traj']} has steps out of order near step {step[keep][0]}")
                out.write(xyz[keep], time=time[keep], step=step[keep], box=box[keep])
                lastStep = int(step[keep][-1])
                nRun += int(keep.sum())
            if not keep.all(): # reached the cutoff
                break
        print(f"  {run['traj']}: {nRun} frames")
        nFrames += nRun
print(f'Wrote {nFrames} frames, last step {lastStep}')

with open(os.path.join(args.outdir, 'joined_from.txt'), 'w') as f:
    for run in runs:
        f.write(f"{os.path.abspath(run['dir'])}\tstart={run['start']}\tcutoff={run['cutoff']}\n")
