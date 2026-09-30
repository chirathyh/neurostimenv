# Healthy reference calibration for the 60 second L23Net protocol

Run this calibration before held-out stimulation confirmation. The completed
S1 discovery favored fixed 14-Hz stimulation on three model seeds, but its
Healthy target was duration-matched rather than matched to the absolute time
of the stimulation outcome. One Healthy qualification trajectory cannot
establish stationarity. This experiment estimates the missing reference; it
does not retest stimulation efficacy or replace the completed S1 result.

## Design

Rerun the original independent Healthy calibration structures, seeds
8101–8116, for 60 seconds. These are the same sixteen structures, not sixteen
additional independent replicates. They remain disjoint from S1 discovery and
future efficacy-confirmation seeds. Every trajectory has MDD=false,
DRUG=false, 1000 cells, 624 MPI ranks, dt=0.025 ms, temperature=34 degrees C,
env_seed=0, and one persistent episode initialized at -80 mV. There is no
active field: the qualified uniform-field interface stays enabled with exactly
zero amplitude throughout. No mechanism or shared integrator is changed.

Exclude 0–8 seconds from all features, including the initial transient and
the historical one-off network input at 4 seconds. Preserve the S1 epochs:

| Purpose | Absolute interval | Duration |
| --- | --- | --- |
| Original eligibility screen | (8, 28] seconds | 20 seconds |
| Short baseline for washout drift audit | (18, 28] seconds | 10 seconds |
| New plateau reference | (29, 49] seconds | 20 seconds |
| New washout reference | (50, 60] seconds | 10 seconds |

Plateau and washout are analysis labels only; these Healthy runs receive no
stimulation. Use the existing continuous causal anti-alias filter, 250-Hz
analysis rate, 4-second Hann Welch windows, 50% overlap, and log10 integrated
theta (4–8 Hz), alpha (8–12 Hz), and low-beta (12–16 Hz) power. Do not count
overlapping spectral segments as independent replicates. The 20-second and
10-second epochs provide nine and four Welch segments respectively.

The new targets use all sixteen Healthy structures and the existing scale-floor
rule. Save sensitivity targets excluding the fixed 6-, 10-, and 14-Hz carriers
plus/minus 0.5 Hz. Retain the original S0 baseline eligibility target unchanged.
Report paired, duration-matched early-to-late changes, two-sided sign-flip
tests and FDR across six drift contrasts. A nonsignificant drift test does not
prove equivalence or stationarity. Drift does not select seeds or alternative
epochs; no target may be fitted to a favorable subset.

## Resources and deadline

Each worker requests normal, 624 CPUs, 256 GB, two hours, and 10 GB jobfs.
The completed 60-second S1 jobs took about 75–82 minutes. Expected actual cost
is approximately 25.6 KSU for sixteen Healthy runs, with a maximum reservation
of 39.936 KSU under the existing normal-queue model of 2 SU per CPU-hour.
Final PBS accounting, not an extrapolation or process-RSS sum, determines cost.

At balances sj53=5.61, fa32=7.37, ny83=60.11 KSU, retaining 1 KSU per project,
the allocator assigns 1, 2 and 13 workers respectively. It queries current
balances by default. All sixteen are independently eligible by default, a
maximum of 9984 CPUs across 208 nodes; the scheduler may run fewer concurrently.
This avoids two serial waves near renewal but does not guarantee queue starts,
completion before renewal, or which allocation period is charged. Do not
change to express or reduce the validated MPI rank count without a new plan.
Use `--max-concurrent 8` if desired, accepting at least two runtime waves.

## NCI commands

Use a clean checkout with no earlier jobs queued or running from it. Do not
update the checkout or virtual environment after submission until all workers
finish. No previously exported experiment variables are needed.

```bash
cd /g/data/ny83/ch9972/NeuroStim/neurostimenv
git switch feature/l23net-sinusoidal-uniform-field
git pull --ff-only origin feature/l23net-sinusoidal-uniform-field
module load python3/3.10.4
module load openmpi/5.0.5
source /g/data/ny83/ch9972/NeuroStim/bin/activate
python3 -m pip check

python3 experiments/l23net_analysis/nci/submit_l23net_reference60.py
```

Inspect the dry-run resource/project plan, then submit once:

```bash
python3 experiments/l23net_analysis/nci/submit_l23net_reference60.py --submit
qstat -u ch9972
```

The submitter automatically locates the latest qualified S1 report and the
exact R1 core identified by the frozen S0 summary hash. Explicit paths are
available as `--qualification /full/path/s1_summary.json` and
`--core /full/path/to/core_suite`. Do not regenerate prerequisites to bypass a
hash failure. If automatic account parsing fails, use freshly checked balances
with `--available-ksu sj53=5.61 fa32=7.37 ny83=60.11` only if still accurate.
Duplicate submission is not deduplicated: inspect the printed suite manifest
and existing job IDs after an interrupted/partial submission before retrying.

## Validation and collection

Before submission, verify each original R1 report and raw trace against the
S0-frozen source summary and save portable prefix fingerprints. After all jobs
finish, the analysis checks native traces, finite EEG, time grids, sparse
spikes, exact zero field, construction, package/source identity, worker exits,
and final PBS accounting. Every new run must reproduce its original R1 first
28 seconds exactly in EEG, dipole and per-window spikes, with the same seed
and construction fingerprints. A mismatch blocks calibration; it is not
silently relabeled as random variation.

The source snapshots are included in the suite, so post-run analysis does not
need the old R1 directory. S1 artifact filenames are deliberately reused for
compatibility with its existing raw-trace auditor; the submission, protocol,
scope and final target distinguish this calibration from S1 discovery.

```bash
python3 experiments/l23net_analysis/analyze_l23net_reference60.py --latest
```

Running analysis too early produces `reference60_status.json` with missing
workers/accounting, never a partial frozen target. A complete success produces
`reference60_summary.json` and `reference60_target.json` with status
`reference_calibrated`. Identical reruns are allowed; a different frozen result
cannot overwrite the first. Copy the entire printed suite to the laptop,
including submission/source/gate JSON, all HDF5 files, worker logs, heartbeat,
environment/mechanism provenance, and PBS stdout/stderr. Use an explicit suite
path when analyzing a transferred directory.

After calibration passes, freeze this target and the discovery-selected
0.4-V/m, 14-Hz axial protocol for a new matched MDD sham/active confirmation
cohort. Do not rerank the completed discovery with the new target. A future
confirmation should include three-band movement, carrier-excluded sensitivity,
rates and washout; it is not permission for phase matching or bandit training.
No efficacy jobs or automatic downstream submissions are included here.
