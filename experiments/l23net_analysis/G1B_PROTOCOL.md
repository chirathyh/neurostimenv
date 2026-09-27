# G1B: paired full-network 28-second pilot

Frozen before inspecting the pilot outcomes. G1A job `179858770.gadi-pbs`
passed exact independent six-second replay at 624 ranks. Its comparison JSON
is checked, copied, and hashed on submission of this pilot.

## Design and execution

| Setting | Frozen value |
|---|---|
| Independent structure seeds | 7101, 7102; `env_seed=0` |
| Conditions per seed | reference (`MDD=false`), MDD-configured (`MDD=true`) |
| Network | full 800 PYR / 50 SST / 70 PV / 80 VIP |
| Integration | fixed `dt=0.025 ms`, configured/effective 34 C |
| Initial state | `tstart=0 ms`, `v_init=-80 mV` |
| Duration | 28,000 ms, 28 contiguous 1,000-ms windows |
| Samples per condition | 1,120,000, sample times `(0,28000]` ms |
| Field/drug | zero field throughout; `DRUG=false` |
| Internal synaptic stimulus | original spreadsheet schedule just after 4 s |
| MPI decomposition | 624 ranks on 13 normal-queue nodes, 48 ranks/node |
| Pair 7101 project | `sj53` |
| Pair 7102 project | `fa32` |
| Allocation per pair | 624 CPUs, 256 GB, 02:30:00 |
| Summary job | `sj53`, 1 CPU, 4 GB, 00:15:00, after both pair jobs finish |

Each pair job runs the reference and MDD conditions sequentially in separate
Python/MPI processes. Both pair jobs can run concurrently; project submission
order does not guarantee scheduler start order. No condition is reinitialized
between its observation windows. The normal queue and rank decomposition are
kept fixed across all conditions. Results use unique suite directories, and a
trajectory refuses to overwrite existing report/trace files.

The two seeds are new pilot structures, disjoint from historical seeds 10--69
and G1A seed 10. Subsequent confirmation and reference calibration must use
different seeds. These are simulator replicates, not biological subjects.

## What the MDD flag actually changes

The existing circuit loader applies:

1. `gmax_MDD = 0.6 * gmax_reference` for all SST-origin recurrent synapses.
2. `g_apical_MDD = 0.6 * g_apical_reference` for pyramidal tonic conductance.
3. For each non-pyramidal population j,

   `g_tonic_MDD,j = g_tonic_ref,j * (1 - 0.4 * S_j / I_j)`,

   where `S_j = sum_SST(gmax_ij * n_contacts_ij * p_connect_ij)` and `I_j`
   sums the same quantity over SST, PV and VIP inputs.

The construction audit checks actual synaptic `gmax`, segment tonic density,
and tonic reversal against these formulas. It hashes recurrent source and
target GIDs, section locations, weights, delays and unchanged synaptic
kinetics, plus OU drive parameters. Existing geometry, synapse layout and
prescribed afferent event fingerprints must also match within each pair.
Expected structural hashes differ across the two pilot seeds.

The legacy OU random generator has no usable `seq()` accessor. The audit does
not query or draw from that generator. Shared seeds and unchanged construction,
OU parameters and stepping, supported by G1A deterministic replay, provide the
common-random-input basis; full OU paths are not recorded. No MOD or HOC files
are edited. The MDD configuration is a specified inhibition manipulation, not
a validated definition of clinical depression.

## Frozen analysis

Use the single ideal EEG channel, returned in volts. Save raw EEG, three-axis
dipole, time, zero field, and stage to HDF5 per window. Keep spike-event hashes
and per-population counts/rates in each window report. Raw spikes and soma
voltages are not accumulated for this pilot.

Both analysis views use equal-duration 20-second outcomes:

- **Legacy:** remove the first 4 seconds; apply the original order-2
  0.1--100-Hz Butterworth `ba`/`filtfilt`; remove another 4 seconds before Welch.
  This deliberately reproduces the old double-trim procedure.
- **Corrected sensitivity:** filter the complete continuous EEG with the same
  Butterworth design in SOS form, then perform one explicit 8-second
  initialization/event-recovery exclusion. Retain `(8,28]` seconds.

In both views Welch uses a Hann window, 0.5-second segments, 50% overlap,
constant detrending and density scaling. Frequency resolution is 2 Hz, matching
the historical analysis. Integrate inclusive endpoints with the trapezoid rule
over theta 4--8 Hz, alpha 8--12 Hz and low beta 12--16 Hz. This resolution is
for replication, not precise individual-frequency estimation.

The old CSV included `t=0`; the persistent trace contains right-endpoint
samples only. No artificial t=0 sample is fabricated. The procedure is a
prospective phenotype replication under the corrected temperature/MPI
lifecycle, not samplewise equivalence to the old simulator.

For each seed and view define the pilot directional composite

`C_s = (1/3) * sum_b log10(P_MDD,s,b / P_reference,s,b)`.

Both seeds must have `C_s > 0` in **both** analysis views. Save each band's
power and paired log ratio, and the PSD arrays (`g1b_psd.npz`). Do not replace
this rule after observing a negative result. This pilot uses equal weights
without standardization: R1's standardized composite needs a separate frozen
reference cohort, which has not yet been simulated.

Two structures provide a technical directional gate. They do not justify
significance, power, efficacy or clinical claims. No bandit or tACS strategy is
tested here.

## Technical and physiological checks

- Correct full population sizes, frozen dt/temperature, matching seeds,
  topology/input fingerprints, expected intervention values and code/environment.
- Finite complete EEG/dipole and exactly 28 windows/1,120,000 samples, monotonic
  fixed-step time, zero applied field and zero residual extracellular voltage.
- Spike vectors drained and unused soma voltage vectors empty after each window.
- Post-recovery 8--28-s mean rate for each population at least 0.01 Hz; PYR no
  more than 50 Hz, and SST/PV/VIP no more than 100 Hz. These are deliberately
  broad engineering plausibility bounds, not calibrated physiology or safety
  prescriptions. A failure requires inspection, not automatic threshold changes.
- Across checkpoints from 8 s onwards, peak aggregate process RSS increase over
  the 8-s checkpoint must be no more than `max(2 GiB, 5% of that checkpoint)`.
  Report the slope as well. This is a bounded-memory diagnostic; aggregate RSS
  can double-count shared pages. Archive the final PBS epilogue to assess the
  authoritative job memory peak and actual headroom. No maximum duration is
  inferred from 28 s alone.

`passed` requires all checks and both positive composites per pair. A valid
completed simulation with a non-positive composite reports
`direction_not_reproduced` and exits normally; the suite does not authorize R1.
Technical failures produce `failed`, a nonzero exit, and available failure
reports. Debug smokes explicitly cannot pass the scientific pilot gate.

## Resource rationale and quota

G1A's PBS peak was approximately 160 GiB; the previous 15-s run used about
153 GiB. A 256-GB request supplies about 96 GiB headroom over G1A's observed
peak and avoids extrapolating its narrow 200-GB allocation to new seeds.
Based on existing timings, allow roughly 60--90 minutes for each pair, with
2.5 hours reserved. Actual runtime for new circuits can differ.

NCI's [queue limits and charging formula](https://opus.nci.org.au/spaces/Help/pages/90308823/Queue+Limits)
give `normal` a rate of 2 SU per resource-hour. CPU count dominates these
requests, so each pair reserves at most `624 * 2.5 * 2 = 3120 SU` = 3.12 KSU.
The summary reserves 0.5 SU. Default totals are:

| Project | Maximum reservation | Expected pair usage at 60--90 minutes |
|---|---:|---:|
| sj53 | 3.1205 KSU | about 1.25--1.87 KSU plus tiny summary |
| fa32 | 3.1200 KSU | about 1.25--1.87 KSU |
| ny83 compute | 0 | 0 |

The shared files remain on `/g/data/ny83`; requesting that storage mount does
not charge the simulation to ny83's compute allocation. Live balances and
queue availability must be checked on NCI. The two-day deadline cannot be
guaranteed by walltime/resource requests. Do not switch to an express queue or
launch duplicate jobs solely to consume allocation.

## NCI commands

The previously repaired G1A environment must still contain NEURON 8.2.3,
LFPy 2.3, NumPy 1.26.3, SciPy 1.11.4 and mpi4py 3.1.5. The worker checks
actually imported versions and `pip check` before expensive simulation.

```bash
cd /g/data/ny83/ch9972/NeuroStim/neurostimenv
git switch feature/l23net-sinusoidal-uniform-field
git pull --ff-only origin feature/l23net-sinusoidal-uniform-field
git status --short

nci_account -P sj53
nci_account -P fa32

bash experiments/l23net_analysis/nci/submit_l23net_g1b.sh
```

The scripts are tracked here, so no manual script copy is required. The wrapper
uses `python3`, verifies G1A, refuses a dirty tracked checkout, records the
submitted commit and prints a unique suite path plus three job IDs. It starts
exactly two pairs. Keep the checkout and virtual environment fixed until all
jobs complete. The worker builds the unchanged mechanism sources in a
job-specific shared directory and verifies loading on every allocated node.

If the G1A folder has moved, set `G1A_REPORT` to its real comparison JSON path
before submitting. Optional positional arguments select the first and second
project, for example `bash .../submit_l23net_g1b.sh sj53 sj53`; defaults honor
the requested sj53/fa32 order and do not use ny83 compute quota.

The submitter prints a ready-to-copy status command using its actual suite
path. Run that command and paste its output when requesting assistance. It
collects phase, progress, stderr, exit codes, recent logs and failure JSON.
For full analysis, copy the complete suite folder, including HDF5 traces, to
the laptop. Preserve `pbs.out` files: `qstat_at_exit.txt` is sampled just before
exit and may lag final scheduler accounting.

After both jobs finish, read `g1b_summary.md`/`g1b_summary.json` in that suite.
The summary job uses `afterany`, so a failed pair is reported rather than
leaving the summary dependent job waiting indefinitely. If only analysis needs
rerunning, use a compute job to execute:

```bash
python3 experiments/l23net_analysis/analyze_l23net_g1b.py --pair "$PAIR_DIRECTORY"
python3 experiments/l23net_analysis/analyze_l23net_g1b.py --suite "$SUITE_DIRECTORY"
```

Use actual directories returned by submission. A partial qsub failure preserves
already submitted job IDs in `submission.json`; inspect those before retrying
so successful conditions are not duplicated.

## Local verification

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
python -m unittest -v tests/test_online_stimulation.py tests/test_online_recording_optimizations.py tests/test_l23net_replay_validation.py tests/test_l23net_g1b.py
mpirun -np 2 python tests/mpi_online_fixed_step_regression.py --duration-ms 28000 --dt-ms 0.025
bash experiments/l23net_analysis/validate_l23net_g1b_local.sh
```

The reduced runner uses five detailed L23Net cells and two MPI ranks, testing
both conditions for six seconds at seed 7101 and one second at seed 7102 with
the production dt. Debug mode inherits the existing different debug afferent
schedule; it validates code paths and pairing, not the full-network phenotype.
The isolated MPI stepping test crosses the full 28-s horizon with 1,120,000
fixed steps on each rank. Unit tests reproduce the literal legacy analysis
at production sampling rate, exercise known power ratios and negative outcomes,
and mock submission to verify project order/dependencies without using NCI.

Verification completed locally on 2026-09-28: all 32 unit tests passed, including
an isolated SciPy 1.11.4 compatibility run; both reduced paired seeds passed;
the 28-s two-rank clock check passed with all 1,120,000 steps; two independent
G1A regression runs remained exactly equal. An injected rank-one audit error
saved its failure JSON and terminated MPI without hanging. Full 1,000-cell,
624-rank validation remains the NCI pilot's purpose.
