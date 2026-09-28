# R1: prospective L23Net replication and reusable independent reference cohort

Frozen before observing R1 outcomes. Prerequisite: the raw-data-audited G1B
result identified by hash in `r1_protocol.py` and `G1B_RESULTS_AUDIT.md`.
No changes to membrane/synaptic mechanisms, the integrator, or field equations.

## Design and purpose

| Cohort | Seeds | Conditions | Trajectories |
|---|---|---|---:|
| Independent calibration | 8101–8116 | reference only | 16 |
| Primary replication | 8201–8216 | matched reference and MDD | 32 |
| Separate precision extension | 8301–8344 | matched reference and MDD | 88 |

The first submission, `core`, runs calibration plus primary replication:
48 trajectories in 24 jobs. Each of eight calibration jobs runs two distinct
reference structures sequentially. Each of 16 primary jobs runs the two
conditions of one structure sequentially in fresh MPI processes.

After the core is technically complete and its calibration and primary analysis
are frozen, a second submission can add the 44 extension pairs. This gives
60 total pairs plus 16 independent reference trajectories, 136 trajectories in
all. Extension eligibility is **not based on primary significance**. The
independent 44-pair and pooled 60-pair results are secondary and cannot replace
the original primary inference. No selection of seeds based on phenotype.

All cohorts are disjoint from G1B and historical seeds 10–69. Keep identical
cell counts (800 PYR, 50 SST, 70 PV, 80 VIP), 624 ranks, dt=0.025 ms, 34 C,
v_init=−80 mV, tstart=0, env_seed=0, DRUG=false, and zero field. Each episode
is one persistent 28,000-ms simulation with 28 one-second observations. The
original internal stimulus after 4 s is retained; effective outcomes are 8–28 s.
The same actual-parameter, pairing, finite-output, time, field-removal,
recording, rate and memory audits used by G1B are retained.

The independent reference cohort is reusable for future stimulation studies.
It must not be replaced with each candidate's own matched reference as a
controller target. R1 freezes a **20-second** target; future 12-s screening and
4-s stimulation endpoints require separately frozen duration/processing-matched
targets derived from these raw reference traces before active-outcome inspection.
Do not silently apply the 20-s target to differently processed short windows.

## Frozen analysis

Both G1B analysis views are retained unchanged:

- Primary: historical double-4-s exclusion, order-2 0.1–100-Hz Butterworth
  `ba`/`filtfilt`, Hann Welch 0.5-s segments with 50% overlap, constant detrend,
  density scaling, inclusive-endpoint trapezoid integration.
- Sensitivity: filter the full record using the corresponding SOS filter,
  then explicitly retain (8,28] s. Identical Welch/band definitions.

Bands are theta 4–8, alpha 8–12 and low beta 12–16 Hz. Retain the right-endpoint
sample convention; do not fabricate t=0. This is prospective phenotype
replication under the corrected online lifecycle, not samplewise equivalence
to the old code. The 2-Hz Welch resolution is not an individual-frequency
estimator suitable for future phase-targeted tACS.

For each view, estimate the reference log10 band-power mean and sample SD from
the 16 calibration structures only. Freeze the numerical scale floor as
`max(0.01, 0.1 * median(reference SD across the three bands))`; use the larger
of this floor and each band's SD. This regularizes near-zero reference variance
without using candidate differences. The primary structure-level outcome is

`g_s = mean_b[(log10(P_MDD,s,b) - log10(P_reference,s,b)) / scale_reference,b]`.

The primary gate is positive mean legacy composite and one-sided exact paired
sign-flip p≤0.05, with positive mean SOS composite as a directional sensitivity
check. Report the two-sided 95% t and structure-bootstrap intervals even if an
interval crosses zero; do not reinterpret a directional test as two-sided
significance. Save paired dz, every paired effect, positive-structure counts,
and all bandwise effects with BH-FDR across the three secondary bands per view.
Inference is conditional on the frozen reference scale; calibration uncertainty
is not included in these intervals. Sign-flip inference assumes exchangeability
under pair swapping/symmetry of paired effects under the null.

The 16-pair test enumerates all 65,536 sign configurations. The 44/60-structure
secondary analyses use 100,000 seeded sign flips with a plus-one correction.
Bootstrap uses 20,000 resamples of complete structures, never EEG windows.
The design assumption dz≈0.7 provides roughly 80–85% directional power at
n=16; this is not a guarantee and is not inferred from G1B's two seeds.

A secondary EEG-only observability audit leaves out both conditions of each
structure. It uses the frozen standardized equal-weight three-band score and
a midpoint threshold learned from the other structures, with higher values
classified MDD. Report balanced accuracy and structure-bootstrap uncertainty;
the latter does not include threshold-refitting uncertainty. This classifier
is not a clinical diagnostic or a treatment policy and does not change the gate.

Technical failures prevent cohort inference; no silent removal of failed seeds.
`phenotype_not_confirmed` is a valid completed negative primary result, not a
software failure. It stops progression to efficacy studies pending review.

## Resource and allocation plan

Each worker: normal queue, 624 CPUs, 256 GB, 02:30:00, 13 nodes, one thread per
rank. Maximum worker reservation: 3.12 KSU at 2 SU per resource-hour. The small
summary job requests 1 CPU/4 GB/30 minutes. Keep the validated decomposition
fixed; more independent jobs, not more ranks per trajectory, provide throughput.

| Stage | Workers | Measured-pilot usage estimate | With 25% runtime contingency | Maximum worker reservation |
|---|---:|---:|---:|---:|
| Core | 24 | 34.26 KSU | 42.83 KSU | 74.88 KSU |
| Extension | 44 | 62.81 KSU | 78.52 KSU | 137.28 KSU |
| Total | 68 | 97.07 KSU | 121.34 KSU | separate staged submissions |

These are estimates from G1B's final PBS epilogues, not promises. A full request
must fit currently available quota even when expected usage is much smaller.
The submitter reads `nci_account -P` for all three projects, allocates by
sj53 → fa32 → ny83 priority, and leaves 1 KSU unreserved in each project,
including headroom for the small summary. It never changes queues to burn quota.
If account text cannot be parsed, it refuses to submit; supply fresh manual
balances using the documented fallback. It does not assume quota automatically
renews for a pending submission.

At most eight workers are eligible concurrently using eight `afterany`
dependency chains. A failed job does not discard independent later structures;
the summary runs after all workers and reports every missing/failing result.
This cap is per submitted suite, not a project-wide cap. Do not submit duplicate
suites. A partial qsub failure preserves accepted IDs in `submission.json` and
must be inspected before any retry.

Eight available slots imply approximately 3.5 hours for the core and 7 hours
for the extension, plus compilation/analysis variability and queue delays.
Finish the core first and inspect it. The extension can then exploit the current
allocation while S0/S1 are prepared. Aim to finish well before the actual renewal
deadline; neither a pending job nor a reservation guarantees useful completed
work before expiry. Use updated `nci_account` output before each submission.

Do not manufacture unnecessary runs to exhaust the balance. This campaign
creates a substantial reusable baseline now. After R1, prioritize the bounded
S0/S1 frequency/phase/dose discovery already described in the reviewer plan;
use renewed allocation for appropriately powered, disjoint held-out S2
confirmation. No automatic bandit launch, active-protocol selection, or
underpowered confirmatory tACS claim is bundled into R1.

NCI charging reference:
https://opus.nci.org.au/spaces/Help/pages/90308823/Queue+Limits

## Commands on NCI

Only pull after G1B has finished. Keep this checkout and its validated virtual
environment unchanged throughout core and extension, including queued jobs.
The extension requires the same commit/environment as the core. Scripts are
tracked: no manual script copy is needed.

```bash
cd /g/data/ny83/ch9972/NeuroStim/neurostimenv
git switch feature/l23net-sinusoidal-uniform-field
git pull --ff-only origin feature/l23net-sinusoidal-uniform-field
git status --short

G1B_REPORT=/g/data/ny83/ch9972/NeuroStim/neurostimenv/results/l23net_g1b_20260928_094057_kCtCZq/g1b_summary.json

# Dry run: prints quota split and creates/submits nothing.
python3 experiments/l23net_analysis/nci/submit_l23net_r1.py \
  --stage core --g1b "$G1B_REPORT"

# Submit once, after inspecting the dry-run plan.
python3 experiments/l23net_analysis/nci/submit_l23net_r1.py \
  --stage core --g1b "$G1B_REPORT" --submit
```

The script prints the actual suite directory and status command. Save them.
Once the core summary job finishes:

```bash
read -r -p "Paste the completed R1 core suite directory: " CORE_SUITE
python3 experiments/l23net_analysis/nci/collect_l23net_r1_status.py "$CORE_SUITE"
cat "$CORE_SUITE/r1_summary.md"

# Inspect the core findings and technical checks before the next submission.
python3 experiments/l23net_analysis/nci/submit_l23net_r1.py \
  --stage extension --g1b "$G1B_REPORT" --core "$CORE_SUITE"
python3 experiments/l23net_analysis/nci/submit_l23net_r1.py \
  --stage extension --g1b "$G1B_REPORT" --core "$CORE_SUITE" --submit
```

If automatic account parsing fails, run the three `nci_account -P PROJECT`
commands and append `--available-ksu sj53=VALUE fa32=VALUE ny83=VALUE`, replacing
each VALUE with its **fresh available KSU**, to the dry-run and submit commands.
Do not reuse stale balances or total allocation values. This fallback is not
permission to oversubscribe an account. Use `--max-concurrent 4` or `6` if desired.

## Artifacts, error handling, and later analysis

Copy each entire suite back, including all worker subdirectories. Core outputs:
`reference_target.json`, `primary_frozen.json`, `core_data_frozen.json`,
`r1_summary.json`, `r1_summary.md`, `r1_psd.npz`, submission/provenance and logs.
Each trajectory has `l23net_r1_run.json`, `l23net_r1_trace.h5`, geometry and Hydra
configuration. Workers save heartbeat, exit status, versions, mechanism hashes,
PBS node allocations and final PBS output. The extension hashes and loads the
three frozen core files; mutable PBS accounting summaries are not hash inputs.

For help, provide the collector output, `r1_summary.md`, failing worker log
tails, rank failure JSON if present, and final PBS epilogues. Do not paste the
entire verbose rank-by-rank report. Do not rerun the whole suite on a partial
failure; inspect the recorded IDs and preserve successful trajectories.

The collector needs only system python3. Scientific reanalysis needs the
validated environment and, on NCI, a compute job rather than a login node.
On the laptop (after activating the repository-parent environment):

```bash
python3 experiments/l23net_analysis/analyze_l23net_r1.py \
  --suite "$LOCAL_CORE" --output-dir "$LOCAL_CORE/reanalysis"
python3 experiments/l23net_analysis/analyze_l23net_r1.py \
  --suite "$LOCAL_EXTENSION" --core "$LOCAL_CORE" \
  --output-dir "$LOCAL_EXTENSION/reanalysis"
```

Frozen numerical files cannot be silently overwritten by a different result.
If a different local numerical stack changes recomputation, preserve NCI's
frozen originals and inspect the discrepancy; do not delete frozen files to
make reanalysis pass. Final PBS accounting is read from epilogues, not inferred
from potentially lagging pre-exit qstat snapshots.

## Local validation

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
python -m unittest -v tests/test_l23net_r1.py tests/test_l23net_g1b.py \
  tests/test_online_stimulation.py tests/test_online_recording_optimizations.py \
  tests/test_l23net_replay_validation.py
bash experiments/l23net_analysis/validate_l23net_r1_local.sh
```

The MPI smoke covers calibration plus matched primary and extension trajectories
with five detailed cells, two ranks, production dt, and two persistent windows
per trajectory. It cannot establish the full-network phenotype. Mock PBS tests
exercise project priority, full-walltime reservations, the concurrency cap,
afterany summary, and preserved IDs after partial submission failure. Statistical
tests cover positive/negative effects, exact/Monte Carlo sign flips, incomplete
cohorts, reference separation, FDR, and preservation of primary results through
extension. Mechanism files are not edited.

Implementation verification on 28 September 2026: 47 unit/regression tests
passed; all five local two-rank trajectories passed the raw-trace and pairing
audit. A separate two-rank 28-s clock regression completed all 1,120,000 fixed
steps (raw NEURON drift about 1.07e-6 ms, correctly mapped to logical time).
A two-rank invalid-cohort test exited nonzero promptly rather than hanging.
PBS submission was tested with mocked qsub/accounting, not by accessing NCI.
