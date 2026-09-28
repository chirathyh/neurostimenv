# G1B completed-result audit — 28 September 2026

Source suite: `results/g1b_results/l23net_g1b_20260928_094057_kCtCZq`.
Submitted code: `bd53b46b690422f7fc32510e092de2c8af39deec`.
Frozen `g1b_summary.json` SHA256:
`5ceb776cd279b9e7d1743182854d194bd29476e34b2ceb77f395ac2a01c74dc8`.

## Decision

**G1B passed its technical and prespecified two-seed directional pilot gates.**
Proceed to disjoint R1 replication and independent reference calibration.
This is not a significance result or a tACS efficacy result.

The local audit independently reread all four HDF5 traces, recomputed their
content hashes and both spectral analyses, and checked paired report contracts,
actual construction fingerprints, zero-field output, sample timing, rates,
resource checks, worker exit codes, environment and mechanism provenance.
No source result files were overwritten during the audit.

Each trajectory contains 28 committed one-second windows and exactly 1,120,000
finite samples at dt=0.025 ms, covering (0,28000] ms. The full 1,000-cell network,
624-rank decomposition, configured/effective 34 C, original internal event
after 4 s, and expected MDD parameter changes were retained. Within each pair,
geometry/connectivity/input construction matched; the two structures differed.
All worker and summary PBS exit statuses were zero; both worker stderr files
were empty. NumPy 1.26.3 and SciPy 1.11.4 were used, with NEURON 8.2.3,
LFPy 2.3, mpi4py 3.1.5 and h5py 3.11.0.

## Phenotype, not a blanket all-band claim

Legacy analysis, effective 8–28 s; percentages are paired MDD/reference power
changes, not changes in EEG amplitude:

| Seed | Theta | Alpha | Low beta | Mean log10 power ratio | Pilot gate |
|---|---:|---:|---:|---:|---|
| 7101 | +38.85% | +27.79% | +57.66% | 0.148920 | passed |
| 7102 | −3.09% | +59.40% | +56.00% | 0.127320 | passed |

The corrected SOS composites were 0.148918 and 0.127320. Independent local
recomputation reproduced the saved band powers within 1e-7 relative tolerance.
The predefined gate required a positive composite in both views for both
structures, not a positive effect in every individual band. Seed 7102's theta
decrease is retained and must not be concealed by selecting bands or seeds.

Post-8-s mean firing rates (reference → MDD, Hz):

| Seed | PYR | SST | PV | VIP |
|---|---|---|---|---|
| 7101 | 0.764 → 1.169 | 5.491 → 7.211 | 10.110 → 15.492 | 3.367 → 6.873 |
| 7102 | 0.731 → 1.108 | 5.167 → 6.651 | 9.916 → 15.096 | 3.284 → 6.629 |

These are mechanistic/engineering plausibility checks, not calibrated clinical
physiology or proof of network desynchronization.

## Authoritative final PBS measurements

| Pair job | Project | Final walltime | Peak memory, PBS GB | Actual SU |
|---|---|---|---:|---:|
| 179985070.gadi-pbs | sj53 | 01:09:02 | 159.80 | 1435.89 |
| 179985071.gadi-pbs | fa32 | 01:08:14 | 158.98 | 1419.25 |
| 179985072.gadi-pbs, summary | sj53 | 00:00:20 | 153.15 MB | 0.01 |

Total charged: 2855.15 SU, approximately 2.855 KSU. These final epilogues
supersede the slightly smaller pre-exit `qstat_at_exit.txt` estimates in the
original summary. Mean pair walltime was 68 min 38 s; mean pair charge was
1.42757 KSU. Integration alone took 1944–1990 s per 28-s trajectory.

Aggregate process RSS rose by 0.84–0.90 GiB from the 8-s checkpoint to its
post-recovery maximum, within the frozen diagnostic limits. Aggregate RSS
double-counts some shared pages and reached about 191 GiB, whereas PBS reported
about 160 GB. Memory growth was small, **not zero**. No maximum safe duration
can be inferred from these 28-s runs. Retain 256 GB for the new seed cohort.

## Claim boundaries and next action

- Two independent model structures are insufficient for inferential replication.
- The MDD flag is the inherited, explicitly audited inhibition intervention;
  the experiment does not establish a clinical definition of depression.
- Legacy OU paths were not recorded. Seed/setup matching and G1A deterministic
  reconstruction support common random inputs without inspecting/mutating RNGs.
- No field was applied in G1B; this result cannot establish tACS efficacy or
  long-duration stimulated stability.
- R1 uses new seeds, freezes a separate reference target, preserves the 16-pair
  primary result, and makes a 60-pair extension explicitly secondary.

See `R1_PROTOCOL.md` for executable commands, quota checks, and staged use of
the current allocation versus later stimulation studies after renewal.
