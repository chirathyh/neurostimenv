# Implementation and claim status — 10 November 2026

Base: `research/ballnstick-h1-h4` at
`855a1c33623573aeae71682dc7740658bdeac401`, verified against origin after fetch.
Feature: `feature/eeg-representation-position2027`.
Implementation commit: `d140ce7cd4e82264d2f6162aa59c053b19977c5d`;
the protocol, runbook and this execution record follow in a documentation commit.
The original modified L23Net checkout was preserved in place. All changes are
new suite/config/test/documentation files; no frozen COSYNE entry point,
configuration, analysis or result was changed.

The exact runtime-source SHA256 verified by both final end-to-end runs is:

`9a1fa6f48f04125143569c062b839859c820c61df17ba90df3487ee3389ae3d0`

The smokes began before the implementation commit, so their provenance records
the then-current base commit plus dirty-source status and the exact content
hash above. That implementation is now committed. Documentation-only commits
do not change the runtime-source hash.

| Claim or capability | Status | Evidence / limit |
|---|---|---|
| Squared-error Bayes-risk decomposition | Analytically established identity | Not a new empirical result or universal EEG sampling bound |
| Poisson/Kalman/variance controls | Implemented and unit-tested | Independent analytical execution passed; no simulator imports needed |
| Trial schema, HDF5 shards, RNG namespaces and grouped splits | Implemented and unit-tested | Float64 units, missing labels, family quotas, corruption/resume guards |
| Causal 500-Hz stream and label-blind noise views | Implemented and unit-tested | Future-perturbation equality; no absolute 1e-8 physical-signal floor |
| Real zero-field adapter and rerun futures | Simulator smoke passed | Exact physical-state/history/pending-event prefixes, explicit 6.3 °C |
| Burst, phase diffusion, kinetics and shared-fraction factors | Implemented; bounded tests passed | Full targeted observability/confound study remains pending |
| All 14 recipe families and distinct order repetitions | Implemented and unit-tested | Every recipe fit/save/reload/evaluate on analytical and actual neural data |
| Two transfer protocols and grouped inference | Implemented and smoke passed | Smoke comparisons excluded from scientific inference |
| Full core pilot | Pending, exploratory when run | 640-trial plan only; not launched |
| Full-state circuit prediction frontier | Pending | Small engineering forks do not establish the scientific curve |
| H2a complementary-task advantage | Pending | No scientific transfer result claimed |
| H2b progression advantage | Pending | Offline order comparison implemented; no preferred outcome assumed |
| Frozen confirmation | Pending; generation guarded | Provisional 4,000-trial profile requires a prospective freeze |
| CUDA / intended workstation | Unexecuted | Local CUDA unavailable; no GPU performance or precision claim |

## Final executed checks

`python -m experiments.eeg_representation.cli validate --profile simulator ...`
completed **70 tests in 20.345 seconds, all passed, none skipped**. This includes
the real-NEURON tests, not mocks. The final command explicitly sets the legacy
`MAIN_PATH` and real-integration opt-in in its child process.

Coverage includes Poisson count/ISI/variance laws, exact Kalman covariance and
history gaps, unchanged RNG state, independent future replay, source/parent/
noise-family separation, tiny physical scales, held-out-label permissions,
all recipe lifecycle tests, positive covariance and proper scores, loss masks,
shuffled-label checks, paired-family label quotas, principal-neural exposure
parity, final-cutoff temporal probes, interrupted checkpoints, incompatible
resumes and import boundaries. Inherited background, synapse-kinetics and
online-stimulation tests contribute 21 of the 70 tests.

Additional inherited checks were run in isolated output directories:

- Isolated-cell polarization: all gates passed.
- Online/legacy 500 ms: 8,000 online samples versus 8,001 legacy samples
  including its left endpoint; correlation 1.0, relative RMS error about
  2.55e-11, configured/effective 6.3 °C, clean field removal and cleanup.
- Reduced A/B three-seed smoke: executed successfully; engineering output
  included accuracy 0.333 and p=1.0. It is not confirmation.
- One-discovery/one-validation stimulation smoke: executed successfully and
  correctly returned **NOT PASSED** (improvement approximately -0.422152,
  rate-safe). No negative result was converted into a success.
- Default 32 E/8 I adapter replay: exact state/history prefix and eight actual
  pending recurrent deliveries preserved at a 46-ms cutoff. External inputs
  use direct event delivery, so an unused NetCon delay is not added to them.
- Four additional actual tiny-circuit 9/11-Hz × D=0.5/2 trials: finite EEG and
  dipoles, 4,000 native samples each, 6.3 °C, zero field, one initialization,
  identical structure hashes. This checks lifecycle, not EEG observability.
- Independent phase-law audit: 64 independent 60-s paths for each of eight
  carrier/diffusion/grid combinations, using 1-ms and 0.5-ms grids. All 24
  lag-coherence comparisons were within four path-level Monte Carlo standard
  errors of exp(-D×lag); 23/24 ordinary 95% intervals covered the law. Maximum
  absolute discrepancy was 0.01433 and maximum increment-variance relative
  discrepancy was 0.001224. Overlapping increments were not independent units.

The standalone analytical control checks passed. The Poisson fork-variance
maximum relative discrepancy was 2.53%; exact Kalman risk checks were within
five Monte Carlo standard errors. The variance-feature classifier and
posterior-variance Gaussian predictor supplied the intended positive controls.
These are numerical checks of declared analytical controls, not H2 results.

## Executed end-to-end pipelines

Both `smoke.yaml` and `simulator_smoke.yaml` completed the actual CLI sequence:
`generate → derive → forks → fit-suite → evaluate → summarize`, with fitting
and evaluation under **both** holdout protocols. Generation and fitting resume
were also executed. Each dataset has 24 trials and 15 recipe runs per protocol:
the 14 base families plus one additional distinct random order. There are only
two possible orders for the two core source tasks, so requesting three random
orders is explicitly capped at two distinct permutations.

Across both datasets: **60 saved model runs, 900 probe-result rows and 12
forecast-model scoring outputs**. Principal non-specialist neural recipes
have identical input-ticket hashes: 14 updates/56 EEG exposures for label
holdout, 10 updates/40 exposures for family holdout in these tiny settings.
Teacher/hybrid extra forwards are counted separately; this is not equal FLOPs.
Summaries contain no scientific smoke hypothesis comparisons; circuit H1,
H2a and H2b remain Pending. The generated summary PNG was visually inspected.

| Measured functional workload | Analytical | Actual NEURON/LFPy |
|---|---:|---:|
| Latent trial count | 24 | 24 |
| Cells | Not a circuit | 2 E / 1 I |
| Generation wall time | 0.877 s | 41.515 s |
| Summed per-trial generation time | Control bookkeeping only | 40.192 s |
| Native trial-shard storage | 2,927,909 bytes | 18,193,699 bytes |
| Fork-command wall time | 2.461 s | 29.674 s |
| Entire output tree | 17,545,468 bytes | 45,541,861 bytes |

The real smoke has 0.25-s burn-in plus 2-s retention. These measurements are
not a 40-cell pilot runtime/storage estimate. The pilot planner's native-array
estimate is pure arithmetic: 5,734,400,000 uncompressed bytes, excluding events,
HDF5 overhead and checkpoints. No full-run CPU estimate is invented.

Local evidence is intentionally outside git: final control logs and validation
JSON, complete command timings/artifact audit, and the two complete output
trees were retained under the task's temporary test directories. Reproduce
them with the runbook rather than treating those temporary paths as required
workstation inputs. Generated arrays, checkpoints and result CSVs are not
committed.

## Unexecuted work and scientific limits

The 640-trial pilot, 4,000-trial confirmation, large 64/128-future campaign,
full factor-development matrix, modulation-depth development sweep, full
model-selection grid, five-seed confirmation, intended GPU and new ML-environment
installation were not run. Their interfaces/configs were checked by help,
planning, the same CPU CLI path, or explicit installation-script syntax checks;
this is not a claim that these campaigns passed. No external dependency blocks
the tested local CPU route. CUDA hardware is the exact unavailable dependency
for GPU validation.

Core downstream baseline/carrier labels reuse source factors and therefore do
not alone establish rich new-task transfer. Coherence, envelope and mechanism
observability remain empirical questions. The 64-ms causal FIR delay must be
retained when interpreting very short lead risks. Source-specific fork modes
condition on the other frozen innovation path and are variance diagnostics;
only the all-source ensemble estimates the stated full-causal-state lower
bound. Neural forks exclude independent future sensor noise. Gaussian forecasts
can miss multimodality. Equal input/update budgets do not ensure equal compute.

An online dataset-arrival experiment, new-circuit support/query calibration,
real EEG checkpoint comparisons, realistic stimulation artifacts and independent
spatial sources are not implemented as scientific studies. They are optional
extensions, not evidence against the core tested implementation. No RL,
clinical efficacy, cortical validation or manuscript-readiness claim is made.
