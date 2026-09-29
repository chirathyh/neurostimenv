# Full-spectrum S0 results and gated 60-second S1

Implemented 29 September 2026. This supersedes the earlier provisional 27-s
S1 design, not the completed R1 analysis. No third-party mechanism was changed.

## S0 result

Re-read and content-hash checked all 136 R1 trajectories: 16 reference
calibration structures, 16 original matched pairs, and 44 extension pairs.
The complete measurement contract was written before analysis, and reference
targets were frozen before evaluating the extension. The fixed criteria were
not relaxed after observing the negative carrier/phase result.

The scientific target is **theta 4--8, alpha 8--12, and low beta 12--16 Hz**.
Alpha is neither the sole endpoint nor the sole stimulation band.

The extension results under the new S0 spectral pipeline are:

| Band | Geometric paired MDD/reference power increase | 95% CI | Stable carrier coverage in MDD |
|---|---:|---|---:|
| Theta | 40.47% | 33.88--47.40% | 7/44 = 15.9% |
| Alpha | 35.56% | 28.20--43.35% | 9/44 = 20.5% |
| Low beta | 79.72% | 69.85--90.16% | 5/44 = 11.4% |

The equal-weight standardized paired composite is 3.200, CI [2.978, 3.422],
positive in 44/44 structures. All three band sign-flip tests reach the
100,000-draw Monte Carlo resolution floor (p/q approximately 1e-5, not an exact
tail probability). The empirical frozen phenotype screen identifies 44/44 MDD
and 41/44 reference extension structures correctly. These are simulator
measurement results, not clinical sensitivity/specificity estimates.

These power estimates differ from R1 because S0 uses a new, prospectively
specified spectral estimator. Do not replace the frozen legacy-compatible R1
numbers with them in the replication section.

### Measurement method

- Native integration remains at dt=0.025 ms (40 kHz).
- Apply an eighth-order, 80-Hz causal low-pass in second-order sections, then
  decimate to 250 Hz, with exact right-endpoint sample alignment. Filter state
  persists between windows. No future samples enter a predecision estimate.
- Exclude all samples at or before 8 s. This includes the initial 4 s and
  recovery from the retained built-in event around 4 s.
- Estimate powers over 8--28 s, using four-second Hann Welch segments with
  50% overlap: 0.25-Hz bin spacing and nine periodograms per 20-s estimate.
  Hann spectral smoothing is broader than one bin; zero-padding is not used
  to manufacture resolution.
- Integrate each band's PSD in V², then take log10 power. Fit independent
  reference mean and scale using only the 16 calibration structures.
- For carrier qualification, require a local peak with at least 3-dB
  prominence, within the band's open interior, and agreement within 0.5 Hz
  across the two baseline halves and the full baseline. Require 80% coverage.
- Audit phase using a causal two-second sinusoidal fit, with carrier selected
  from preceding data only, against held-back future fits at 1- and 4-s
  horizons. Future fits are an offline scoring convention, not latent neural
  phase truth and never controller inputs.

The carrier criterion fails in every band. Accepted phase predictions are
also sparse: only three theta, one alpha and zero low-beta predictions per
horizon among 132 possible extension predictions. Those small selected samples
cannot establish reliable phase control. Existing 28-s recordings cannot
qualify a 22-s open-loop phase forecast after the required 20-s baseline.

**Decision: fixed-frequency exploration is permitted; frequency-matched and
phase-matched/antiphase arms are blocked.** This is a negative result for this
measurement method and protocol, not proof that all possible estimators or
short-cadence controllers must fail. A later phase study needs new qualification;
it must not be silently substituted into this S1 experiment.

### Why longer records help, and their limits

Across extension trajectories, RMS log10 band-power discrepancy from the same
record's 20-s estimate averaged approximately 0.133--0.137 for 4-s records,
0.065--0.068 for 8 s, and 0.039--0.043 for 12 s. These overlapping-data comparisons
are descriptive sensitivity measures, not independent noise-variance estimates.
Longer averaging improves stability but does not make biological fluctuations
disappear or make a broad spectrum's phase predictable.

The second 10-s half versus first 10-s half geometric power ratios were:
reference [0.970, 0.972, 0.911], MDD [0.967, 0.944, 0.984], in theta/alpha/low-beta
order. Eight-second exclusion is explicit; complete stationarity after 8 s
is not proven. The longer qualification therefore measures later sham/reference
drift instead of assuming it away.

Raw/derived local artifacts are under
`results/l23net_s0_full_spectrum_release/`: frozen contract, targets, gate,
per-structure observations, decimated EEG, spectra and `s0_spectra.png`.
The smaller portable gate is checked in as
`frozen/s0_full_spectrum_v1.json`. It includes targets, decisions, source hashes,
and the inherited scientific/environment contract; NCI need not access laptop
paths embedded as provenance. Earlier local iterations changed provenance/FDR
reporting and plot layout, not measurement thresholds or scientific results.

## S1 episode and endpoint

| Interval | Role | Included in primary PSD? |
|---|---|---|
| 0--4 s | Initial transient | No |
| 4--8 s | Recovery from inherited event | No |
| 8--28 s | 20-s baseline and prospective phenotype screen | Separate baseline |
| 28--29 s | Raised-cosine onset ramp | No |
| 29--49 s | 20-s constant-amplitude stimulation plateau | Yes |
| 49--50 s | Raised-cosine offset ramp | No |
| 50--60 s | Ten-second no-field washout | Separate recovery analysis |

One network, one initialization, continuous channel/synaptic/event state.
Actions are broadcast identically across all ranks. Every counterfactual is a
fresh full-history replay, with exact prestimulation EEG, dipole and spike
fingerprint checks. The production contract fixes 624 ranks, 34 C, dt=0.025 ms,
DRUG=false, inherited synaptic activity and the same EEG forward model as R1.

For amplitude E0=0.4 V/m and f in {6,10,14} Hz:

\[
\mathbf E(t)=E_0 w(t)\sin[2\pi f(t-28)]\hat{\mathbf n},
\]

with t in seconds, one absolute 28--50-s raised-cosine block, one-second ramps,
and zero field outside the block. The sinusoid is continuous across 1-s online
windows; the ramp does not restart each window. Default direction is +z.
The orientation control uses +x and is not assumed to be a null in branched
L23Net morphologies. Tissue amplitude is not a scalp-current or clinical dose.

For bands b in {theta, alpha, low beta}, define

\[
x_b=\log_{10}\!\int_{B_b}\widehat S_{EEG}(f)\,df,\qquad
d_H(\mathbf x)=\sqrt{\frac13\sum_b[(x_b-\mu_{H,b})/s_{H,b}]^2},
\quad g_s(a)=d_H(\mathbf x_s^{sham})-d_H(\mathbf x_s^a).
\]

Positive benefit means movement toward the frozen reference, not equivalence
with Healthy physiology. All three bands have equal standardized weight;
the metric does not fit a full covariance matrix. Keep bandwise changes,
population rates, sparse spike times, dipole, fundamental-excluded outcomes,
and washout as separate audits. Fundamental exclusion removes f±0.5 Hz from
both target calibration and each compared trajectory; integration never bridges
across the excluded gap.

Targets are matched to the 20-s baseline/outcome and 10-s washout durations.
They are calibrated at earlier absolute times in R1. Qualification has a
predeclared 1.5-fold late/early no-field band-power drift tolerance, but this is
an engineering screen on one structure, not population stationarity equivalence.
Before confirmatory S2, either obtain long, absolute-epoch-matched independent
reference calibration or establish adequate target transfer with a stronger
stationarity study. Do not refit targets on active outcomes.

## Submission stages and gates

### Stage 1: qualification — three 60-s trajectories

Seed 8451, previously unused:

1. Healthy/reference sham;
2. MDD sham;
3. MDD 0.4-V/m, 10-Hz field.

The active qualification deliberately bypasses the phenotype screen, but not
the baseline rate guard: this is an engineering test, not an efficacy sample.
It must deliver the field. Checks cover full duration, finite streamed data,
exact matching prehistory, actual extracellular coupling, analytical waveform,
field removal, sparse spike integrity, intended Healthy/MDD intervention,
20% paired population-rate guardrails, late no-field drift, persistent RSS
growth and final PBS status/memory. PBS memory must leave at least 10% of the
256-GB request unused. Missing PBS epilogues block qualification until reanalysis.

### Stage 2: discovery — thirteen 60-s trajectories

Only a hash-validated `status=qualified` result from unchanged source code and
the same S0 target permits submission. Three independent seeds 8401--8403 each
receive sham and fixed 6/10/14-Hz counterfactuals. Seed 8401 also receives a
transverse 10-Hz control. Failed baseline phenotype/safety screens map to sham
and remain recorded; they are not retrospectively removed after outcomes.

This is directional discovery, not a powered efficacy result. Three structures
cannot give a one-sided all-positive exact sign-flip p below 0.125. No automatic
S2 or bandit permission is issued. Do not extend the seed list or adjust the
carrier/dose after inspecting outcomes without a separately frozen next stage.

## Memory, compute and limitations

- Native-rate EEG, dipole, field and sample times are streamed to HDF5 every
  second. Sparse per-population spike times/GIDs are separately streamed.
- Only 250-Hz EEG is retained for this bounded 60-s protocol: 15,000 samples on
  rank zero, not a lifetime native-rate voltage matrix on every rank.
- The inherited recorder-draining and single dipole probe checks are retained.
- The 60-s bound is deliberate. It is not an unlimited-duration streaming claim.
- One worker runs one trajectory, with 624 CPUs, 256 GB, normal, 02:30:00.
  Baseline R1 extrapolation is about 1.51 KSU per trajectory. Allow approximately
  1.5--2 KSU with active overhead, subject to qualification measurements.
- Maximum worker reservation is 3.12 KSU: qualification 9.36 KSU, discovery
  40.56 KSU. Actual expected combined use is roughly 24--32 KSU. The submitter
  re-reads current project balances and retains 1 KSU per project.
- Project priority is sj53, fa32, ny83. With the supplied balances, all three
  qualification jobs fit on sj53. Recompute allocation after they finish.
- Queue delays and pre-renewal completion are not guaranteed. Do not change
  the checkout while its jobs are queued/running. A failed dependency blocks
  later workers in that lane; inspect/cancel those dependent jobs, do not
  silently release them.

## Commands on NCI

No previously defined terminal variables are required for the first stage:

```bash
cd /g/data/ny83/ch9972/NeuroStim/neurostimenv
git switch feature/l23net-sinusoidal-uniform-field
git pull --ff-only
module load python3/3.10.4
module load openmpi/5.0.5
source /g/data/ny83/ch9972/NeuroStim/bin/activate
python3 -m pip check

python3 experiments/l23net_analysis/nci/submit_l23net_s1.py --stage qualification
python3 experiments/l23net_analysis/nci/submit_l23net_s1.py --stage qualification --submit
```

The first invocation is a dry run. Mechanism compilation, shared-filesystem
visibility, environment and unit checks happen inside the worker; no mechanism
source is edited. Do not run `nrnivmodl` manually in the checkout.

After **all three jobs** finish and their PBS output files are finalized:

```bash
python3 experiments/l23net_analysis/analyze_l23net_s1.py --latest qualification
```

This prints the selected full directory and creates its `s1_summary.json`.
Use a positional suite path instead of `--latest` if you need an older suite.
If `status` is `qualified`, inspect the drift, rates, memory and errors, then:

```bash
python3 experiments/l23net_analysis/nci/submit_l23net_s1.py --stage discovery --qualification-latest
python3 experiments/l23net_analysis/nci/submit_l23net_s1.py --stage discovery --qualification-latest --submit
```

After discovery finishes:

```bash
python3 experiments/l23net_analysis/analyze_l23net_s1.py --latest discovery
```

Copy back the **entire selected suite**, including `submission.json`,
`s0_gate.json`, `qualification.json` where present, `s1_summary.json`, and every
worker directory. For quick triage share the printed summary, each `pbs.out`
epilogue, `pbs.err`, `worker_exit_code.txt`, and any `failure_rank_*.json` or
`failure_summary.json`. Retain HDF5 traces/spikes for full validation.

## Local verification

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export MAIN_PATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m unittest -v tests.test_l23net_s0_s1 tests.test_l23net_r1 \
  tests.test_l23net_g1b tests.test_l23net_replay_validation \
  tests.test_online_stimulation tests.test_online_recording_optimizations
bash experiments/l23net_analysis/validate_l23net_s1_local.sh
```

Completed verification on 29 September 2026:

- All 62 unit/regression tests passed. The synthetic 60-s test also streams
  native-rate artifacts and independently recomputes primary and
  fundamental-excluded scores from disk; it rejects changed scores.
- All six four-second L23Net episodes passed in
  `results/s1_local_CTAoZ2`: serial sham, two-rank sham, and two-rank theta,
  alpha, low-beta and transverse-alpha stimulation. Each contains exactly
  160,000 samples at dt=0.025 ms. Active fields reached 0.4 V/m, coupled to
  extracellular potential, matched the sham prehistory, and were removed.
- The corrected S1 two-rank sham exactly matches an independent legacy G1B
  replay (`results/s1_legacy_compare_5RUwzd`) in EEG, dipole, sample/field
  times, field values and every window's spike fingerprints.
- Testing caught and fixed a new S1 screening error: the environment's
  `firing_rates` mapping includes spike counts as well as Hz fields. The
  guard now extracts only `_firing_rate_hz` values. The original R1/S0
  results were unaffected; a dedicated regression test preserves the fix.
- BallAndStick isolated-cell polarization and online/legacy regressions
  passed. Reduced A/B and stimulation-reachability workflows completed;
  their deliberately tiny scientific samples did not pass their efficacy
  gates and are not reported as positive biological results.
- Shell syntax, Python compilation, frozen-S0 content/source hashes, quota
  dry run, and partial-submission failure handling were checked.

The reduced local episodes and synthetic 60-s analysis test check software,
not scientific efficacy or full-network memory. The actual 60-s full-network
qualification must still run on NCI. Third-party mechanism kinetics were not
audited or modified. Ideal EEG lacks stimulation artifacts and sensor noise;
neither low EEG power nor a successful engineering run is a treatment claim.
