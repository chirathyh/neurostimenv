# CF0b: continuous-alpha phase measurement and rhythm screening

## Purpose and limits

CF0 found accurate continuous carrier estimates, but only 69% of recent phase
estimates passed its amplitude/RMS criterion. Its three B references also
showed that carrier confidence is not a rhythm-presence screen. CF0b preserves
that completed negative result by hash; it does not rerate CF0 as positive.

This follow-up compares existing measurement pipelines, then asks whether a
frozen pipeline and an independently calibrated rhythm screen generalize to
new circuit structures. **No stimulation is applied. No treatment policy is
trained. Neither a pass nor improved confidence coverage establishes H5.**

H1–H4 runners, controllers, network defaults and the binary carrier estimator
are unchanged. FS0 and its cell mechanisms are not imported. The proposed
fixed-location/operating-point FS0 follow-up remains a separate future study.

## 1. Hash-locked replay: no new neuronal simulations

Required input is the complete original `ballnstick_cf0_full/cf0` folder,
including raw traces, original unit-noise vectors, frozen carrier estimator,
tables, configuration and `run_complete.json`. Verify every manifest hash and
the exact conclusion SHA256. Read original files without changing them.

Replay all 15 episodes: three structures, each with two continuous carriers
crossed with D={0.5,2.0}, plus homogeneous B. Retain the CF0-selected smooth
multitaper carrier estimator and every frequency/abstention parameter. Recompute
its decisions and verify exact agreement with the source table. Each saved
trajectory contains 30 s baseline and 8 s stimulation-free continuation after
the original 1 s burn-in. Reconstruct observed EEG exactly as

\[
y_n=x_n+s u_n,\quad
s=0.25\,\mathrm{SD}(x_{\rm baseline})/\mathrm{RMS}(u_{\rm baseline}),
\]

with the original raw-rate AR(1) noise path, rho=0.95. This is an inherited
measurement stress test, not a calibrated clinical noise spectrum. Noise
scaling never uses continuation data.

### Two predeclared phase candidates

Both initialize from the preceding 1 s and subsequently use preceding 0.5-s
tails at 125-ms boundaries. Carrier selection uses the 30-s observed baseline;
accepted incorrect carriers are retained. Hidden generator frequency does not
enter either phase estimator.

1. `cf0_raw`: CF0 regression on intercept, trend and sine/cosine quadratures;
   amplitude/SD cutoff 0.10.
2. `h4_original`: the historical H4 primitives: detrend the available tail,
   fourth-order 0.5–100-Hz Butterworth forward/backward filtering within that
   tail, antialias resample to 500 Hz, then Fourier quadrature projection;
   amplitude/RMS cutoff 0.03, as used by the inherited H4 code path.

The H4 path is reproduced exactly, including its sample-time convention; it
is not silently replaced by a new estimator. Filtering is **causal at each
decision boundary** because only the preceding tail is supplied. It is not a
streaming zero-phase filter, and its boundary effects must be assessed.

The cutoffs are pipeline-dependent engineering conventions, not comparable
calibrated posterior confidence levels. Save a crossed pipeline-by-cutoff
{0.03,0.10} diagnostic with coverage, noise error, common-reference error and
large-error rate. Those extra combinations cannot be selected. The study does
not interpret a coverage increase caused by a lower cutoff as better accuracy.

### Error audits and reference definitions

For a local quadrature fit

\[
y(t)=b_0+b_1t+C\cos(2\pi\hat f t)+S\sin(2\pi\hat f t),
\]

the cosine phase at boundary T is

\[
\hat\theta(T)=\operatorname{wrap}[2\pi\hat f T+\operatorname{atan2}(-S,C)].
\]

Save three distinct errors; do not call them interchangeable:

- **Noise attribution:** observed versus neural-only estimates using the same
  pipeline, frequency and past tail. This is not an absolute phase error.
- **Common offline reference:** a 1-s centered raw neural-EEG quadrature
  regression at the same EEG-selected carrier, evaluated at T. Its support is
  [T−0.5,T+0.5] s. This reference deliberately uses future neural samples for
  scoring only. It reduces past-window lag asymmetry between candidates, but
  remains a smoothed phase convention—not instantaneous biological truth or
  latent afferent ground truth. Very weak oscillations have ambiguous phase.
- **125-ms forward error:** extrapolate the causal phase at the selected
  carrier to T+0.125 s; compare with the same centered offline reference at
  that time. No future samples enter the extrapolator.

Reference estimates are absent when the recording lacks the required future
support. Save validity flags and input timestamps. There are 65 causal
boundaries per full continuation, 61 contemporaneous reference comparisons
and 60 forward comparisons. Save the reference amplitude/SD as an ambiguity
audit rather than censoring difficult windows or treating it as ground truth.

### Frozen replay gates and selection

Average windows within context, crossed contexts within structure, then weight
structures equally. Require, on all rhythmic contexts (not only screen-positive
ones):

- actionable fraction >=0.80 overall, >=0.70 in each D, >=0.60 in each structure;
- accepted noise-attribution mean absolute error <=20 degrees;
- unconditional common-reference mean absolute error <=45 degrees;
- accepted common-reference errors >90 degrees in <=10% of valid comparisons;
- unconditional forward-reference mean absolute error <=60 degrees;
- finite context summaries, safe rates and exact zero field.

These are explicitly **new exploratory engineering tolerances**, specified
before replay, not biologically validated safety limits or a relaxation of the
original CF0 verdict. Candidate selection uses passing candidates only, then
lowest unconditional common-reference error, then noise-attribution error and
name for deterministic ties. Noise-free EEG is a development/scoring reference,
never an input to the deployable estimator.

Freeze the selected profile, complete criteria and unchanged carrier estimator
before any new simulations. If neither passes, `execution=full` finishes with
a negative result and runs **zero** new neuronal simulations. A smoke can
exercise downstream plumbing after failure but is always unqualified.

## 2. B-only rhythm-presence calibration

Only after the replay gate passes, run 19 new homogeneous B structures. The
score is the maximum pooled aperiodic-adjusted continuous-carrier evidence
over 8–12 Hz from observed baseline EEG. Calibrating this maximum accounts for
the frequency search; it is not a threshold at a known hidden frequency.

For n=19 and alpha=0.05, freeze the

\[
k=\lceil(n+1)(1-\alpha)\rceil=19
\]

order statistic of the B scores. A new observation must **strictly exceed**
this cutoff to be rhythm-positive. Ties abstain. Too few calibration structures
produce no finite cutoff and abstention, not a relaxed threshold. No A outcomes,
hidden carrier correctness or validation recordings choose the cutoff.

Under exchangeable B structures, this rank rule provides a marginal null
exceedance bound, not a conditional guarantee for the realized threshold or a
clinical diagnostic specificity claim. Safety/field-integrity failure also
stops before validation.

## 3. Disjoint frozen qualification

Six new structures each provide two independent continuous carrier draws
(one in 8–10 Hz and one in 10–12 Hz), both D values, and matched B:

| Stage | Independent structures | Episodes per structure | New episodes |
|---|---:|---:|---:|
| Replay | 3 existing | 5 | 0 |
| B calibration | 19 new | 1 | 19 |
| Frozen qualification | 6 new | 5 | 30 |
| **Total new, if replay/calibration pass** | **25** | | **49** |

New structure namespaces start at 368101 and 368201; private drive/noise seeds
are separately derived and checked against all planned CF0 namespaces and one
another. The local default H1–H4 network is compared with the frozen CF0
network configuration. Every new episode retains **1 s burn-in + 30 s baseline
+ 8 s zero-field continuation**. The reused online collector calls its last
blocks stimulation/washout; `stimulate=False` is unconditional throughout.
Initialize once per episode and retain the online recorder/event queues.

Repeat measurement and phase audits with no refitting. Require the frozen
profile's phase gates, >=80% unconditional carrier accuracy/coverage, >=90%
accepted-carrier accuracy within 0.25 Hz, >=75% accuracy in each D, >=50% in
every structure, >=80% rhythm sensitivity/specificity point estimates, >=75%
joint rhythm-positive/carrier-accepted coverage in A, rate safety and zero field.

Rhythm specificity is scored **before** carrier or phase rejection; abstention
cannot conceal false phenotype positives. The eventual active-treatment
fallback is sham whenever rhythm, carrier or current phase confidence fails.
This study itself applies no treatment. It establishes no reference-B target
for a new efficacy endpoint and no dose-selection policy.

### Statistics and claim limits

Structures, not windows or carriers, are the independent unit. Save context
and structure endpoints, equal-structure means and 2,000 structure-bootstrap
intervals for phase coverage/error. With only three replay and six validation
structures these intervals are exploratory. No confirmatory superiority claim
is made for one phase estimator, and no multiplicity-adjusted treatment test
is relevant because no treatment is given.

Report exact binomial confidence limits for held-out B specificity. Even
6/6 negatives have a wide interval (lower 95% bound about 0.54). Thus this
small qualification **cannot validate 95% specificity**. A pass permits
further bounded controller testing; it is not powered clinical IAF validation,
tACS efficacy, clinical EEG artifact robustness, desynchronization or H5.

## Artifacts and reproducibility

Save configuration, code/package provenance, upstream hashes, predetermined
contexts, frozen measurement/screen JSON, all rejection decisions, original
new neural/noise traces and hashes, spectral/window evidence, raw phase windows,
context/structure summaries, matched-cutoff diagnostics and runtime. Generate
PNG/PDF alpha-region PSD/evidence, phase comparison, first-context phase trace
and calibration/held-out screening figures. Figure examples use the first
structure/context by saved order, never the best-looking response.

Write `run_complete.json` last with artifact hashes. Scientific negatives are
completed runs. Incomplete or modified sources fail preflight. Exceptions abort
all MPI ranks; existing nonempty study output directories cannot be overwritten.

## Workstation commands

The full experiment keeps CF0's **8 MPI ranks** to avoid a change in rank/layout
as an uncontrolled factor. The previous special H5-P0 folder is not needed.
CF0 results should already be at `/home/u7041472/Documents/chirath/results/`.

```bash
cd /home/u7041472/Documents/chirath/depression-simulator/neurostimenv
git switch research/ballnstick-h1-h4
git pull --ff-only origin research/ballnstick-h1-h4
source /home/u7041472/Documents/chirath/depression-simulator/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg HYDRA_FULL_ERROR=1

time mpiexec --bind-to core --map-by core -n 8 \
  python experiments/ballnstick_analysis/run_ballnstick_cf0b.py \
  experiment.name=ballnstick_cf0b_full \
  env=ballnstick analysis=ballnstick_cf0b \
  analysis.source_cf0.result_dir=/home/u7041472/Documents/chirath/results/ballnstick_cf0_full/cf0 \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true experiment.tqdm=false
```

Output: `/home/u7041472/Documents/chirath/results/ballnstick_cf0b_full/cf0b`.
Based on CF0's 15 episodes in about 19 minutes, the 49-episode path is roughly
**60–90 minutes on the same workstation with 8 ranks**, not a benchmark promise.
Replay failure stops in seconds/minutes. Normal machine load can change this.

### Replay only on the laptop (no MPI or neural simulations required)

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python experiments/ballnstick_analysis/run_ballnstick_cf0b.py \
  experiment.name=ballnstick_cf0b_replay \
  env=ballnstick analysis=ballnstick_cf0b analysis.execution=replay \
  env.simulation.obs_win_len=1000 experiment.plot=true experiment.tqdm=false
```

### Engineering smoke (cannot qualify)

```bash
mpiexec -n 2 python experiments/ballnstick_analysis/run_ballnstick_cf0b.py \
  experiment.name=ballnstick_cf0b_smoke \
  env=ballnstick analysis=ballnstick_cf0b env.simulation.obs_win_len=1000 \
  analysis.smoke=true analysis.design.calibration_structures=2 \
  analysis.design.validation_structures=1 \
  analysis.timeline.baseline_steps=4 analysis.timeline.stimulation_steps=1 \
  experiment.plot=true experiment.tqdm=false
```

The smoke performs real source replay and seven short new episodes. Two B
calibration structures necessarily yield abstention, and one validation
structure cannot pass the full requirements. Those negative scientific checks
are expected and must not be interpreted as an implementation failure.
Smoke structure seeds are offset by 10,000, with drive/noise seeds derived
from those offsets, so engineering tests never expose the full validation set.
An early plumbing test used 367xxx seeds; those are also disjoint from the
final 368xxx study and were not used to select or tune any criteria.
