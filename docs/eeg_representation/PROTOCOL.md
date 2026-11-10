# EEG representation study protocol, version 1

This suite is separate from COSYNE H1–H4. It uses explicit task IDs and the
R0/Ralpha/Rbeta_low/Rbeta_high regime names. Every new empirical hypothesis is
pending. A functional smoke cannot qualify a scientific hypothesis. The
40-cell circuit is a mechanistic stochastic test bed, not validated human
cortex, depression, treatment, or a human predictability bound.

## Questions and falsification

Representation H1 concerns a target- and information-dependent prediction
frontier. For squared error, Bayes risk is `E[tr Cov(Y|I)]/d`; excess risk is
the squared distance from the conditional mean. New unobserved innovations
cannot be removed by additional model capacity. This identity does not imply
a universal EEG sampling interval, failure of distribution prediction, or H2.
An exogenous-future oracle differs from a causal full-state oracle, clean EEG
history, noisy EEG history, and compressed EEG history. Independent sensor
garbling cannot improve population Bayes risk for the same target; equality
is possible. Finite fitted models need not obey that ordering exactly.

H2a compares complementary source supervision with raw reconstruction and
stronger spectral, latent, probabilistic and equal-supervision hybrid recipes.
H2b compares progression with joint, reverse and independently randomized task
orders, holding final data, labels, examples and label exposures fixed.
H2a does not entail H2b. A reconstruction, hybrid or probabilistic model that
matches task-supervised transfer narrows/falsifies H2a. Joint or alternative
orders matching progression narrows/falsifies H2b. A successful PSD baseline
does not support a claim that sophisticated representation learning was needed.

Core held-out baseline-versus-carrier labels are combinations of source rhythm
and carrier labels. They test accessibility and label transfer, not a novel
independent semantic factor. Stronger complementary transfer requires the
targeted coherence/envelope/kinetics/shared-participation extensions and their
observability checks. Preserve weak or negative tasks.

## Designs and seed separation

The core pilot is 8 structures × 4 regimes × 20 trials = 640 trials, with
2 s burn-in and 12 s retained EEG. The provisional confirmation is 40 new
structures × 4 × 25 = 4,000, split 24/6/10 structures before any windows.
Pilot is exploratory. Confirmation generation is refused until a copied
configuration has been frozen; final test access additionally requires
`evaluation.allow_test: true`. Freeze counts, all task/budget axes, source
order, estimator choices and the meaningful-gain margin before generation.
The provisional margin is 0.02 normalized transfer area, not a selected result.
Use pilot structure-level variance and runtime for the eventual precision plan.

`configs/eeg_representation/seed_registry.json` reserves disjoint ranges and
records the audited LFPy multiplication bound. Structure seeds remain below
429496. Private events, initial phase, phase innovations, burst envelope,
sensor noise, fork histories and futures have named namespaces. Conditions
are paired within a structure/trial family, and recording order is shuffled
independently. No seed, regime string, hidden phase, spikes, future events or
mechanism variable enters the EEG model tensor. Privileged arrays are stored
in a separate HDF5 group.

R0 is homogeneous Poisson drive. Rhythmic carriers initially are 10, 16 and
24 Hz at depth 0.04 and D=0. Carrier frequency is not afferent event rate.
The inherited E/I mean intervals remain 40/30 ms. Individual synapses retain
independent event streams conditional on a shared rate phase. Initial phase
is uniform by trial. If development EEG does not expose the carriers, only a
bounded 0.04/0.08/0.16 depth sweep on new development structures is appropriate;
freeze the resulting setting or report failure before confirmation. No code
automatically tunes a generator to make tasks pass.

`extensions.yaml` is an opt-in targeted subset: nearby 9/11 Hz, D=0.5/2,
tonic/burst envelope, I-to-E decay ×0.8/1.2, and shared fraction 0.5/1.0.
It generates homogeneous baseline once. Diffusion is not amplitude bursting.
The burst envelope is a two-state continuous-time Markov process with bounded
depth, default mean on/off dwells 500/1000 ms, and its own RNG. Its nominal
high-state duty cycle is 1/3. The tonic candidate m=0.08/sqrt(3) matches the
input envelope's expected squared depth, not necessarily output EEG power,
realized event count, spectral linewidth, or rates. Quantify remaining
confounds on development recordings. Kinetics changes analytically preserve
conductance-time area, not charge, firing rates, or EEG.

## Causal observations and targets

Native traces and times are float64 in V, nA µm and ms, sampled on `(start,stop]`.
The single channel is a projection of one population dipole. Copying it into
more channels would add no spatial information. Save raw arrays, events,
spikes, phase and structure audit arrays; full dynamic state is captured only
at fork decision times. Analytical controls have no physical dipole; their
compatibility array is explicitly marked unavailable/zero, never interpreted.

A causal Kaiser FIR antialiases and decimates to 500 Hz. Filtering includes
burn-in, then crops it. The FIR uses 64×decimation+1 taps with a 225-Hz cutoff,
64-ms group delay and 128-ms support. Timestamps describe availability; the
delay is not removed. Filter coefficients, initial state and final history
are recorded. A future perturbation test verifies unchanged earlier inputs.
There is no `filtfilt`, latent phase preprocessing, or future-event access.
For filtered fork targets, the causal state includes the fixed filter history.
Very short leads refer to this delayed available stream, not instantaneous
unfiltered physiology.

Each transfer protocol fits channel means/scales and noise reference power
using eligible training recordings only. Float64 calculations use no absolute
1e-8 floor; genuinely zero variance is an error. Model inputs are normalized
float32. There is no per-window amplitude normalization or implicit V-to-µV
conversion. White Gaussian sensor variants use 20/10/0 dB defined by
`10 log10(training pooled demeaned channel power / noise variance)`, with
diagonal channel covariance. Noise is independent of the latent trial.
Different SNR views share one standardized noise path and the parent's split;
they are repeats. The realized per-trial SNR may differ from this label-blind
reference SNR. Drift/artifacts/correlated noise are not silently substituted.

Forecast inputs end at cutoff; waveform targets have support
`[cutoff+lead, cutoff+lead+block)`, default block 128 ms. Leads are
2/10/**26**/50/100/250/500/1000 ms: 25 ms cannot be represented exactly on a
500-Hz grid and is explicitly replaced, not silently rounded. Histories
0.25/1/4 s are declared; one frozen `history_s` and `lead_ms` defines a fit.
Run other settings with separate resolved configs/output roots. Envelope and
power operators declare finite target support. Lower raw variance of a
smoothed target is not automatically better normalized predictability.
Clean-target and future-noisy-recording prediction are explicitly different
configurations; future sensor noise adds an independent error term. The saved
biophysical fork ensemble scores ideal neural targets; its variance excludes
that extra sensor term and must not be presented as the exact risk of a future
noisy recording.

## Future forks and analytical controls

The first backend reruns the same prefix. It never restores pre-generated
future random events with SaveState or clears delayed recurrent spikes.
At grid-aligned decisions it compares structure/morphology/synapse hashes,
external history, segment voltage, HH gates, Exp2Syn states, drive phase,
prefix spikes and pending recurrent events caused before cutoff. The inherited
LFPy external-event path uses direct `NetCon.event(t)` delivery: its unused
NetCon delay must not be added to event times. Event and phase innovations can
be varied independently. A finer phase-grid test checks interpolation safety.

Default forks use 10 states per regime across training structures and 64
independent futures, extending a nested convergence subset to 128. Report
sample covariance trace/d per state, independent disjoint-pair half-square
differences, Monte Carlo intervals, and structural variation separately.
All pairwise differences are not independent. Leave-one-out oracle means
avoid scoring a future with itself; finite-ensemble error is reported.
These are full-causal-state risk lower bounds for EEG-only inference, not
the exact nonlinear EEG Bayes limit or proof of a model-capacity plateau.

The independent controls cover exponential Poisson shot noise, a damped
linear-Gaussian oscillator with exact state/clean-history/noisy-history Kalman
risks, and trial-dependent Gaussian variance. The last control retains variance
classification despite an optimal zero conditional-mean forecaster; it shows
objective mismatch, not inevitable collapse. A variance classifier and
posterior-variance Gaussian predictor provide positive controls. See `ANALYTICAL_CONTROLS.md`.

## Training, transfer and inference

All principal neural recipes use the same patch Transformer (default 128
dimensions, four layers/heads, 64-ms patches) and pooled representation; actual
total parameter counts are checked against 2M. The TCN is a selectable common
sensitivity architecture. Encoder checkpoints are saved separately from heads.
Recipes include PSD/bandpower/envelope features, source specialist, masked
MSE/L1, multiscale spectral/envelope reconstruction, momentum contrastive
learning, low-rank-plus-positive-diagonal Gaussian forecasts, diagonal/AR
Gaussian references, joint/progressive/reverse/random supervised training and
reconstruction with exactly the same source heads/labels. Gaussian forecasts
cannot represent all multimodal futures. Reconstruction is bidirectional only
within its declared history; forecasting receives causal history only.

The implemented order comparison is offline: every recipe has the same
frozen final recording pool. A separate online dataset-arrival experiment is
not implemented and would require the same arrival schedule for all methods.
Progression reserves replay tickets from the full source pool for later
stages; capacity, sampling, ordering, unique labels, exposures and schedule
hashes are recorded. Random-order repetitions vary order separately from the
paired optimizer seed, capped at the number of distinct task permutations. Final exposure parity is tested; extra hybrid/teacher
forward passes and measured runtime must still be disclosed. Do not claim
equal FLOPs from equal optimizer updates. Principal non-specialist neural
recipes share the same source-eligibility EEG tickets; source labels define
this matched sampling design even for losses that do not consume those labels.
Feature models and the single-task specialist are separately accounted baselines. Mask fraction/gap (32/128/512 ms)
and auxiliary weights can be changed only in comparably budgeted development
searches; save the exact configs. No amplitude-invariant augmentation is used
for variance-defined tasks. Source orders are prespecified candidates unless
a development-only selection record is supplied.

`leave-task-label-out` shares the pretraining EEG pool but withholds declared
downstream labels. `leave-generator-family-out` also excludes held-out families
from pretraining recordings and normalization. Downstream probes can learn
labels from training/development structures, including the held-out family;
test circuits never fit parameters. Strong family-only scores are distinguished
from all-family scores; one-class slices are Pending. This is not new-circuit
few-shot calibration. That optional support/query protocol is not implemented.

Frozen linear and shallow nonlinear probes use development-only hyperparameter
selection. Independent paired trial families, not windows or condition siblings, determine
nested label budgets. Optional limited-label encoder adaptation and a random-initialized
downstream specialist are enabled with `evaluation.adaptation_steps`; they
are reported separately from the frozen-probe primary endpoint. Retention and
early learning curves are separate from final transfer. Noise/gain stress
scores reuse frozen probes and training scales.

The primary classification aggregate is the equally weighted mean of
task-specific balanced-accuracy areas over a frozen log independent-label
budget axis, divided by its log range. Capped/incomplete axes remain Pending.
Continuous normalized error/R² are separate. Average paired optimizer seeds
inside structure, then weight circuits equally and resample circuits. Report
paired intervals, sign flips and FDR-controlled secondary task contrasts.
Windows, noise views and optimizer seeds are not additional circuits.
Forecast scores include target-matched MSE/NMSE, proper NLL with normalization
units, calibration/coverage and CRPS/energy; distribution diagnostics do not
establish conditional correctness.

Resume requires matching schema, resolved config, source and shard hashes.
Writes are atomic and a coordinator lock prevents concurrent output-root
writers. Workers own separate trial shards. There are no destructive legacy
folder helpers. Missing jobs and negative findings stay visible. Aggregation
uses real result files only; smoke values remain separate from empirical
claims and LaTeX Pending cells are never converted to zero.
