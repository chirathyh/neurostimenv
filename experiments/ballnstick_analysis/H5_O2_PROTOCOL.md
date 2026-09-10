# H5-O2: screened montage response reassessment

## Decision and scope

This is a small, new, active response map after the completed positive
H5-O1S rhythm-screening validation. It is **not H5 confirmation and does not
train a learning policy**. It asks two separate questions:

1. With the repaired noisy-EEG screen and spatial estimator frozen, does the
   analytical EEG-geometry rule improve the neural-current spectral endpoint
   over sham and the strongest fixed field profile?
2. Is there a practically useful, independent-future response opportunity
   **beyond that analytical rule**, sufficient to justify another policy study?

Success on the first question alone demonstrates rule-based contextual
control, not an advantage of machine learning. A negative answer to the
second question is a useful stopping result. No further generator, cutoff,
controller, or action tuning is permitted on these outcomes.

## Why this is a defensible but deliberately limited tACS model

The causal chain is a local extracellular field, spatial membrane
polarization, altered membrane/synaptic currents and possibly spike timing,
followed by a current-dipole EEG forward model. Unlike a point neuron, the
soma-plus-apical-cable model permits differential polarization and dendritic
filtering. Simplified spatial models are useful for this mechanistic purpose;
they do not establish that this particular network reproduces human alpha
physiology. See [Aspart et al. (2016)](https://arxiv.org/abs/1603.04881).

For local compartment position r in micrometres and tissue electric field E
in V/m, the imposed extracellular potential is

    v_ext(r,t) [mV] = -0.001 E(t) · r.

This is an extracellular potential gradient, not an injected somatic current
or a scalp-current prescription. An end-to-end extracellular voltage
difference is not itself the resulting transmembrane polarization.

With R(theta) the population's local-to-head rotation and L the three-sensor
leadfield, the reciprocal transforms are

    E_local(t) = R(theta)^T E_head(t),
    y_neural(t) = L R(theta) p_local(t).

Cells and synapse locations remain in the canonical local coordinate system.
Rotating physical cells would change existing z-dependent synapse-placement
rules; that would confound orientation with a different circuit. Sham local
dipole and spike histories must remain identical across paired orientations.

The two actions represent idealized local-field solutions from two possible
montages/current distributions. Real multi-electrode tACS can change local
field direction through its electrode-current pattern, but this experiment
does **not** solve that mapping. A real implementation would need a subject
head model, electrode placement/contact constraints, current conservation,
current limits and off-target exposure assessment. Intracranial measurements
support weak intracranial fields with strong spatial/montage dependence;
they do not clinically validate our profiles or amplitude. See
[Huang et al. (2017)](https://elifesciences.org/articles/18834.pdf).

The two equal-norm 0.2-V/m peak fields have head-frame directions 0 and 60
degrees. Their axial projections onto a straight cable at theta are

    E_axial,0 = 0.2 cos(theta),
    E_axial,60 = 0.2 cos(theta - 60 degrees).

This immediately supplies a strong non-learning baseline: estimate theta
from EEG, then choose the closest profile (boundary 30 degrees). The
intervention is likely dominated by this projection in a single straight
column. Making orientation observable does not guarantee any further
information that ML could exploit. That is why the analytical comparator
is indispensable.

## Frozen biological and observation settings

- Forty cells: 32 E, 8 I; one soma and one apical cable each. The intrinsic
  canonical Hodgkin–Huxley soma is shared by E/I populations; outgoing
  synapses and afferent drive distinguish populations. Temperature remains
  6.3 degrees C, the channel model's reference temperature. This is not a
  calibrated mammalian-temperature depression microcircuit.
- The neuronal integration step remains 0.0625 ms; online base windows are
  1000 ms. No reinitialization occurs at feedback boundaries.
- A uses independent conditional Poisson afferents with modulation depth
  m=0.04, shared rhythmic afferent fraction q=1 and carrier f in {9,11} Hz.
  B is the homogeneous, mean-rate-matched afferent reference. Neither is a
  clinical diagnostic label. Recurrence and intrinsic cell parameters do not
  change between these states.
- Afferent phase obeys dphi = 2 pi f dt + sqrt(2D) dW, with
  D=0.5 rad^2/s. Conditional rate lambda_j(t)=lambda_0,j[1+m cos(phi(t))]
  remains nonnegative and has the same ensemble mean. Each afferent keeps
  its private event stream. A shared phase is not copied Poisson events.
- The external afferent generator does not receive stimulation feedback.
  tACS changes the network's response, not the externally specified phase
  generator. Autonomous oscillator entrainment claims would go beyond this
  model.
- Three idealized sensors and the known four-sphere leadfield are frozen.
  Noise is additive independent AR(1) across sensors, coefficient 0.95, with
  the same absolute scale at all sensors: 0.25 times the **raw baseline
  vertex neural RMS**. Unit paths are normalized using only the predecision
  prefix; amplitude scaling is fixed from baseline throughout the episode.
  This is not a 25% RMS guarantee after filtering or during suppression.
- The rank-one cross-spectral estimator assumes one source, a known source
  location/leadfield, and equal-spectrum independent sensor noise. It is not
  a general cortical-source inverse solution or a connectivity biomarker.
- The controller sees noisy EEG but no direct stimulation artifact. A real
  concurrent EEG-tACS controller needs separate artifact handling and latency
  validation. At 10 Hz, uncompensated 10/20-ms delays correspond to 36/72
  degrees. Those delays are not modeled or silently compensated here.
  Uncertainty-aware causal phase estimation is a principled later extension,
  not another parameter to optimize in this map; see
  [Wodeyar et al. (2021)](https://elifesciences.org/articles/68803).

## What is frozen from H5-O1S

The loader hash-checks its conclusion, completion marker, B targets, rhythm
rule, protocol, provenance, configuration, trajectory audit and PSD table.
Only this result folder is required; earlier ancestry is embedded provenance.

1. The observed-EEG alpha-excess threshold remains 0.05 log10 units above
   the frozen population-B baseline mean.
2. Maximum pooled multitaper residual evidence must strictly exceed
   **2.5411671943274263 dB**, the B-only calibrated rhythm cutoff.
3. Spatial confidence, multitaper carrier confidence, and recent phase
   actionability must pass. Failure maps to sham. Hidden carrier correctness
   cannot decide enrollment, and accepted incorrect carriers remain included.
4. The analytical action is the observed CSD angle's nearest field profile.
   No old auto-power threshold or newly fitted classifier is used.
5. Efficacy uses the **neural-audit** B outcome target,
   mu_B = -20.822591909008796 in geometry-normalized log10 alpha power.
   Screening uses the separately frozen **observed** baseline target. The
   positively biased short-window noisy B outcome target is not substituted
   into the neural endpoint. No new B calibration or seed-specific B is used.

These source targets were originally estimated from only three B structures.
Target uncertainty is therefore a limitation, not removed by having many
active simulations. The new map reports overshoot and noisy-versus-neural
endpoint differences explicitly. No claim about deployable noisy rewards
is made.

## Experimental structure and timing

Three new structures (429451--429453 with experiment.seed=1), two carriers,
and four orientations (0,20,40,60 degrees) give 24 candidate contexts.
There is one afferent history per structure/carrier. Orientations are paired
views of the same local circuit, not independent biological subjects.

Each eligible context has four independent postdecision continuations and
exactly three treatment arms:

| Arm | Field peak | Field direction | Controller |
|---|---:|---|---|
| Sham | 0 V/m | irrelevant | no active field |
| Profile z | 0.2 V/m | head-frame 0 degrees | frozen H4 fast tracker |
| Profile 60 | 0.2 V/m | head-frame 60 degrees | same tracker |

Every arm follows one persistent episode:

| Time | Purpose |
|---|---|
| 0--1 s | Burn-in, excluded from endpoint |
| 1--31 s | Stimulation-free observed EEG: phenotype, carrier and orientation |
| 31--40 s | Single fixed-profile intervention; 0.5-s onset/offset ramps |
| 31.5--39.5 s | Central 8-s primary outcome, excluding ramps |
| 40--42 s | Field removed; separate physiological recovery audit |

The carrier is selected from 30-s predecision EEG. All active arms initialize
from the preceding one-second phase estimate, then use a 0.5-s history every
125 ms, targeting pi-relative phase. The phase-correction horizon is fixed
at 250 ms and frequency correction bounded at +/-2 Hz. Phase and envelope
remain continuous; the profile and amplitude never switch within an episode.
The nominal carrier is fixed, but the instantaneous frequency changes during
bounded phase correction. This is not a sequence of phase-discontinuous
sinusoids.

The first future's full sham is also used for screening, but the screen
function receives **only its baseline EEG**. The prospective screening file
is written before active simulations. Its future sham outcome is never
passed into screening. That sham is reused, not simulated twice.

With all contexts eligible, there are

    3 structures x 2 carriers x 4 angles x 3 arms x 4 futures = 288 episodes.

If e of 24 contexts enroll, there are 24 + 11e executed episodes and 12e
paired outcome rows. Rejected contexts receive their screening sham only.
They are reported, not imputed as extra successful control cases.

Within a context/future, all actions share the exact prestimulation history
and future external randomness. Across futures, only postdecision neural
drive and sensor-noise paths change. Recurrence may change naturally after
stimulation. Episodes are deterministic replays up to the decision boundary,
not resumed from an incomplete membrane-only snapshot. Original unit-noise
paths, seeds and hashes are saved, and reconstruction is checked against
the actual online controller observations.

## Endpoints, baselines and decision rules

For each intervention compute the frozen CSD geometry-normalized neural
log-alpha value y and absolute target distance

    L = | y - mu_B |.

This symmetric distance penalizes suppression below B as well as residual
excess power. Improvement relative to sham is L_sham - L_active; larger is
better. Alpha suppression itself, noisy EEG measurements, firing rates,
fixed-carrier spike PPC, phase actionability, waveform continuity, exact field
removal and stochastic washout recovery are separate audits.

The fixed-carrier PPC audit is (|sum exp(i phase_k)|^2 - N)/(N(N-1)), using
spikes in the central endpoint and the same EEG-selected carrier for every
arm, including the first reused sham. It measures population locking to that
carrier, not pairwise neuron-neuron desynchronization. An EEG-power decrease
can reflect altered/cancelling transmembrane currents without reduced spike
synchrony; report this distinction rather than calling every decrease
"desynchronization".

The full-information empirical oracle selects the smaller mean loss for
each context. It is optimistic and not deployable. The mandatory replication
selects the context-specific profile on futures 1--2, evaluates it on 3--4,
then repeats with the split reversed. The best fixed comparator is also
chosen on the selection futures only; no arm is discarded to help
personalization. The exact expected uniform-random-active baseline is the
average of the two active losses.

The analytical EEG rule is frozen and uses no response labels. Report both
its advantage over fixed/sham and the response selector's incremental
advantage over this analytical rule. Residual opportunity requires practical
support for both profiles across multiple contexts/structures and at least
0.01 log-distance advantage over **both** fixed and analytical comparators,
including both independent-future splits. The 0.01 criterion is an engineering
decision threshold, not a clinically calibrated effect size or "1% power".
Do not reintroduce a blanket 75% realized-winner gate: expected benefit,
independent-future replication and regret are more relevant than noisy
single-realization winners.

The selected active profile must also improve over sham in both splits.
Analytical-rule success and residual learning opportunity are reported
separately: if neural response contradicts the maximum-projection heuristic,
an independently beneficial alternative can still justify policy development.
The heuristic is a strong comparator, not a presumed law of treatment efficacy.

All means first average futures/contexts within structures, then weight
structures equally. Save structure bootstrap intervals and exact paired
sign-flip audits. With three structures the smallest one-sided exact p-value
is 0.125. This stage is deliberately **not statistically powered confirmation**.
There is no multiplicity-adjusted confirmatory claim. A later policy study
would need whole-structure out-of-sample prediction and a new, powered,
hash-frozen confirmation of incremental benefit.

## Outputs

The runner saves prospective screening, complete context/action/future
metrics, expected response maps, two independent-future splits,
structure-level effects and inference, phase updates, processed three-channel
EEG, original paired unit-noise paths, representative raw dipoles/EEG/fields
and spike times, frozen inputs, runtime, provenance and a recursive SHA256
completion manifest.

PNG/PDF figures cover alpha PSDs (4-s Welch segments, 0.25-Hz bins), response
curves and residual opportunity, spatial/noise/mechanism diagnostics, firing
and washout, and actual controller waveforms. B PSD curves are explicitly
labeled as frozen stimulation-free reference baselines, not new matched B
outcomes. A power-only plot cannot establish a spike-timing mechanism.

## Workstation command

The current research branch is `research/ballnstick-h1-h4`. Do not reset a
dirty worktree; inspect `git status` first.

```bash
cd /home/u7041472/Documents/chirath/depression-simulator/neurostimenv
git status --short
git fetch origin
git switch research/ballnstick-h1-h4
git pull --ff-only origin research/ballnstick-h1-h4
source /home/u7041472/Documents/chirath/depression-simulator/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg HYDRA_FULL_ERROR=1
set -o pipefail
mpiexec -n 16 --bind-to core --map-by core \
  python experiments/ballnstick_analysis/run_ballnstick_h5_screened_montage_response_mapping.py \
  experiment.name=ballnstick_h5_screened_montage_response_mapping_full \
  experiment.seed=1 env=ballnstick \
  analysis=ballnstick_h5_screened_montage_response_mapping \
  analysis.source_h5o1s.result_dir=/home/u7041472/Documents/chirath/results/ballnstick_h5_rhythm_screening_validation_full/h5_rhythm_screening_validation \
  env.simulation.obs_win_len=1000 experiment.plot=true experiment.tqdm=false \
  2>&1 | tee /home/u7041472/Documents/chirath/results/h5_o2_terminal.log
```

Use a new experiment.name if a previous output directory exists. Earlier
H5-P0's exceptional location under `neurostimenv/results` is not needed by
this runner. Expected output is
`/home/u7041472/Documents/chirath/results/ballnstick_h5_screened_montage_response_mapping_full/h5_screened_montage_response_mapping`.

The full run retains the source experiment's **16 MPI ranks**, not a claim
of optimal scaling. The inherited network initialization contains rank-based
random seeding, so a different rank count changes the circuit realization
even with the same structure seed. The runner rejects that unplanned full-run
change; smokes may use fewer ranks. MPI partitions the 40-cell network; it
does not execute 16 entire contexts concurrently. Do not multiply worker
count by an assumed linear speedup. Runtime and progress are recorded.

## Local implementation verification

Before release, 86 targeted tests passed, covering the new mapping logic,
previous screening/spatial/montage methods, online stimulation and H1--H4
analyses. The isolated-cell polarization validation passed. The online/legacy
regression passed with relative EEG RMS error approximately 2.55e-11.

Two shortened 12-outcome MPI smokes completed with two and four ranks
(approximately 8.2 and 16.7 minutes on the laptop; these are not workstation
runtime predictions). The final smoke had finite arrays, five PNG/PDF figure
pairs and 50 verified artifact hashes. Recorded field and EEG sample counts,
causal updates, all-sensor pairing, frozen amplitude/controller settings and
field removal were checked. The final analysis/plotting code was also replayed
against those saved outputs after presentation and audit refinements.

Eleven previously saved full-duration O1S examples, including accepted and
rejected measurements, reproduced the frozen screen/evidence/orientation
and eight-second endpoint through the new measurement interface. This checks
30-second measurement compatibility without pretending that shortened neural
smokes establish scientific efficacy. The full 288-outcome study has **not**
been run locally, and no success of H5 is asserted.
