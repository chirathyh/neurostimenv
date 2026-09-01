# NeuroStimEnv contributor guidance

## Current objective

Develop the BallAndStick circuit as a transparent mechanistic test bed for:

1. detecting a matched-seed change in I-to-E inhibition from simulated EEG; and
2. testing, on held-out circuit seeds, whether causal stimulation can move the
   reduced-inhibition circuit toward the reference circuit.
3. demonstrating a separate toy entrainment problem in which an asynchronous
   stochastic-drive state A and a weak rhythmically driven reference state B
   are distinguishable from ideal simulated EEG, then testing whether uniform
   field tACS moves held-out A circuits toward the frozen B EEG state.

Work from first principles, report negative results, and do not describe this
40-cell model as a validated model of depression, human cortex, or treatment.

Use the repository-parent virtual environment:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
```

Run commands from this repository root.

## Scientific validity boundaries

- The morphology is one soma plus one apical cable. Every E and I cell has the
  same intrinsic canonical Hodgkin-Huxley soma; E/I identity is defined by
  outgoing synapses and population-specific drive.
- Canonical `hh` kinetics are referenced to 6.3 °C. Use
  `env.network.celsius=6.3` and `online.temperature_mode=configured`.
  A mammalian-temperature study requires a mammalian channel model; changing
  `h.celsius` to 36.5 is not a valid substitute.
- Condition B is only the explicit intervention
  `network.inhibition_scale=0.5` on I-to-E conductance. It is not, by itself, a
  biological definition of depression.
- Simulation seeds quantify variability within this model, not biological
  subjects or clinical uncertainty.
- The generated EEG is an ideal, artifact-free forward-model signal. Real
  stimulation-EEG would contain a much larger stimulation artifact.
- In the entrainment-state experiment, A and B are not health/disease labels.
  They differ only in afferent event-time statistics at an unchanged expected
  mean rate. The B-generating modulation is not the tACS actuator.

## Parameter and unit conventions

- Length: µm; time: ms; membrane voltage: mV; NEURON conductance weight: µS.
- EEG returned by the environment: V.
- Legacy-compatible online field action:
  `[ac_amplitude_v_per_m, frequency_hz]`.
- Scientific mapping actions may additionally contain signed
  `dc_offset_v_per_m`, non-negative `ac_amplitude_v_per_m`, `phase_rad`, and
  either a named `montage` or a three-component `field_direction`.
- The default field is spatially uniform along `+z`, aligned with the
  somatodendritic axis. Field amplitude is specified at the modeled tissue.
- `point_source_current` is retained only for legacy regression. Its amplitude
  is mA injected by a microscopic point source and must not be interpreted as a
  scalp-current dose.
- A 1000-ms online window at `dt=0.0625 ms` contains exactly 16,000 samples,
  using `(t_start, t_stop]`. A waveform contains the additional left endpoint.

## Online lifecycle

- Construct one network and call `finitialize()` exactly once per episode.
- Select an action only after the previous window's observation is available.
- Advance to absolute time boundaries without reinitializing membrane,
  channel, synaptic, recurrent, or event-queue state.
- `reset_online()` may rebuild a new episode; never call it between decisions.
- Keep the manual fixed-step recorder. With NEURON 8.2.3, `Vector.record()` or
  `Vector.play()` attached after initialization does not become active without
  `frecord_init()`, which would disturb persistent LFPy recorders.
- Preserve the private-LFPy-transform version guard and exact segment order.
- Keep `close()` idempotent.

## Statistical design rules

- Treat the circuit seed as the statistical unit; never treat adjacent EEG
  windows from one trajectory as independent replicates.
- Use matched A/B seeds (common random numbers) and verify identical
  connectivity/background realizations.
- Burn in before extracting features. Estimate low-frequency EEG features from
  several seconds, not a 100-ms window.
- Predefine EEG features and report paired effects, confidence intervals,
  permutation p-values, FDR, and held-out discriminability.
- Keep E/I firing rates as mechanistic/guardrail variables, not as deployable
  EEG-only RL state unless invasive observations are explicitly intended.
- Tune stimulation only on discovery seeds. Freeze the primary protocol and
  evaluate it on disjoint validation seeds.
- Reachability requires held-out distance-to-A improvement, shift alignment,
  and rate safety. A failed criterion is a valid outcome.
- Before continuous-action RL, show a smooth and reproducible dose-response.
  Otherwise use a finite discrete set containing sham and held-out validated
  protocols.

## Primary experiment entry points

- A/B characterization:
  `experiments/ballnstick_analysis/run_ballnstick.py`
- Stimulation discovery and validation:
  `experiments/ballnstick_analysis/run_ballnstick_stimulation_sweep.py`
- Fixed-protocol mechanism and confounding confirmation:
  `experiments/ballnstick_analysis/run_ballnstick_stimulation_mechanism.py`
- Causal metric and selective-I controllability ladder:
  `experiments/ballnstick_analysis/run_ballnstick_controllability_ladder.py`
- Isolated-cell field-polarization validation:
  `experiments/ballnstick_analysis/validate_ballnstick_polarization.py`
- Signed-field open-loop system identification:
  `experiments/ballnstick_analysis/run_ballnstick_field_controllability.py`
- Reversible weak-field entrainment (T1):
  `experiments/ballnstick_analysis/run_ballnstick_tes_entrainment.py`
- Online/legacy regression:
  `experiments/ballnstick_analysis/validate_online_legacy.py`
- Single online episode:
  `experiments/ballnstick_analysis/run_ballnstick_online.py`
- Asynchronous-to-entrained state reachability:
  `experiments/ballnstick_analysis/run_ballnstick_entrainment_state.py`
- EEG-primary asynchronous-to-reference reachability:
  `experiments/ballnstick_analysis/run_ballnstick_eeg_reachability.py`
- Hierarchical EEG-only frequency/phase/amplitude identification:
  `experiments/ballnstick_analysis/run_ballnstick_hierarchical_tacs.py`
- Shared afferent phase-diffusion validation (D0):
  `experiments/ballnstick_analysis/run_ballnstick_phase_diffusion_validation.py`
- Causal phase-maintenance audit after D1 (D1-R):
  `experiments/ballnstick_analysis/run_ballnstick_phase_refresh_audit.py`
- H4 causal phase-tracker bandwidth discovery (H4-BW):
  `experiments/ballnstick_analysis/run_ballnstick_phase_refresh_bandwidth_discovery.py`
- Targeted 1-s/125-ms cadence and reliability discovery (H4-BW2):
  `experiments/ballnstick_analysis/run_ballnstick_phase_refresh_cadence_discovery.py`
- Disjoint adaptive phase-maintenance confirmation (H4-C):
  `experiments/ballnstick_analysis/run_ballnstick_h4_confirmation.py`
- Frozen EEG phase-increment observability confirmation (D0b):
  `experiments/ballnstick_analysis/run_ballnstick_phase_increment_confirmation.py`
- Phase-diffusion full-information action mapping (D1):
  `experiments/ballnstick_analysis/run_ballnstick_phase_diffusion_action_map.py`
- Frozen-frequency phase-invariant EEG confirmation:
  `experiments/ballnstick_analysis/run_ballnstick_phase_invariant_tacs.py`
- Single-action conditional expected-dose mapping:
  `experiments/ballnstick_analysis/run_ballnstick_single_action_dose_map.py`
- Monotone EEG-severity threshold discovery:
  `experiments/ballnstick_analysis/run_ballnstick_severity_threshold_discovery.py`
- Frozen EEG-severity single-action rule confirmation:
  `experiments/ballnstick_analysis/run_ballnstick_severity_threshold_confirmation.py`
- EEG-conditioned frequency/relative-phase feasibility map (F0-FP):
  `experiments/ballnstick_analysis/run_ballnstick_frequency_phase_feasibility.py`
- Disjoint stationary H1--H3 confirmation after F0 (S1-C):
  `experiments/ballnstick_analysis/run_ballnstick_stationary_h1_h3_confirmation.py`
- EEG-relative alpha-suppression toy proof of concept:
  `experiments/ballnstick_analysis/run_ballnstick_alpha_suppression.py`
- Frozen alpha-suppression confirmation on new seeds:
  `experiments/ballnstick_analysis/run_ballnstick_alpha_suppression_confirmation.py`
- Exploratory frozen-phase alpha dose/mechanism audit:
  `experiments/ballnstick_analysis/run_ballnstick_alpha_suppression_dose_audit.py`
- Prospectively screened frozen-dose alpha confirmation:
  `experiments/ballnstick_analysis/run_ballnstick_alpha_suppression_screened_confirmation.py`
- Crossed-seed EEG-context dose feasibility gate (CL0):
  `experiments/ballnstick_analysis/run_ballnstick_context_dose_feasibility.py`
- Common-probe contextual-dose system identification (CL1-P):
  `experiments/ballnstick_analysis/run_ballnstick_context_probe_feasibility.py`
- Held-out EEG-trajectory contextual-dose confirmation (CL1-C):
  `experiments/ballnstick_analysis/run_ballnstick_context_trajectory_confirmation.py`
- Scientific rationale and commands:
  `experiments/ballnstick_analysis/SCIENTIFIC_EXPERIMENTS.md`

## Required checks after relevant changes

```bash
python -m unittest -v tests/test_online_stimulation.py

python experiments/ballnstick_analysis/validate_ballnstick_polarization.py \
  experiment.name=ballnstick_cellular_polarization \
  env=ballnstick \
  analysis=ballnstick_polarization \
  experiment.plot=false \
  experiment.tqdm=false

python experiments/ballnstick_analysis/validate_online_legacy.py \
  experiment.name=ballnstick_online_legacy_smoke \
  env=ballnstick \
  env.simulation.duration=500 \
  env.simulation.obs_win_len=250 \
  experiment.plot=false \
  experiment.tqdm=false
```

Also run reduced A/B and stimulation-reachability smokes from
`SCIENTIFIC_EXPERIMENTS.md`. Confirm finite EEG, exact window counts,
monotonic time, current-window spike accounting, configured/effective
temperature equality, clean zero-field removal, and no cleanup exception.

For the controllability ladder, preserve the interpretation hierarchy:
I-to-E interpolation validates the output metric; selective I-population
background drive is a mechanistic positive-control actuator; neither is TES.
Freeze the discovery-ranked multiplier before held-out validation.

For uniform-field control, first validate isolated-cell polarization, then run
the one-seed broad field screen and two-seed targeted replication. Do not run
the full multi-seed field experiment unless both quick stages show the same
direction, beyond-synthetic improvement, seed consistency, and rate safety.
The controllable-span projection is necessary but not sufficient; actual
available actions must improve distance to A.

For the fixed 0.5-V/m, 10-Hz follow-up, keep calibration and confirmation
seeds disjoint.  Do not replace the primary fundamental-excluded endpoint with
the raw relative-gamma endpoint after inspecting results.  Distinguish acute
directional modulation from entry into the A-equivalence region.

For the entrainment-state pilot, A and B must retain identical cells,
recurrence, inhibition scale, mean afferent rate, and background synaptic
weights. B differs only through sinusoidal modulation of independent Poisson
afferent rates; A+tACS must retain modulation depth zero. Calibrate only the B
modulation depth on seeds disjoint from validation, keep the tACS action fixed,
and treat E-population PPC target-distance reduction as the primary endpoint.

For T1 entrainment, keep the tissue field at or below 0.8 V/m, use one
persistent baseline-stimulation-washout episode plus a same-seed sham, and use
population PPC difference-in-differences as the primary endpoint. Freeze the
discovery action before disjoint validation. Raw EEG power at the stimulation
fundamental, PLV without a finite-spike control, or a one-seed response cannot
establish entrainment.

For EEG-primary entrainment-state reachability, fit and freeze the A-to-B EEG
mapping using unstimulated A/B discovery seeds only. Keep the policy action set
to sham and a small number of frozen axial doses; orientation and matched-sine
arms are controls, not policy actions. Treat seeds, not adjacent 1-s windows,
as independent units. Report the ideal forward-model EEG conclusion separately
from the fundamental-excluded and observation-only-sinusoid robustness checks.

For hierarchical tACS identification, keep target-learning, frequency-screen,
phase-screen, and validation circuit seeds disjoint. The selector may inspect
only EEG-derived features and action metadata; never expose the hidden B input
frequency, modulation depth, spikes, or rates. Use phase-invariant spectral
distance for frequency selection, then phase-sensitive EEG distance for phase
selection. Freeze both before held-out amplitude validation. Report a negative
discovery screen even if a subsequently selected least-bad action happens to
improve validation. Require excluded-space A/B observability before describing
fundamental-excluded movement as B-like, and match both Fourier quadratures in
the observation-only control.

For the frozen 60-Hz phase-invariant follow-up, absolute Fourier phase is not
an A/B state variable and must not be optimized in asynchronous A. Keep the
60-Hz frequency, zero-phase convention, 0.8-V/m primary amplitude, calibration
feature, and full validation seeds frozen before confirmation. The
matched-observation sine and frequency-excluded analysis are measurement
audits, not substitutes for the ideal neural-only EEG primary endpoint. Do not
fit a bandit until held-out EEG movement passes; contextual control additionally
requires a replicated context-by-action interaction.

For shared phase diffusion, use one population-level latent afferent phase
following `dphi = 2*pi*f*dt + sqrt(2D)dW`, but retain conditionally independent
Poisson events for every background synapse. D changes phase coherence and
spectral linewidth; it does not by itself create burst-amplitude dynamics.
Validate the SDE, mean-rate invariance, multi-second ideal-EEG observability,
and recent causal phase estimation in stimulation-free D0. Freeze D0 outputs
before implementing D1. Do not tune diffusion levels using tACS outcomes or
describe the latent process as an explicit thalamic or clinical disorder model.

For the D1-R causal phase-maintenance audit, preserve the failed D1 result by
hash and use new seed namespaces. Keep both active arms at the same 0.2-V/m
amplitude, EEG-selected 9/11-Hz carrier, axial montage, and pi-relative phase
target. The one-time arm initializes phase once; the refreshed arm estimates
phase only from the preceding rolling ideal-EEG tail. Correct phase with a
bounded frequency slew while retaining oscillator and block-envelope
continuity; never introduce phase jumps at update boundaries. Sham, frequency,
diffusion, and independent futures remain paired controls. This experiment is
an ideal-neural-EEG mechanism/reliability audit, not a clinical closed-loop
claim, policy confirmation, or contextual-bandit experiment.

For H4-BW, hash-lock the failed D1-R result and reuse its frozen population-B
target without recalibration. Use new seed namespaces, the paired 9/11-Hz by
low/high-diffusion grid, four independent futures, and the exact controller
ladder sham/one-time/1-s--250-ms/0.5-s--250-ms/0.5-s--125-ms. Give every active
arm the same one-second initialization, 0.2-V/m axial field, EEG-selected
carrier, and pi-relative phase target. Keep the phase-correction horizon fixed
at 250 ms while varying observation cadence so feedback gain is not silently
changed. Freeze a short-history controller only if it passes the predeclared
endpoint, future-reproducibility, phase-error, variance, cross-structure,
cross-diffusion, causality, continuity, and safety gates. Otherwise stop before
the 12-structure H4 confirmation. H4-BW is discovery, not a bandit or a
confirmatory result.

For H4-BW2, preserve the completed negative H4-BW result by hash and use new
seed namespaces. Add only the missing 1-s/125-ms controller to the frozen
sham/one-time/1-s--250-ms/0.5-s--125-ms comparison; do not reinterpret the
earlier H4-BW gate. Give all active arms the same one-second initialization,
EEG-selected carrier, 0.2-V/m axial field, pi-relative target, and 250-ms
correction horizon. Use a new disjoint homogeneous-B calibration only for the
duration-matched eight-second outcome target; keep the frozen prestimulation
eligibility target unchanged. Evaluate every controller with the same causal
one-second phase auditor at common 250-ms boundaries. Estimate reliability
from within-context paired controller-minus-one-time effects over six futures,
not marginal outcome variance. Freeze a controller only if it passes the
predeclared endpoint, paired-future, paired-variance, structure, diffusion,
phase, causality, continuity, and safety gates. Otherwise stop before H4
confirmation. H4-BW2 remains discovery and is not a bandit.

For H4-C, hash-lock the positive H4-BW2 conclusion, selected-controller JSON,
duration-matched B target, provenance, screening, metrics, updates, and
selection table. Freeze the selected 0.5-s/125-ms controller without further
ranking or refitting. Compare exactly sham, one-time phase initialization, and
the selected controller at the same 0.2-V/m axial amplitude, EEG-selected
9/11-Hz carrier, pi-relative phase target, one-second initialization, 250-ms
correction horizon, and continuous bounded-frequency waveform. Prospectively
screen stimulation-free EEG before active outcomes and enroll twelve new
independent structures with complete 9/11-Hz by low/high-diffusion coverage;
use six paired future continuations as repeats. The primary structure-level
contrast is selected refreshed control versus one-time initialization. Test
selected versus sham only in a fixed sequence after the primary passes. Keep
phase-error reduction, future-wise reliability, rate safety, causality,
continuity, and washout as mandatory confirmation gates. Save manuscript PSD,
context, structure, temporal, phase, safety, inference, power, provenance, and
PNG/PDF figure artifacts. H4-C can establish only deterministic adaptive
feedback in ideal neural EEG under toy phase diffusion; it is not learned
prediction, a contextual bandit, a clinical result, or proof that adaptation
is necessary against all possible open-loop controllers.

For D0b, treat the failed D0 global-resultant endpoint as discovery only. Load
and hash-lock the D0 conclusion, generator, and EEG table; freeze the one-step
demodulated EEG phase-increment coherence and its D0-derived threshold before
new simulations. Keep D={0.5, 2.0} rad^2/s, modulation depth 0.04, and the
9/11-Hz grid unchanged. Select the carrier from ideal pre-action EEG rather
than the hidden generator label. Use six new circuit structures as independent
units and two afferent histories as repeats. Require a practical paired effect,
exact structure-level sign-flip evidence, cross-structure consistency, frozen-
threshold discriminability, temporal signal-to-noise, frequency observability,
and rate safety before D1. D0b applies no tACS and cannot establish control.

For D1, hash-lock the positive D0b observability result and positive F0
frequency/phase map. Keep D={0.5,2.0} rad^2/s, modulation depth 0.04, and the
9/11-Hz generator grid unchanged. Estimate the slow EEG context C1 from the
complete 12-s stimulation-free baseline, but initialize action phase only from
the most recent causal 1-s EEG. The deterministic controller selects the
9/11-Hz carrier from EEG and uses the frozen pi-relative axial montage; the
full-information action grid is exactly sham/0.2/0.4 V/m, with one constant
action per intervention. Split both private Poisson events and the shared
phase-diffusion path at the decision boundary, preserve an identical history
across actions and futures, and estimate expected response from multiple
independent futures. The only learnable policy feature is predecision EEG C1;
frequency and recent phase are signal-processing inputs, while hidden D,
generator frequency, spikes, and rates remain audits. Require practical
optimal-action reversals, a diffusion-by-dose interaction, cross-structure
LOSO advantage, and failure under structure-preserving C1 shuffling before a
disjoint policy confirmation. D1 is system identification, not a bandit.

For single-action contextual-dose mapping, observe only stimulation-free EEG,
then apply one discrete amplitude for the complete intervention; do not use an
active probe or switch doses after onset. Keep sham as a causal comparator.
Split background-event randomness at the decision boundary so every action
shares an identical history while multiple independent future continuations
estimate conditional expected response. Never define a learnable oracle from
one realized future. Latent afferent modulation depth creates explicit toy
state heterogeneity but cannot enter the EEG policy.

For monotone severity-threshold discovery, retain a 12-s stimulation-free
baseline, the frozen {0.2, 0.4}-V/m active set, at least three crossed
structures, two histories, and three independent postdecision futures. The
only primary policy variable is log10 alpha excess over the frozen B mean.
Select a threshold using discovery outcomes, evaluate it by leaving out whole
structures, and require replicated low-dose opportunities across structures
and histories. A discovery pass only freezes a candidate for a new disjoint
confirmation; it is not a contextual-bandit result.

For frozen severity-rule confirmation, load the CDM2-D threshold and source
files by hash and never refit the threshold, feature, target, action set, or
success criteria. Circuit structure is the independent statistical unit;
histories and postdecision futures are repeats. Keep the single primary
contrast as frozen EEG-threshold policy versus fixed 0.4 V/m. Confirmation
requires the prespecified practical effect, one-sided exact structure-level
sign-flip evidence, and cross-structure consistency. Fixed 0.2 V/m, sham,
oracle, t-test, Wilcoxon, and bootstrap results are secondary audits and cannot
replace a failed primary criterion.

For F0 frequency/phase mapping, keep the elevated-alpha generator and tACS
actuator causally distinct. A has mean-rate-matched afferent modulation at 9
or 11 Hz with a randomized continuous phase; B has homogeneous afferents.
Screen only prestimulation ideal EEG against a B population target. Cross the
fixed 0.4-V/m field with {9, 11} Hz and {0, pi} EEG-relative phase, plus sham,
and apply exactly one constant action per replay. Pair every action through an
identical predecision trajectory and estimate conditional response with
independent futures. Generator frequency and spike PPC are post hoc audits,
not policy inputs. Require frequency and phase crossover interactions, an
EEG-rule advantage over the best fixed arm, and failure under context shuffling
before advancing. This three-structure experiment is directional system
identification, not a bandit or confirmatory evidence.

For S1-C stationary H1--H3 confirmation, hash-lock all declared F0 source
files and use seed namespaces disjoint from F0. H1 is an unconditional paired
A/B phenotype analysis across all candidate structures; H2/H3 are conditional
on prospectively screen-positive structures, and eligibility may use only the
12-s stimulation-free ideal EEG. Retain the frozen 0.4-V/m, {9,11}-Hz by
{0,pi}-relative-phase grid and use four independent postdecision futures.
Structure is the inferential unit; carrier and futures are repeats. The H3
rule selects the EEG-detected frequency at pi relative phase and must be
compared with sham, the F0-frozen best fixed action, and the uniform expected
outcome over all four active actions. Do not fit or call this a contextual
bandit. Keep the exact sign-flip primary tests, prespecified practical
thresholds, FDR-controlled frequency/phase audits, structure-preserving
context shuffle, 4-s/0.25-Hz paper PSDs, rate safety, orientation audit, and
field-removal audit. A failed fixed-sequence hypothesis is a valid outcome.

For the exploratory 10-Hz alpha dose audit, retain the frozen A/B generator,
frequency, axial montage, and EEG-relative 180-degree phase. Sham plus the
predeclared 0.2, 0.4, 0.6, and 0.8-V/m grid is a directional mechanism screen,
not confirmation. Evaluate washout independently of acute-effect direction,
and preserve the exact Fourier-vector cross/induced decomposition. Any ranked
dose must be frozen and tested on disjoint seeds before it becomes an RL
action. Do not exceed 0.8 V/m to rescue a negative screen.

For screened alpha confirmation, freeze the EEG eligibility threshold and the
0.4-V/m protocol before examining candidate seeds. Screen only an unstimulated
A episode using ideal EEG alpha, pre-action phase quality, and rate safety;
never use an active outcome, hidden PPC, or the seed-specific B counterfactual
for eligibility. Save every rejection and report screening yield. Generate B
and active controls only after enrollment, interpret inference as conditional
on screen eligibility, and do not equate toy-seed screening yield with human
biomarker prevalence.

For CL0 contextual-dose feasibility, keep structure, afferent-drive, and
absolute-phase seed namespaces explicit. Hold absolute phase fixed in the
crossed variance audit, screen before active outcomes, and use only the frozen
B population target rather than a seed-specific B counterfactual. The action
set is sham/0.2/0.4 V/m with frequency, montage, and EEG-relative phase fixed.
Evaluate the exploratory EEG-only rule by leaving out complete structure seeds;
never split drive sessions from one structure across training and evaluation.
An oracle opportunity plus cross-fitted improvement over fixed 0.4 V/m is only
a directional gate. Freeze any resulting policy and confirm it on new
structure seeds before calling it a contextual bandit result.

For CL1-P, calibrate the short-probe B target only on disjoint population
reference seeds. Every eligible active replay must share the exact baseline
and 0.2-V/m probe history and may diverge only at the decision boundary, where
the escalation arm transitions smoothly to 0.4 V/m. The frozen rule may use
only the duration-matched ideal-EEG probe response; never expose the paired
counterfactual, spikes, rates, structure seed, drive seed, or a seed-specific B
simulation. Require practical optimal-action reversals across multiple
structures, improvement over fixed escalation, and failure after context
shuffling before implementing a contextual bandit. A failed CL1-P gate means
the tested 10-Hz amplitude task still does not justify contextual learning.

For CL1-C, keep the CL1-P timing, 0.2-V/m common probe, 0.2/0.4-V/m action
set, and zero trajectory threshold frozen. The primary policy observes only
matched baseline-minus-probe ideal-EEG alpha power and must use seed namespaces
disjoint from CL1-P. Require improvement over both fixed doses at the
structure-seed level. Treat the baseline-only rule and paired sham-trajectory
rule as attribution audits, and do not call CL1-C a contextual bandit.

For the alpha-suppression toy experiment, A is an elevated-alpha state created
only by mean-rate-matched 10-Hz modulation of independent Poisson afferents and
B is the homogeneous-Poisson low-alpha reference. These are not depression and
healthy labels. Calibrate the A modulation depth without stimulation, discover
an EEG-relative field-phase offset on disjoint seeds, then freeze it before
validation. Estimate phase causally from the preceding EEG; randomize the
hidden afferent phase by seed. Treat multi-second 8--12-Hz power as primary,
and spike PPC/rates as hidden mechanism/safety checks. Sham, opposite-phase,
and transverse-field arms are required. Report ideal neural-only EEG control
separately from the complex observation-only sinusoid audit; the latter cannot
validate simultaneous tACS-EEG without an explicit artifact model.

For alpha-suppression confirmation, treat the entire pilot—including its four
validation seeds—as exploratory protocol generation. Load and hash the pilot's
frozen target/protocol JSON files; never recalibrate modulation depth or select
phase, frequency, amplitude, or montage in confirmation. Use only new circuit
seeds and exact paired sign-flip inference. Audit baseline phase estimability
by extrapolating independent first-half and second-half estimates to the same
action boundary. A positive fixed-action result does not establish a
contextual policy: require a held-out context-by-action interaction before RL.
