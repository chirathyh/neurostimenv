# BallAndStick scientific experiments

## Bottom line

This circuit is appropriate for testing software causality and a narrowly
defined mechanistic hypothesis: whether scaling I-to-E conductance changes a
small E/I network's idealized EEG, and whether a weak uniform electric field
can reverse that change. It is not yet appropriate for claims about depression,
clinical EEG biomarkers, or treatment efficacy.

The legacy-compatible scientific online action remains:

```python
action = [field_amplitude_v_per_m, frequency_hz]
result = environment.step_online(action, duration_ms=1000.0)
eeg_window_v = result["eeg_v"]
rates = result["firing_rates"]       # validation only, not EEG-only RL state
```

Open-loop system-identification experiments use the richer mapping API:

```python
action = {
    "montage": "axial",
    "dc_offset_v_per_m": -0.5,       # signed local tissue field
    "ac_amplitude_v_per_m": 0.3,     # non-negative sine amplitude
    "frequency_hz": 10.0,
    "phase_rad": 0.0,                # phase at this action onset
}
```

Named montages currently map to idealized local field directions. They are not
scalp electrode montages until replaced by head-model-derived vectors.

Each call advances the same NEURON state through `(t_start, t_stop]`. No future
action waveform is created and no state is reinitialized.

## LFPy: what changed and what remains

The project still uses LFPy 2.3. It supplies:

- `Network`/the online `NetworkEnv` subclass and `NetworkCell`;
- recurrent and external `Synapse` objects;
- cell and network geometry;
- `RecExtElectrode` and `CurrentDipoleMoment` transformations; and
- the segment-to-probe forward operators used to compute dipole moments.

The online path bypasses only LFPy's monolithic `Network.simulate()` control
flow. It precomputes the same probe transforms once, calls NEURON `fadvance()`
(or MPI `psolve()`) at each fixed step, reads `i_membrane_`, and returns one
causal window at a time.

Two legacy issues were made explicit:

1. In the pinned LFPy implementation, `Network.simulate()` retained the
   configured `Network.celsius=36.5` but did not set global `h.celsius`;
   effective legacy kinetics remained at NEURON's 6.3 °C default. The canonical
   `hh` equations use a Q10 referenced to 6.3 °C. The configuration now says
   6.3 °C and online initialization explicitly sets it. This preserves the
   effective legacy model without pretending it is mammalian-temperature
   physiology.
2. NEURON 8.2.3 record/play vectors attached after `finitialize()` did not
   advance until `frecord_init()`. Reinitializing recorders at every action
   boundary would disturb persistent state, so the online loop records membrane
   currents and assigns extracellular voltage explicitly at each fixed step.

The no-stimulation current setup was rechecked over 500 ms: after dropping the
legacy-only `t=0` sample, online and legacy EEG correlation was effectively
1.0 and relative RMS error was `2.55e-11`.

References:

- [LFPy Network documentation](https://lfpy.readthedocs.io/en/latest/_modules/LFPy/network.html)
- [NEURON `celsius` and `dt` documentation](https://www.neuron.yale.edu/neuron/static/docs/help/neuron/neuron/nrnoc.html)
- [NEURON canonical HH/Q10 example](https://neuron.yale.edu/neuron/docs/hodgkin-huxley-using-rxd)

## Parameter audit

| Parameter | Current value | Assessment |
|---|---:|---|
| Population | 32 E, 8 I | Preserves an 80/20 split, but 40 cells is a toy finite-size network. |
| Morphology | 30×30 µm soma, 1000×3 µm apical cable | Polarizable and dipole-generating, but not a reconstructed cortical cell. |
| Intrinsic channels | canonical `hh` soma; passive apical cable | Numerically transparent; not mammalian E/I cell-type physiology. |
| Temperature | 6.3 °C effective/configured | Correct for canonical HH reference kinetics; not physiological body temperature. |
| Fixed step | 0.0625 ms | Resolves the fast synapses reasonably for this screen. A 0.03125-ms sensitivity run changed 1-s rates by roughly 4–5%; repeat claimed effects at half-step. |
| E/I connection probability | EE/EI 0.10; IE/II 0.50 | Dense I probability avoids uninhibited E cells in an 8-I-cell network. |
| Multapses | E mean 2; I mean 1 | Expected I contact count is approximately preserved from the old `p=0.1`, mean-5 setup while coverage is less heterogeneous. |
| Synaptic kinetics | AMPA-like 0.2/1.8 ms, 0 mV; GABA-A-like 0.1/9 ms, −80 mV | Qualitatively plausible fast excitation/inhibition; not fitted to a cortical area or cell class. |
| Weights | EE/EI 0.001 µS; IE/II 0.010 µS; CV 0.10 | Produces E/I balance in this toy; not experimentally fitted. |
| Delays | 1.5±0.3 ms, minimum 0.3 ms | Plausible local-circuit scale. |
| Condition B | multiply IE weight by 0.5 | Clear and reproducible large perturbation; not a disease-specific estimate. |
| Background drive | 64 AMPA-like synapses/cell; E interval 40 ms, I 30 ms | Calibrated from the old ~41/41 Hz regime to about 3.3/8.0 Hz in the seed-1 first second. The stronger I drive compensates for identical intrinsic E/I models. |
| Spatial extent | 100-µm radius; aligned apical axes | Suitable for a coherent toy dipole; overstates morphological alignment. |
| EEG head model | four spheres; source at z=78 mm; sensor at z=90 mm | Standard idealized volume conductor, one channel only; no montage variability or artifacts. |
| Stimulation | uniform +z sinusoid, 0.2–2 V/m, 5–40 Hz grid | Field-at-tissue parameterization is defensible. The 2-V/m level is exploratory, not a conventional-dose claim. |
| Control window | 1 s | Suitable for action timing; spectral state should aggregate multiple windows. |
| Analysis epoch | 2-s burn-in + 8-s A/B analysis | Gives 0.5-Hz Welch bins with several segments; still short for stable delta estimates. |

The old 64×100-Hz external drive delivered 6,400 arrivals/s/cell and produced
about 41 Hz in both populations. The new E/I intervals produce approximately
1,600/2,133 arrivals/s/cell and a lower-rate operating point. This calibration
is a model choice, not an empirical fit.

Conventional 2-mA human transcranial stimulation has been measured at up to
about 0.8 V/m in cortex. Therefore the experiment varies field at the tissue
rather than treating a current injected by a point source 10 µm from a cell as
equivalent to scalp mA:

- [Huang et al., eLife 2017](https://elifesciences.org/articles/18834)
- [Radman et al., Brain Stimulation 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2797131/)

## Experiment 1: does reduced I-to-E inhibition change the circuit?

### Confirmatory question

For matched circuit seed `k`, compare:

```text
A_k: inhibition_scale = 1.0
B_k: inhibition_scale = 0.5
```

Connectivity, background event trains, weights' standardized random draws, and
initial conditions are matched. Stimulation is off. The circuit is the
statistical unit.

Primary outputs:

- predefined EEG features after burn-in;
- paired B−A effect, bootstrap confidence interval, paired sign-flip p-value,
  Cohen's dz, and FDR;
- E/I rates as mechanistic checks; and
- leave-one-circuit-pair-out EEG-only A/B classification with within-pair label
  permutation.

The classifier addresses whether an EEG state contains out-of-sample condition
information. It does not establish an RL reward or treatment target.

### Full run

```bash
source /home/chirath/Documents/depression-simulator/bin/activate

python experiments/ballnstick_analysis/run_ballnstick.py \
  experiment.name=ballnstick_ab_full \
  env=ballnstick \
  analysis=analysis \
  env.simulation.obs_win_len=1000 \
  analysis.n_circuits=20 \
  analysis.n_steps=10 \
  analysis.burn_in_steps=2 \
  experiment.plot=true
```

### Smoke run

```bash
python experiments/ballnstick_analysis/run_ballnstick.py \
  experiment.name=ballnstick_ab_smoke \
  env=ballnstick \
  analysis=analysis \
  env.simulation.obs_win_len=500 \
  analysis.n_circuits=3 \
  analysis.n_steps=3 \
  analysis.burn_in_steps=1 \
  analysis.n_bootstrap=100 \
  analysis.n_permutations=100 \
  analysis.classifier_permutations=20 \
  experiment.plot=false
```

Outputs are under
`../../results/<name>/ab_eeg_analysis/analysis/`, including
`statistical_comparison.csv`, `condition_discriminability.json`, per-circuit
features, reference distributions, PSD summaries, and plots.

Do not formulate the RL task from this comparison unless the full run shows:

- a reproducible mechanistically sensible A/B shift;
- uncertainty narrow enough to resolve it;
- held-out EEG information above chance; and
- robustness to `env.network.dt=0.03125` and modest background-drive changes.

## Experiment 2: can stimulation move B toward A?

### Design

The script performs two stages with disjoint seeds:

1. Discovery: evaluate a predeclared amplitude×frequency grid in fresh
   Condition-B episodes. Each episode contains sham burn-in followed by one
   causal fixed protocol.
2. Validation: freeze the discovery-ranked protocols and evaluate them on
   untouched seeds with matched A and B-sham counterfactuals.

For a compact predeclared EEG feature vector, define:

```text
target shift       = z(A) - z(B_sham)
stimulation shift  = z(B_stim) - z(B_sham)
distance gain      = 1 - d(B_stim, A) / d(B_sham, A)
alignment          = cosine(target shift, stimulation shift)
```

Scaling is fit only from discovery A/B-sham references. The primary protocol
passes reachability only if its held-out bootstrap CI for mean distance gain is
above zero, median alignment is positive, and every validation episode passes
E/I rate guardrails.

### Full run

```bash
python experiments/ballnstick_analysis/run_ballnstick_stimulation_sweep.py \
  experiment.name=ballnstick_stim_reachability_full \
  env=ballnstick \
  analysis=ballnstick_stimulation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true
```

The default full grid is amplitudes
`[0.2, 0.5, 0.8, 1.2, 2.0] V/m` and frequencies
`[5, 10, 20, 40] Hz`, with four discovery and eight validation seeds.

### Smoke run

```bash
python experiments/ballnstick_analysis/run_ballnstick_stimulation_sweep.py \
  experiment.name=ballnstick_stim_reachability_smoke \
  env=ballnstick \
  analysis=ballnstick_stimulation \
  env.simulation.obs_win_len=500 \
  analysis.burn_in_steps=1 \
  analysis.stimulation_steps=2 \
  analysis.discovery.n_seeds=1 \
  analysis.validation.n_seeds=1 \
  'analysis.discovery.amplitudes_v_per_m=[0.8]' \
  'analysis.discovery.frequencies_hz=[10.0]' \
  analysis.discovery.top_k=1 \
  analysis.validation.n_bootstrap=100 \
  experiment.plot=false
```

The smoke completed and correctly returned `NOT PASSED` for its single
0.8-V/m, 10-Hz validation episode (distance gain −0.422, alignment −0.769).
That verifies negative-result handling; it is not a powered scientific result.

Outputs are under
`../../results/<name>/stimulation_reachability/`, including discovery and
validation seed-level tables, protocol summaries, a response-surface plot, and
`reachability_conclusion.json`.

## Experiment 2b: fixed-protocol mechanism and confounding confirmation

The original sweep nominated 0.5 V/m at 10 Hz as a secondary lead.  Do not
search the grid again on the same outcome.  This follow-up freezes that action
and asks whether it produces a circuit-mediated response rather than merely
adding rewarded spectral power.

The design uses eight calibration seeds and 24 disjoint confirmation seeds.
Every confirmation seed has matched A-sham, B-sham, B-parallel-field, and
B-perpendicular-field episodes.  An observation-only control adds a pure
10-Hz sinusoid to B-sham EEG until its alpha-band power matches the active
parallel episode; it never changes the neural circuit.

Each full episode contains:

```text
4 s burn-in -> 8 s baseline -> 8 s stimulation -> 8 s post-stimulation
```

A single 250-ms raised-cosine onset/offset envelope spans the complete
stimulation block.  It is not restarted at one-second window boundaries.

The primary feature distance uses total and relative-gamma power after zeroing
PSD bins within 1 Hz of the 10-Hz fundamental.  The raw endpoint is retained as
a secondary confounding diagnostic.  The runner also reports spike phase
locking, relative and absolute rate safety, matched-baseline identity, and the
post-stimulation response.

Directional modulation and A-like reachability are separate decisions:

- mechanistic modulation requires the lower 95% confidence bound to exceed a
  prespecified 10% improvement, at least 80% positive seeds, positive
  alignment, safe rates, a response beyond both controls, a population PLV
  increase, and exact matched baselines;
- A-like reachability additionally requires at least 80% of seeds to lie
  within 0.5 calibration standard deviations of A for every primary feature.

### Full confirmatory run

```bash
python experiments/ballnstick_analysis/run_ballnstick_stimulation_mechanism.py \
  experiment.name=ballnstick_stimulation_mechanism_confirmatory \
  env=ballnstick \
  analysis=ballnstick_stimulation_mechanism \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true
```

This is a long run: on the current single-process setup, budget approximately
six to eight hours.  Do not combine its 24 confirmation seeds with the prior
eight validation seeds because the endpoint and protocol are different.

### Smoke run

```bash
python experiments/ballnstick_analysis/run_ballnstick_stimulation_mechanism.py \
  experiment.name=ballnstick_stimulation_mechanism_smoke \
  env=ballnstick \
  analysis=ballnstick_stimulation_mechanism \
  env.simulation.obs_win_len=500 \
  analysis.timeline.burn_in_steps=1 \
  analysis.timeline.baseline_steps=1 \
  analysis.timeline.stimulation_steps=1 \
  analysis.timeline.post_steps=1 \
  analysis.calibration.n_seeds=1 \
  analysis.confirmation.n_seeds=1 \
  analysis.n_bootstrap=100 \
  analysis.n_permutations=100 \
  analysis.save_raw_eeg=false \
  analysis.save_spikes=false \
  experiment.plot=false
```

Outputs are under `../../results/<name>/stimulation_mechanism/`.  The complete
configuration is also preserved by Hydra.  Important files include:

- `calibration_epoch_features.csv` and `confirmation_epoch_features.csv`;
- `confirmation_window_features.csv`;
- `confirmation_reachability_by_seed.csv` and its summary;
- `confirmation_paired_contrasts.csv`;
- `baseline_causality_checks.csv`;
- per-seed raw EEG, PSD, and spike archives; and
- `experiment_conclusion.json`.

The model has static synapses.  A post-stimulation effect is reported as a
negative-control/aftereffect endpoint but is not required for acute online
compensation and should not be interpreted as plasticity.

## Implication for RL action and state spaces

Start with no RL until Experiment 2 passes on held-out seeds.

If it passes:

- Begin offline with a discrete action set: sham plus only held-out validated
  protocols. This makes coverage and counterfactual support explicit.
- Use a continuous `[amplitude, frequency]` action only if the discovery
  response surface is smooth across amplitude, stable across seeds, and the
  optimum is not a boundary artifact. Normalize amplitude and frequency to
  `[-1, 1]` for the agent while retaining physical units in logs.
- Constrain field amplitude to the validated range. Do not map V/m to scalp mA
  without a subject/montage-specific current-flow model.
- Build state from rolling, training-normalized EEG features (log power and
  relative band powers or ratios, entropy, and previous action). The current
  single-window `OnlineNeuronEnv["observation"]` is a compatibility feature
  vector and should be replaced before production RL.
- Define reward from held-out distance to the A reference with rate/saturation
  penalties. Do not reward one hand-selected band alone.

If the response is noisy or multimodal, keep the action discrete. If no
protocol passes, the current actuator/model pair is not a supported RL task;
change the model or stimulation mechanism before trying a stronger learner.

## Experiment 3: causal controllability ladder

The fixed 0.5-V/m, 10-Hz confirmation showed that its raw EEG improvement was
reproduced by an observation-only 10-Hz sinusoid and disappeared when the
fundamental was excluded. Before screening another electric-field grid, this
experiment separates endpoint validity from actuator controllability.

The first stage varies the actual I-to-E conductance scale through
`[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`. On held-out matched seeds, EEG distance to A
should rank-monotonically decrease as the causal parameter approaches A. If it
does not, revise the EEG state before any further control or RL study.

The second stage keeps Condition B at `inhibition_scale=0.5` and multiplies
only the excitatory background-synapse weight onto I cells. The Poisson event
trains and recurrent realization stay matched within each seed. Discovery
seeds rank the configured multipliers; only the frozen top protocols run on
disjoint validation seeds. This is a population-selective mechanistic positive
control, not transcranial stimulation or a clinically interpretable dose.

### Full run

```bash
python experiments/ballnstick_analysis/run_ballnstick_controllability_ladder.py \
  experiment.name=ballnstick_controllability_ladder_full \
  env=ballnstick \
  analysis=ballnstick_controllability_ladder \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true
```

The default design uses eight discovery and 24 disjoint validation seeds, a
4-s burn-in, and an 8-s analysis epoch. Budget approximately six to ten hours
on the current single-process setup. Results are written under
`../../results/<name>/controllability_ladder/`.

### Smoke run

```bash
python experiments/ballnstick_analysis/run_ballnstick_controllability_ladder.py \
  experiment.name=ballnstick_controllability_ladder_smoke \
  env=ballnstick \
  analysis=ballnstick_controllability_ladder \
  env.simulation.obs_win_len=500 \
  analysis.timeline.burn_in_steps=1 \
  analysis.timeline.analysis_steps=2 \
  analysis.discovery.n_seeds=1 \
  analysis.validation.n_seeds=2 \
  'analysis.causal_interpolation.inhibition_scales=[0.5,0.75,0.9,1.0]' \
  'analysis.selective_i_drive.background_weight_multipliers=[1.0,1.10]' \
  analysis.selective_i_drive.top_k=1 \
  analysis.n_bootstrap=100 \
  analysis.n_permutations=100 \
  analysis.save_raw_eeg=false \
  experiment.plot=false
```

The smoke only checks execution and output integrity; its one discovery and
two validation seeds cannot support a scientific conclusion. The conclusion
file reports separate gates for causal-metric validity, directional selective-I
control, and entry into the A-equivalence region. A selective-I success only
justifies the next uniform-field transfer-function experiment; it is not
evidence that TES can restore I-to-E inhibition.

## Regression and numerical checks

```bash
python -m unittest -v tests/test_online_stimulation.py

python experiments/ballnstick_analysis/validate_online_legacy.py \
  experiment.name=ballnstick_online_legacy_smoke \
  env=ballnstick \
  env.simulation.duration=500 \
  env.simulation.obs_win_len=250 \
  experiment.plot=false \
  experiment.tqdm=false
```

For any positive scientific result, rerun the selected comparisons with:

```bash
env.network.dt=0.03125
```

and report whether the sign, effect size, selected protocol, and conclusion
remain stable.

## Experiment 4: signed-field controllability identification

This stage checks the actuator before any further RL work. It preserves the
fundamental-excluded two-feature endpoint, evaluates signed DC, phase-controlled
AC, mixed DC+AC, and field direction, and ranks AC actions only after subtracting
an observation-only matched-sinusoid control. All primary actions obey
`abs(DC) + AC <= 0.8 V/m`.

First validate the isolated cell:

```bash
python experiments/ballnstick_analysis/validate_ballnstick_polarization.py \
  experiment.name=ballnstick_cellular_polarization \
  env=ballnstick \
  analysis=ballnstick_polarization \
  experiment.plot=false \
  experiment.tqdm=false
```

Then run the approximately six-minute, one-seed broad screen:

```bash
python experiments/ballnstick_analysis/run_ballnstick_field_controllability.py \
  experiment.name=ballnstick_field_controllability_quick \
  env=ballnstick \
  analysis=ballnstick_field_controllability \
  env.simulation.obs_win_len=1000 \
  experiment.plot=false \
  experiment.tqdm=false
```

If that screen has a signal, freeze its leading AC/DC candidates and run the
two-seed targeted replication:

```bash
python experiments/ballnstick_analysis/run_ballnstick_field_controllability.py \
  experiment.name=ballnstick_field_controllability_targeted_quick \
  env=ballnstick \
  analysis=ballnstick_field_controllability_targeted \
  env.simulation.obs_win_len=1000 \
  experiment.plot=false \
  experiment.tqdm=false
```

Only if both quick stages retain a consistent, aligned, beyond-synthetic and
rate-safe response should the disjoint four-seed discovery/eight-seed
validation run be started:

```bash
python experiments/ballnstick_analysis/run_ballnstick_field_controllability.py \
  experiment.name=ballnstick_field_controllability_full \
  env=ballnstick \
  analysis=ballnstick_field_controllability_full \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true
```

Outputs are under `../../results/<name>/field_controllability/`. The SVD
projection in `controllable_subspace.json` describes directional span only;
the held-out per-action reachability criteria remain primary.

## Experiment 5: T1 reversible tES entrainment

T1 asks a different question from reduced-I-to-E reachability: can a realistic
weak uniform field acutely and reversibly entrain spike timing in the unchanged
40-cell network? It does not call the entrained state healthy, depressed, or
treated.

Every active run is one persistent A-B-A episode:

```text
burn-in -> sham baseline A -> tACS state B -> stimulation-free washout A'
```

A same-seed sham trajectory controls for time drift. Discovery maps a
predeclared amplitude/frequency grid at no more than `0.8 V/m`; the highest
mean excitatory-population PPC gain that meets the discovery rate criterion is
frozen before disjoint validation. Validation repeats that action against a
transverse-field negative control, neighbouring frequencies, and the discovery
doses at the frozen frequency. Dose-response smoothness is therefore checked
again on held-out circuits.

The primary endpoint is the active-minus-sham difference-in-differences in
excitatory-population pairwise phase consistency (PPC), an unbiased
transformation of PLV with respect to spike count. Raw EEG power at the driven
fundamental is secondary. A matched observation-only sinusoid is saved so a
spectral response cannot be mistaken for spike entrainment.

### Reduced software smoke

This checks the complete online lifecycle and output schema. One discovery and
one validation seed, one dose, and one frequency cannot pass the scientific
criteria by design.

```bash
python experiments/ballnstick_analysis/run_ballnstick_tes_entrainment.py \
  experiment.name=ballnstick_tes_entrainment_smoke \
  env=ballnstick \
  analysis=ballnstick_tes_entrainment \
  env.simulation.obs_win_len=500 \
  analysis.timeline.burn_in_steps=1 \
  analysis.timeline.baseline_steps=2 \
  analysis.timeline.stimulation_steps=2 \
  analysis.timeline.washout_steps=2 \
  analysis.timeline.block_ramp_ms=100 \
  analysis.discovery.n_seeds=1 \
  analysis.validation.n_seeds=1 \
  'analysis.discovery.amplitudes_v_per_m=[0.8]' \
  'analysis.discovery.frequencies_hz=[10.0]' \
  analysis.validation.include_frequency_neighbors=false \
  analysis.validation.include_dose_controls=false \
  analysis.phase_null.n_surrogates=100 \
  analysis.n_bootstrap=100 \
  analysis.n_permutations=100 \
  analysis.save_raw_eeg=false \
  analysis.save_spikes=false \
  experiment.plot=false \
  experiment.tqdm=false
```

### Quick amplitude-frequency exploration

Use this before the full design. It maps three doses and four frequencies on
two discovery seeds, then performs a four-seed targeted replication. Treat it
as exploratory because four validation circuits cannot satisfy the configured
minimum of eight.

```bash
python experiments/ballnstick_analysis/run_ballnstick_tes_entrainment.py \
  experiment.name=ballnstick_tes_entrainment_quick \
  env=ballnstick \
  analysis=ballnstick_tes_entrainment \
  env.simulation.obs_win_len=1000 \
  analysis.timeline.burn_in_steps=2 \
  analysis.timeline.baseline_steps=3 \
  analysis.timeline.stimulation_steps=4 \
  analysis.timeline.washout_steps=3 \
  analysis.discovery.n_seeds=2 \
  analysis.validation.n_seeds=4 \
  'analysis.discovery.amplitudes_v_per_m=[0.2,0.5,0.8]' \
  'analysis.discovery.frequencies_hz=[10.0,20.0,40.0,60.0]' \
  analysis.phase_null.n_surrogates=500 \
  analysis.n_bootstrap=1000 \
  analysis.n_permutations=2000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Proceed to the full experiment only if the quick run shows a positive
selected-action PPC gain, axial specificity, at least one reproducible
frequency comparison, and rate safety. The quick run is a stop/go screen, not
confirmation.

### Full discovery and held-out validation

```bash
python experiments/ballnstick_analysis/run_ballnstick_tes_entrainment.py \
  experiment.name=ballnstick_tes_entrainment_full \
  env=ballnstick \
  analysis=ballnstick_tes_entrainment \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

The defaults use four discovery and 12 validation seeds, four field strengths,
five frequencies, a 2-s burn-in, 4-s baseline, 6-s stimulation block, and 4-s
washout. Results are written to
`../../results/<name>/tes_entrainment/`. Important files are:

- `discovery_action_summary.csv`: dose-frequency mapping;
- `selected_protocol.json`: protocol frozen before validation;
- `validation_action_seed_metrics.csv`: circuit-level statistical units;
- `validation_control_comparisons.csv`: axial-versus-control contrasts; and
- `experiment_conclusion.json`: prespecified mechanistic checks.

An observed PLV without a positive held-out PPC difference-in-differences is
not sufficient. A passing T1 result establishes acute generic entrainment only;
it does not establish a lasting after-effect or rescue of the earlier
reduced-inhibition condition.

## Experiment 6: asynchronous-to-entrained state reachability

This minimal follow-up makes the target state and the tACS actuator physically
distinct. It does not reuse reduced I-to-E inhibition:

```text
A: homogeneous independent Poisson afferent event times
B: independent Poisson afferents with a weak sinusoidally modulated rate
A+tACS: A's homogeneous afferents plus an axial uniform AC field
```

All three retain `inhibition_scale=1.0`, identical cells, recurrent wiring and
weights, background synaptic weights, and mean afferent event rates. The B
reference is generated by stochastic synaptic events and never receives tACS.
The A+tACS circuit retains modulation depth zero, so the field cannot silently
change the setting used to construct B. Spike times remain outputs of the HH
cells; the code does not prescribe output spikes.

Two disjoint calibration seeds select the smallest rate-matched afferent
modulation depth closest to the predeclared E-PPC target of 0.02. The tACS
action is not tuned: it is fixed from T1 at 0.8 V/m and 60 Hz. Four held-out
matched circuit seeds then compare A, B, A+axial tACS, and an A+transverse-field
orientation control. This is a low-cost mechanistic pilot; a positive result
must later be confirmed with at least eight new seeds.

Run the pilot from the repository root:

```bash
python experiments/ballnstick_analysis/run_ballnstick_entrainment_state.py \
  experiment.name=ballnstick_entrainment_state_pilot \
  env=ballnstick \
  analysis=ballnstick_entrainment_state \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/ballnstick_entrainment_state_pilot/entrainment_state/`.
`selected_reference.json` explicitly records which settings differ between A
and B and the separate tACS action. `validation_seed_metrics.csv` contains the
circuit-level target-distance results, and `experiment_conclusion.json`
contains the predefined pilot checks.

The primary endpoint is reduction of absolute E-PPC distance from A toward B,
not equality of synaptic mechanisms. Passing establishes acute functional
state reachability only. It does not establish structural conversion, a
preferred biological state, a persistent after-effect, or an EEG-observable RL
state. EEG observability must be tested separately before RL.

## Experiment 7: EEG-primary A-to-B-like tACS reachability

This experiment performs that missing EEG-observability test without changing
the circuit definitions after seeing a stimulation result. A is homogeneous
independent Poisson drive. B has the already fixed 0.04-depth, 60-Hz modulation
of independent afferent event probability, with the same expected mean rate.
Both conditions have identical cells, recurrence, synaptic weights and
`inhibition_scale=1.0`. B never receives tACS. Every stimulated arm retains A's
homogeneous afferent process.

The design follows two observations from the supplied tACS papers. Weak-field
tACS may alter spike timing more consistently than mean firing rate, so spike
PPC is a hidden mechanistic endpoint and rates are guardrails. Concurrent EEG
at the stimulation frequency is also vulnerable to direct periodic signal
contamination. Consequently the experiment gives two separate conclusions:

1. ideal forward-model EEG reachability, which is the toy-environment primary
   question; and
2. robustness after excluding the 60-Hz bins and against a sine added only to
   A's recorded observation.

Four discovery seeds see unstimulated A and B only. Their predeclared EEG
features define a standardized A-to-B centroid axis. The mapping is frozen
before six disjoint validation seeds see A, B, axial tACS, and controls. The
primary validation action is fixed at 0.8 V/m, 60 Hz and -90 degrees. That
quadrature phase is a coarse correction specified from the preceding pilot,
not selected on these validation data. A 0.5-V/m axial action gives one dose
check. A transverse 0.8-V/m field is a mechanistic orientation control and an
observation-only matched sine is a contamination control; neither belongs to
the eventual agent action set.

The primary seed-level endpoint is reduction in absolute distance from A to
the discovery-frozen B EEG centroid. The 1-s window accuracy is an
observability diagnostic, not an independent-replicate statistical test.
Firing rates and spike timing are never inputs to the EEG state or reward.

Run the compact held-out pilot from the repository root:

```bash
python experiments/ballnstick_analysis/run_ballnstick_eeg_reachability.py \
  experiment.name=ballnstick_eeg_reachability_pilot \
  env=ballnstick \
  analysis=ballnstick_eeg_reachability \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are saved under
`../../results/ballnstick_eeg_reachability_pilot/eeg_reachability/`.
`frozen_eeg_state_mapping.json` records the discovery-only mapping;
`validation_eeg_reachability.csv` contains seed-level endpoints;
`validation_window_metrics.csv` contains the online-window observations;
`validation_hidden_mechanism.csv` contains non-agent mechanistic checks; and
`contextual_bandit_transition_table.csv` is a sham/0.5/0.8-V/m dataset for a
later bandit demonstration.

Do not fit a bandit merely because that transition table exists. First require
held-out A/B observability and positive seed-consistent EEG movement. A failure
of the fundamental-excluded or matched-sine checks must be reported as evidence
that the ideal-simulator result is not yet robust to realistic concurrent-EEG
measurement. Only after a positive pilot should the frozen design be repeated
with at least 12 new validation seeds.

## Experiment 8: hierarchical EEG-only tACS identification

This experiment asks a harder but still compact question: can a controller
identify useful frequency, phase, and amplitude settings without being told
that B's hidden stochastic input is modulated at 60 Hz? The hidden generator
configuration is used only to simulate B and for a post-hoc audit. It is never
passed to the selector.

The hierarchy avoids a costly and statistically opaque Cartesian sweep:

```text
4 target seeds:       unstimulated A/B EEG -> generic spectral target
2 frequency seeds:    40, 60, 80 Hz at 0.8 V/m and phase 0
2 phase seeds:        0, 90, 180, 270 degrees at selected frequency
6 validation seeds:   sham, 0.5, 0.8 V/m at frozen frequency/phase
                      plus transverse and observation-only controls
```

All four seed sets are disjoint. Frequency selection uses only distance in a
predeclared vector of band powers, so it cannot exploit spike timing or a
frequency-specific target feature. After frequency is selected, a
phase-sensitive A/B EEG model adds the observed sine and cosine quadratures.
Phase is then frozen before held-out amplitude validation. Spikes and firing
rates remain hidden mechanistic and safety variables.

The state endpoint is full standardized Euclidean distance to the discovery-B
centroid. This avoids calling a point B-like merely because it moved along one
A-to-B projection while diverging in another feature direction. The
concurrent-EEG audit is also stricter than Experiment 7: the observation-only
sine matches both the cosine and sine coefficients of real tACS. The excluded
endpoint can pass only when held-out A and B are themselves distinguishable
after the selected-frequency bins are removed.

The six paper-inspired figures are deliberately limited to results that are
interpretable in this toy model:

1. validation PSD for A, B, selected tACS, and the observation control;
2. EEG-only frequency and phase selection curves;
3. held-out EEG, PPC, and firing-rate dose responses;
4. representative A/tACS E-spike rasters and phase histograms;
5. baseline-stimulation-washout EEG-state trajectory; and
6. paired real-tACS versus observation-only artifact controls.

Run the consolidated laptop pilot:

```bash
python experiments/ballnstick_analysis/run_ballnstick_hierarchical_tacs.py \
  experiment.name=ballnstick_hierarchical_tacs_pilot \
  env=ballnstick \
  analysis=ballnstick_hierarchical_tacs \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/ballnstick_hierarchical_tacs_pilot/hierarchical_tacs/`.
The primary files are `frozen_hierarchical_protocol.json`, the target,
frequency and phase CSVs, `validation_reachability.csv`,
`validation_hidden_mechanism.csv`, `experiment_conclusion.json`, and six
numbered PNG figures.

This remains a go/no-go pilot. A selected frequency or phase is not evidence
when every discovery action was harmful; those discovery-positive checks are
therefore explicit. Do not fit a contextual bandit from the generated policy
table unless a later new-seed dataset demonstrates that baseline EEG predicts
which dose outperforms the best fixed action.

## Experiment 9: phase-invariant EEG tACS confirmation

The hierarchical pilot recovered the hidden 60-Hz target from EEG and found a
reproducible spike-PPC response, but its absolute sine/cosine phase target did
not generalize. That negative result is preserved. This follow-up tests the
narrower and more appropriate stationary-state hypothesis: A and B differ in
the strength of a 60-Hz oscillation, irrespective of an arbitrary time-origin
phase, and a frozen 60-Hz axial field may move A toward that phase-invariant B
distribution.

The protocol is frozen before the run at 60 Hz, phase zero by convention, and
0.8 V/m. Four new calibration seeds see only unstimulated A/B EEG and fit a
one-dimensional log-band-power target. They also audit whether the strongest
generic A/B spectral shift is again 60 Hz, but cannot change the protocol.
Eight disjoint validation seeds then evaluate A, B, A+axial tACS, and an
A+transverse orientation control. The default deliberately omits the 0.5-V/m
arm to reduce runtime; it can be enabled as a secondary arm without changing
the primary hypothesis.

The primary endpoint is movement toward the calibration-frozen B centroid in
log EEG power around 60 Hz. Relative band power and Fourier resultant
magnitude are secondary phase-invariant summaries. Raw cosine/sine
quadratures are not state variables. E-cell PPC, firing rates, and washout are
hidden mechanism/safety checks and never enter the EEG target or reward.

Run a three-seed directional gate before committing to confirmation. These
offsets are deliberately separate from the full-run seeds:

```bash
python experiments/ballnstick_analysis/run_ballnstick_phase_invariant_tacs.py \
  experiment.name=ballnstick_phase_invariant_tacs_gate \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_invariant_tacs \
  env.simulation.obs_win_len=1000 \
  analysis.calibration.n_seeds=2 \
  analysis.calibration.seed_offset=142000 \
  analysis.validation.n_seeds=3 \
  analysis.validation.seed_offset=143000 \
  experiment.plot=true \
  experiment.tqdm=false
```

The gate cannot pass the minimum-sample checks and is not confirmation.
Proceed only if the frozen 60-Hz feature is positive in calibration, axial
tACS increases that feature in at least two of three validation seeds, the
direction is closer to B, PPC is not directionally adverse, and rates remain
safe. Do not change the full-run feature, phase, or amplitude after inspecting
the gate.

Run the predeclared eight-seed confirmation with new seeds:

```bash
python experiments/ballnstick_analysis/run_ballnstick_phase_invariant_tacs.py \
  experiment.name=ballnstick_phase_invariant_tacs_confirmatory \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_invariant_tacs \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are saved under
`../../results/<name>/phase_invariant_tacs/`. The frozen mapping is recorded in
`frozen_phase_invariant_protocol.json`; seed-level primary results are in
`validation_reachability.csv`; direct phase-invariant features are in
`validation_phase_invariant_feature_effects.csv`; hidden spike/rate checks are
in `validation_hidden_mechanism.csv`; and `experiment_conclusion.json` contains
the predefined confirmation and measurement-audit conclusions.

The complex matched-observation sine remains an explicit limitation audit.
It is expected to reproduce a same-frequency power endpoint, so superiority
to that synthetic observation is not required for the ideal neural-only EEG
claim. Conversely, a positive ideal result must not be described as robust
simultaneous tACS-EEG. If realistic concurrent recording is required, it needs
an explicit artifact forward model and additional measurement information;
removing the 60-Hz band is not a valid primary endpoint when that band defines
B itself.

## Experiment 10: EEG-relative alpha suppression toy problem

This experiment replaces the failed 60-Hz phase-invariant reachability target
with a different, explicitly operational problem. It does not retrofit the
BallAndStick network into a depression model.

```text
A: identical circuit plus mean-preserving 10-Hz modulation of independent
   Poisson afferent rates (elevated-alpha toy state)
B: identical circuit plus homogeneous Poisson afferents (low-alpha reference)
action: 0.8-V/m, 10-Hz uniform field along the somatodendritic axis
goal: acutely reduce A's ideal EEG 8--12-Hz power toward B
```

The hidden afferent phase is randomized by circuit seed. In each active
episode, four seconds of preceding EEG are used to estimate and extrapolate
the 10-Hz phase to the intervention boundary. The discovery action is one of
four relative phase offsets; it is not an absolute simulator-clock phase. A
500-ms raised-cosine onset/offset is used, and those ramps are removed before
the primary six-second block is summarized. The remaining five seconds give
0.5-Hz Welch bins and four overlapping 2-s segments. Two-second bins are saved
only as a shorter-window observability audit.

Calibration, phase discovery, and validation use disjoint seeds. The primary
held-out endpoint is the paired reduction in log 8--12-Hz EEG power and the
paired movement toward B. The selected phase must also outperform its opposite
phase and the same field applied transversely. Exact 10-Hz EEG amplitude,
alpha-peak prominence, E-population PPC, firing rates, baseline equality, and
washout are secondary mechanism/safety checks. A complex observation-only
sinusoid is fitted as a measurement audit: failure to beat it blocks a claim
about concurrent artifact-contaminated tACS-EEG, but is kept separate from the
ideal neural-only forward-model conclusion.

Run the low-cost four-seed directional pilot:

```bash
python experiments/ballnstick_analysis/run_ballnstick_alpha_suppression.py \
  experiment.name=ballnstick_alpha_suppression_pilot \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_alpha_suppression \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Do not change the target, selected phase, amplitude, or controls after seeing
the pilot. The four-seed pilot cannot provide a two-sided exact sign-flip
p-value below 0.05. If all directional checks pass, add/use a confirmation-only
run that reads `frozen_alpha_target.json` and `frozen_tacs_protocol.json` and
evaluates at least eight new validation seeds; rerunning phase discovery and
calling the result confirmation would be invalid.

Results are written to
`../../results/<name>/alpha_suppression/`. The key files are
`frozen_alpha_target.json`, `frozen_tacs_protocol.json`,
`phase_discovery_summary.csv`, `validation_seed_metrics.csv`,
`validation_summary.csv`, `observation_only_complex_match_audit.csv`,
`two_second_eeg_bins.csv`, `experiment_conclusion.json`, and four numbered
figures.

## Experiment 11: frozen alpha-suppression confirmation

The pilot constructed a reproducible elevated-alpha state and found that the
180-degree EEG-relative action reduced alpha on all four validation seeds, but
all phases had increased alpha on its two phase-screen seeds. The pilot is
therefore exploratory as a whole. The next valid question is whether the
specific hypothesis suggested by its validation split replicates without any
new calibration or action selection.

This runner loads `frozen_alpha_target.json` and
`frozen_tacs_protocol.json` from Experiment 10, checks the expected 0.04 input
modulation depth, 10-Hz frequency, 0.8-V/m amplitude, axial montage and
180-degree EEG-relative phase, and records hashes of both source files. It then
tests exactly five arms on eight new matched circuit seeds: B, A sham, frozen
180-degree axial tACS, opposite 0-degree axial tACS, and the 180-degree
transverse control.

The primary endpoint is paired log 8--12-Hz power suppression by the frozen
action. Confirmation requires a positive bootstrap interval, at least six of
eight positive seeds and an exact two-sided sign-flip p-value no greater than
0.05. A-to-B distance movement is assessed separately. Opposite-phase and
orientation controls form a two-test FDR family. Rates, washout and exact
pre-action trajectory equality remain guardrails.

The four-second baseline is additionally divided into two halves. Each half's
10-Hz phase is independently extrapolated to the stimulation boundary. The
phase policy is considered measurable only when at least 75% of seeds have
less than 45 degrees split-half disagreement and a 10-Hz Fourier-resultant to
EEG-RMS ratio of at least 0.05. The hidden afferent phase is saved solely to
audit the simulated transfer lag; it never determines the action.

Run from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_alpha_suppression_confirmation.py \
  experiment.name=ballnstick_alpha_suppression_confirmation \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_alpha_suppression_confirmation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

The serial form is scientifically identical but slower: remove
`mpiexec -n 4`. Results are written under
`../../results/<name>/alpha_suppression_confirmation/`. The primary files are
`frozen_protocol_provenance.json`, `baseline_phase_quality.csv`,
`confirmation_seed_metrics.csv`, `confirmation_summary.csv`,
`experiment_conclusion.json`, and three numbered figures.

A positive result supports only acute phase-dependent control in ideal
neural-only simulated EEG for this toy circuit. It does not establish a
depression mechanism, clinical efficacy, persistent plasticity, continuous
action controllability, or a contextual RL advantage. Do not combine pilot
and confirmation seeds for the primary p-value or retune after inspecting the
confirmation.

## Experiment 12: exploratory frozen-phase dose and mechanism audit

The independent confirmation in Experiment 11 reproduced the elevated-alpha
A state and found phase-dependent spike-timing modulation, but the frozen
0.8-V/m, EEG-relative 180-degree action increased rather than suppressed alpha
on average. Repeating that action with additional seeds is therefore not the
next scientific step. This experiment tests the prospective mechanistic
hypothesis that a weaker field may preserve spike desynchronization while
reducing the coherent membrane-current contribution to ideal EEG.

The A/B generator, 10-Hz frequency, axial montage, and EEG-relative
180-degree phase convention are loaded from the frozen pilot files and cannot
be recalibrated here. Four new matched exploratory seeds receive B sham, A
sham, and A with 0.2, 0.4, 0.6, or 0.8 V/m. Four seeds make this a directional
screen only: even perfect sign consistency cannot yield a two-sided exact
sign-flip p-value below 0.05.

For every dose, the primary metric is A-sham minus active log 8--12-Hz EEG
power. Movement toward B, exact 10-Hz resultant, alpha-peak prominence,
E/I-population PPC, firing rates, phase quality, pre-action identity, and
field removal are saved. Washout is now audited independently of whether the
acute effect was beneficial. The exact 10-Hz Fourier-vector change is
decomposed using

```text
|a + d|^2 - |a|^2 = 2 a·d + |d|^2
```

where `a` is A-sham EEG and `d` is the matched active-minus-sham response. The
cross term can be suppressive, whereas the induced-component term is always
non-negative. The three-component total current-dipole trace from the online
simulator is also analyzed. These are neural source currents in the ideal
forward model, not a model of electrode artifact.

Run the four-seed directional audit from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_alpha_suppression_dose_audit.py \
  experiment.name=ballnstick_alpha_suppression_dose_audit \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_alpha_suppression_dose_audit \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

The serial form is scientifically identical but slower: remove
`mpiexec -n 4`. Results are written under
`../../results/<name>/alpha_suppression_dose_audit/`. Important outputs are
`dose_seed_metrics.csv`, `dose_metric_summary.csv`, `dose_guardrails.csv`,
`dose_response_models.json`, `exploratory_candidate_protocol.json`,
`experiment_conclusion.json`, and three numbered figures.

The runner may rank one dose using only ideal EEG directional effects and
action metadata, but explicitly labels it exploratory. A passing gate does not
confirm controllability. Freeze a passing dose and use new circuit seeds for a
confirmation-only experiment before adding it to a bandit action set. If no
dose suppresses alpha consistently, do not increase the field beyond 0.8 V/m
to rescue the result; instead test a small quadrature phase map or conclude
that this toy circuit does not support the intended alpha-power control.

## Experiment 13: prospectively screened 0.4-V/m confirmation

Experiment 12 selected 0.4 V/m as an exploratory candidate, with all four
seeds showing alpha suppression and movement toward B. The response depended
on baseline phenotype magnitude: a circuit with a weak A/B difference could
overshoot B. This confirmation therefore estimates treatment effect in a
prospectively defined, biomarker-positive and phase-actionable toy subgroup,
analogous to enrolling only individuals who exhibit the mechanism targeted by
an intervention.

Candidate seeds are considered in a fixed order. Each first receives one
unstimulated A screening episode. Eligibility requires:

1. multi-second ideal EEG alpha power classified as A by the frozen threshold
   learned in Experiment 10;
2. a stable, measurable 10-Hz phase in the preceding baseline; and
3. baseline E/I firing rates within the predeclared safety ranges.

The screen cannot inspect tACS outcomes, hidden PPC, or the seed-specific B
counterfactual. This is important both statistically and translationally: a
real participant's untreated measurement may be compared with a previously
defined reference distribution, but their personal healthy counterfactual is
not observable. Excluded candidates and reasons are saved in
`screening_audit.csv`; they receive no active stimulation. The first eight
eligible seeds are enrolled, from at most twenty prospective candidates.

Only after enrollment does the runner simulate B for target-distance
evaluation and apply the frozen 10-Hz, 0.4-V/m, EEG-relative 180-degree axial
protocol. Opposite-phase and transverse controls use the same dose. The
primary statistical unit remains the enrolled circuit seed, and all claims
are conditional on screen eligibility. Screening yield quantifies this toy
model's seed variability and must not be interpreted as human biomarker
prevalence.

Run from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_alpha_suppression_screened_confirmation.py \
  experiment.name=ballnstick_alpha_suppression_screened_confirmation \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_alpha_suppression_screened_confirmation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written under
`../../results/<name>/alpha_suppression_screened_confirmation/`. Key outputs
are `screening_audit.csv`, `screened_confirmation_seed_metrics.csv`,
`screened_confirmation_summary.csv`, `frozen_candidate_provenance.json`,
`experiment_conclusion.json`, and three numbered figures.

Confirmation requires at least six of eight positive paired alpha effects, a
positive bootstrap interval, an exact two-sided sign-flip p-value no greater
than 0.05, and independently positive movement toward B. Phase and orientation
controls are FDR-corrected. Rates, exact pre-action equality, and
sign-independent field removal remain guardrails. Passing supports only acute
ideal-EEG control in the prospectively screened toy subgroup; it does not
validate a diagnostic for depression or establish benefit in humans.

## Experiment 14: crossed-seed EEG-context dose feasibility (CL0)

Experiment 13 established a strong fixed 0.4-V/m effect in a prospectively
screened subgroup, but three of eight circuits overshot the frozen B target.
CL0 asks the necessary question before implementing a contextual bandit:
does pre-action ideal EEG predict whether sham, 0.2 V/m, or 0.4 V/m will
finish closer to the frozen B population target than the fixed 0.4-V/m policy?

This is a three-by-three crossed directional audit. The structure seed controls
cell placement, synapse locations, recurrent topology, weights, delays, and
multapses. The drive seed independently controls per-synapse stochastic
Poisson event timing. Absolute hidden 10-Hz phase is assigned through a third
seed namespace and held fixed so the drive factor isolates event-time
variability. Every matched action replay retains the same structure and drive.

Each context first receives an unstimulated A screening episode. Ineligible
contexts receive no active intervention. Eligible contexts receive only the
frozen 10-Hz, axial, EEG-relative 180-degree action at 0.2 and 0.4 V/m; the
screening episode is the sham counterfactual. The target is the frozen B mean,
never a seed-specific B simulation. Primary EEG contexts are alpha excess
above B and the coherent 10-Hz fraction of alpha power. Hidden rates and PPC
remain mechanism/safety audits.

The full-information counterfactual oracle first establishes whether any
action-selection opportunity exists. An arm-specific ridge rule is then
evaluated by leaving out entire structure seeds, not individual drive
sessions. Passing requires a practical nonfixed oracle opportunity, multiple
selected actions, and directional cross-fitted improvement over fixed
0.4 V/m. With three structure seeds this is a low-cost gate, not statistical
confirmation or an RL result.

Run from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_context_dose_feasibility.py \
  experiment.name=ballnstick_context_dose_feasibility \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_context_dose_feasibility \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written under
`../../results/<name>/context_dose_feasibility/`. Important outputs are
`screening_audit.csv`, `context_action_metrics.csv`,
`context_counterfactual_summary.csv`, `cross_validated_context_policy.csv`,
`structure_level_policy_comparison.csv`, `seed_variance_decomposition.csv`,
`mechanistic_context_correlations.csv`, `exploratory_context_model.json`,
`experiment_conclusion.json`, and three numbered figures.

If the feasibility gate fails, retain fixed 0.4 V/m and do not fit a bandit
from these contexts. If it passes, the next experiment must freeze the context
features, preprocessing, action set, reward, and selection rule, then compare
that policy with fixed 0.4 V/m on disjoint structure seeds and independent
drive sessions.

## Experiment 15: common-probe contextual-dose feasibility (CL1-P)

CL0 showed that passive baseline EEG did not reliably identify the sole
context in which 0.2 V/m outperformed 0.4 V/m. CL1-P tests the mechanistic
alternative that the response to a weak common probe reveals the local
input--output gain. This remains paired system identification, not RL.

The B target used during the short probe is first calibrated on four disjoint
homogeneous-Poisson population-reference seeds. This is necessary because the
two-second probe endpoint and frozen five-second decision endpoint have
different spectral-estimator sampling distributions. It is a population mean,
never a seed-specific counterfactual.

Each crossed A context receives an unstimulated screening replay. Eligible
contexts are then replayed twice with identical structure, drive, phase,
baseline, and 0.2-V/m probe. At the decision boundary, one replay maintains
0.2 V/m and the other makes a 500-ms raised-cosine transition to 0.4 V/m.
Both retain the frozen 10-Hz axial field and EEG-relative 180-degree phase.
The five-second decision analysis windows are duration-matched to the frozen B
endpoint.

The frozen probe rule maintains 0.2 V/m when the duration-matched probe alpha
estimate is at or below the calibrated B mean and otherwise escalates. Passing
requires practical low-dose opportunities across at least two structures,
multiple prospectively selected actions, positive structure-level improvement
over fixed escalation, and performance beyond shuffled probe contexts. Hidden
PPC and rates remain mechanism/safety audits. A failure means that this
homogeneous 10-Hz amplitude task still does not justify a contextual bandit.

Run from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_context_probe_feasibility.py \
  experiment.name=ballnstick_context_probe_feasibility_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_context_probe_feasibility \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written under
`../../results/<name>/context_probe_feasibility/`. Important outputs are
`probe_target_calibration.csv`, `frozen_probe_target.json`,
`screening_audit.csv`, `context_epoch_eeg_and_hidden_metrics.csv`,
`context_action_metrics.csv`, `context_counterfactual_summary.csv`,
`structure_level_policy_comparison.csv`, `probe_context_shuffle_null.csv`,
`frozen_protocol_provenance.json`, `experiment_conclusion.json`, and three
numbered figures.

## Experiment 16: held-out EEG-trajectory dose confirmation (CL1-C)

CL1-P established a counterfactual action crossover: four eligible contexts
preferred maintaining 0.2 V/m and four preferred escalating to 0.4 V/m. Its
prespecified absolute-target rule nevertheless escalated every context. A
single post-CL1-P EEG-only hypothesis is now frozen before new simulation:

\[
\Delta_{\mathrm{trajectory}}
=\log_{10}P_{\alpha,\mathrm{matched\ baseline}}
-\log_{10}P_{\alpha,\mathrm{active\ probe}}.
\]

The rule maintains 0.2 V/m when
\(\Delta_{\mathrm{trajectory}}>0\), and otherwise escalates to 0.4 V/m. This
is an operational state-history rule. It must not be described as a causal
probe-susceptibility biomarker because CL1-P showed that part of the trajectory
was also present during sham.

CL1-C retains the exact CL1-P baseline, probe, decision, and washout timing.
It uses six new structure seeds crossed with three new afferent-drive seeds.
The runner loads and hashes the completed CL1-P outputs, rejects overlapping
seed namespaces, and performs no rule fitting on confirmation outcomes. Both
post-probe actions are simulated with identical predecision histories; this
scores the frozen policy, both fixed-dose comparators, and the counterfactual
oracle under common random numbers.

The structure seed is the inferential unit. Primary confirmation requires the
frozen rule to select both actions, beat fixed 0.2 and fixed 0.4 V/m by the
predeclared mean margin, improve in at least four of six structure groups,
pass an exact structure-level sign-flip test and a shuffled-context test, and
reduce regret to the oracle. A baseline-only rule using the previously frozen
A mean and a paired sham-trajectory rule are attribution audits. Their results
do not replace the primary fixed-dose comparisons.

Run from the repository root after CL1-P has completed:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_context_trajectory_confirmation.py \
  experiment.name=ballnstick_context_trajectory_confirmation_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_context_trajectory_confirmation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written under
`../../results/<name>/context_trajectory_confirmation/`. Important outputs
are `discovery_rule_audit.json`, `screening_audit.csv`,
`context_epoch_eeg_and_hidden_metrics.csv`, `context_action_metrics.csv`,
`heldout_context_policy_summary.csv`, `structure_level_policy_comparison.csv`,
`trajectory_context_shuffle_null.csv`, `frozen_protocol_provenance.json`,
`experiment_conclusion.json`, and four numbered figures.

A primary pass confirms only that a frozen EEG-history dose rule outperforms
both fixed doses in this ideal neural-only toy system. It supports running a
separate two-action contextual-bandit trial; it is not itself RL and it does
not establish a human stimulation result.

## Experiment 17: single-action conditional dose map (CDM1-S)

CL1-C showed realized 0.2-versus-0.4-V/m action opportunities, but its active
probe trajectory did not predict the better action on held-out structures.
CDM1-S removes the probe and action switching. Each intervention replay uses
exactly one amplitude, selected once after a six-second stimulation-free EEG
baseline and held for the complete stimulation block. Sham is a paired causal
comparator, not a required future policy action.

The experiment explicitly crosses three mean-rate-matched toy alpha states
(afferent modulation depths 0.02, 0.04, and 0.06) with circuit structures.
The latent modulation depth is not an EEG feature and must never be supplied
to a future policy. It exists to create transparent mild, moderate, and strong
oscillatory states instead of relying on random seeds to accidentally produce
a learnable context--action interaction. Cells, recurrence, inhibition,
synaptic weights, mean afferent rate, 10-Hz frequency, and afferent phase are
otherwise unchanged across states.

The principal methodological addition is a decision-boundary random-stream
split. Background events before tACS depend on a history seed and are exactly
identical across every action and future replay of one context. Events after
the decision depend on a separately recorded future seed. Two independent
future continuations are run for each sham/0.1/0.2/0.3/0.4-V/m action. This
estimates conditional expected action response instead of defining the oracle
from one unpredictable Poisson realization.

The exploratory policy context contains only four phase-invariant features
from the preceding ideal EEG: alpha excess over the frozen B population mean,
coherent 10-Hz fraction, alpha peak prominence, and alpha-power temporal
standard deviation across three non-overlapping two-second windows. The EEG
phase estimate aligns the frozen 180-degree waveform but is not a dose feature.
Hidden state depth, spikes, PPC, and rates cannot enter the policy.

Run the low-cost directional screen from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_single_action_dose_map.py \
  experiment.name=ballnstick_single_action_dose_map_quick \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_single_action_dose_map \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

The default screen contains six contexts, two independent futures, and five
counterfactual actions: at most 60 persistent 15-second episodes, with fewer
active episodes when a baseline fails the frozen EEG/phase screen. Results are
written under `../../results/<name>/single_action_dose_map/`. Key outputs are
`predecision_screening_audit.csv`, `future_action_metrics.csv`,
`conditional_expected_dose_map.csv`, `context_expected_action_summary.csv`,
`state_observability_and_dose_summary.csv`,
`exploratory_loso_eeg_policy.csv`, `protocol_provenance.json`,
`experiment_conclusion.json`, and four numbered figures.

This is a directional system-identification gate, not a bandit result. Proceed
to a larger, disjoint policy confirmation only if multiple active doses are
expected-optimal in reproducible contexts, the expected oracle practically
beats the best fixed action, realized optima agree across independent futures,
and the EEG-only leave-one-structure-out diagnostic improves directionally on
the best fixed dose. Do not select a protocol from a single future realization.

## Experiment 18: expanded monotone EEG-severity discovery (CDM2-D)

CDM1-S established dose-dependent ideal-EEG control but not predictable
adaptation. Fixed 0.4 V/m was expected-optimal for three of four eligible
contexts; one lower-severity circuit preferred 0.2 V/m because 0.4 V/m
overshot B. Its four-feature arm-specific regression was underdetermined and
worse than fixed 0.4 V/m. CDM2-D therefore tests the narrower mechanistic
hypothesis that predecision alpha excess alone determines whether a weaker or
stronger single action is appropriate.

The screen-negative 0.02 modulation-depth state is retained as a specificity
control. The 0.04 and 0.06 states are crossed with three new structure seeds
and two new predecision-history seeds. Every eligible context has a 12-s
stimulation-free baseline, divided into six non-overlapping 2-s estimates, and
is replayed under sham, 0.2 V/m, and 0.4 V/m. Each action is held for the whole
intervention. Three independent postdecision Poisson continuations estimate
conditional expected response without changing the observed past.

Discovery fits only the preregistered monotone rule

\[
a(x)=
\begin{cases}
0.2\ \mathrm{V/m}, & x < \tau,\\
0.4\ \mathrm{V/m}, & x \geq \tau,
\end{cases}
\]

where \(x\) is stimulation-free log10 alpha excess over the frozen B mean.
No other EEG feature, state label, modulation depth, spike statistic, or rate
may enter threshold selection. Prediction is evaluated by leaving out entire
structure seeds. The low-dose opportunity must appear in multiple structures
and histories, both actions need practical support, the cross-validated rule
must beat frozen fixed 0.4 V/m, and true EEG context must outperform shuffled
context. The completed CDM1-S inputs are hashed, and all CDM2-D seeds are new.

Run from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_severity_threshold_discovery.py \
  experiment.name=ballnstick_severity_threshold_discovery_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_severity_threshold_discovery \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written under
`../../results/<name>/severity_threshold_discovery/`. Key outputs are
`predecision_screening_audit.csv`, `future_action_metrics.csv`,
`conditional_expected_binary_dose_map.csv`,
`context_threshold_discovery_table.csv`,
`crossvalidated_threshold_policy.csv`,
`crossvalidation_fold_thresholds.csv`,
`structure_level_policy_comparison.csv`, `alpha_context_shuffle_null.csv`,
`candidate_threshold_protocol.json`, `experiment_conclusion.json`, and four
numbered figures.

A passing candidate remains exploratory. Its threshold, preprocessing, action
set, and reward must be frozen before a separate experiment on disjoint
structures and histories. If the discovery gate fails, retain fixed 0.4 V/m
and do not train a contextual bandit from these data.

## Experiment 19: disjoint frozen severity-rule confirmation (CDM2-C)

CDM2-C loads the CDM2-D conclusion, candidate, and provenance by SHA-256
and freezes the complete rule: choose 0.2 V/m when predecision log10 alpha
excess is below 0.3801312721, otherwise choose 0.4 V/m. The threshold, B
reference, EEG preprocessing, actions, phase tracking, endpoint, comparator,
and criteria are not re-estimated.

Twelve new circuit structures are crossed with two new histories and three
new postdecision futures. Circuit structure is the independent unit;
histories and futures reduce conditional-response noise but do not increase
the inferential sample size. The minimally important mean advantage is 0.01
log10 and the planning structure SD is 0.013 log10, giving
\(d_z=0.769\). An a priori one-sided paired-t approximation gives 80.3% power
at 12 structures. This powers a large proof-of-concept benefit. Powering the much
smaller exploratory cross-validated estimate (approximately \(d_z=0.2\))
would require about 156 structures and is outside this toy study's scope.

There is one primary contrast: structure-averaged distance under the frozen
rule versus fixed 0.4 V/m. Confirmation requires mean advantage of at least
0.01 log10, one-sided exact structure-level sign-flip \(p\leq0.05\), positive
advantage in at least 75% of structures, all 12 structures remaining
analyzable, both actions retaining at least 20% support, and all design and
safety checks passing. Paired t, interval, bootstrap, Wilcoxon, fixed-0.2,
sham, and oracle results are secondary audits and cannot rescue the primary
gate.

Run from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_severity_threshold_confirmation.py \
  experiment.name=ballnstick_severity_threshold_confirmation_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_severity_threshold_confirmation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written under
`../../results/<name>/severity_threshold_confirmation/`. The primary files
are `frozen_threshold_policy_outcomes.csv`,
`structure_level_primary_contrast.csv`, `statistical_inference.json`,
`frozen_confirmation_protocol.json`, and `experiment_conclusion.json`.
A pass permits a subsequent contextual-bandit experiment; it is not itself a
trained or tested bandit.

## Experiment 20: frequency/relative-phase feasibility map (F0-FP)

The failed frozen severity rule showed that prestimulation alpha magnitude did
not reproducibly predict whether 0.2 or 0.4 V/m was better on new structures.
F0-FP tests a more directly observable and mechanistically matched source of
context: the dominant alpha frequency. It is deliberately a small
full-information system-identification experiment, not another bandit fit.

The low-alpha reference B uses homogeneous independent Poisson afferents. The
elevated-alpha toy state A uses the same cells, recurrence, inhibition,
synaptic weights, and expected afferent rate, but weakly modulates the afferent
event probability at either 9 or 11 Hz. Its continuous afferent phase is
randomized by context. Thus frequency and phase are dynamic toy-circuit
properties rather than health/disease labels, and the state generator is not
the tACS actuator.

Six disjoint B seeds calibrate a two-dimensional ideal-EEG target from log10
power near 9 and 11 Hz. Each A context is screened before stimulation for an
elevated spectral phenotype and a stable EEG phase; hidden generator labels,
spikes, rates, and action outcomes cannot determine enrollment or the policy.
Three independent circuit structures are crossed with both hidden frequencies.
For every eligible context, two independent postdecision futures are replayed
under five arms:

- sham;
- 9 Hz at 0 or pi relative to the preceding EEG;
- 11 Hz at 0 or pi relative to the preceding EEG.

Every active arm uses the same axial 0.4-V/m tissue field and one constant
action for the complete six-second intervention. Relative phase is causally
estimated from the preceding six-second EEG; it is not a fixed absolute phase
and it is not the hidden afferent phase.

The primary F0 questions are whether prestimulation EEG identifies 9 versus
11 Hz, whether matched-frequency anti-phase control beats both mismatched
frequency and matched in-phase controls, and whether the frozen EEG rule
"choose the detected frequency at pi relative phase" beats the best fixed
active arm at the structure level. A context-label shuffle is an attribution
audit. Hidden E-population PPC, firing rates, exact paired baseline identity,
phase tracking, and washout are mechanism/safety checks. With only three
structures, all criteria are directional feasibility gates; p-values would
not support confirmatory efficacy claims.

Run from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_frequency_phase_feasibility.py \
  experiment.name=ballnstick_frequency_phase_feasibility_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_frequency_phase_feasibility \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written under
`../../results/<name>/frequency_phase_feasibility/`. Key outputs are
`reference_B_calibration.csv`, `prospective_screening.csv`,
`context_action_future_metrics.csv`, `expected_context_action_map.csv`,
`frequency_phase_crossover_summary.csv`,
`eeg_rule_vs_fixed_comparison.csv`,
`structure_level_policy_comparison.csv`,
`frequency_context_shuffle_null.csv`, `protocol_and_provenance.json`,
`experiment_conclusion.json`, and four numbered figures.

Advance only if both frequency states remain EEG-observable, frequency and
phase crossover effects have the predicted sign across structures, the EEG
rule beats the best fixed action, and shuffled context loses that advantage.
Otherwise report that this minimal frequency/phase action set lacks a
replicated contextual opportunity; do not increase model complexity merely to
force a positive bandit result.

## Experiment 21: stimulation-free shared phase-diffusion validation (D0)

D0 adds a minimal source of nonstationarity to the weak rhythmic afferent
drive. One latent phase is shared by E- and I-population afferent intensities,
while individual synapses retain private Poisson events:

\[
d\phi = 2\pi f\,dt + \sqrt{2D}\,dW, \qquad
\lambda_{pj}(t)=\lambda_{0p}[1+m\sin\phi(t)].
\]

This is a phenomenological shared upstream rhythm, compatible with fluctuating
long-range or thalamocortical drive, but it is not an explicit thalamic model.
Phase diffusion changes coherence time and linewidth; it does not create true
amplitude bursts. Expected mean afferent rates, cells, recurrence, weights, and
inhibition remain fixed, and D0 applies no electric field.

Three independent structures are crossed with 9 and 11 Hz and the
preregistered candidate levels D=0, 0.5, and 2 rad^2/s. The 12-s baseline gives
0.083-Hz raw-periodogram spacing and twelve causal 1-s phase estimates. D0
checks the SDE increment variance `2*D*dt` and audits the finite-record
coherence against `exp(-D*tau)`, as well as independent private event streams,
mean-rate invariance, frequency recovery, phase stability, spectral
concentration, held-structure low/high classification, signal relative to
within-trajectory temporal noise, causal recent-phase measurability, and firing
rate safety. Circuit structure is the statistical unit.

Run the short lifecycle smoke first:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_phase_diffusion_validation.py \
  experiment.name=ballnstick_phase_diffusion_validation_smoke \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_diffusion_validation \
  analysis.smoke_test=true \
  analysis.timeline.baseline_steps=2 \
  analysis.crossed_design.n_structure_seeds=1 \
  experiment.plot=false \
  experiment.tqdm=false
```

The smoke confirms execution only. Run the frozen full D0 experiment with:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_phase_diffusion_validation.py \
  experiment.name=ballnstick_phase_diffusion_validation_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_diffusion_validation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written under
`../../results/<name>/phase_diffusion_validation/`. The primary files are
`phase_diffusion_eeg_metrics.csv`, `eeg_periodograms.csv`,
`private_afferent_event_audit.csv`, `latent_phase_trace_10ms.csv`,
`frozen_phase_diffusion_generator.json`, `experiment_conclusion.json`, and
three numbered figures.

Do not run or fit D1 unless the full D0 gate passes or a disjoint confirmation
validates a replacement EEG endpoint frozen from D0. A D0 failure means the
candidate generator or observation is not yet adequate; diagnose it without
tACS and never use stimulation outcomes to tune D or the observation.

## Experiment 22: frozen phase-increment observability confirmation (D0b)

D0 validated the shared phase-diffusion generator, mean-rate invariance,
frequency visibility, and ordered group-average EEG effects. It nevertheless
failed its preregistered individual-circuit observability gate because a single
global 12-s circular phase resultant was noisy across structures. That negative
gate remains reported. Post-hoc D0 analysis nominated a more direct observation
of the mechanism: successive increments of one-second, carrier-demodulated EEG
phase,

\[
C_1=\frac{1}{K-1}\sum_{k=1}^{K-1}
\cos\!\left(\theta_{k+1}-\theta_k\right).
\]

The phase `theta_k` is obtained after the carrier has been selected from the
ideal pre-action EEG on the frozen {9,11}-Hz candidate grid. The generator's
hidden frequency is used only to audit selection accuracy. D0 gave
`C1_low=0.7021899103`, `C1_high=0.3339978710`, and a midpoint threshold of
`0.5180938907`; these values and all three D0 source files are hash-locked in
D0b. They are never refitted on confirmation data.

D0b retains only the frozen low/high candidates D={0.5,2.0} rad^2/s, the 0.04
modulation depth, both frequencies, the unchanged BallAndStick circuit, and a
12-s stimulation-free EEG baseline. Six new circuit structures are crossed
with two new afferent histories. Frequencies and histories are repeated
measurements; the structure remains the inferential unit. The minimum useful
paired effect is 0.15 and the D0 discovery structure SD is 0.12114, giving
`d_z=1.238`. Six structures provide 82.87% one-sided paired-t planning power;
the prespecified exact sign-flip test is the primary randomization inference.

Confirmation additionally requires at least 5/6 positive structure effects,
75% frozen-threshold balanced accuracy, above-chance classification in at
least 5/6 structures, signal larger than within-trajectory temporal variation,
90% frequency recovery, recent phase measurability, and rate safety. Tests at
0.5- and 1-s phase-estimation intervals are action-cadence audits, not extra
primary outcomes. Two frequency-specific sign-flip audits use Benjamini-
Hochberg FDR. D0b has one primary contrast and applies no electric field.

Run the full confirmation from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_phase_increment_confirmation.py \
  experiment.name=ballnstick_phase_increment_confirmation_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_increment_confirmation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written under
`../../results/<name>/phase_increment_confirmation/`. Primary outputs are
`confirmation_eeg_metrics.csv`, `paired_context_effects.csv`,
`frozen_threshold_classification.csv`,
`structure_level_primary_effects.csv`, `statistical_inference.json`,
`frequency_level_FDR_audits.csv`, `frozen_endpoint_provenance.json`,
`experiment_conclusion.json`, and four numbered figures.

Advance to D1 full-information system identification only if the final output
says both `Frozen EEG phase-increment observability: CONFIRMED` and
`Ready for D1 system identification: YES`. A failed D0b result means the phase-
diffusion context is not sufficiently reproducible from this ideal-EEG history;
do not tune D or the threshold using tACS outcomes to force a crossover.

## Experiment 23: phase-diffusion full-information action map (D1)

D1 begins only after D0b confirmed that the frozen phase-increment endpoint is
observable from ideal EEG. It also hash-locks the positive F0 result that
identified EEG-matched carrier frequency and pi-relative phase as a defensible
minimal tACS convention. Neither source is refitted in D1.

The toy state generator remains

\[
d\phi = 2\pi f\,dt + \sqrt{2D}\,dW, \qquad
\lambda_{pj}(t)=\lambda_{0p}[1+0.04\sin\phi(t)],
\]

with `f={9,11}` Hz and `D={0.5,2.0}` rad²/s. A separately calibrated B
population has homogeneous Poisson afferents. All cells, recurrence, mean
afferent rates, weights, and inhibition are unchanged. The circuit labels are
not health or disorder labels.

Each D1 context provides a 12-s stimulation-free ideal-EEG history. The full
history estimates the frozen slow context

\[
C_1=\frac{1}{K-1}\sum_k \cos(\theta_{k+1}-\theta_k),
\]

while only the most recent one-second EEG initializes the field phase at the
decision boundary. This separation prevents obsolete early-baseline phase
from controlling a phase-diffusing circuit. EEG chooses the nearest carrier
on the frozen `{9,11}`-Hz grid and every active arm uses the frozen
EEG-relative antiphase convention. Frequency and recent phase are deterministic
signal-processing inputs; only C1 is considered a learnable policy feature.

The full-information action set is exactly `{sham, 0.2, 0.4}` V/m. One action
is held for the complete intervention. Every counterfactual replay has the
same circuit, predecision Poisson events, and predecision latent phase path.
At the action boundary, both private Poisson events and shared Brownian phase
increments split into independent future streams. Two futures estimate each
conditional expected action response rather than selecting an action from one
lucky realization. Structure is the independent unit; frequency, diffusion,
and futures are repeated measurements.

The primary reward is negative absolute distance between post-action ideal-EEG
log alpha power and the duration-matched B population mean. Prospective
eligibility requires elevated predecision alpha, a measurable recent carrier,
and rate safety, but may not use hidden D, hidden generator frequency, spikes,
or stimulation outcomes. D1 maps every action before fitting an exploratory
leave-one-structure-out C1 rule. Advancement requires practical expected
optimal-action reversals across structures, reproducible future-wise winners,
a diffusion-by-dose interaction, advantage over the best fixed action, and
loss of that advantage when C1 is shuffled within structure. A positive D1 is
still system identification; it freezes a candidate for disjoint confirmation
and is not a contextual-bandit result.

Run the reduced lifecycle smoke with:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_phase_diffusion_action_map.py \
  experiment.name=ballnstick_phase_diffusion_action_map_smoke \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_diffusion_action_map \
  analysis.smoke_test=true \
  analysis.smoke_context_limit=2 \
  analysis.reference_calibration.n_seeds=1 \
  analysis.crossed_design.n_structure_seeds=1 \
  analysis.crossed_design.n_future_continuations=1 \
  analysis.timeline.baseline_steps=4 \
  analysis.timeline.stimulation_steps=2 \
  analysis.timeline.washout_steps=1 \
  analysis.timeline.block_ramp_ms=250 \
  analysis.timeline.stimulation_analysis_trim_ms=250 \
  analysis.screening.minimum_alpha_excess_log10=-10 \
  analysis.screening.minimum_recent_resultant_to_rms=0 \
  analysis.context_shuffle.n_permutations=20 \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

The smoke verifies execution only. Run the complete directional action map:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_phase_diffusion_action_map.py \
  experiment.name=ballnstick_phase_diffusion_action_map_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_diffusion_action_map \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Outputs are stored in
`../../results/<name>/phase_diffusion_action_map/`. Do not train a contextual
bandit unless the run reports both `D1 full-information action map: PASSED` and
`Ready for disjoint contextual policy confirmation: YES`. A failed gate means
that observable phase diffusion did not create a reproducible contextual
opportunity for this action set; do not add hidden D to the policy or tune the
generator using stimulation outcomes.

## Experiment 24: causal phase-maintenance audit (D1-R)

D1-R follows the frozen D1 failure without overwriting or reinterpreting it.
It asks whether the poor future-wise action reproducibility arose partly
because the tACS phase was initialized once and then allowed to drift away
from a phase-diffusing endogenous carrier. Three paired controllers are used:
sham, the D1 one-time phase initializer, and causal phase refreshing. Both
active arms use the same EEG-selected carrier, pi-relative target, axial
montage, and 0.2-V/m field, so controller timing is the only active contrast.

At update boundary \(t_k\), the refreshed controller estimates the EEG carrier
phase from the immediately preceding one-second ideal-EEG tail. Let
\(\theta_k^*\) be the desired field phase and \(\theta_s(t_k)\) the current
stimulator oscillator phase. It computes

\[
e_k=\operatorname{angle}\{\exp[i(\theta_k^*-\theta_s(t_k))]\},
\qquad
\Delta f_k=\operatorname{clip}\left(
\frac{e_k}{2\pi T_u},-2,2\right),
\]

and applies \(f_k=\hat f+\Delta f_k\) over the next \(T_u=250\) ms. This is a
phase-continuous frequency slew, not a reset: the first field sample after an
update equals the previous endpoint. A single raised-cosine block envelope is
also continuous across every controller window. The one-time arm calculates
the same later phase estimates for auditing but does not use them.

The full run uses 12 disjoint B references, three new structures, both 9/11-Hz
carriers, both frozen diffusion levels, and four independent postdecision
futures per context-controller pair. The primary endpoint is four-second ideal
EEG log-alpha distance to the duration-matched B mean; one-second trajectories
audit temporal stability. Advancement requires reduced phase error, a
practical refreshed-versus-one-time advantage, positive structure coverage,
future-wise winner reproducibility, no increase in future variance, rate
safety, and exact field removal. This remains exploratory and assumes
artifact-free concurrent EEG.

Run the lifecycle smoke with:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_phase_refresh_audit.py \
  experiment.name=ballnstick_phase_refresh_audit_smoke \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_refresh_audit \
  analysis.smoke_test=true \
  analysis.smoke_context_limit=2 \
  analysis.reference_calibration.n_seeds=1 \
  analysis.crossed_design.n_structure_seeds=1 \
  analysis.crossed_design.n_future_continuations=1 \
  analysis.timeline.baseline_steps=4 \
  analysis.timeline.stimulation_steps=3 \
  analysis.timeline.washout_steps=1 \
  analysis.timeline.block_ramp_ms=250 \
  analysis.timeline.stimulation_analysis_trim_ms=500 \
  analysis.screening.minimum_alpha_excess_log10=-10 \
  analysis.screening.minimum_recent_resultant_to_rms=0 \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Run the complete directional audit with:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_phase_refresh_audit.py \
  experiment.name=ballnstick_phase_refresh_audit_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_refresh_audit \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Outputs are saved under
`../../results/<name>/phase_refresh_audit/`. Proceed to a new context-action
map only if the final output reports `Causal phase-refresh mechanism: PASSED`
and `Ready for reliable context-action remapping: YES`.

Generate the paired controller-mechanics figure after either the smoke or full
run with:

```bash
python \
  experiments/ballnstick_analysis/plot_ballnstick_phase_refresh_example.py \
  --result-dir \
  ../../results/ballnstick_phase_refresh_audit_full/phase_refresh_audit \
  --display-duration-s 2
```

The upper traces are target-frequency EEG carriers reconstructed from the
rolling one-second causal Fourier estimates; they are not raw broadband EEG.
The lower traces are the phase-continuous field commands reconstructed from the
saved oscillator phases and interval frequencies. A PNG, PDF, and scope
metadata JSON are written into the result directory.

## Experiment 25: stationary H1--H3 disjoint confirmation (S1-C)

S1-C is the confirmatory stationary-carrier experiment that follows the
positive F0 system-identification result. It does not use phase diffusion and
does not train a bandit. It evaluates a fixed sequence of three claims in ideal
neural-only simulated EEG:

1. **H1, observable phenotype.** A has mean-rate-matched 9- or 11-Hz
   sinusoidal modulation of conditionally independent Poisson afferents at
   frozen depth 0.04; B has homogeneous afferents. All cells, recurrence,
   conductances, and expected afferent rates remain equal. Sixteen independent
   candidate structures provide paired A/B estimates. Four-second Hann-Welch
   segments give 0.25-Hz PSD resolution and paper figures show 1--30 Hz and a
   5--15-Hz alpha zoom.
2. **H2, causal tACS modulation.** Before any stimulation result is generated,
   structures are screened using only 12 s of stimulation-free ideal EEG. The
   first 12 structures for which both carriers pass the frozen phenotype and
   phase-quality screen are enrolled. Each enrolled context is replayed under
   sham and the complete fixed 0.4-V/m axial grid
   \(f\in\{9,11\}\) Hz by EEG-relative
   \(\Delta\phi\in\{0,\pi\}\), with four independent future continuations.
   The primary contrast is matched-frequency antiphase versus sham. Frequency
   and phase crossover tests are FDR-controlled secondary causal audits; one
   transverse replay per context audits orientation.
3. **H3, frozen one-decision EEG rule.** The F0 rule detects 9 versus 11 Hz
   from the prestimulation EEG and applies the detected carrier at relative
   phase \(\pi\). Without refitting, it is compared with sham, the hash-locked
   F0 best fixed action (`f9_antiphase`), and the uniform expected outcome of
   the four active frequency/phase actions. A structure-preserving frequency
   shuffle tests whether the EEG context, rather than a generic active-field
   benefit, explains its advantage.

The F0 conclusion, provenance, raw future metrics, and B calibration are
SHA-256 locked. S1-C uses disjoint reference, structure, history, phase, trial,
and future seeds. The circuit structure is the statistical unit; carriers,
actions, and futures are repeated measurements. The a-priori design has 84.7%
power at paired \(d_z=0.70\) for H1 (16 structures) and 82.9% power at
\(d_z=0.80\) for H2/H3 (12 structures), using a one-sided paired-t planning
approximation. Primary inference includes exact structure-level sign flips,
paired t and Wilcoxon sensitivity analyses, and structure bootstrap intervals.
These calculations do not imply power for smaller effects.

Run the complete confirmation with:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 4 python \
  experiments/ballnstick_analysis/run_ballnstick_stationary_h1_h3_confirmation.py \
  experiment.name=ballnstick_stationary_h1_h3_confirmation_full \
  experiment.seed=100000 \
  env=ballnstick \
  analysis=ballnstick_stationary_h1_h3_confirmation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/<name>/stationary_h1_h3_confirmation/`. The runner refuses to
overwrite a nonempty result folder. The documented base seed 100000 keeps the
scientific confirmation namespace separate from the seed-1/2 software smokes
used during implementation. It saves long-form and structure-level CSV
tables, frozen targets, power/provenance and inference JSON, plus seven figures
as both 300-dpi PNG and vector PDF. The fixed-sequence conclusion requires H1
before H2 and H2 before H3; a failed hypothesis is reported without changing
the frozen endpoints. The interpretation is restricted to a screened subgroup
of toy circuits and artifact-free simulated EEG. It is not evidence for a
clinical disorder, treatment efficacy, artifact-robust concurrent EEG, or a
trained contextual bandit.

## Experiment 26: H4 controller-bandwidth discovery (H4-BW)

H4-BW follows the completed D1-R mechanism audit without reinterpreting its
failed future-wise reliability gate. D1-R showed that its rolling one-second,
250-ms phase tracker reduced EEG-derived phase error and improved the expected
four-second distance to B, but the realized refreshed-versus-one-time winner
agreed across only 0.6875 of paired futures, below the frozen 0.75 gate. H4-BW
therefore selects a controller on new discovery seeds before the planned
12-structure H4 confirmation.

Five paired arms are run: sham, one-time phase initialization, the existing
1-s/250-ms tracker, a 0.5-s/250-ms tracker, and a 0.5-s/125-ms tracker. Every
active arm starts from the same final one-second prestimulation phase estimate
and uses the same EEG-selected carrier, 0.2-V/m axial field, and pi-relative
phase target. Only post-onset tracking differs. The two short-history arms use
the known EEG-selected frequency to estimate phase; the 0.5-s tail is not used
to discover frequency.

The phase error (e_k) is converted to a phase-continuous frequency slew using
a correction horizon fixed at 250 ms for every refresh rate,

\[
\Delta f_k=\operatorname{clip}\left(
  \frac{e_k}{2\pi(0.25\ {\rm s})},-2,2\right)\ {\rm Hz}.
\]

This prevents a 125-ms observation cadence from silently doubling feedback
gain. The full discovery uses three independent structures crossed with 9/11
Hz and low/high diffusion, one history, and four independent futures. It
contains 240 paired controller episodes if all 12 contexts pass screening.
Selection requires a practical benefit over one-time initialization, positive
structure and diffusion coverage, at least 0.75 paired-future wins, no increase
in future variance, noninferiority to the current tracker, lower phase error,
actionable half-second estimates, and all causal/safety/continuity checks. A
candidate within 0.01 log10 of the best passing endpoint uses the slower
250-ms update by the frozen parsimony rule. No p-value from this discovery is a
confirmation claim.

Run the complete bandwidth discovery with eight MPI ranks:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 8 python \
  experiments/ballnstick_analysis/run_ballnstick_phase_refresh_bandwidth_discovery.py \
  experiment.name=ballnstick_phase_refresh_bandwidth_discovery_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_refresh_bandwidth_discovery \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/<name>/phase_refresh_bandwidth_discovery/`. The selected
controller and complete frozen rule are saved in
`frozen_controller_candidate.json`. Proceed to H4 confirmation only if the
runner reports `Ready for disjoint 12-structure H4 confirmation: YES`. H4-BW
is a deterministic ideal-neural-EEG controller discovery experiment, not a
clinical concurrent-EEG result, contextual bandit, or disease model.

## Experiment 27: targeted 1-s/125-ms cadence discovery (H4-BW2)

The completed H4-BW experiment selected no controller. Its two faster
candidates shortened the phase-estimation history at the same time as they
changed update cadence, so estimator variance and control cadence could not be
separated. H4-BW2 preserves that negative result by file hash and adds the
missing `refresh_1000ms_125ms` arm. This controller retains the more stable
one-second phase estimator while issuing a new causal estimate every 125 ms.

Five paired arms are evaluated: sham, one-time initialization, the existing
1-s/250-ms controller, the new 1-s/125-ms controller, and the prior
0.5-s/125-ms arm. Each active arm uses the same one-second initialization,
EEG-selected 9/11-Hz carrier, 0.2-V/m axial field, pi-relative phase target,
500-ms onset/offset ramps, and fixed 250-ms correction horizon. Thus the new
contrast isolates update cadence without silently changing controller gain or
phase-history length.

The primary outcome is the absolute ideal-EEG log-alpha distance to a new,
disjoint, duration-matched homogeneous-B population target over eight seconds
after ramp trimming. The original D1-R prestimulation target remains frozen
for eligibility. Twelve B references are calibrated before any active outcome.
Every controller is also audited with the same causal one-second phase
estimator at common 250-ms boundaries, avoiding the window-dependent phase
error comparison that affected H4-BW.

The crossed discovery has three independent circuit structures, two carriers,
two diffusion levels, one history per structure/state, and six paired future
continuations per arm (360 total episodes if all 12 contexts enroll, including
the B calibration). Reliability is the within-context standard deviation and
win fraction of the paired controller-minus-one-time effect. A 125-ms arm is
frozen only if it passes every predeclared endpoint, cross-structure,
cross-diffusion, paired-future, paired-variance, common-phase, causality,
continuity, washout, and rate-safety gate. A tie within 0.01 log10 favors the
one-second estimator. This is discovery; it cannot confirm H4.

Run the full H4-BW2 discovery with eight MPI ranks:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 8 python \
  experiments/ballnstick_analysis/run_ballnstick_phase_refresh_cadence_discovery.py \
  experiment.name=ballnstick_phase_refresh_cadence_discovery_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_phase_refresh_cadence_discovery \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/<name>/phase_refresh_cadence_discovery/`. Proceed to a disjoint
12-structure H4 confirmation only if the runner reports
`Ready for disjoint 12-structure H4 confirmation: YES`.

## Experiment 28: disjoint adaptive phase-maintenance confirmation (H4-C)

H4-C freezes the controller selected by H4-BW2: a 0.5-s causal ideal-EEG phase
estimate refreshed every 125 ms, after a common one-second initialization.
The field remains 0.2 V/m, axial, EEG-carrier matched at 9 or 11 Hz, and
pi-relative to the measured neural phase. Frequency corrections remain bounded
to +/-2 Hz with a fixed 250-ms correction horizon and a phase-continuous field.
No controller parameter, target, endpoint, eligibility rule, or action is
selected in H4-C.

Sixteen new candidate structures are available to preserve a prospectively
screened sample of twelve. A structure is enrolled only if all four 9/11-Hz by
low/high-diffusion contexts satisfy the frozen stimulation-free EEG screen.
The three counterfactual arms are sham, one-time phase initialization, and the
frozen 0.5-s/125-ms controller. Each enrolled context has six paired future
continuations. Structure is the independent statistical unit; frequency,
diffusion and future are repeated measurements. Screening stops the study
before active outcomes if twelve complete structures cannot be enrolled.

The prespecified primary effect for structure s is

\[
d_s = \operatorname{mean}_{f,D,u}
\left[D_{\mathrm{one\mbox{-}time}}-D_{\mathrm{refresh}}\right],
\qquad
D=\left|\log_{10}P_{\alpha}-\mu_B\right|.
\]

Confirmation requires a mean primary advantage of at least 0.01 log10, a
one-sided exact structure-level sign-flip p value at most 0.05, positive
effects in at least two thirds of structures, nonadverse effects under both
diffusion levels, at least 0.75 paired-future wins, and all mechanistic and
safety gates. Refreshed control versus sham is a fixed-sequence secondary
contrast tested only after the primary passes. The twelve-structure design has
82.9% one-sided t-approximation power for a prespecified standardized paired
effect of dz=0.8. Exact permutation inference, t and structure-bootstrap
intervals, and Wilcoxon audits are all saved, but only the exact test is the
primary significance decision.

The runner saves long and summarized manuscript PSD tables, context- and
structure-level effects, diffusion summaries, one-second EEG trajectories,
causal phase tracking, full future metrics, power and provenance JSON, a
manuscript statistical table, and six figures in both 300-dpi PNG and vector
PDF. This experiment can confirm only that the frozen deterministic refreshed
controller improves ideal neural EEG under the toy phase-diffusion generator.
It does not establish clinical efficacy, artifact robustness, a disease model,
learned multi-step prediction, a contextual bandit, or superiority to every
possible open-loop controller.

Run the full confirmation with eight MPI ranks:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 8 python \
  experiments/ballnstick_analysis/run_ballnstick_h4_confirmation.py \
  experiment.name=ballnstick_h4_confirmation_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h4_confirmation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to `../../results/<name>/h4_confirmation/`. H4 is confirmed
only if the runner reports `H4 adaptive phase maintenance: CONFIRMED`.

## Experiment 29: H5-P0 controller-profile feasibility map

H5-P0 is the required system-identification stage before fitting an H5
machine-learning policy. It asks a narrower causal question: after frequency
and phase are handled by the frozen H3/H4 signal-processing rules, is there a
replicable EEG-observable context in which the preferred phase-tracker
bandwidth changes? A negative result means this action space does not justify
learning, even though H4 itself remains confirmed.

The new biological state axis is the fraction \(q\) of background afferents
participating in one population-shared rhythmic rate modulation. For synapse
\(j\),

\[
\lambda_j(t)=\lambda_0\left[1+m I_j
\cos\phi(t)\right],\qquad
d\phi=2\pi f\,dt+\sqrt{2D}\,dW,
\]

where \(m=0.04\), \(f\in\{9,11\}\) Hz,
\(D\in\{0.5,2.0\}\,\mathrm{rad^2/s}\), and exactly a fraction
\(q\in\{0.5,1.0\}\) has \(I_j=1\). The remaining afferents have
\(I_j=0\) and retain homogeneous Poisson drive at \(\lambda_0\). Thus \(q\)
changes population coherence rather than mean afferent rate. Every synapse has
its own Poisson event stream; only the latent rate phase is shared. The q=0.5
set is an exact nested subset of q=1 under common random numbers.

The two active actions are complete controller profiles, not mixtures or
within-episode choices: (i) the conservative 1-s/250-ms phase tracker and (ii)
the H4-confirmed responsive 0.5-s/125-ms tracker. Both use the same EEG-selected
9/11-Hz carrier, pi-relative phase target, 0.2-V/m axial field, one-second
initialization, fixed 250-ms correction horizon, ramps, and eight-second
endpoint. Sham is retained for causal and washout audits. Selecting a profile
therefore changes only the estimator/controller bandwidth.

To represent a minimal measurement limitation, the policy-facing EEG is

\[
y_k=x_k+\sigma\eta_k,\qquad
\eta_k=0.95\eta_{k-1}+\sqrt{1-0.95^2}\,\epsilon_k,
\]

with noise RMS frozen to 25% of baseline neural-EEG RMS. The context features
and causal phase tracker see \(y\); the scientific efficacy endpoint remains
the ideal neural-only EEG \(x\). This is a controlled robustness model, not a
fitted human EEG or stimulation-artifact model. Predecision observation noise,
neural history, topology, and afferent history are identical across
counterfactual profiles; independent postdecision futures estimate expected
response.

The full exploratory design has three independent structures, the complete
2-frequency by 2-diffusion by 2-shared-drive grid, and four paired futures per
profile: 24 contexts and 288 total episodes including sham. The deployable EEG
feature candidates are phase-invariant coherence, linewidth/concentration,
alpha excess, and recent resultant magnitude. A leave-one-structure-out
classifier audits whether q is observable, but no policy is fitted. The
full-information oracle is defined only from each profile's mean response over
futures. Progression requires both profiles to be optimal in replicated
contexts and structures, a mean oracle advantage of at least 0.01 log10 over
the best fixed profile, a q-by-profile response interaction, cross-structure
opportunity, and at least 0.75 future-wise winner agreement, along with causal,
continuity, rate, washout, carrier-detection, and noisy-EEG observability gates.
This is discovery, not statistical confirmation of H5.

Run the full H5-P0 map with eight MPI ranks:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 8 python \
  experiments/ballnstick_analysis/run_ballnstick_h5_controller_profile_feasibility.py \
  experiment.name=ballnstick_h5_controller_profile_feasibility_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h5_controller_profile_feasibility \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/<name>/h5_controller_profile_feasibility/`. Proceed to policy
development only if the runner reports
`Contextual controller-profile opportunity: PASSED`. Even then, H5 requires a
frozen policy and disjoint confirmation against the best fixed profile and the
H4 rule; this experiment neither trains nor tests that policy.

## Experiment 30: H5-I0 robust IAF measurement validation

H5-P0 stopped for three reasons, one of which precedes any treatment-policy
question: its raw whole-baseline periodogram selected the correct 9/11-Hz
carrier in only 66.7% of eligible noisy-EEG contexts. H5-I0 therefore applies
no stimulation and asks whether a more defensible individual-alpha-frequency
(IAF) measurement pipeline generalizes before another response map is run.
It hash-locks the negative H5-P0 outputs and retains the same mean-rate-matched
9/11-Hz by low/high-phase-diffusion by q=0.5/1.0 shared-drive grid and the same
AR(1) observation noise with RMS equal to 25% of neural-EEG RMS.

After one second of burn-in, each circuit provides 30 seconds of stimulation-
free observed EEG. Two additional one-second compatibility epochs also remain
at exactly zero field and are not used for IAF estimation. The robust spectral
pipeline divides the 30-second record into four-second Hann epochs with 50%
overlap, log-transforms each epoch PSD before averaging, fits an aperiodic
background on 6--8 and 12--14 Hz sidebands, and smooths the residual spectrum.
The two selectable estimators are (i) the maximum of that smoothed residual
and (ii) a bounded Gaussian fit to that peak. Peak prominence and agreement of
the 9/11-Hz decision across subwindows provide an explicit identifiable/not-
identifiable decision. The exact H5-P0-style 12-second raw periodogram and a
30-second raw periodogram are benchmarks only and cannot be selected; this
separates the benefit of a longer observation from the robust estimator.

Three discovery structures, each with the complete eight-context crossed
grid, rank the robust estimators using only their known simulator labels. The
complete selected method and thresholds are written to
`frozen_iaf_estimator.json` before any confirmation structure is simulated.
Six new structures then provide 48 confirmation contexts. Structure is the
independent unit; frequency, diffusion, and q are repeated measurements. The
primary validation requires at least 0.90 carrier accuracy, at least 0.80
identification coverage, at least 0.90 accuracy among identified contexts,
and at least 0.80 accuracy in every frequency, diffusion, and shared-drive
stratum. At least five of six structures must individually reach 0.75
accuracy. Subwindow agreement, recent one-second phase actionability, finite
EEG, exact zero field, and firing-rate safety are mandatory checks.

This is a computational measurement validation, not a clinical IAF study.
The continuous peak estimate is mapped to the already frozen finite 9/11-Hz
action grid, hidden generator frequency is used only for scoring, and no
machine-learning policy is fitted. Proceed to an H5-P1 stimulation-response
opportunity map only if the disjoint confirmation passes.

Run the full H5-I0 experiment with eight MPI ranks:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1

mpiexec -n 8 python \
  experiments/ballnstick_analysis/run_ballnstick_h5_iaf_measurement_validation.py \
  experiment.name=ballnstick_h5_iaf_measurement_validation_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h5_iaf_measurement_validation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/<name>/h5_iaf_measurement_validation/`. The full design contains
72 stimulation-free network episodes: 24 discovery and 48 confirmation. If no
robust estimator passes discovery, the runner stops after the first 24 rather
than spending compute on confirmation.

## Experiment 31: H5-I0b multitaper pooled-evidence measurement validation

H5-I0 correctly stopped after discovery: its best Gaussian log-Welch method
mapped 22/24 contexts to the right 9/11-Hz carrier and was correct in every
accepted context, but accepted only 17/24 contexts against the frozen 18/24
coverage requirement. H5-I0b preserves that result by hash and tests the
specific measurement hypothesis that phase-diffusion-broadened carrier
evidence is present but is poorly summarized by a single spectral maximum and
a hard vote across short windows. It changes no circuit, generator, noise, or
stimulation parameter, and every run remains at exactly zero field.

The primary spectrum is an eigenvalue-weighted five-taper DPSS estimate of the
complete 30-s noisy baseline. Aperiodic log power is fit on the same 6--8 and
12--14-Hz sidebands used in H5-I0. The estimator integrates the residual dB
evidence with a cosine kernel over 9+/-0.75 and 11+/-0.75 Hz and selects the
larger of the two action-specific scores. Six-second, 50%-overlapping
multitaper windows provide graded temporal evidence. Unlike H5-I0, window
contributions retain their magnitude: a weak contradictory interval cannot
outvote a strong carrier-consistent interval merely by count. Explicit
minimum residual evidence, score margin, and soft-support thresholds retain a
causal abstention option.

The frozen H5-I0 Gaussian estimator is a benchmark and cannot be selected.
Six new structures (48 complete crossed contexts) select between the
whole-record and robust-temporal pooled-evidence candidates. If neither passes
the discovery gate, the experiment stops. Otherwise, the complete estimator
is frozen before twelve disjoint structures provide 96 confirmation contexts.
The confirmation requires >=0.90 overall accuracy, >=0.80 decision coverage,
>=0.90 accepted accuracy, <=0.10 wrong active-selection rate, accuracy across
frequency/diffusion/shared-drive strata, structure-level consistency, phase
actionability, zero field, and firing-rate safety. Structure is the
independent unit. The paired Gaussian comparison uses an exact structure-level
sign-flip audit; neural-only EEG is saved and analyzed only to attribute noisy
measurement failures and never enters estimator selection.

The runner saves processed neural/noisy EEG for every context, full estimator
tables, representative multitaper spectra, temporal evidence, structure-level
statistics, provenance, and manuscript PNG/PDF figures. This is still a
measurement experiment: it applies no tACS and trains no machine-learning
policy.

Run the full experiment with sixteen physical-core MPI ranks on the
workstation:

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

mpiexec -n 16 --bind-to core --map-by core python \
  experiments/ballnstick_analysis/run_ballnstick_h5_multitaper_measurement_validation.py \
  experiment.name=ballnstick_h5_multitaper_measurement_validation_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h5_multitaper_measurement_validation \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/<name>/h5_multitaper_measurement_validation/`. A complete run
contains at most 144 stimulation-free network episodes: 48 discovery and 96
confirmation. A failed discovery gate is a valid stopping result and must not
be rescued by changing thresholds after inspecting these outcomes.

## Experiment 32: H5-P1 frozen-carrier controller response mapping

H5-I0b removed the discrete carrier-measurement bottleneck, but it did not
repair the two response-opportunity failures in H5-P0. H5-P1 therefore remains
a full-information system-identification experiment and does not train a
machine-learning policy. It asks whether better causal carrier measurement
reveals a practically important, repeatable context-by-controller interaction
that was obscured in H5-P0.

The state generator remains exactly the mean-rate-matched 9/11-Hz by
`D={0.5,2.0}` rad2/s by `q={0.5,1.0}` shared-afferent grid. Each context starts
with one second of burn-in and 30 seconds of stimulation-free EEG. The frozen
H5-I0b DPSS multitaper estimator selects 9 or 11 Hz using noisy EEG only. Its
evidence, margin, and temporal-support rejection rules are unchanged. A
rejected carrier, absent elevated-alpha screen, nonactionable recent phase, or
unsafe baseline rate invokes the prespecified sham fallback; hidden generator
frequency is used only to audit measurement accuracy.

Eligible contexts are replayed through four independent paired postdecision
futures. Every future compares sham, the conservative 1-s-history/250-ms-update
tracker, and the H4-confirmed responsive 0.5-s-history/125-ms-update tracker.
Both active arms use the EEG-selected carrier, a 0.2-V/m axial field, the same
pi-relative phase target and 250-ms correction horizon, and one fixed
controller profile for the complete eight-second intervention. Controller
updates use only preceding noisy EEG and preserve waveform continuity. The
efficacy endpoint is ideal-neural-EEG distance to the frozen duration-matched
population-B target; a one-second zero-field washout audits reversibility.

Six independent circuit structures provide 48 screened contexts. Frequency,
diffusion, shared-drive level, and the four future continuations are repeats;
structure remains the inferential unit. Before policy development, H5-P1
requires both profiles to win practically in multiple contexts and structures,
the post-hoc expected-outcome oracle to improve by at least 0.01 log10 over
both the best fixed profile and the frozen H4 responsive profile, at least 75%
future-wise winner agreement, and positive opportunity across structures.
Shared drive must remain observable from predecision EEG. Associations between
the predeclared phase-invariant EEG features and the paired relative response
use within-structure centering, structure-preserving permutation tests, and
Benjamini--Hochberg FDR. Any selected response feature is exploratory and must
be frozen for later policy development and disjoint policy confirmation.

The runner saves complete screening, future-level, expected-response,
controller-update, trajectory, structure, observability, feature-association,
provenance, and conclusion tables. It also saves PNG/PDF carrier-screening,
representative PSD, EEG-context, controller-response, interaction,
structure-opportunity, future-reliability, and phase-tracking figures.

Run on the workstation with sixteen physical-core MPI ranks:

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

mpiexec -n 16 --bind-to core --map-by core python \
  experiments/ballnstick_analysis/run_ballnstick_h5_response_mapping.py \
  experiment.name=ballnstick_h5_response_mapping_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h5_response_mapping \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to `../../results/<name>/h5_response_mapping/`. If all 48
contexts enroll, the run contains 576 full network episodes: one prospective
sham screen plus eleven additional paired counterfactual replays per context.
At sixteen workstation ranks, budget approximately 8--10 hours. Exclusions
reduce the active replay count. A failed H5-P1 gate is a valid stopping result
and does not establish that a learned policy is needed.

## Experiment 33: H5-P2A causal phase-tracker bias--variance discovery

H5-P1 found that the shared-afferent fraction was observable from noisy EEG,
but the two H4-derived controller profiles differed by only 0.0068 log10 on
the post-hoc oracle endpoint, their realized winner agreement was only 0.643,
and no predecision EEG feature survived the response-association gate. A
principled H5 task needs an observable variable that changes which action is
causally preferable. H5-P2A therefore tests that prerequisite at the
measurement layer before running another large stimulation experiment.

The candidate mechanism is the standard tracking bias--variance trade-off.
The latent afferent phase remains

```text
d phi(t) = 2 pi f dt + sqrt(2 D) dW(t),
```

so higher `D` demands recent measurements. In contrast, additive correlated
sensor noise makes a short phase-estimation history more variable, whereas a
long history averages that noise at the cost of lagging a changing phase.
This motivates a prespecified crossover: at low measurement noise and high
diffusion, the 0.5-s-history/125-ms-update tracker should have lower causal
phase error; at high measurement noise and low diffusion, the
1-s-history/250-ms-update tracker should have lower error. These are two fixed
controller-profile actions, not continuously tuned hyperparameters.

H5-P2A applies no electric field. Six new independent circuit structures are
crossed with carriers `{9,11}` Hz and diffusion `{0.5,2.0}` rad2/s, with the
shared rhythmic-afferent fraction fixed at `q=1.0`. This produces 24 network
episodes. After one second of burn-in, one persistent 38-second neural-EEG
record is collected. The first 30 seconds supply the already frozen H5-I0b
multitaper 9/11-Hz carrier estimate. The subsequent eight seconds are held
later in time for causal tracker evaluation. The final two one-second online
epochs remain zero-field compatibility/washout audits, so the complete neural
episode is 41 seconds and never contains stimulation.

For each neural trajectory, a unit-RMS AR(1) observation-noise path with
coefficient 0.95 is generated once and normalized from predecision samples
only. Three paired observed-EEG views are then formed without resimulating the
network:

```text
y_r(t) = x_neural(t) + r * RMS_pre[x_neural] * epsilon(t),
r in {0.25, 0.50, 0.75}.
```

Using the same `epsilon(t)` at every `r` makes noise severity the only changed
measurement variable. This is an engineering sensitivity model, not a claim
that real EEG artifacts are AR(1) or have these exact amplitudes.

Both causal trackers use only preceding noisy EEG and are compared on common
125-ms boundaries. The primary audit reference is the exact simulated latent
afferent phase plus a circular-mean neural-EEG phase offset estimated only in
the first 30 seconds. This hidden reference is never passed to either tracker.
The same-profile observed-versus-neural phase difference separately quantifies
measurement error. Importantly, every carrier accepted by the frozen
estimator is tracked; hidden correctness cannot exclude a difficult case.

The low-noise anchor is fixed at 0.25. The smallest candidate high-noise level
among `{0.50,0.75}` is frozen only if all predeclared gates pass: carrier
coverage at least 0.80; accepted carrier accuracy at least 0.90; tracker
actionability at least 0.80; at least 0.02-rad mean advantage in each expected
direction; at least 0.05-rad summed crossover contrast; each direction positive
in at least four of six structures; long-history reduction of observed-versus-
neural error; coherent neural-to-latent phase transfer; finite rate-safe EEG;
and exact zero field. This is a discovery gate and does not use a tACS outcome.

The runner saves carrier/noise tables, all causal 125-ms tracker boundaries,
context and structure summaries, the frozen candidate record with upstream
hashes, provenance, representative multitaper PSD data, and seven PNG/PDF
figures covering carrier robustness, phase-error profiles, the crossover,
measurement attribution, structure directions, PSD evidence, and a temporal
phase-error trace.

Run on the workstation with sixteen physical-core MPI ranks:

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

mpiexec -n 16 --bind-to core --map-by core python \
  experiments/ballnstick_analysis/run_ballnstick_h5_phase_tracker_tradeoff.py \
  experiment.name=ballnstick_h5_phase_tracker_tradeoff_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h5_phase_tracker_tradeoff \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/<name>/h5_phase_tracker_tradeoff/`. Budget approximately
20--35 minutes at sixteen physical-core MPI ranks based on the existing
workstation H5-I0b timing; filesystem and MPI scaling can widen this estimate.
Proceed to a new active H5-P2B response map only if the final line reports
`H5-P2A phase-tracker trade-off: PASSED`. A negative result means this tested
phase-tracking mechanism does not justify controller-profile learning.

## Experiment 34: H5-P2B active phase-tracker response mapping

H5-P2A established a stimulation-free bias--variance crossover: the
0.5-s/125-ms tracker was preferable when phase diffusion was high and sensor
noise was low, whereas the 1-s/250-ms tracker was preferable when diffusion
was low and sensor noise was higher. H5-P2B asks the necessary causal question
that H5-P2A could not answer: does that measurement-layer trade-off transfer
to a meaningful difference in the neural response to tACS? This is still
full-information system identification. It does not train a policy.

The generator is frozen at a fully shared (`q=1.0`), mean-rate-matched
rhythmic afferent drive, crossed over carrier `{9,11}` Hz and phase diffusion
`D={0.5,2.0}` rad2/s. The measurement layer is crossed independently at the
H5-P2A-selected AR(1) noise fractions `{0.25,0.50}` with coefficient 0.95.
For each structure/frequency/diffusion combination, both noise conditions use
the same neural generator seeds, future seeds, and standardized noise path;
only the path's RMS scale changes. This paired design isolates measurement
severity from biological process noise.

Each context has one second of burn-in and 30 seconds of stimulation-free
observed EEG. The frozen H5-I0b DPSS multitaper method selects 9 or 11 Hz. The
prospective screen has no access to stimulation outcomes, hidden carrier, or
hidden diffusion. Estimator abstention, failure of the frozen elevated-alpha
screen, nonactionable recent phase, or unsafe baseline rate invokes sham. An
accepted but incorrect carrier is not removed after comparison with the hidden
label. Predecision phase-invariant spectral features are augmented by causal
tracker innovations, resultants, and fast--slow phase disagreement, all
computed from the preceding noisy EEG only.

Every eligible context is replayed over four independent paired future
continuations. Each future compares exactly:

- sham;
- the conservative 1-s phase-history/250-ms-update controller; and
- the responsive 0.5-s phase-history/125-ms-update controller.

The two active controllers use the same EEG-selected carrier, 0.2-V/m axial
field, pi-relative target, one-second initialization, and 250-ms bounded
correction horizon. Each controller profile remains fixed throughout the
eight-second intervention; updates use only preceding observed EEG and the
field waveform remains continuous. A one-second zero-field washout audits
field removal. Efficacy is evaluated from ideal neural-only EEG as distance to
the frozen eight-second population-B alpha-power target, while noisy EEG is
the deployable measurement supplied to signal processing.

Six new independent circuit structures yield 48 screened contexts before any
exclusion. Four futures and the frequency/diffusion/noise conditions are
paired repeats; circuit structure remains the inferential unit. Advancement
requires both controller profiles to have practical optimal contexts across
multiple structures, fast-controller benefit in the high-diffusion/low-noise
corner, slow-controller benefit in the low-diffusion/high-noise corner, a
practical crossover and post-hoc expected oracle advantage over the best fixed
profile, at least 75% future-wise optimal-profile agreement, cross-structure
support, and alignment between causal phase-error advantage and neural tACS
response. Candidate EEG response associations use within-structure centering,
structure-preserving permutations, and Benjamini--Hochberg FDR. At least one
predeclared noisy-predecision-EEG feature must pass this exploratory mapping
gate before any policy is developed.

The runner saves prospective screening, carrier-by-noise, future-level and
expected response maps, one-second neural-EEG trajectories, every causal phase
update, structure summaries, feature associations, frozen-source hashes,
provenance, and seven PNG/PDF figures including the representative PSD and the
active controller-response crossover.

Run on the workstation with sixteen physical-core MPI ranks:

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

mpiexec -n 16 --bind-to core --map-by core python \
  experiments/ballnstick_analysis/run_ballnstick_h5_phase_tracker_response_mapping.py \
  experiment.name=ballnstick_h5_phase_tracker_response_mapping_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h5_phase_tracker_response_mapping \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/<name>/h5_phase_tracker_response_mapping/`. If all 48 contexts
enroll, the design contains 576 full network episodes: one screen/sham episode
and eleven additional action/future replays per context. Budget approximately
8--10 hours at sixteen physical-core MPI ranks, comparable to H5-P1. A failed
H5-P2B gate is evidence that the P2A estimator crossover did not create a
reliable active-control learning opportunity and is not permission to tune a
policy on hidden labels or the same outcomes.
