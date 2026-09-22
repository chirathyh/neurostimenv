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

## Experiment 35: H5-Dose-P0 bounded contextual dose opportunity

The negative H5-P2B result showed that selecting between the conservative and
responsive phase trackers provides too little active-response headroom for a
useful learner. H5-Dose-P0 changes the action, not the classifier: it freezes
the H4-confirmed responsive controller and asks whether field amplitude has a
reproducible context-dependent dose--response. This is a bounded mechanism and
full-information feasibility map, not a policy-training or H5 confirmation
experiment.

The biological state manipulation remains external to the actuator. Every
background synapse retains its independent conditionally Poisson event stream,
but either one half or all afferents participate in a shared mean-rate-matched
rhythm (`q={0.5,1.0}`) at modulation depth 0.04. Carrier is crossed at 9 and
11 Hz; phase diffusion is fixed at `D=0.5 rad2/s` to avoid creating an
unnecessarily large factorial design. Thus `q` changes the aggregate coherent
afferent component, approximately `q*m`, and is not an intrinsic neuronal
susceptibility or clinical diagnosis.

After one second of burn-in, 30 seconds of stimulation-free ideal neural EEG
feed the frozen H5-I0b multitaper carrier estimator and the predeclared
phenotype/phase screen. No active outcome, spike, hidden carrier, hidden `q`,
or seed-specific B simulation may determine eligibility. An ineligible or
abstaining context maps to sham. Every eligible context is replayed across four
independent paired postdecision futures under exactly four actions:

- sham (`0 V/m`);
- `0.1 V/m`;
- `0.2 V/m`; and
- `0.4 V/m`.

Every active action uses the same frozen 0.5-s phase history, 125-ms update,
250-ms bounded correction horizon, EEG-selected carrier, pi-relative phase
target, and axial uniform field. A single dose remains fixed for the entire
nine-second intervention. A 0.5-s onset and offset ramp are excluded, leaving
the central eight seconds as the endpoint matched to the frozen population-B
target. Two seconds of zero-field washout follow.

Three new circuit structures crossed with two carriers and two `q` levels give
12 contexts. Four actions by four futures yield 192 main network episodes if
all contexts enroll; a single transverse 0.4-V/m episode is an orientation
audit. Futures 1--2 select the preferred dose and futures 3--4 evaluate that
choice, then the split is reversed. This prevents an empirical oracle from
being selected and evaluated on the same stochastic outcomes. Circuit
structure is the inferential unit; frequencies, `q` levels, and futures are
paired repeats.

The primary opportunity gate requires consequential optimal-dose reversals
across contexts, structures, and both shared-drive levels; mean oracle
headroom of at least 0.01 log-distance over the strongest best fixed dose; and
at least 0.01 held-out-future advantage with cross-structure support. A
leave-one-structure-out threshold on one predeclared phase-invariant EEG
feature is retained only as an exploratory observability audit. A preferred
0.02 headroom target is advisory rather than a gate. No biological meaning is
assigned to 0.01 without later calibration.

The mechanism audit distinguishes three possibilities. Exact Fourier-vector
decomposition separates destructive interaction with the ongoing neural EEG
dipole from the induced coherent component. Total current dipole plus
representative E/I soma and distal-apical voltages/currents verify field
polarization. Spike timing and firing rates test whether spectral suppression
also reflects network timing rather than only population-current
cancellation. Exact zero field after the action and stochastic physiological
washout recovery are reported separately. The 0.4-V/m arm receives the same
causality, waveform-continuity, rate, and orientation audits as weaker doses.

Run on the workstation with sixteen physical-core MPI ranks:

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

mpiexec -n 16 --bind-to core --map-by core python \
  experiments/ballnstick_analysis/run_ballnstick_h5_dose_opportunity.py \
  experiment.name=ballnstick_h5_dose_opportunity_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h5_dose_opportunity \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/<name>/h5_dose_opportunity/`. The runner saves prospective
screening, future-level and expected dose maps, the independent-future split,
the exploratory LOSO threshold audit, EEG and current-dipole decompositions,
representative membrane traces, phase updates, orientation/safety tables,
frozen-source hashes, and PNG/PDF figures. Continue to policy development only
if the final contextual dose opportunity gate passes; otherwise retain the
negative result and consider at most one separately declared biological
sensitivity extension.

## Experiment 36: H5-K0 inhibitory-kinetics susceptibility dose opportunity

H5-Dose-P0 found a nonlinear causal dose response but essentially no
contextual headroom: 0.2 V/m was optimal in six of seven enrolled contexts,
the only 0.1-V/m preference was smaller than 0.01 log-distance and failed an
independent-future replay, and five of six partial-drive contexts failed the
prestimulation phenotype screen. H5-K0 preserves that negative experiment by
hash and performs the one bounded biological sensitivity extension declared
in advance. It remains exploratory full-information system identification;
it does not train or establish a machine-learning policy.

The afferent generator is held at the reliably screen-positive setting: full
shared drive (`q=1`), modulation depth 0.04, `D=0.5 rad2/s`, and carrier 9 or
11 Hz. The crossed circuit property is only the recurrent I-to-E inhibitory
`Exp2Syn` decay:

- short decay: `tau2=0.8*9=7.2 ms`;
- long decay: `tau2=1.2*9=10.8 ms`.

NEURON's `Exp2Syn` difference-of-exponentials kernel is normalized to unit
peak. Its conductance-time area for time constants `tau1<tau2` is

```text
t_peak = tau1*tau2/(tau2-tau1) * log(tau2/tau1)
A(tau1,tau2) = (tau2-tau1) /
  (exp(-t_peak/tau2)-exp(-t_peak/tau1)).
```

Every realized I-to-E peak weight is multiplied by
`A(0.1,9)/A(0.1,tau2_new)`. This preserves the exact conductance-time area of
each I-to-E event while changing its temporal profile. I-to-I kinetics,
connectivity, background synapses, afferent event paths, and all tACS settings
remain unchanged. Area preservation is not charge preservation because
synaptic current also depends on membrane voltage and inhibitory driving
force. The 0.8/1.2 values are exploratory sensitivity levels rather than a
validated biological or clinical range.

The timeline and action protocol are identical to H5-Dose-P0: one-second
burn-in, 30-second stimulation-free ideal neural EEG, nine seconds of tACS
with 0.5-second ramps, the central eight-second endpoint, and two-second
washout. Every eligible context receives sham and constant 0.1-, 0.2-, and
0.4-V/m axial actions under the frozen 0.5-s-history/125-ms-update fast phase
controller, EEG-selected carrier, and pi-relative target. Three new structures
crossed with two carriers and two kinetics levels yield 12 contexts; four
actions and four paired futures yield 192 outcomes if all contexts enroll.

Before stimulation outcomes are interpreted, the runner requires both
kinetics states to remain screen-positive, paired baseline alpha differences
no larger than 0.10 log10, E/I rate differences no larger than 0.75/1.50 Hz,
and leave-one-structure-out discrimination from a frozen small set of
phase-invariant prestimulation EEG features. Stimulation opportunity requires
reciprocal state-level optimal-dose margins of at least 0.01, practical
nonfixed contexts across at least two structures, at least 0.01 mean oracle
headroom over the strongest best fixed dose, and at least 0.01 advantage when
the preferred dose is selected on futures 1--2 and evaluated on futures 3--4
(and vice versa). Failure is a stopping result, not permission to tune these
kinetics values on the same outcomes.

Run on the workstation with sixteen physical-core MPI ranks:

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

mpiexec -n 16 --bind-to core --map-by core python \
  experiments/ballnstick_analysis/run_ballnstick_h5_inhibitory_kinetics_dose_opportunity.py \
  experiment.name=ballnstick_h5_inhibitory_kinetics_dose_opportunity_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h5_inhibitory_kinetics_dose_opportunity \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/<name>/h5_inhibitory_kinetics_dose_opportunity/`. In addition
to the dose, future-split, current, membrane, spike, rate, phase, provenance,
and safety outputs from H5-Dose-P0, the experiment saves paired baseline
kinetics audits, LOSO kinetics-state predictions, analytic conductance-area
metadata, and prestimulation PSD figures for both kinetics states.

## Experiment 37: H5-O0 EEG-observable orientation--montage opportunity

### Question

The dose and inhibitory-kinetics branches did not produce enough repeatable
contextual headroom to justify learning a policy. H5-O0 therefore asks a more
fundamental, bounded question: does a variable that enters the weak-field
coupling directly create a useful context--action interaction?

The crossed context is the common somatodendritic orientation of the toy
population in head coordinates, either 0 or 60 degrees. The two active actions
are equal-magnitude (0.2 V/m) head-frame field profiles at 0 or 60 degrees.
They stand in for two precomputed electrode-montage/current solutions. H5-O0
does not optimize scalp currents and its field values remain tissue-level
simulator settings rather than clinical dose prescriptions.

For a unit somatodendritic axis `n(theta)` and a unit montage-field direction
`e(phi)`, the first-order polarization scale is

```text
E_axial = 0.2 V/m * |n(theta) dot e(phi)|.
```

The prespecified projection matrix is

```text
                       field 0 deg   field 60 deg
orientation 0 deg          1.0            0.5
orientation 60 deg         0.5            1.0
```

This is a positive-control source of response heterogeneity, not an A/B
disease manipulation. A remains the full-shared-drive rhythmic state with
modulation depth 0.04, `D=0.5 rad2/s`, and carrier 9 or 11 Hz. B remains the
homogeneous, mean-rate-matched afferent reference. Recurrent circuitry,
synaptic kinetics, cells, phase controller, frequency estimator, amplitude,
and relative phase are frozen.

### Coordinate and EEG model

Morphology and synapse locations remain in one canonical local-column frame.
This is important because the BallAndStick synapse-placement rule depends on
local z; physically rotating LFPy cells would otherwise change which segments
receive recurrent and background synapses. With the proper rotation
`R_y(theta)` from local column to head coordinates, the implementation uses

```text
p_head(t) = R_y(theta) p_local(t)
E_local(t) = R_y(theta)^T E_head(t).
```

The first expression is used by both the causal online phase tracker and the
offline three-channel FourSphere EEG. The second is the reciprocal field
transform used by NEURON. Existing configurations omit the optional dipole
rotation, preserving the H1--H4 EEG path exactly. The code verifies that a
matched pair of coordinate-frame orientations retains identical spike trains,
rates, connectivity/event realizations, and local dipole norm.

The ideal three-sensor array contains a vertex sensor and symmetric right/left
x--z sensors on the 90-mm outer sphere. The causal controller uses the sensor
with the largest prestimulation alpha power. Phase-invariant carrier-band
power fractions across all three sensors form the deployable context audit;
hidden orientation is used only as an evaluation label.

### Protocol and comparisons

Each episode uses a one-second burn-in, 30-second stimulation-free baseline,
nine-second intervention with 0.5-second onset/offset ramps, the central eight
seconds as the endpoint, and a two-second washout. Three new circuit structures
are crossed with two carriers and two orientations, yielding 12 A contexts.
Three new reference structures are simulated at both orientations before any
active outcome to calibrate orientation-specific homogeneous-B targets.

Every eligible A context receives exactly three paired arms over four
independent postdecision futures:

- sham (0 V/m);
- the 0-degree field profile at 0.2 V/m;
- the 60-degree field profile at 0.2 V/m.

Both active arms use the frozen H4-confirmed 0.5-second-history/125-ms-update
phase-maintenance controller, the EEG-selected 9/11-Hz carrier, and a
pi-relative target. This gives 144 action--future outcomes, in addition to the
12 prospective screening episodes and six B-reference calibration episodes.
Futures are paired across all three arms. The preferred montage is selected on
futures 1--2 and evaluated on futures 3--4, and vice versa.

The full gate requires carrier and recent-phase actionability; LOSO orientation
discrimination from prestimulation phase-invariant EEG topography; complete
paired action/future data; matched-profile benefit in both orientations; use
of both profiles by the expected-outcome oracle; at least 0.01 log10 distance
headroom over the best fixed profile; at least 0.01 log10 independent-future
benefit with cross-structure support; projection-ordered representative
cellular polarization; rate safety; causal phase updates; continuous fields;
and exact field removal. The oracle is post hoc and not a deployable policy.
Passing H5-O0 would justify a disjoint policy-development study, not establish
H5.

### Full workstation command

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

mpiexec -n 16 --bind-to core --map-by core python \
  experiments/ballnstick_analysis/run_ballnstick_h5_montage_orientation_opportunity.py \
  experiment.name=ballnstick_h5_montage_orientation_opportunity_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h5_montage_orientation_opportunity \
  analysis.source_h5k0.result_dir=../../results/ballnstick_h5_inhibitory_kinetics_dose_opportunity_full_v2/h5_inhibitory_kinetics_dose_opportunity \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/ballnstick_h5_montage_orientation_opportunity_full/h5_montage_orientation_opportunity/`.
The runner saves prospective screening, paired orientation-invariance audits,
multichannel A/B and active PSDs, topographies, the complete action--future
map, matched/mismatched crossovers, future-split validation, representative
cellular polarization, safety/phase audits, provenance, and seven figures in
both PNG and PDF.

## Experiment 38: H5-O1D noisy-EEG montage-policy development

### Question and scientific scope

H5-O0 established a large, independently replicated full-information
opportunity: at fixed amplitude, carrier, phase controller, and network state,
the tissue-field profile aligned more closely with the modeled population axis
than the mismatched profile and improved the ideal neural-EEG endpoint. H5-O1D
asks the next distinct question: can a small model infer which of those two
precomputed profiles to use from causal, noisy, stimulation-free EEG on an
unseen circuit structure?

This is an intentionally bounded policy-development experiment. Population
orientation is a toy source of subject-level electric-field susceptibility;
the actions represent two fixed candidate montage/current solutions at the
modeled tissue. The study does not infer a realistic cortical normal, solve a
head-model current-optimization problem, include tACS recording artifacts, or
claim a clinical benefit.

### Frozen mechanism and new context grid

The positive H5-O0 outputs are hash-locked before any new simulation. The A
generator remains a full-shared-drive, mean-rate-matched Poisson state with
modulation depth 0.04, phase diffusion `D=0.5 rad2/s`, and a 9- or 11-Hz
carrier. B remains the homogeneous afferent reference. The two active actions
remain 0- and 60-degree head-frame field profiles, both with amplitude
0.2 V/m, the EEG-selected carrier, pi-relative phase, and the H4-confirmed
0.5-second-history/125-ms-update controller.

H5-O1D adds only intermediate axes at 20 and 40 degrees to the frozen H5-O0
endpoints at 0 and 60 degrees. The mechanistic projection is therefore graded:

```text
E_axial(theta, phi) = 0.2 V/m * |cos(theta - phi)|,
theta in {0, 20, 40, 60} deg, phi in {0, 60} deg.
```

Four new structures are crossed with two carriers and four orientations,
giving 32 contexts. Three additional structures calibrate homogeneous-B
targets at the four orientations before active outcomes. Each eligible A
context receives sham and both active profiles over four paired postdecision
futures. If all contexts enroll, this is 32 screening episodes, 12 reference
episodes, and

```text
32 contexts * 3 arms * 4 futures = 384 action--future outcomes.
```

Each episode has a one-second burn-in, a 30-second stimulation-free baseline,
nine seconds of intervention with 0.5-second onset/offset ramps, the central
eight seconds as the efficacy endpoint, and two seconds of washout. Within a
context and future, all arms have the same predecision neural trajectory,
standardized observation-noise realization, private Poisson continuation, and
latent-phase continuation. Only the stimulation profile differs.

### Observation and learned rule

The three-sensor ideal neural EEG from H5-O0 is augmented with frozen moderate
AR(1) sensor noise at 0.25 of baseline neural RMS and coefficient 0.95. The
vertex and two side-sensor noise paths are independent, but use a common
absolute scale estimated from the vertex prestimulation signal. Scaling is
fixed at the decision boundary. Ideal EEG remains available only for the
efficacy endpoint and attribution audits.

The carrier is selected by the frozen H5-I0b multitaper estimator. The only
learned inputs are two phase-invariant, noisy-predecision-EEG features at that
carrier:

```text
x1 = carrier-band power fraction at the vertex,
x2 = right carrier-band power fraction - left carrier-band power fraction.
```

For the active profiles `z` and `60`, define the paired response label

```text
d(x) = E[L(z) - L(60) | x],
```

where `L` is ideal neural-EEG log10 distance to the frozen orientation-specific
B target. A positive prediction selects the 60-degree profile and a nonpositive
prediction selects the z profile. One ridge-linear model with fixed penalty
1.0 estimates this contrast. This is supervised offline policy development
from full paired action outcomes; it is not an online contextual-bandit trial.

### Leakage-resistant evaluation and comparators

The primary analysis leaves out one complete circuit structure. Within each
fold, response labels are learned from futures 1--2 of the other structures;
the selected profile is scored only on futures 3--4 of the held-out structure.
The reverse split is a prespecified robustness audit. A final candidate is fit
only after evaluation, using futures 1--2 across all development structures,
and is saved for possible disjoint confirmation without claiming performance
from that fitted object.

The learned policy is compared with:

- sham;
- uniform random selection over the two active profiles;
- the best fixed active profile learned inside each training fold;
- the frozen H5-O0 analytical topography-threshold rule; and
- a post-hoc full-information oracle, used only as an upper-bound audit.

A structure-preserving permutation shuffles EEG context rows only within each
structure while retaining the paired response map. This tests whether policy
benefit depends on the correct context--response association rather than
structure identity or global action imbalance.

The gate requires noisy-EEG carrier coverage and accuracy, profile
observability, complete pairing, use of both actions, at least 0.01 log10 mean
advantage over the best fixed profile, positive benefit in at least 75% of
structures, small oracle regret, benefit over uniform random selection,
context-shuffle specificity, noninferiority to the frozen analytical rule,
reverse-split robustness, rates, causality, waveform continuity, and field
removal. A pass freezes a candidate for new-seed confirmation; a failure stops
that branch without post-hoc feature or threshold changes.

### Full workstation command

```bash
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

mpiexec -n 16 --bind-to core --map-by core python \
  experiments/ballnstick_analysis/run_ballnstick_h5_montage_policy_development.py \
  experiment.name=ballnstick_h5_montage_policy_development_full \
  experiment.seed=1 \
  env=ballnstick \
  analysis=ballnstick_h5_montage_policy_development \
  analysis.source_h5o0.result_dir=../../results/ballnstick_h5_montage_orientation_opportunity_full/h5_montage_orientation_opportunity \
  env.simulation.obs_win_len=1000 \
  experiment.plot=true \
  experiment.tqdm=false
```

Results are written to
`../../results/ballnstick_h5_montage_policy_development_full/h5_montage_policy_development/`.
The runner saves the source hashes, B targets, screening table, noisy and
neural multichannel PSDs, orientation invariance, LOSO observability, complete
action--future map, expected response map, held-out policy evaluation,
structure-preserving shuffle, fold models, frozen candidate, checks,
provenance, and seven PNG/PDF manuscript-oriented figures.

## Experiment 39: H5-O1M spatial measurement and paired-reference audit

### Why this experiment, rather than another policy fit?

The completed H5-O1D result is negative and is preserved by seven file hashes.
Its primary learned advantage was approximately 0.00177 log-distance units,
below the declared 0.01 engineering margin. Ideal-EEG orientation decoding was
much stronger than noisy decoding, but even its in-sample full-information
oracle had only approximately 0.00623 headroom. Better measurements therefore
cannot be assumed to establish useful treatment personalization.

This small, stimulation-free study addresses two measurement weaknesses:
independent sensor noise biases normalized auto-power topography, and the
previous screen indexed B targets by the simulated (hidden) orientation.
It also removes stochastic differences between B reference trajectories
previously run independently at different orientations. No existing result,
H1--H4 runner, network mechanism, treatment dose, or success criterion is
rewritten. H5-O1M is a new measurement study, not a rescue of H5-O1D.

### Design and timing

The 40-cell canonical local circuit, synapses, mean afferent rates and
6.3-degree canonical HH kinetics are unchanged. Rhythmic A has q=1,
modulation depth 0.04, D=0.5 rad²/s, and a 9- or 11-Hz carrier. Homogeneous B
has zero rhythmic modulation at the same expected mean afferent rate. These
are toy spectral states, not validated disease and health labels.

| Stage | Independent structures | Neural conditions per structure | Neural episodes |
| --- | ---: | --- | ---: |
| Population-B calibration | 3 | B | 3 |
| Disjoint measurement evaluation | 4 | B, A9, A11 | 12 |
| Total | 7 | | 15 |

Every persistent episode is **1 s burn-in + 30 s baseline + 9 s later sham +
2 s final sham**. The inherited internal epoch key `stimulation` contains
zero-field sham only. The central eight seconds of that nine-second block
are retained solely for duration-matched B calibration audits. All deployable
features and screens use the preceding 30 seconds, never those later samples.
One B trajectory is needed per structure because a homogeneous generator has
no 9/11-Hz carrier. Matched A/B contexts share structure, private-drive,
phase-path and future seeds; their afferent rate functions differ as intended.

Each canonical dipole trajectory is projected offline to 0, 20, 40 and
60 degrees. This is exact for the zero-field coordinate-transform model: the
local neural dynamics do not depend on orientation. Three independent
three-sensor observation-noise paths are shared across these rotations. This
gives 96 A evaluation views and 48 B evaluation views, plus 36 calibration
views. Neural-only and noisy attribution views are analyzed with both fixed
estimators, giving 720 measurement rows. These rows are **not 720 independent
circuits**; the inferential unit is the four evaluation structures.

The three sensors have independent AR(1) noise with coefficient 0.95 and
equal absolute scale, fixed to 0.25 of the prestimulation vertex neural RMS.
As in H5-O1D this does not mean identical relative noise at the weaker side
sensors. The unit paths are normalized before the decision boundary and the
scale remains fixed afterwards. Original unit-noise vectors, seeds and hashes
are saved, and the same paths are reused for orientation comparisons.

### Fixed estimators and the source of additional information

With the known three-sensor FourSphere leadfield H, an oriented single local
column gives

```text
y(t) = g(theta) u(t) + n(t),
g(theta) = H [sin(theta), 0, cos(theta)]^T.
```

Integrate the real cross-spectral matrix over 8--12 Hz. Spectra use 4-second
Hann windows with 50% overlap on the 30-second baseline (0.25-Hz bin spacing;
the taper still determines effective resolution). Under independent sensor
noise of equal spectrum,

```text
E[S_alpha] = P_u g(theta) g(theta)^T + N_alpha I.
```

Noise adds to each auto-spectrum. In expectation it does not add to
off-diagonal cross-spectra. The common neural-current source produces
cross-channel covariance, including the polarity information that is lost by
using only squared amplitudes. This is a volume-conducted source pattern,
**not evidence for connectivity between three neural populations**.

The primary estimator `equal_noise_csd_rank1` fits the leading eigenvector to
known normalized leadfield patterns on a frozen 0--60-degree grid in 0.25-degree
steps. It estimates global neural alpha energy as

```text
P_hat = max(lambda_max - mean(lambda_2, lambda_3), 0).
```

It receives no true orientation, condition, hidden carrier, ideal EEG, or
actual noise vector. It estimates the noise floor from the lower eigenvalues.
An estimated signal fraction below 0.10 or a leadfield-pattern residual above
0.25 causes abstention. These are predeclared engineering tolerances, not
physiologically calibrated diagnostic cutoffs.

The comparator `auto_power_template` fits the same known forward-model
templates using only normalized diagonal powers. It uses the same 8--12-Hz
matrix and is **not an exact rerun** of H5-O1D's carrier-band ridge policy.
Neither estimator is selected or fitted on new neural outcomes. The nearest
field-profile label, separated at 30 degrees, is a geometry audit only; the
best physiological intervention need not be the geometrically nearest field.

### EEG-only screening and B calibration

Let G(theta)=sum_s g_s(theta)^2. Remove the estimated geometry gain using

```text
z = log10(P_hat) - log10(G(theta_hat)/G(0)).
```

For the auto-power comparator, total alpha power replaces P_hat. Three
calibration B structures determine a single population mean z_B for each
estimator and attribution view, averaging repeats within each structure first.
Freeze these targets before simulating the evaluation structures. The
phenotype screen is `z - z_B >= 0.05`; it never indexes a threshold by the
hidden orientation, and never uses the matched evaluation B as a target.

Carrier identification retains the frozen H5-I0b multitaper pooled-evidence
9/11-Hz method on the vertex. Recent phase actionability is assessed from the
last second only. Treatment eligibility would require the phenotype screen,
spatial confidence, carrier acceptance, and phase confidence together. Failed
confidence means a sham fallback. Since this is a measurement study all
episodes actually remain sham, including the eligible ones. Accepted incorrect
carriers remain errors, not retrospective exclusions. B specificity is scored
on the phenotype decision separately, so carrier abstention cannot inflate it.

### Gates, figures, and next decision

Predeclared descriptive gates require observed-EEG geometry balanced accuracy
at least 0.80, mean angle error at most 10 degrees, acceptance coverage at least
0.80, accepted balanced accuracy at least 0.80, and improvement over the
auto-power comparator. Also require carrier coverage 0.80, accepted carrier
accuracy 0.90, phase actionability 0.80, A sensitivity 0.60, B specificity 0.80,
phenotype balanced accuracy 0.75, stable decisions across noise repeats,
paired reference invariance, finite EEG, rate bounds and zero field.

Report metrics per structure and per observed/neural attribution view. A
structure bootstrap and exact paired sign-flip test describe uncertainty in
the spatial-accuracy difference. Four structures imply a minimum one-sided
exact p-value of 1/16=0.0625: this is deliberately not a powered confirmatory
study, and statistical significance is not a gate. No electrode, orientation,
window, cutoff or estimator search follows inspection of the outcomes.

Saved PNG/PDF figures show separate A9/A11/B PSDs; estimated versus true
orientation; structure-level spatial error and accuracy; EEG-only phenotype
screening; paired B calibration; sensitivity/specificity; and vertex/side-sensor
PSD noise attribution. CSV/JSON tables include every view, structure summaries,
inference, full configuration and source provenance. Compressed artifacts retain
canonical dipoles, processed neural/noisy EEG and original noise vectors
(allow roughly 1 GB disk space for the full study). `run_complete.json` is
written last, after all figures, tables and the conclusion; it records runtime
and hashes the top-level artifacts.

A pass permits only a **small new paired-response reassessment** with the
corrected measurement/target. It does not establish extra oracle headroom or
H5. That later study must retain the strongest fixed montage and the analytical
geometry-based comparator, use independent-future selection/evaluation, and
still show a practically useful context-by-action interaction. A failure
should stop the current noisy-montage branch, not trigger a large ML run.
Even a pass relies on a known source location/head model, one oriented source,
and independent equal-spectrum sensor noise. Multiple sources, correlated
noise, unknown anatomy, and tACS recording artifacts remain outside scope.

### Workstation full command

Only the completed H5-O1D result folder is read at runtime. Its embedded
upstream provenance is retained; no older result-directory overrides are
needed. Run from the repository root after activating its parent environment:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

time mpiexec -n 16 --bind-to core --map-by core python \
  experiments/ballnstick_analysis/run_ballnstick_h5_spatial_measurement_audit.py \
  experiment.name=ballnstick_h5_spatial_measurement_audit_full \
  experiment.seed=1 env=ballnstick \
  analysis=ballnstick_h5_spatial_measurement_audit \
  analysis.source_h5o1d.result_dir=../../results/ballnstick_h5_montage_policy_development_full/h5_montage_policy_development \
  env.simulation.obs_win_len=1000 experiment.plot=true experiment.tqdm=false
```

Output: `../../results/ballnstick_h5_spatial_measurement_audit_full/h5_spatial_measurement_audit/`.
The full command runs 15 episodes; it is not a smoke. Additional MPI ranks
parallelize cells within a trajectory, not the offline noise views. With only
40 cells, 16 is a reasonable starting point; 32 is not guaranteed faster.
Reuse neither an occupied experiment name nor a smoke output as a scientific
source. The runner refuses to overwrite an existing populated result folder.

### Local engineering smoke

```bash
python -m unittest -v tests.test_h5_spatial_measurement_audit

mpiexec -n 2 python \
  experiments/ballnstick_analysis/run_ballnstick_h5_spatial_measurement_audit.py \
  experiment.name=ballnstick_h5_spatial_measurement_audit_smoke \
  experiment.seed=1 env=ballnstick \
  analysis=ballnstick_h5_spatial_measurement_audit \
  analysis.smoke_test=true \
  analysis.timeline.baseline_steps=4 \
  analysis.timeline.stimulation_steps=2 \
  analysis.timeline.washout_steps=1 \
  analysis.measurement_design.noise_repeats=1 \
  env.simulation.obs_win_len=1000 experiment.plot=true experiment.tqdm=false
```

This executes four shortened zero-field episodes and exercises the complete
artifact pipeline. Its conclusion is always `SMOKE COMPLETED`, never a
positive measurement gate or permission for policy confirmation. Scientific
criteria are not relaxed to make the short smoke pass.

## Experiment 40: H5-O1S — null-calibrated rhythm-presence screening

### Motivation and the one change being tested

H5-O1M completed correctly but failed its phenotype-specificity gate. The
equal-noise cross-spectral estimator substantially improved spatial decoding;
the noisy B specificity was only 35/48 = 0.729. The inherited criterion,
alpha power at least 0.05 log10 above a small calibration population's mean,
is a target-excess criterion, not a test that a rhythmic component is present.
Choosing between 9 and 11 Hz also does not test whether either oscillation
exists. Some homogeneous B trajectories have elevated stochastic alpha power.

This follow-up preserves that negative result. It adds a **B-only calibrated
rhythm-presence prerequisite**, using the already implemented multitaper
aperiodic-adjusted evidence. It does not choose a cutoff from the apparent
gap in the old A/B scores, lower the old specificity requirement, replace the
population-B efficacy target, fit a montage policy, or apply any tACS.

Everything else remains frozen: the 40-cell canonical-HH model at 6.3 C,
9/11-Hz A generator, modulation depth 0.04, full rhythmic-afferent fraction,
phase diffusion 0.5 rad2/s, homogeneous mean-rate-matched B, four orientations
(0/20/40/60 degrees), known-head three-sensor forward model, equal-absolute
independent AR(1) noise (coefficient 0.95, vertex baseline RMS fraction 0.25),
and the fixed cross-spectral and multitaper estimators. Ideal EEG remains an
attribution audit; deployable measurements use observed noisy EEG only.

### Prospective protocol and calibration

Run 19 new homogeneous-B calibration structures. Each produces one persistent
42-second zero-field episode: 1-second burn-in, 30-second baseline, 9-second
continued sham, and 2-second final sham. The central eight seconds of the
nine-second interval remain a duration-matched measurement audit, not an input
to screening. Project the same canonical dipole offline to all four angles
and apply three paired noise realizations. Normalize noise from baseline only
and retain the original unit-noise vectors and their hashes.

For each observed baseline, the frozen multitaper method produces pooled
residual evidence around each candidate carrier. Define

$$s(X)=\max_{f\in\{9,11\}} E_f(X),\qquad
  M_i=\max_{\text{prespecified angle/noise views}} s(X_i).$$

Here $E_f$ is the existing aperiodic-adjusted spectral evidence in dB, not
raw alpha power or a hidden-frequency label. Across the $n=19$ calibration
structures, freeze the ordered cutoff

$$r=\lceil(n+1)(1-0.05)\rceil=19,\qquad c=M_{(r)}.$$

Rhythm presence requires $s(X)>c$, with ties negative. No A episode or active
outcome participates in this fit. The grouped rank construction bounds the
marginal probability that a new exchangeable B structure has any score above
the cutoff by 1/20. This averages over the random calibration set and new
structure: it does **not** guarantee 95% conditional specificity for whichever
cutoff happens to be fitted. This is why a separate validation set is needed.
The bound applies to this finite view grid and toy B population, not arbitrary
subjects, sessions, correlated real-world noise, or new recording geometries.

The full screen is

$$\mathrm{phenotype}(X)=
 [s(X)>c]\ \land\ [\widehat{\log_{10}P_\alpha}(X)-\mu_B\ge0.05].$$

The cross-spectral alpha estimate and the H5-O1M population mean $\mu_B$ are
unchanged. Treatment eligibility additionally requires spatial confidence,
accepted carrier identification, and recent causal phase actionability.
Rejection implies a future sham fallback. Specificity is scored **before**
these confidence exclusions: an estimator abstention cannot conceal a
phenotype false positive.

The runner writes and hashes `frozen_rhythm_presence_rule.json` before
simulating any validation trajectory. It then evaluates 30 disjoint structures,
each with matched B, A9 and A11 episodes. One afferent history per structure
is paired across those conditions; noise realizations are repeated measurement
views, not new independent circuit structures. The exact workload is

$$19+30\times3=109\text{ full zero-field episodes}.$$

Validation provides 360 B views and 720 A views, but only 30 independent
structure units. All 49 structures and their seed namespaces are new relative
to H5-O1M. The source result folder alone is needed at runtime; its embedded
older provenance is retained without reopening earlier result folders.

### Prespecified inference and pass/fail interpretation

The primary outcome for each held-out B structure is whether **all** its
12 angle/noise views are phenotype-negative. Let $K$ count clean structures.
Test $H_0:p_{\rm clean}\le0.80$ against $p_{\rm clean}>0.80$, with an exact
one-sided binomial test at 0.05. For 30 structures, the rejection region is
$K\ge28$: exact test size is 0.04418 and anticipated power is 0.81218 if
$p_{\rm clean}=0.95$. Thirty is a prospective choice, not a sample-size
adjustment after the results. Power depends on the unverified 0.95 assumption.
The runner saves exact Clopper–Pearson intervals and a structure-bootstrap
description of sensitivity, specificity, balanced accuracy and coverage.

Mandatory additional guards retain A phenotype sensitivity and treatment
coverage of at least 0.60 **for each carrier**, mean B-view specificity 0.80,
screen balanced accuracy 0.75, spatial balanced accuracy 0.80, angle error at
most 10 degrees, carrier coverage 0.80, accepted carrier accuracy 0.90, recent
phase actionability 0.80, finite measurements, unchanged noise scaling,
geometry-invariant B audit, paired firing-rate bounds and exact zero field.
These are separate predeclared guards, not alternative significant endpoints.
Rejecting every circuit can therefore pass specificity but cannot pass the
experiment. No diagnostic cutoff is selected using evaluation sensitivity.

The primary test is about a stronger endpoint than mean view specificity:
repeatability of a negative screen across all prespecified views of a B
structure. There is only one neural history per structure, so the study does
not validate within-person longitudinal screening. It also retains favorable
single-source, known-anatomy and equal-spectrum independent-noise assumptions.

A pass permits a **small disjoint active-response reassessment**, keeping the
strongest fixed montage and analytical EEG geometry rule as comparators.
It does not establish treatment crossovers, ML superiority, clinical screening,
or H5. A failure is retained: do not move the cutoff or remove difficult A/B
structures after inspection. In particular, better anatomy decoding need not
increase the small action-selection headroom previously observed in H5-O1D.

### Saved artifacts

The output includes per-view and per-structure CSVs, the B calibration maxima,
frozen rule and original B target, exact primary inference and power plan,
source/configuration hashes, runtime and a final completion manifest. Ten
PNG/PDF figure pairs show A9/A11/B PSDs, neural/noisy side-sensor PSDs, spatial
accuracy, alpha-only diagnostics, the joint alpha/rhythm screen, B specificity
by structure, and calibration versus held-out evidence distributions.
Figures 01–07 preserve the inherited alpha-only/measurement diagnostics;
Figures 08–10 explicitly show the new screening test. Canonical dipoles,
processed EEG and original unit-noise arrays are saved for reproducibility.
Allow approximately 8 GB of free disk space.

### Workstation full command

Run from the repository root, with the parent virtual environment activated:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

time mpiexec -n 16 --bind-to core --map-by core python \
  experiments/ballnstick_analysis/run_ballnstick_h5_rhythm_screening_validation.py \
  experiment.name=ballnstick_h5_rhythm_screening_validation_full \
  experiment.seed=1 env=ballnstick \
  analysis=ballnstick_h5_rhythm_screening_validation \
  analysis.source_h5o1m.result_dir=../../results/ballnstick_h5_spatial_measurement_audit_full/h5_spatial_measurement_audit \
  env.simulation.obs_win_len=1000 experiment.plot=true experiment.tqdm=false
```

Output: `../../results/ballnstick_h5_rhythm_screening_validation_full/h5_rhythm_screening_validation/`.
This is the full 109-episode study, not a smoke. Scaling the previous 15-episode
workstation measurement audit (974 seconds) suggests about two hours before
additional processing/I/O variation; budget roughly 2–3 hours at 16 ranks.
This is an estimate, not a benchmark of the new full run. Offline spectral
analysis is done on rank zero, so increasing MPI ranks does not accelerate
every part of the workload. The runner refuses to overwrite populated outputs.

### Local engineering tests and smoke

```bash
python -m unittest -v tests.test_h5_rhythm_screening_validation

mpiexec -n 2 python \
  experiments/ballnstick_analysis/run_ballnstick_h5_rhythm_screening_validation.py \
  experiment.name=ballnstick_h5_rhythm_screening_validation_smoke \
  experiment.seed=1 env=ballnstick \
  analysis=ballnstick_h5_rhythm_screening_validation \
  analysis.smoke_test=true \
  analysis.timeline.baseline_steps=4 \
  analysis.timeline.stimulation_steps=2 \
  analysis.timeline.washout_steps=1 \
  analysis.measurement_design.noise_repeats=1 \
  env.simulation.obs_win_len=1000 experiment.plot=true experiment.tqdm=false
```

The smoke runs four shortened zero-field episodes and generates all artifacts.
One calibration structure cannot supply a finite 95% grouped-rank cutoff:
the correct smoke behavior is an explicit abstain-all rule, not a loosened
scientific threshold. Its conclusion is always `SMOKE COMPLETED`. Unit tests
separately exercise finite 19-structure calibration, strict ties, B-only/noisy-
only fitting, hidden-label exclusion, confidence versus phenotype rejection,
the exact power calculation, all-reject failure, and full-design validation.

## H5-O2: frozen-screen active montage reassessment

Entry point: `run_ballnstick_h5_screened_montage_response_mapping.py`.
Configuration: `analysis=ballnstick_h5_screened_montage_response_mapping`.
Detailed rationale, equations, limitations and workstation command are in
[H5_O2_PROTOCOL.md](H5_O2_PROTOCOL.md).

The completed H5-O1S screen is frozen by hash; only its result directory is
required. Use three new structures, 9/11 Hz, orientations 0/20/40/60 degrees,
and four paired futures, comparing sham and the two frozen 0.2-V/m field
profiles with the same H4 fast phase controller. At most 288 episodes are run.
The screen and controller use noisy EEG; efficacy uses the frozen neural-only
B target. There is no new reference calibration, learned model or generator
change. First establish analytical EEG-geometry control, then ask whether any
residual benefit survives independent-future evaluation beyond that strong
rule and the best fixed profile. Three structures are discovery only.

Local integration smoke (12 shortened episodes if both contexts enroll):

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg HYDRA_FULL_ERROR=1
mpiexec -n 2 python experiments/ballnstick_analysis/run_ballnstick_h5_screened_montage_response_mapping.py \
  experiment.name=ballnstick_h5_screened_montage_response_mapping_smoke \
  experiment.seed=1 env=ballnstick \
  analysis=ballnstick_h5_screened_montage_response_mapping \
  analysis.smoke_test=true analysis.smoke_force_eligible=true \
  analysis.smoke_context_limit=2 analysis.crossed_design.n_future_continuations=2 \
  analysis.timeline.baseline_steps=4 analysis.timeline.stimulation_steps=2 \
  analysis.timeline.washout_steps=1 env.simulation.obs_win_len=1000 \
  experiment.plot=true experiment.tqdm=false
```

The integration smoke exercises artifacts and both profiles; forcing can
never count as scientific enrollment and never permits a full-run pass.
Keep source thresholds unchanged. The independent-future synthetic tests
also exercise response replication, a misleading in-sample oracle, geometry
success without residual H5 opportunity, and equal structure weighting.

## CF0 and FS0: independent measurement and cellular qualification

See [CF0_FS0_PROTOCOL.md](CF0_FS0_PROTOCOL.md) for equations, gates, counts,
artifacts, and workstation/local commands. CF0 tests continuous-alpha carrier
measurement on 15 discovery and (only after a pass) 15 disjoint qualification
episodes. FS0 tests an opt-in tonic conductance in 35 isolated-cell cases.
Neither changes H1--H4 or runs L23Net. Neither establishes H5 or authorizes a
network/ML experiment after a failed qualification.
