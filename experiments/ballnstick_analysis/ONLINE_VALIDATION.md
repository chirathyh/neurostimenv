# Online BallAndStick implementation validation

For the current scientific configuration, parameter audit, experiment designs,
and commands, see [SCIENTIFIC_EXPERIMENTS.md](SCIENTIFIC_EXPERIMENTS.md).

## Implementation findings

The causal path was developed and validated with Python 3.10.12, NEURON 8.2.3,
LFPy 2.3, and NumPy 1.26.3.

Two implementation issues motivated the online fixed-step loop:

1. Temporary `Vector.record()` objects attached after the episode's only
   `finitialize()` remained empty until `frecord_init()`. Reinitializing all
   recorders at every window would interfere with persistent LFPy spike/soma
   recorders.
2. `Vector.play()` registered after initialization likewise did not begin
   playback merely by continuing `fadvance()`. The small BallAndStick online
   path therefore assigns cached extracellular values explicitly at each
   left-step boundary.

`OnlineNeuronEnv` initializes one network per episode, caches LFPy probe
transforms and segment order, enables fast membrane current, then advances with
`fadvance()` (single process) or fixed-boundary `psolve()` (MPI).

Observation samples and spikes use `(t_start, t_stop]`. At `dt=0.0625 ms`, a
1000-ms window returns 16,000 samples; a waveform has 16,001 values because it
includes both control boundaries.

## Temperature finding

The pinned LFPy `Network.simulate()` stored `Network.celsius=36.5` but did not
set global `h.celsius`, so the effective legacy value remained NEURON's
6.3 °C default. Canonical `hh` rates are referenced to 6.3 °C. The current
configuration explicitly sets both configured and effective temperature to
6.3 °C. This preserves the old effective kinetics and exposes the model's real
semantics. A mammalian channel model is required for physiological-temperature
claims.

## Current same-seed legacy regression

With the current circuit construction and no stimulation, a 500-ms run
produced 8,001 legacy samples (including `t=0`) and 8,000 online samples.
After dropping the legacy initial sample:

```text
EEG correlation:        0.9999999999999996
relative RMS error:     2.5463503981515646e-11
maximum absolute error: 5.2479724330996715e-20 V
configured/effective:   6.3/6.3 °C
```

Reproduce with:

```bash
python experiments/ballnstick_analysis/validate_online_legacy.py \
  experiment.name=ballnstick_online_legacy_smoke \
  env=ballnstick \
  env.simulation.duration=500 \
  env.simulation.obs_win_len=250 \
  experiment.plot=false \
  experiment.tqdm=false
```

## Focused waveform regression

```bash
python -m unittest -v tests/test_online_stimulation.py
```

Coverage includes exact zero current, point-source mA-to-nA conversion,
uniform-field V/m waveforms, endpoint count, and phase-continuous action
boundaries.
