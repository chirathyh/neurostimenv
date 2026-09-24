# L23Net sinusoidal uniform-field tACS: phase-one integration

## Scope

This phase makes the existing causal BallAndStick actuator available to
L23Net. It addresses two implementation criticisms in the reviewer report:

1. active tACS is a sampled sinusoid, not a square/biphasic pulse train; and
2. the neural actuator is a spatially uniform tissue electric field, not a
   microscopic point source whose current is presented as a scalp dose.

The historical `NeuronEnv` point-source path remains untouched for regression.
New L23Net field work should use `OnlineNeuronEnv`. The smoke test below is a
technical verification only; its 300-ms trajectory cannot test steady state,
entrainment, efficacy, plasticity, or a clinical protocol.

## Online field updates

`OnlineNeuronEnv.step_online()` accepts a new action after each completed
observation window without rebuilding the network or calling `finitialize()`.
Amplitude, frequency, phase, signed DC offset, montage, and field direction can
therefore be changed while membrane, channel, synaptic, recurrent, and event-
queue state persist. Within each window the extracellular potential is updated
at every fixed integration step. This is simulation-time closed-loop control;
it does not yet expose a wall-clock command channel into an already running PBS
job. The feedback cadence is the chosen observation-window duration.

## Model equations

The scalar field waveform is

\[
E(t)=w(t)\left[E_{\mathrm{DC}}+E_{\mathrm{AC}}
\sin\!\left(2\pi f(t-t_0)+\phi_0\right)\right],
\]

where field amplitudes are in V/m at the modeled tissue, `w(t)` is an optional
raised-cosine envelope, `f` is in Hz, and time in the sine is in seconds. For a
uniform field direction `d` with unit norm,

\[
\mathbf E(t)=E(t)\mathbf d,\qquad
V_e(\mathbf r_i,t)=-\mathbf E(t)\cdot
(\mathbf r_i-\mathbf r_{\mathrm{ref}}).
\]

NEURON uses millivolts and the morphology uses micrometres, hence

\[
V_{e,i}[\mathrm{mV}]
=-10^{-3}E(t)[\mathrm{V/m}]\,
\mathbf d\cdot(\mathbf r_i-\mathbf r_{\mathrm{ref}})[\mu\mathrm m].
\]

The membrane voltage remains

\[
V_m(\mathbf r_i,t)=V_i(\mathbf r_i,t)-V_e(\mathbf r_i,t).
\]

The reference is the segment-centroid of each cell. This is a gauge choice and
does not affect polarization in the current circuits, which have chemical
synapses but no gap-junction or ephaptic coupling. It must be revisited if
electrical or extracellular inter-cell coupling is added.

## Workstation smoke commands

Activate the repository-parent environment and run from the repository root:

```bash
source /home/chirath/Documents/depression-simulator/bin/activate

python experiments/l23net_analysis/validate_l23net_tacs.py \
  experiment.name=l23net_tacs_serial_smoke \
  env=hl23net \
  analysis=l23net_tacs_smoke \
  experiment.debug=true \
  experiment.tqdm=false \
  env.debug_n_neurons.PYR=1 \
  env.debug_n_neurons.SST=1 \
  env.debug_n_neurons.PV=1 \
  env.debug_n_neurons.VIP=1 \
  env.network.dt=0.125 \
  env.simulation.obs_win_len=100 \
  env.simulation.duration=300

mpiexec -n 2 python experiments/l23net_analysis/validate_l23net_tacs.py \
  experiment.name=l23net_tacs_mpi2_smoke \
  env=hl23net \
  analysis=l23net_tacs_smoke \
  experiment.debug=true \
  experiment.tqdm=false \
  env.debug_n_neurons.PYR=1 \
  env.debug_n_neurons.SST=1 \
  env.debug_n_neurons.PV=1 \
  env.debug_n_neurons.VIP=1 \
  env.network.dt=0.125 \
  env.simulation.obs_win_len=100 \
  env.simulation.duration=300
```

The validator checks exact sample counts, strictly increasing time, finite
ideal EEG, configured/effective temperature equality, a literal sinusoid,
non-zero field coupling on every occupied rank, MPI time agreement, morphology
projection onto the requested direction, and exact zero-field washout.

## Full-circuit NCI resource profile

`profile_l23net_tacs_full_scale.py` is the next technical gate. Its frozen
default is the complete 1,000-cell MDD-configured circuit at `dt=0.025 ms` in
one persistent 15-s episode:

| Phase | Duration | Purpose |
|---|---:|---|
| burn-in | 4 s | reproduce the historical discarded transient |
| baseline | 4 s | retain a multi-second unstimulated reference |
| stimulation | 5 s | 0.5 V/m, 10 Hz, axial field with 0.5-s edge ramps |
| washout | 2 s | verify exact field removal and continued dynamics |

The run advances in one-second windows. This bounds temporary probe arrays and
writes `l23net_tacs_full_scale_profile.json` after every window, so a walltime
or memory failure still leaves the last completed timing and memory snapshot.
The companion chunked HDF5 trace appends and flushes ideal EEG, total dipole,
field samples, and phase labels after every completed window. Its committed
sample/window attributes identify the durable prefix after an interrupted job.
The JSON records construction time, per-window time, local
cell/segment balance, process RSS by node, waveform checks, temperature,
firing-rate guardrails, non-zero active polarization, and exact washout.
The legacy full-circuit `STIM_PARAM` table also schedules one-off events to
subsets of PYR/PV/VIP cells just after 4 s. Consequently, the retained 4--8 s
reference follows the historical transient boundary but is not a perfectly
stationary spontaneous baseline. This must be held fixed in later paired runs
or redesigned as an explicit protocol choice.

The first attempted full NCI profile (job `179714811.gadi-pbs`) was not a valid
performance run. Mechanisms were compiled under node-local `$PBS_JOBFS` on the
launch node, so ranks on the other nodes could not load them and exited while
the remaining ranks waited in MPI collectives. Its elapsed time, CPU use, and
memory peak therefore cannot be extrapolated to a healthy full-network run.

The replacement technical gate uses the complete 1,000-cell circuit at the
production `dt=0.025 ms`, but advances only 500 ms: 100-ms burn-in, 100-ms
baseline, 200-ms stimulation (two 10-Hz cycles), and 100-ms washout. It compiles
the unchanged mechanisms into a job-specific directory on shared `/g/data`,
then starts one preflight rank on every allocated node and requires every rank
to load and instantiate representative mechanisms before network construction.
The conservative first gate requests 624 CPUs and 2470 GB but launches 256 MPI
ranks across all 13 nodes. This is a correctness and memory-safety allocation,
not an efficiency claim or a scientific stimulation experiment. A longer
resource profile and short rank-count benchmark should follow only after this
gate passes.

Rank count must remain fixed for matched scientific comparisons because the
current circuit RNG is rank-local. CPU efficiency can be studied separately
with short 128/256/384/512-rank jobs, accepting that those timing runs construct
different circuit realizations. The chosen production rank count must then be
frozen before paired scientific simulations are generated.

The old point-source implementation materialized segment-by-time extracellular
arrays. Uniform-field storage now scales as `O(N_segments + N_time)` instead
of `O(N_segments * N_time)`. The online path detaches LFPy's unused full-rate
soma-voltage recorders, drains spike vectors after every window, evaluates only
the current-dipole forward transform, and streams the retained trace to disk.
Remaining runtime includes Python reads and assignments for each local segment
at every fixed step. The PBS epilogue's
job-level `Memory Used` value is authoritative. Summed process RSS in the JSON
can double-count shared pages and its linear duration projection is only a
planning estimate, not a safe maximum.

## Assumptions and limitations

- `cortical_depth = +z` is a configurable circuit-coordinate convention, not a
  SimNIBS-derived subject-specific vector. A later head-model bridge must pass
  the local vector field and tissue amplitude explicitly.
- The field is uniform over the microcircuit (quasistatic, homogeneous local
  approximation). It does not model field gradients at the column scale.
- The EEG is ideal neural-only output. Stimulation artifact, electrode
  referencing, sensor noise, and artifact removal are not modeled.
- Tissue V/m is not interchangeable with stimulator mA. The recent comparison
  paper's approximate 1-mA-to-1-V/m relation belongs to its own COMSOL cube and
  is not transferred as a calibration here.
- The short smoke uses no onset/offset ramp so an active non-zero endpoint can
  verify application and subsequent removal. Scientific protocols should use a
  continuous block envelope and durations justified for their endpoint.
- Uniform-field storage is factorized into a field time series and a geometric
  coupling vector per local cell. This removes the prohibitive
  segment-by-time stimulation matrix, but full L23Net runtime and memory still
  require profiling on NCI before a production experiment.
- L23Net currently seeds rank-local NumPy streams with `SEED + RANK`, as
  expected by LFPy's local-postsynaptic connectivity construction. A fixed
  seed is reproducible only for a fixed MPI decomposition: changing the rank
  count can change positions, connectivity, weights, delays, and synapse
  locations. Freeze the MPI rank count within every matched scientific
  comparison unless circuit construction is redesigned to be rank-invariant.
- `env.network.syn_activity=true` is kept in full-profile commands for
  provenance compatibility with historical jobs, but the current L23Net setup
  constructs its OU/background mechanisms unconditionally; this switch does
  not enable or disable them for L23Net.
- Legacy `env.ts.method`, `env.ts.type`, and point-electrode fields remain in
  the shared YAML but are not used by `OnlineNeuronEnv` in uniform-field mode;
  the active waveform and geometry are controlled by `env.online`.
