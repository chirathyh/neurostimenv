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
