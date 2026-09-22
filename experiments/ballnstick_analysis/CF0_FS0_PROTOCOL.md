# CF0 and FS0: independent qualification for further H5 work

Neither study changes H1–H4, the old binary carrier estimator, circuit defaults,
morphology, or stimulation controller. No L23Net simulation is run. These are
measurement/mechanism studies, **not H5 demonstrations**.

## Isolation and outputs

Entry points: `run_ballnstick_cf0.py` and `run_ballnstick_fs0.py` in this folder;
configs: `configs/analysis/ballnstick_cf0.yaml` and `ballnstick_fs0.yaml`;
new utilities/mechanisms: `qualification/`. FS0-specific NMODL mechanisms
compile into its **result directory**, not the legacy circuit directory. Only
FS0-created isolated cells receive them; no network runner imports them.

Both studies save resolved configuration, source hashes, package versions,
prespecified cases, metrics, traces, PNG/PDF figures, and elapsed wall time.
`run_complete.json` is written **last**, with artifact hashes. A scientific
negative is still a completed run; a crash is not. Existing nonempty result
directories are rejected; use a new `experiment.name` for reruns. MPI failures
abort the whole job, including root-only analysis failures.

Neither study needs upstream results. The workstation's older H5-P0 folder
under `neurostimenv/results/` need not be moved.

## CF0: continuous-alpha measurement

### Generator and protocol

Retain 32 E/8 I cells, canonical HH at 6.3 °C, recurrence, background weights,
and mean afferent rates. All afferents participate in the rate rhythm (`q=1`,
`m=0.04`), but their Poisson events remain conditionally independent:

\[
\lambda_j(t)=\lambda_{0,j}[1+0.04\sin\phi(t)],\qquad
d\phi=2\pi f_0\,dt+\sqrt{2D}\,dW_t.
\]

Time here is seconds and D is rad²/s. The existing phase path uses a 1-ms grid;
neural integration remains 0.0625 ms. The carrier stays fixed within an
episode while phase diffuses. This is an externally imposed afferent rhythm,
not an autonomous oscillator that stimulation can change directly.

Per structure, draw one carrier uniformly in (8,10) Hz and one in (10,12) Hz,
without rounding, then cross with `D={0.5,2.0}`. Save draws before simulation.
Add one homogeneous-Poisson B episode (`m=0`, unchanged mean rate). B's nominal
frequency is a placeholder, never scored as a true carrier.

| Stage | Structures | Rhythmic episodes/structure | B episodes/structure | Total |
|---|---:|---:|---:|---:|
| Discovery | 3 | 2 carriers × 2 D levels | 1 | 15 |
| Disjoint qualification | 3 | 2 carriers × 2 D levels | 1 | 15 |

Qualification runs only after discovery passes: **15 or 30 neuronal episodes**,
not 30 subjects. Structure namespaces begin at 365101 and 365201; afferent and
observation namespaces are separately recorded.

Each persistent episode: **1 s burn-in + 30 s EEG baseline + 8 s
stimulation-free continuation = 39 s**, initialized once. The reused collector
names the last blocks `stimulation` (7 s) and `washout` (1 s), but
`stimulate=False` is unconditional: **all CF0 intervals have zero field**.

### Measurement, freezing, and gates

Observed EEG is neural EEG plus stationary AR(1) noise:

\[
u_n=.95u_{n-1}+\sqrt{1-.95^2}\epsilon_n,\quad y_n=x_n+s u_n,
\quad s={.25\,\mathrm{SD}(x_{baseline})\over\mathrm{RMS}(u_{baseline})}.
\]

Save/hash the original noise vector. Scale uses baseline only and stays fixed
during continuation. Noise is defined at raw 16 kHz: correlation time is about
1.22 ms. This inherited simulator stress test is **not a calibrated clinical
alpha-band SNR**. Decisions use noisy EEG; neural-only results are attribution.

Antialias resample baseline alone to 500 Hz. Combine whole-record DPSS
multitaper evidence with graded evidence from overlapping 6-s windows. Fit
log power versus log frequency in 4–7 and 13–20 Hz, then pool adjusted evidence
over 8–12 Hz candidates spaced 0.025 Hz. The fine grid and 32-s FFT padding
are **interpolation, not additional spectral resolution**; save taper bandwidth.

Predeclared candidates: `NW=2,K=3,sigma=0.20 Hz` and
`NW=3,K=5,sigma=0.35 Hz`. Confidence requires ≥0.5-dB evidence, ≥0.2-dB margin
over a competitor ≥0.75 Hz away, and temporal evidence-centroid SD ≤0.6 Hz.
Abstention maps to **sham**, not an arbitrary active frequency.

Discovery ranking: complete gate, accepted accuracy, unconditional accuracy,
coverage, then smaller unconditional MAE. Hidden labels score candidates but
never enter the estimator. Freeze/hash parameters and all thresholds before
qualification; no refitting or post-failure threshold rescue.

At baseline end and every 125 ms during continuation, regress preceding raw
EEG on an intercept, linear trend, and both carrier quadratures. Initial
history is 1 s; later history is 0.5 s. No Hilbert transform or future samples
are used. Save boundaries, input timestamps, amplitude/RMS confidence, and
phase differences from the same-frequency neural-only estimate. **That
estimate is not latent afferent ground truth.** This is measurement
qualification, not a test of an active continuous-frequency H4 controller.

Engineering gates: ≥80% coverage; ≥90% accepted estimates within 0.25 Hz;
≥80% unconditional accuracy; ≥75% accuracy/coverage in each D level; no
structure below 50% accuracy; ≥80% phase actionability (amplitude/RMS ≥0.1);
safe E/I rates; zero field. A 0.25-Hz error accumulates 11.25° in 125 ms;
diffusion and sensing error add further uncertainty.

Average within structure first. Report MAE, accuracy, coverage, D-stratified
results, observation attribution, and exploratory structure-level t intervals
for MAE. Three structures are not powered confirmation; windows are repeats.
B acceptance is an audit: **three B examples per stage cannot validate a
rhythm-presence screen**. Later treatment needs a frozen phenotype screen and
B target; passing CF0 does not authorize treating accepted B records.

Artifacts: context/structure metrics, phase audits, frozen estimator, raw neural
EEG/original noise, processed EEG, whole/window evidence, alpha-region PSD
panels, measurement-summary figures. Hidden-carrier annotations are audits.

## FS0: tonic conductance and cellular field sensitivity

Use the unchanged HH soma/passive 1-mm apical cable at 6.3 °C. Only FS0 inserts
a reduced **equilibrium rectifying tonic conductance** into the cable:

\[
I_t=\bar g_t o_\infty(V)(V+75),\quad o_\infty={a\over a+b},\quad
a=50F[.1(V+20)],\quad b=20F[-.08(V-10)],
\]
\[
F(x)={x\over1-e^{-x}},\qquad F(0)=1.
\]

V is mV, density S/cm², current mA/cm². The I–V relation is inspired by
`setup/circuits/L23Net/mod/tonic.mod`; stable analytic limits replace its
problematic near-zero cases. Verify compiled versus Python activation. Save
the source relaxation time `1/(a+b)` to audit the equilibrium approximation;
do not claim transplanted mammalian channel kinetics or Q10.

Read-only checks verify L23 PYR apical density 0.000938 S/cm² and leak
0.0000954 S/cm² against the tracked spreadsheet/HOC. Preserve the
**tonic-to-leak slope-conductance ratio at −65 mV**:

\[
r={\partial I_{t,L23}/\partial V\over g_{pas,L23}},\qquad
\bar g_{t,toy}={r g_{pas,toy}\over
\partial[o_\infty(V)(V+75)]/\partial V\vert_{-65}}.
\]

Toy leak 0.0002 S/cm² gives reference density 0.001966457 S/cm² and slope
ratio 0.38473. Compare **legacy (no insertion), low (0.6×reference), reference**.
The 0.6 factor is an L23-inspired sensitivity perturbation, not a diagnosis or
validated population range. These are not H1–H4 A/B labels. There is no
artificial field-gain multiplier.

Each case: **2 s settling + 3 s field/probe + 1 s washout**, central 2-s
harmonic endpoint, 500-ms field ramps. Retain
`V_ext[mV] = -0.001 E[V/m] dot r[um]`.

| Cases | Count |
|---|---:|
| 3 conditions × (8,10,12 Hz) × (0.1,0.4 V/m) | 18 |
| Condition-matched shams | 3 |
| 10-Hz excitatory conductance probes | 3 |
| Somatic current steps for input resistance | 3 |
| Zero-conductance legacy equivalence | 1 |
| Transverse 10-Hz/0.4-V/m controls | 3 |
| dt/2 and finer spatial discretization, legacy/reference | 4 |
| **Total deterministic isolated-cell cases** | **35** |

The mid-apical probe is `g=0.00005[1+0.5 sin(2 pi 10 t)] uS`, reversal 0 mV:
a small-input transfer assay, **not tACS, recurrence, or a Poisson generator**.
Resistance uses a 0.001-nA step. Neither diagnostic is a policy action.

Record soma/distal voltages, total z-dipole, tonic/probe currents, conductance,
and field. Fit complex harmonic transfer relative to the recorded input.
Compare field/synaptic transfer at 10 Hz. Save rest and resistance: tonic
changes affect operating point as well as incremental conductance. This is
not a pure shunt assay at artificially matched resting voltage.

Checks: finite subthreshold responses, temperature, exact field removal,
recovery within 0.01 mV, time/space complex-gain error ≤5%, zero-inserted
equivalence, transverse null. The sensitivity criterion is ≥5% low/reference
gain difference at a measured site/frequency: an **engineering threshold**,
not biological significance. Report every contrast/sign; do not increase
conductance or lower the gate after looking.

No between-subject p-values apply. Different field gain need not imply
different optimal doses: synaptic transfer may change similarly. Reduced ideal
EEG power need not mean desynchronization; field-induced membrane currents
also contribute. FS0 does not establish network efficacy, clinical safety,
or H5. Save calibration/I–V audit, cell metrics, convergence, sensitivity and
field/synaptic comparisons, traces, and PNG/PDF figures.

## Workstation commands

```bash
cd /home/u7041472/Documents/chirath/depression-simulator/neurostimenv
git switch research/ballnstick-h1-h4
git pull --ff-only origin research/ballnstick-h1-h4
source /home/u7041472/Documents/chirath/depression-simulator/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg HYDRA_FULL_ERROR=1

mpiexec --bind-to core --map-by core -n 8 \
  python experiments/ballnstick_analysis/run_ballnstick_cf0.py \
  experiment.name=ballnstick_cf0_full env=ballnstick \
  analysis=ballnstick_cf0 env.simulation.obs_win_len=1000 \
  experiment.plot=true experiment.tqdm=false

mpiexec --bind-to core --map-by core -n 8 \
  python experiments/ballnstick_analysis/run_ballnstick_fs0.py \
  experiment.name=ballnstick_fs0_full env=ballnstick \
  analysis=ballnstick_fs0 experiment.plot=true experiment.tqdm=false
```

Outputs: `/home/u7041472/Documents/chirath/results/ballnstick_cf0_full/cf0`
and `/home/u7041472/Documents/chirath/results/ballnstick_fs0_full/fs0`.
FS0 invokes `nrnivmodl` automatically; a C/C++ toolchain is required. CF0
distributes the network; FS0 distributes cell cases. Start at 8 ranks; more
need not help a 40-cell network. Keep MPI size fixed within a dataset because
rank-dependent RNG/layout need not be invariant across MPI sizes.

For the laptop use the local venv and `-n 2`. Reduced CF0 smoke:

```bash
mpiexec -n 2 python experiments/ballnstick_analysis/run_ballnstick_cf0.py \
  experiment.name=ballnstick_cf0_smoke env=ballnstick analysis=ballnstick_cf0 \
  env.simulation.obs_win_len=1000 analysis.smoke=true \
  analysis.design.structures_per_stage=1 analysis.timeline.baseline_steps=4 \
  analysis.timeline.stimulation_steps=1 experiment.plot=true experiment.tqdm=false
```

Smoke exercises both stages after a failed gate but **cannot qualify**. Full
FS0 is already inexpensive. Review both outputs before authorizing a network
response map or ML study.
