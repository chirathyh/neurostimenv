# L23Net reviewer-response experiment plan

## Decision in brief

Do not run another broad MPI scaling sweep. The 15-s, 624-rank profile passed
the exit checks below, so the sinusoidal uniform-field implementation is ready
for focused validation before scientific inference:

1. an exact deterministic no-field replay check at the frozen MPI rank count;
2. a small full-network healthy/reference versus reduced-inhibition pilot using
   the historical 28-s protocol.

After those gates, proceed to a prospective replication and then to a gated
stimulation study. Do not begin with a bandit. First establish that the
reduced-inhibition phenotype is observable, that a uniform-field sinusoid can
causally move the prespecified EEG endpoint, and that different prestimulation
EEG contexts genuinely favour different actions. A contextual bandit is only
scientifically justified if the last condition holds on independent circuit
structures.

This plan treats circuit structure/seed as the statistical unit. Time windows,
stochastic continuations, frequencies, and actions are repeated observations
within a structure and must not be counted as independent samples.

## Scope and terminology

- **Reference circuit** means the existing L23Net configuration with
  `env.simulation.MDD=false`.
- **Reduced-inhibition or MDD-configured circuit** means the existing
  `env.simulation.MDD=true` configuration. In code this multiplies all
  SST-originating recurrent synaptic peak conductances by 0.6, multiplies
  pyramidal apical tonic inhibition by 0.6, and proportionally reduces tonic
  inhibition in non-pyramidal populations. It is a model manipulation, not a
  definition or validated model of clinical depression.
- **Ideal EEG** means the single-sensor four-sphere forward-model signal. It
  contains neural currents induced by the field but no direct stimulation
  artifact, sensor noise, unrelated sources, or reference-electrode model.
- Field amplitudes are local tissue electric fields in V/m. They are not scalp
  current in mA and must not inherit the old SimNIBS-to-point-source dose labels.
- EEG-relative in-phase/antiphase refers to temporal field phase relative to
  the simulated EEG. It is not the same as the relative phase between two scalp
  electrodes in a clinical montage.

## Evidence that informs the plan

### Current technical implementation

The current online L23Net path already provides the two reviewer-requested
changes:

\[
E(t)=w(t)E_0\sin\theta_a(t),\qquad
\mathbf E(t)=E(t)\mathbf d,
\]

\[
V_{e,i}(t)=-10^{-3}\,\mathbf E(t)\cdot
           (\mathbf r_i-\mathbf r_{c})\quad\text{mV},
\]

where segment and reference coordinates are in micrometres, the field is in
V/m, and the cell-centred subtraction is a gauge choice. The field is updated
at each fixed NEURON step, while actions can be changed between completed
observation windows without reinitializing membrane, channel, synaptic,
recurrent, or event-queue state.

The bounded-memory path now streams EEG, dipole, field, and stage information
to HDF5; drains spike vectors; disables unused full-lifetime soma-voltage
recorders; and evaluates only the current-dipole forward probe. These changes
remove the known avoidable duration-dependent storage, but the 15-s run remains
the required empirical check of memory behaviour at production precision.

### Strong-scaling result

The completed 2-s profiles all passed. The two most relevant operating points
were:

| MPI ranks | Integration wall time for 2 simulated s | PBS peak memory | Short-run allocated core-hours | Interpretation |
|---:|---:|---:|---:|---|
| 288 | 357.1 s | 75.7 GiB | 35.6 | lower short-job allocation cost, slower |
| 624 | 135.5 s | 92.2 GiB | 43.3 | fastest; fixed startup cost dominates the 2-s accounting |

At 624 ranks the median inactive and active costs were approximately 64.3 and
78.5 wall-seconds per simulated second, respectively. Once a run exceeds about
5 simulated seconds, the fixed startup overhead is amortized and 624 ranks are
also projected to use fewer allocated core-hours than 288 ranks for this
implementation. Therefore, subject to the 15-s result, use 624 MPI ranks and
624 CPUs for the scientific campaign and never mix rank counts within paired
comparisons. The circuit random streams currently depend on MPI rank.

The completed 15-s job used 160,318,012 KiB (approximately 153 GiB) and
plateaued after construction. Adding approximately 25% headroom gives a
practical 200-GB request. Do not return to the earlier 2,470-GB request unless
measured PBS memory requires it. The PBS `resources_used.mem` value, not summed
process RSS, is authoritative.

### Historical replication audit

The published baseline data consist of 60 reference and 60 MDD-configured
files, each containing 1,120,001 samples (28.000025 s including the left
endpoint) at `dt=0.025 ms`. The two archived cell-position files are byte
identical, supporting positional matching, although the archive does not by
itself prove that every connection and stochastic input was paired.

The archived data reproduce very large paired condition effects under the
legacy analysis:

| Band | MDD/reference mean-power ratio | Paired standardized effect \(d_z\) | Positive pairs |
|---|---:|---:|---:|
| theta, 4--8 Hz | 1.365 | 2.20 | 58/60 |
| alpha, 8--12 Hz | 1.431 | 2.34 | 60/60 |
| low beta, 12--16 Hz | 1.694 | 3.38 | 60/60 |

These are historical planning data, not evidence for the new implementation
and not estimates of biological effect size.

There are four reproducibility hazards in the old entry points:

1. The checked-out `run_healthy.sh` and `run_mdd.sh` set
   `experiment.debug=True`, producing 80 PYR, 5 SST, 7 PV, and 8 VIP cells
   rather than the published 1,000-cell network. An earlier Git revision used
   `debug=False`; the present scripts must not be submitted unchanged.
2. `experiments/case_study/replication.py` crops the first 4 s before filtering
   and then crops another 4 s before Welch analysis. The plotted baseline is
   therefore effectively 8--28 s, whereas the paper describes a single 4-s
   transient exclusion.
3. The legacy `NeuronEnv` mutates the shared `MPI_VAR` seed dictionary in its
   constructor. A loop labelled with seeds 10--69 therefore uses cumulative,
   not simply 10--69, internal seeds. Reference and MDD jobs appear to have
   followed the same sequence, which preserves pairing, but filenames alone do
   not record the resolved streams.
4. `Circuit_param.xls` schedules one-off synaptic input to subsets of PYR, PV,
   and VIP cells just after 4 s even in nominal no-stimulation runs. This is
   part of the historical circuit protocol. The later 8--28-s analysis mostly
   avoids its immediate transient, but the data should not be described as a
   completely unperturbed resting baseline.

The new study will retain a literal legacy analysis for reproducibility and
add a clearly labelled corrected/sensitivity analysis. It will not conceal
these discrepancies or alter the archived result retrospectively.

## Gate 0: completed 15-s profile

This technical profile passed; it is not an efficacy experiment. Its acceptance
criteria were:

- JSON `status` is `passed`, `errors` is empty, and the resolved commit and
  mechanism hashes are saved;
- all 1,000 cells are present: 800 PYR, 50 SST, 70 PV, and 80 VIP;
- exactly 15 windows and 600,000 samples are committed at `dt=0.025 ms`;
- the HDF5 committed window/sample attributes agree with the JSON;
- EEG and dipole samples are finite and time is strictly monotonic;
- configured and effective temperatures are both 34 degrees C;
- the active field is a phase-continuous 10-Hz sinusoid with the requested
  0.5-V/m peak and 0.5-s block ramps;
- extracellular polarization is non-zero on occupied ranks during stimulation
  and exactly zero throughout washout;
- unused soma-voltage and drained spike vectors remain bounded at every
  checkpoint;
- PBS reports a normal exit, no OOM/timeout, and a memory peak with adequate
  headroom under the requested 256 GiB;
- the post-warm-up memory checkpoints do not show an unexplained monotone
  increase large enough to endanger a 28-s run.

The gate passed, so general scaling tests stop here.

## Gate 1: two focused validation runs

### G1A. Deterministic current-path no-field replay

Purpose: establish that independently reconstructed current-path trajectories
are exactly reproducible and that the saved seed/structure manifest really
identifies the same circuit.

- The reduced two-rank local test passed with exact structure, spike, EEG,
  dipole, time, and zero-field hashes across two independent processes.
- Run two identical six-second full-network zero-field trajectories at 624
  ranks with one frozen seed. Six seconds crosses the historical event just
  after four seconds.
- Save resolved global/rank-local seeds plus rank-wise geometry,
  synapse-layout, afferent-event, and recurrent-NetCon fingerprints.
- Require exact equality of the fingerprints, spike-event hashes, committed
  sample/window counts, and every retained HDF5 dataset.

The historical path is not used as a bitwise oracle. It effectively used
NEURON's 6.3-C default, its LFPy 2.3 MPI loop calls `h.fadvance()` rather than
the corrected `ParallelContext.psolve()`, and its unstimulated path omitted the
extracellular mechanism that active/sham online counterfactuals retain at zero.
Prospective phenotype replication in G1B is the valid bridge to the old study.

Pass: exact equality of the two independent current-path runs. A failure blocks
the replication.

### G1B. Full-network paired 28-s pilot

Run two new circuit structures, each under reference and MDD-configured
conditions, with no extracellular field:

- 1,000 cells, `dt=0.025 ms`, 34 degrees C, 624 ranks;
- 28 s continuous simulation;
- historical internal stimulus schedule retained;
- exact matched structure/event seed manifest across each pair;
- streamed output and resource checkpoints.

Apply both analysis views:

- **Legacy reproduction:** the exact historical filter/Welch procedure and
  effective 8--28-s interval.
- **Corrected sensitivity:** one explicitly declared transient/recovery
  exclusion and a numerically stable SOS filter; use duration-matched records
  in both conditions.

Pass: finite output, exact pairing metadata, bounded memory, plausible rates,
and the MDD-minus-reference direction is positive for the prespecified
three-band composite in both pilot structures. Two structures are a technical
directional gate and provide no inferential claim.

## Experiment R1: prospective reference versus reduced-inhibition replication

### Design

- Use 16 new paired circuit structures. This is the confirmatory replication
  cohort; do not reuse the two pilot structures.
- Replay every structure once as reference and once as MDD-configured using
  matched circuit and external-randomness namespaces.
- Use the exact 28-s, 1,000-cell, `dt=0.025 ms`, 34-degree-C historical
  protocol and the frozen 624-rank decomposition.
- Use a separate set of 16 reference-only structures to calibrate downstream
  EEG targets. Do not use a candidate structure's matched reference replay to
  define its own treatment target.
- If the aim is a literal sample-size replication of the prior figure, extend
  the paired cohort from 16 to 60 only after the 16-pair result and resource
  report are frozen. The 60-pair extension should not replace or redefine the
  primary analysis.

The archived paired effects are much larger than needed for this gate. A design
assumption of \(d_z=0.70\) gives approximately 80--85% directional power near
16 structures, while retaining enough sign-flip permutations for exact
inference. Using 60 pairs is valuable for figure precision and literal
replication, but it is not required merely to rediscover effects historically
estimated at \(d_z>2\).

### Endpoints

For each band \(b\in\{\theta,\alpha,\text{low beta}\}\), compute log power on
the frozen analysis interval. The primary directional composite is

\[
g_s^{(R1)}=\frac{1}{3}\sum_b
\frac{\log_{10}P_{s,b}^{\mathrm{MDD}}-
      \log_{10}P_{s,b}^{\mathrm{ref}}}
     {\max(\sigma_{b,\mathrm{ref}},\sigma_{\min})}.
\]

The scale floor \(\sigma_{\min}\) must be fixed from the independent reference
calibration before candidate inference. Positive values indicate the expected
elevated-power phenotype. Report each band's paired difference and ratio as
secondary endpoints with false-discovery-rate control; report population
firing rates as mechanistic guardrails, not controller observations.

Use an exact structure-level sign-flip test, a two-sided 95% confidence
interval, paired \(d_z\), the number of positive structures, and a
structure-level bootstrap interval. Do not treat frequency bins or time windows
as replicates.

### Outputs to freeze

- raw streamed ideal EEG and dipole traces;
- the literal and corrected analysis tables;
- the independent reference mean, scale, duration, and hash;
- per-population rates and numerical exclusion flags;
- all seed, rank, package, code, mechanism, and PBS provenance;
- a compute table in wall-hours, node-hours, allocated core-hours, peak memory,
  storage, and simulated-seconds-per-wall-hour.

## Experiment S0: stimulation protocol and observability gate

The 0.5-V/m, 10-Hz technical profile does not identify an effective treatment.
Before applying tACS, ask whether the MDD-configured prestimulation EEG supports
the parameterization that the proposed strategies require.

### Episode

Use one persistent 27-s trajectory:

| Epoch | Time | Purpose |
|---|---:|---|
| initialization/transient | 0--4 s | historical settling interval |
| recovery from the built-in 4-s event | 4--8 s | prevent event transient entering the controller baseline |
| prestimulation baseline | 8--20 s | phenotype screen, peak-frequency and phase estimation |
| stimulation or sham | 20--25 s | 0.5-s onset/offset ramps |
| primary endpoint | 20.5--24.5 s | central 4 s, excluding ramps |
| washout | 25--27 s | exact field removal and short-term recovery audit |

The independent reference set must supply a duration-matched 12-s screening
reference and a separate duration-matched 4-s outcome reference.

### Observability checks

From prestimulation EEG only:

1. confirm that the MDD-configured circuit lies outside the frozen reference
   region in the three-band feature space;
2. estimate the dominant 8--12-Hz peak using a prespecified estimator with
   at least 0.5-Hz resolution;
3. require peak choice to be stable across the two 6-s baseline halves;
4. require adequate Fourier-resultant-to-RMS magnitude and acceptable
   extrapolated phase agreement before using an EEG-relative phase action.

If peak/phase observability fails, do not label an action "frequency matched"
or "antiphase." Fall back to a phase-invariant fixed-frequency comparison and
report the negative observability result.

The stimulation phase convention is

\[
\widehat\phi_k=
\operatorname{wrap}\left[2\pi f_a t_k-
\operatorname{atan2}(S_k,C_k)\right],
\qquad
\theta_k^*=\operatorname{wrap}
\left(\widehat\phi_k+\frac{\pi}{2}+\Delta\phi\right),
\]

with \(\Delta\phi=0\) for EEG-relative in-phase and \(\pi\) for antiphase.
The \(\pi/2\) term converts the EEG cosine convention to the field sine
convention; it is not a physiological delay.

## Experiment S1: small balanced causal action map

This is system identification, not a bandit experiment. Use three new circuit
structures and one matched postdecision continuation per structure. Reconstruct
each counterfactual from the beginning with identical structure, baseline, and
postdecision external randomness; recurrent spikes may diverge causally after
field onset.

Use a balanced, minimally redundant action set at one frozen amplitude:

| Arm | Purpose |
|---|---|
| sham | causal negative control |
| fixed 10-Hz open-loop sinusoid | clinically motivated non-personalized comparator; not claimed as a clinical dose |
| EEG-peak-matched in-phase | phase-specific comparator |
| EEG-peak-matched antiphase | candidate mechanistic strategy |
| frequency-mismatched antiphase | frequency-specificity control |

Use 0.4 V/m as the provisional discovery amplitude because it is within the
reviewer-highlighted weak-field range and was effective in the BallAndStick
stationary study. The current 0.5-V/m run is only a technical maximum. If a
dose check is necessary, compare sham, 0.2, and 0.4 V/m on one or two discovery
structures before freezing the action map; do not tune dose on confirmation
structures.

Map the EEG peak to a symmetric prespecified action grid, for example 8, 10,
and 12 Hz. Do not repeat one frequency at several doses while representing the
others once. The mismatch mapping (for example the farther of the two remaining
grid points) must be fixed before outcomes are observed.

For the first discovery structure only, repeat the candidate action with a
transverse field. This is a geometry null for the circuit, not a realistic
alternative scalp montage or a policy arm.

### Primary causal endpoint

Let \(\mathbf x\) contain prespecified log theta, alpha, and low-beta powers,
and let \(\boldsymbol\mu_H,\boldsymbol\sigma_H\) be the independent,
duration-matched reference target. Define

\[
d_H(\mathbf x)=
\left[\frac{1}{3}\sum_b
\left(
\frac{x_b-\mu_{H,b}}
{\max(\sigma_{H,b},\sigma_{\min})}
\right)^2\right]^{1/2},
\]

\[
g_s(a)=d_H(\mathbf x_s^{\mathrm{sham}})-d_H(\mathbf x_s^a).
\]

Positive benefit means movement toward, not equivalence with, the reference
EEG distribution. Overshoot is penalized. The primary endpoint retains the
stimulation frequency because neural activity at that frequency is part of the
mechanism; a prespecified fundamental-excluded distance is a mandatory
robustness analysis. Neither endpoint can distinguish spike entrainment from
changes in transmembrane-current magnitude or spatial cancellation, so firing
rates and spike timing are supporting audits only.

Advance only if one active action shows:

- benefit over sham in all or nearly all three discovery structures;
- a practically relevant margin fixed before confirmation;
- the expected frequency/phase ordering rather than a nonspecific response to
  any field;
- firing rates within simulator bounds and within 20% of paired sham;
- exact field removal and substantial washout of the acute EEG effect;
- no dependence on one anomalous structure.

A failed screen is a valid stopping result. Do not search additional frequencies
or phases on the same structures until a positive result appears.

## Experiment S2: frozen held-out tACS confirmation

Freeze the amplitude, frequency grid, phase rule, endpoint, analysis code, and
practical margin from S1. Evaluate on disjoint structures. A provisional
minimum of 12 structures is reasonable only for a large paired effect around
\(d_z=0.8\); it is not adequate to establish the old paper's reported small
effect of approximately 0.23.

Approximate one-sided paired design requirements at alpha 0.05 and 80% power
are:

| Assumed paired effect \(d_z\) | Approximate structures required | Interpretation |
|---:|---:|---|
| 0.8 | 10; use at least 12 | large, practically compelling simulated effect |
| 0.5 | 25 | moderate effect |
| 0.23 | 117 | effect comparable to the small legacy theta result |

Use pilot variance and a prespecified smallest effect of scientific interest to
perform a structure-level simulation-based power calculation. Do not use
post-hoc power. If compute limits cap the study at 12 structures, state that the
study is an efficacy gate for large simulated effects and cannot exclude smaller
ones.

The minimum confirmation arms are sham, the frozen S1 action, the fixed 10-Hz
comparator, and a uniformly randomized active action assigned from the balanced
grid before outcomes. Random assignments should be balanced across structures
and repeated-future identifiers. Add a second active clinical/mechanistic
comparator only if S1 provides a clear reason.

Primary inference is the paired frozen-action versus sham benefit. Test the
random and fixed comparisons in a prespecified sequence or control their false
discovery rate. Average futures within structure before inference. Require a
positive practical mean, exact sign-flip evidence, two-sided confidence
interval, positive effects in at least two-thirds of structures, rate safety,
and field-removal/washout checks.

## Experiment A: algorithm comparison, only after S2

### What can be compared fairly

1. **Sham:** no field.
2. **Uniform random:** a pre-generated balanced random choice over the frozen
   active action grid.
3. **Fixed 10 Hz:** the non-personalized comparator.
4. **EEG-informed deterministic rule:** select the nearest frozen frequency
   from prestimulation EEG and use the frozen phase rule.
5. **Non-contextual bandit:** learn the best population-level fixed arm from
   discovery structures only, freeze it, then evaluate that selected arm on
   held-out structures.
6. **Contextual bandit:** include only after a discovery action map demonstrates
   a reproducible context-by-action interaction and leave-one-structure-out
   benefit over the best fixed arm.

An ordinary multi-armed bandit has no mechanism to personalize actions. In a
stationary environment its held-out policy is simply a data-efficient estimate
of the best fixed arm, so it should not be expected to outperform a correctly
estimated exhaustive best-fixed comparator. Its scientifically relevant metric
is sample efficiency and regret during discovery. A contextual method can add
value only if observed EEG predicts differential action response.

### Compute-efficient evaluation

- Generate a balanced counterfactual action table on discovery structures
  first. Use it to compare random search, exhaustive grid search, epsilon-greedy,
  UCB/Thompson sampling, and the deterministic EEG rule offline under identical
  sampled outcome sequences. This algorithm benchmark requires no additional
  NEURON runs.
- Freeze every algorithm, feature, hyperparameter, and stopping rule before
  held-out simulation.
- On each held-out structure, simulate only the action selected by each frozen
  strategy plus the paired controls. Deduplicate identical selected actions.
- Preserve equal field-amplitude/exposure constraints. Otherwise a strategy
  may win merely by delivering more stimulation.
- Compare strategy outcomes at the structure level. Adjacent decisions within
  one trajectory are not additional subjects.

If the context-action opportunity gate fails, stop at the non-contextual and
deterministic comparisons. A high-capacity contextual bandit fitted to roughly
12 circuit structures would be statistically indefensible even if many windows
or futures were available.

## NCI execution layout and projected cost

The following use the completed 15-s r624 measurement:

- the 15-s mixed profile took approximately 18.4 wall-minutes and 155 aggregate
  CPU-hours;
- at 48.75 simulated seconds per wall-hour, a 28-s trajectory projects to about
  34.5 integration minutes; a no-field run should be slightly faster;
- 16 paired 28-s reference/MDD trajectories plus 16 independent reference
  calibrations (48 trajectories) are therefore roughly 16,000 core-hours;
- extending to 60 pairs would require 120 paired trajectories before separate
  calibration, roughly 40,000 core-hours.

Use these only for allocation planning. Record actual build, inactive,
stimulation, and total times for every run.

Recommended PBS organization:

- one structure per array element;
- for R1, run reference then MDD as separate fresh `mpirun` processes in the
  same array element so pairing is explicit and NEURON state cannot leak;
- for counterfactual action maps, run each arm as a fresh process from the same
  frozen seed manifest;
- checkpoint after every trajectory and make reruns idempotent;
- throttle array concurrency to the available project allocation;
- request 624 CPUs, 624 MPI ranks, and the memory selected from the 15-s peak;
- compile mechanisms once into a shared, commit-specific read-only directory
  and preflight one rank per node before launching the network.

## Reproducibility and reporting requirements

Every result directory must contain:

- resolved Hydra configuration and explicit epoch/action schedule;
- Git commit plus a hash of any uncommitted patch;
- mechanism source/build hash;
- module list, Python/NEURON/LFPy/NumPy/SciPy versions, and environment lock;
- NCI job ID, node file, `qstat` resource request and exit resource usage;
- independent namespaces for structure, history, action, future, and algorithm
  randomness;
- resolved per-rank seeds and MPI size;
- cell-position and connectivity summaries/hashes;
- committed HDF5 sample/window counts and JSON validation report;
- exclusions determined without treatment outcomes;
- frozen-target and frozen-analysis hashes.

Resolve the current NCI SciPy/NumPy compatibility warning before scientific
runs. The module-provided SciPy seen in earlier logs required NumPy below 1.25
while the environment loaded NumPy 1.26.4. Use one internally consistent,
version-locked environment and fail preflight on import warnings or unexpected
package paths.

## Stop/go sequence

1. **15-s profile passes:** no more general scaling; run G1A.
2. **G1A passes:** run the two-pair 28-s G1B pilot.
3. **G1B passes:** run and freeze R1 plus the independent reference target.
4. **R1 confirms observability:** run S0 and S1 on discovery structures.
5. **S1 shows causal, specific, safe movement:** power and run S2 on held-out
   structures.
6. **S2 confirms efficacy:** benchmark random, fixed, individualized, and
   non-contextual bandit strategies.
7. **Only a replicated context-by-action interaction justifies a contextual
   bandit.** Otherwise report that the simpler rule or fixed strategy is the
   appropriate model for this action space.

## Principal limitations that remain even after a positive study

- The local uniform field is a controlled microcircuit approximation, not a
  subject-specific SimNIBS montage or current-to-field calibration.
- L23Net cell orientations and the single EEG sensor constrain spatial
  generalization; a transverse field is only a model-geometry control.
- The ideal EEG omits the dominant concurrent tACS artifact and measurement
  latency. Offline forward-backward filtering is causal with respect to a
  completed decision boundary but is not a deployable zero-latency filter.
- Five seconds tests acute modulation. Without plasticity mechanisms it cannot
  establish effects of clinical multi-minute or repeated-session protocols.
- Movement toward a reference EEG distribution does not prove restoration of
  the hidden circuit mechanism, clinical benefit, or biological safety.
- Seeds quantify uncertainty across this simulator's circuit realizations, not
  human participants or clinical heterogeneity.
- The historical built-in 4-s synaptic event complicates the term
  "resting-state." It is retained for replication and isolated from the new
  controller baseline by a recovery interval.

## Source material used

- `neurostimenv-preprint.pdf`: historical L23Net protocol, action set, reward,
  sample counts, and claimed interpretation.
- `neurostimenv-reviews.pdf` and the supplied reviewer summary: controls,
  waveform/field, compute, action-bias, statistical, and ecological-validity
  concerns.
- `tACS_Mechanistic_Study.pdf`: persistent episodes, uniform-field equations,
  matched counterfactual futures, independent reference calibration,
  frequency/phase controls, random and fixed comparators, structure-level
  inference, and staged discovery/confirmation.
- Current repository implementation and the archived 60-pair case-study data.
