# Baseline audit for the EEG representation suite

Audited base: `research/ballnstick-h1-h4`, commit
`855a1c33623573aeae71682dc7740658bdeac401` (the isolated implementation worktree
starts at this commit). The original checkout and its uncommitted work are
preserved. This document was written before simulator changes.

## Source inspection

`configs/env/ballnstick.yaml` specifies 32 E and 8 I cells, 64 external synapses
per cell, `dt=0.0625 ms`, E/I mean external intervals 40/30 ms, canonical HH at
6.3 C, and recurrence probabilities 0.10/0.10/0.50/0.50. Cells share the HH soma
and passive apical morphology. The source is at `[0,0,78000]` um within the
79/80/85/90-mm four-sphere head, observed at `[0,0,90000]` um. EEG is returned
in V; dipoles are nA um. A single projected dipole does not provide realistic
independent multichannel spatial information.

`setup/circuits/ballnstick/utils.py` already provides private local NumPy
Poisson streams, rate-modulation thinning, shared rate-phase diffusion,
split-history/independent-future events, aligned phase-grid splits, shared
rhythm participation, and I-to-E kinetics with analytic conductance-time-area
normalization. These features are reused; the old jittered helper is not used.
Shared phase does not mean shared event realizations, and equal conductance
area does not mean equal synaptic charge or rates.

`env/models/neuron/env_online.py` and `networkenv_online.py` build once and
initialize once per episode. The manual recorder advances to absolute
boundaries and retains membrane, HH, synapse and pending recurrent-event
state. Windows contain `(start,stop]`; 1000 ms contains 16000 native samples.
The private LFPy transformation/segment order and version-sensitive recording
path are already protected. Temperature is set explicitly in configured mode.
Close is idempotent. No old bug is presumed to persist from public main.

Existing studies seed global NumPy/Python RNGs with `structure_seed*10000 +
rank`; the new serial adapter therefore rejects seeds outside 0..429495 (a conservative bound shared with the schema).
External event RNGs use a separate drive seed. The inherited phase seed is
currently derived from that drive seed, so independent phase-vs-event fork
variation needs an opt-in adapter override. Phase interpolation is safe before
a fork only when the split is aligned to the phase grid; the new suite rejects
unaligned splits. A shared bounded burst-amplitude envelope is new and must be
explicitly opt-in, with tonic calls delegated unchanged to inherited code.

## New-suite work required

Add isolated adapters for zero-field trials, independent phase namespaces,
bounded stochastic burst envelopes, deterministic prefix replay with
structure/event/state/pending-cause checks, float64 physical storage, and a
causal antialiasing model stream. Add typed manifests, grouped splits,
training-only normalization/noise references, declared tasks and learning/
evaluation workflows outside frozen COSYNE runners. Preserve all legacy
configuration defaults and entry points. Do not use destructive legacy output
folder helpers or biological/clinical labels for this toy circuit.

## Executed versus pending

The initial version above was written before implementing the adapter. The
following engineering checks were subsequently executed on the preserved
NEURON 8.2.3 / LFPy 2.3 CPU environment. They are not empirical study results:
The source audit is not evidence of observability, forecast performance,
transfer benefit, a successful curriculum, or a physiological prediction bound.


- Inherited background/kinetics/online stimulation tests: 21 passed, including
  private-event streams, modulation, phase increment variance and split replay.
- Isolated-cell polarization validation: all gates passed.
- Online/legacy 500-ms regression: 8000 online samples, 8001 legacy samples
  including t=0, correlation 1.0, relative RMS error 2.55e-11, configured and
  effective temperature both 6.3 C.
- Documented reduced A/B smoke: completed on three structures (software check;
  held-out accuracy 0.333, permutation p=1). It does not establish observability.
- Documented one-discovery/one-validation stimulation smoke: completed and
  correctly returned NOT PASSED. No frozen conclusion or threshold was edited.
- New real adapter: tiny and full 32-E/8-I zero-field episodes returned finite
  float64 EEG/dipoles, exact counts, monotonic time and clean repeated close.
  Two 250-ms default-size episodes returned 4000 samples each. First-call
  runtime was 9.76 s including imports and concurrent regression activity;
  this is not a pilot or a campaign runtime estimate.
- Exact fork equality was checked for separate event/phase variations, both
  1-ms and 0.5-ms phase grids, all segment HH/passive and Exp2Syn states,
  geometry/locations/weights/delays, input histories and output spikes. A
  40-cell cutoff at 46 ms was chosen to exercise eight actually pending
  recurrent deliveries; their causal histories remained identical.
- Causal filtering passed future-perturbation equality. Atomic two-future
  shard generation and verified resume were exercised on actual neural data.

Initial legacy-run invocations failed because Hydra's historical relative
`../../results` location was unwritable from the isolated worktree. They were
rerun with both Hydra and experiment output directories pointed at new `/tmp`
locations. No old output directory was deleted or reused.

## External event semantics and compilation

Installed LFPy 2.3 `Cell._load_spikes` schedules the supplied external delivery
schedule with `NetCon.event(tdeliver)`. The local NEURON API documentation
explicitly says that this call ignores `NetCon.delay`. These delivery times
must therefore **not** be interpreted as source spikes plus the stored default
1-ms NetCon delay. The adapter's pending-event audit applies actual conduction
 delays only to recurrent source spikes. Past external deliveries are already
consumed; future external schedules are regenerated by deterministic-prefix
rerun. No event queue is cleared, restored or exposed to a learned model.

BallAndStick uses built-in HH, passive, extracellular and Exp2Syn mechanisms.
This installed LFPy path does not use VecStim and needs no new NMODL build.
Do not compile the unrelated L23Net/FS0 mechanism trees for this suite. A real
adapter smoke is the relevant dependency check; testing for `h.VecStim` would
incorrectly reject this validated environment.

## Seed-range audit

For committed default experiment seed 1, 61 configured intervals were computed
from each `seed_offset`/`structure_seed_offset` and its neighboring count (and
stage counts), rather than comparing literal offset strings. The original A/B
range is 1--20; the other intervals below 60000 are 1001--1004, 2001--2008,
40001--40008 and 50001--50024. All other offset-defined intervals begin above
60000. Explicit CF0/CF0b structure starts are above 365000. Proposed new ranges
31000--31005, 32000--32007, 33000--33039 and 34000--34005 have no intersection
with these default ranges and satisfy `structure_seed * 10000 < 2**32`.
This source-derived audit does not prove absence of unpublished historical CLI
overrides. The dataset namespace and actual structural splits are recorded in
its manifest, and a workstation user must inventory any additional local seed
registry before a scientific campaign.
