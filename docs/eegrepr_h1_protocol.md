# H1 information campaign v2: protocol and interpretation

This implementation follows the attached 12 November 2026 H1 implementation brief and revised ICML position draft. It is a separate campaign, not an alteration of the COSYNE study or a new H2 leaderboard. A positive implementation test is not evidence for a biophysical information limit.

## Preservation and provenance

The base is `4532e19f29cfcf318f8afbe7c25850f29981803e` on `feature/eeg-representation-position2027`. Development uses the new `codex/eegrepr-h1-information-v2` branch. The user's active COSYNE checkout was on `feature/l23net-sinusoidal-uniform-field` at `ec3c6de`, with unrelated edits; it was left untouched. No existing physics, EEG-v1 module, pilot configuration or output is modified. `h1-audit` recomputes the legacy source hash and inspects a supplied pilot root. Local access does not establish that `/data/u7041472/eegrepr-results/core-pilot` is available on the workstation. The local expert-review ZIP was inspected again: its pilot manifest has 640 trial records and the expected commit/source/config hashes, but zero native HDF5 or fitted checkpoints under the pilot directory. Missing historical predictions cannot be recovered by copying summary files.

Already present before this work: private Poisson streams; continuous simulation; rate modulation and shared phase diffusion; full HH/synapse/phase/pending-event prefix checks; sample covariance trace, independent-pair half-MSE and finite-M-corrected leave-one-out reference. H1 delegates to those implementations. The legacy CLI remains the `legacy_filtered_stream_v1` regression protocol. Its causal 2049-tap filtered-stream targets and scalar definitions are not silently relabelled as new physical targets.

Every new output has schema `eegrepr-h1-2.0`, resolved configuration, branch/commit, source-content hash, dirty status/diff hash, dependency versions, generator seeds, target fingerprints and checksummed artifacts. An explicit resume requires the same config and scientific source; corrupted outputs fail. Failed simulations retain an error record and halt. Repeats do not silently drop circuits. Arrays are numeric HDF5/NPZ, with no object arrays or arbitrary training-pickle dependency.

## Physical target and causal observations

The default native integration remains 0.0625 ms, 32 E/8 I cells and canonical HH at 6.3 C. The default episode has 2 s burn-in and 4 s observed history, ending at a true physical cutoff of 6000 ms. The primary future support is **(6010, 6138] ms**. Each consecutive 2 ms block of 32 native EEG samples is averaged, producing 64 values at 500 Hz. This is a declared lossy box-average projection, not ideal anti-aliasing and not the original 16 kHz waveform. A native-waveform operator is also implemented.

All predictors and the fork reference see this same target. Future target values may use their declared future block; input preprocessing and fitted normalization never use those values. Input EEG is filtered causally with a Hamming FIR, nominal .5–30 Hz, 2049 taps at 16 kHz. Support is 128 ms and group delay is 64 ms. Availability timestamps remain at their actual acquisition times. The finite kernel does not sharply resolve .5 Hz; the named edges are design cutoffs, not a measured brick-wall response. Rate comparisons share this kernel, including its finite transition band. No backward delay correction supplies unavailable data.

TargetSpec records support, output grid, operator, units, normalization identity, physical cutoff/history, input acquisition/filter hash, nuisance settings, context/population/split, and code identity. Lead and output grids must align exactly: a 2 ms lead with a 100 Hz **output target** is rejected. A 100 Hz **input** with a 500 Hz target can legitimately predict a 2 ms physical lead; input and output axes are distinct. Scalar and frequency coordinates do not receive fictitious waveform timestamps. Multi-job runs use one cutoff determined by the longest selected history; shorter histories are suffixes of exactly that physical prefix.

Normalizers fit clean pre-cutoff blocks from training circuits only. Input quantization ranges also use training only. A single scalar target scale preserves the affine conversion `physical MSE = standardized MSE * scale^2`; target offsets may be coordinate-specific. Physical and standardized targets retain float64 for small-innovation scoring; neural minibatches explicitly cast to float32. Log-power targets have squared log-unit losses, not V-squared losses. RMS, natural-log total power, Hann/constant-detrended 4–30 Hz log band power, complex FFT and magnitude-only FFT are separately registered operators. These scalar conventions differ from the old pilot/fork conventions and must not be subtracted across versions. At 128 ms the DFT bin spacing is 7.8125 Hz; this is not a universal impossibility bound for sinusoid estimation.

The full orthonormal FFT is stored as `sqrt(2)*concat(real, imag)`. The factor preserves **mean** squared error when the real coordinate count doubles; without it, per-coordinate MSE would halve. Waveform and full-FFT equivalence is verified against physical arrays and IDs. Magnitudes and powers lose phase and change the estimand.

## States, futures and statistical units

A circuit seed fixes structure. A separately seeded prefix fixes a causal context within that circuit. It does not create a new independent circuit. Each context has independent score/reference future RNG streams after its decision boundary. RNG state, external precomputed future arrays and their seeds are provenance, not realistic observations or causal physical-state information. Already caused delayed recurrent events are retained. No stimulation is applied in this campaign.

Generation currently replays the entire prefix and stops at the last required native sample (rounded up only to the 1 ms generator phase grid). There is no unvalidated state-restoration cache. Engineering runs replay one complete input schedule exactly as a numerical control. That privileged replay is not a forecast competitor. Prefix equality is checked again when panels are derived. Circuit/trial/future/acquisition variants remain within one split.

Report equal-context averages inside a circuit and equal circuit weights across circuits; high-M contexts do not gain population weight. Future-level Monte Carlo error, circuit variability, observation ambiguity and model-seed disagreement are separate quantities. Optimizer seeds never increase the number of circuits. Small development panels cannot establish narrow population confidence intervals.

## P1: exact controls and matched predictors

The exact linear-Gaussian oscillator supplies finite-history Kalman and state oracles, allowing all four risk terms to be separated and checking observation noise, process noise, history compression, fitting error, leads, blocks and affine units. The filtered-Poisson control checks shot-noise covariance and sample/pair/leave-one-out estimators. Known finite-M predictions use predeclared stochastic tolerances. Raw control draws and analytic predictions are exported.

The nonlinear benchmark includes an empirical-variance training mean, a prospective standardized N(mean_train,I) reference, persistence, ridge/Wiener, harmonic extrapolation, a fitted delay-coordinate linear state-space baseline, shared-encoder MSE and shared-encoder diagonal-Gaussian objectives. The state-space baseline is an approximation fitted to EEG, not a privileged physical-state Kalman filter. The harmonic baseline integrates its predicted sinusoid over the declared output bins. Its D=0 motivation does not guarantee model correctness in nonlinear circuits. Train/development circuits supply model parameters and tuning only. Models are frozen before evaluation futures are scored.

The two neural objectives start from the same encoder and mean decoder. Gaussian variance parameters are additional and counted; parameter tolerance failures are disclosed, not hidden by padding other baselines. Candidate counts, updates, exposures, forward passes, estimated dense FLOPs and measured times/memory are saved separately. Fixed update counts do not establish equal compute. Few-example fitting, shuffled-target behavior and an analytic learnable signal are diagnostic gates; a poor forecast far above the reference remains possible model error.

For each matched context, score mean MSE, finite-state sample variance, pair estimates, corrected combined excess, MC standard errors and oracle finite-M behavior. The corrected excess is `||prediction - score_mean||^2/d - sample_variance/M`; negative estimates are retained. Independent reference means are uncertain; corrected leave-one-out behavior is explicitly labelled. Aggregate risk/R0 and skill use matched training-mean test risk. Gaussian NLL, marginal CRPS, energy, PIT, coverage and sharpness have numeric score components and sampled draws. Scores are available in physical and standardized coordinates.

Biophysical forks identify `v_C` and the combined `R_m - v_C`. They do **not** separately identify observation, compression and fitting gaps, the true EEG-history Bayes floor, or a human biological threshold. Conditional-mean and distributional prediction remain mathematically well-defined even when an individual future is not recoverable.

## P2: replication and context coverage

Nested subsamples use M=1,2,4,8,16,32,64,128 only when the saved score pool supports them. Mean/variance error against an independent reference is pending without that pool. M=1 direct sample variance is unavailable, rather than zero. Reference uncertainty and finite-pool reuse are reported; containment of a finite reference mean is not exact-truth coverage.

A pooled EEG-only conditional Gaussian uses cross-fitted training-circuit residuals from one future per context, then scores held-out circuits. This is a qualified distribution model, not a zero-variance straw man. Its covariance includes possible hidden-state ambiguity and is assessed by held-out proper scores/calibration. It is not scored against state-fork covariance as if those conditioning sets were identical. Exact hidden-state controls test the surviving ambiguity term as M grows.

The optional fixed-budget extension fits actual nested allocations (K,M)=(256,1),(128,2),(64,4),(32,8),(16,16), each with 256 training outcomes. Four circuits by four regimes with 16 prefixes per stratum need 768 unique training continuations in the union. This is additional to development and held-out reference/scoring data. Existing 16 evaluation contexts cannot manufacture this coverage. Its learners share a fitted model family and held-out panel; repeated outcomes are not counted as new prefixes or subjects.

## P3 and P4: declared task information and scale

See [P3/P4 implementation notes](eegrepr_h1_p3_p4_notes.md) for representations, quantizers, decoder fitting and task definitions. Exact controls include the ten-bit nuisance example, equal-rate control, variance-only classification/MSE example, Fano and Gaussian-channel/rate-distortion checks. All information outputs use bits. No deterministic-continuous latent MI percentage is estimated. `H(T)-CE` is decoder-dependent; differences of lower bounds are not certified MI losses.

The bounded compression profile uses dimensions 8/32, code budgets 32/128 bits and label budgets 8/all; the dedicated optional compression config adds dimension 128, 512-bit codes and label budget 32. Impossible dimensional/codebook combinations are unsupported. A fixed B-bit packet is an operational upper bound, not a joint entropy estimate. Training-fitted quantizers report clipping and dequantization. Direct EEG readouts provide a comparator; decoder weakness is not evidence that a representation contains no information.

Scale-core executes seven named jobs; scale-targeted executes 25, rather than a factorial inferred from unused lists. Each target gets its own R0, floor and units; input-rate/precision comparisons verify physical targets and context/future IDs actually match. Task retention is joined to risk, compute and latency where computed. Noise uses 20/10/0 dB on observed history only, with clean future targets held fixed. Rates, target averaging and latent bits are distinct axes.

Nonzero diffusion, bursts and kinetics are later generator-validation extensions, not executed results. Their labels are exported if configured, but constant factors in D=0/default-kinetics data are unsupported tasks. No clinical labels are invented. Development-only mean-rate, stationarity, observability and label-independent nuisance audits are required before claiming new mechanism tasks; no full factorial or clinical conclusion is supplied by this implementation.

## Plans and paper mapping

| Configuration | Independent evaluation circuits | Evaluation contexts | Evaluation continuations | Additional train/development | Explicit target/input jobs |
|---|---:|---:|---:|---:|---:|
| timing_gate | 2 | 4 | 8 | 0 | 1 |
| bounded_development | 4 | 16 | 256 | 256 + 64 | 1 |
| expanded_reference | 4 | 16 | 3,584 | 256 + 64 | 1 |
| fixed_budget_extension | 4 | 16 | 3,584 | 768 + 64 | 1 |
| scale_core | 4 | 16 | 256 | 256 + 64 | 7 |
| scale_targeted | 4 | 16 | 256 | 256 + 64 | 25 |
| confirmation_plan | 12 | 96 | 36,864 | 512 + 256 | 1 |

The eight preselected expanded contexts receive 256 reference plus 128 score futures each. The other eight receive 32 + 32. Selection is balanced over four circuits and four regimes and fixed before any model error. The confirmation plan uses new disjoint structures and five optimizer seeds; >=12 circuits is provisional, not a power guarantee. Freeze endpoints and refine this design after development precision estimates, using a fresh config/root if changed.

Historical arithmetic: 3584 * 210.86 s / 3600 = 209.93 **summed old-trial hours**, not measured H1 cost. Workstation plans require a checksummed full-network eight-continuation gate and report measured RSS/storage, serial concurrency, duration extrapolation, actual job counts and fit counts. Fitting, derivation and scheduling overhead are disclosed separately. Replaying prefixes remains part of the cost. No local tiny-network timing qualifies the gate.

| Paper claim/table | Machine-readable evidence | Figures |
|---|---|---|
| H1a risk accounting | controls + benchmark per-context NPZ, models, fits, per-circuit CSV | controls; summary/*_matched_risk |
| H1b finite M and context coverage | replication JSON/NPZ, reference uncertainty, candidate allocations | replication/*/replication |
| H1c task-specific finite-rate retention | information controls, codebooks/codes, readout predictions/logits and surface | compression/rate_distortion_task_surface |
| H1d scale/precision tradeoffs | scale/executed_scale_table.json, exact paired arrays, compute/latency | scale figures + FFT controls |
| H2 | explicitly work in progress | no new main leaderboard |

Every phase distinguishes complete, pending, unsupported and failed items. Compact ZIPs include target/prediction/code arrays, metadata, source snapshot, plots and hashes; native HDF5 is excluded explicitly by default and can be included with `h1-export --include-native`. Paper results remain pending until the workstation evidence passes its diagnostics. Negative findings must be retained.
