# H1 finite compression and scale diagnostics

This note describes the implementation in `experiments/eeg_representation/h1/compression.py` and `scale.py`. It does not report a completed full-network H1 campaign. The frozen COSYNE implementation and original pilot outputs are unchanged by these modules.

## Scope and shared data

`run_compression(config, output_root, device='cpu', resume=False)` and `run_scale(config, output_root, resume=False)` consume **actually derived** jobs through the common `dataset.list_jobs` / `load_panel` interface. Configuration lists do not establish execution. Each panel carries context, circuit, future, target and normalization identifiers; numerical arrays use NPZ without object arrays or arbitrary pickle loading.

The target and acquisition registries own physical operators, filtering, availability, normalization and timestamps. These modules do not introduce another EEG simulator or another target definition. In particular, the primary target is the registered future block of native-sample box averages. It is a declared common projection, not an instantaneous native signal or ideal anti-aliasing. Recorded causal input-filter latency is not subtracted from sample availability.

Training, development and evaluation circuits must be disjoint. The panel normalizers are fitted to training pre-cutoff data. Representation fitting and latent projection/quantization use training histories; forecast encoder supervision uses the designated first independent training future. Evaluation outcomes are not inputs to representation fitting, quantizers or readout selection.

## Exact controls

The controls produce JSON and PNG/PDF figures where applicable:

* Enumerate all 1,024 values of ten independent fair bits, `V=(T,N1,...,N9)`. The nuisance-nine-bit code retains nine input bits and zero task bits. The task-three-bit code retains three input bits and the complete one-bit task. The nuisance-three-bit code provides the equal-rate comparison and retains zero task bits. Fractions such as 9/10 apply only to this finite exact example.
* A variance-only class has equiprobable contexts with conditional futures `{-1,+1}` or `{-3,+3}`. Both conditional means are zero. Retaining or discarding the class gives the same optimal mean-MSE, five, while task CE changes from one to zero bits and optimal discrete future CE changes from two to one bits.
* The scalar Gaussian channel reports `I=.5 log2(1+sigma²/tau²)` and posterior MSE `sigma² tau²/(sigma²+tau²)`. The scalar Gaussian rate-distortion expression `sigma² 2^(-2I)` agrees in this qualified control. It is not asserted as a universal EEG equality.
* Fano's necessary bound is implemented for balanced `K>=2`, with `0<=delta<=1-1/K`, in bits. Evaluating the formula at a finite-sample observed error is an illustrative estimate, not a statistical certificate about its population value.
* Full orthonormal complex FFT preserves per-future squared error and sample covariance trace. Tests include correlated vectors, odd/even lengths and affine unit scales. Loss divides by the original waveform dimension. The shared target registry's real-valued FFT packing multiplies concatenated real/imaginary parts by `sqrt(2)` so mean error over `2D` coordinates equals the original `D`-coordinate error.
* A quarter-cycle phase shift gives different waveforms and identical full power spectra. It demonstrates information removal by a different target; it does not establish a neural forecasting limit.

## Operational quantizer and feasible points

The representation bank reuses feature18, PCA, an untrained shared encoder, masked-MSE and forecast representations. `benchmark.get_representations` saves its numeric matrices, weights/configuration and provenance. It reuses the already frozen same-job deterministic forecast encoder when available; otherwise it fits the explicitly recorded small diagnostic forecast representation. This is not a new H2 curriculum sweep.

For each requested dimension, a PCA rotation/truncation is fitted on training latent vectors. Requested dimensions above the source dimension or `N_train-1` are unsupported; dimensions are never silently padded. Numerical rank is recorded. A fixed coordinate may still occupy code bits, which is why dimensionality and transmitted bits are not information estimates.

Each coordinate receives `floor(B/d)` bits; the first `B mod d` coordinates receive one additional bit. This implementation supports one through sixteen bits per coordinate, with byte-aligned total budgets. Training coordinate quantiles 0.005 and 0.995 define the default range. Values are clipped to that frozen range, rounded to one of the declared uniform levels, and dequantized with the same stored bounds. Constant coordinates use one level while retaining their declared transmitted allocation. There is no stochastic latent noise or entropy model.

The implementation writes the actual packed `uint8` codes, verifies a packing/unpacking round trip and checks **exactly B transmitted bits per window**, excluding the shared codebook/model overhead. Codebook bytes and all model artifacts are separately inspectable. This is an operational rate upper bound, not measured joint entropy or a universal retained-information fraction. Train/development/evaluation coordinate and example clipping fractions and dequantization MSE are recorded separately.

| Dimension | 32 bits/window | 128 bits/window | 512 bits/window |
|---|---:|---:|---:|
| 8 | 4 bits/coordinate | 16 bits/coordinate | Unsupported: would require 64 bits/coordinate |
| 32 | 1 bit/coordinate | 4 bits/coordinate | 16 bits/coordinate |
| 128 | Unsupported: fewer than one bit/coordinate | 1 bit/coordinate | 4 bits/coordinate |

These are coding feasibility conditions, not guarantees that a specific bank has that dimension or training rank. For example, the 18-feature bank supports requested dimension eight, not 32 or 128. Unsupported points remain explicit records in the surface.

## Readouts and interpretation

Direct EEG-history readouts and dequantized latent readouts use the same declared regularization grid and readout family. Linear heads use ridge regression or logistic regression. Flexible heads use a training-fitted standardized random Fourier feature map followed by the same head types. Their random feature widths, weights, phase, training-fitted scale and all head coefficients are numeric artifacts. Parameter counts and measured fit/inference times are disclosed; different input dimensions do not imply matched parameter count or compute.

Available tasks are:

1. Reconstruction of the observed, standardized history, with its own training-mean reference and physical-voltage MSE conversion.
2. Prediction of the **same shared future TargetSpec**. A mean head is fitted on the first valid training future per prefix and scored against every valid evaluation future. A declared homoskedastic diagonal Gaussian is calibrated using development residuals. Shared `scoring.gaussian_scores` supplies Gaussian NLL, marginal CRPS/PIT/coverage, energy score and independent numeric draws. The distribution is EEG-conditioned; its variance is not asserted to equal causal-state fork variance.
3. Rhythm and carrier classes. Carrier scoring excludes homogeneous contexts; class-set mismatches and constant training classes are unsupported rather than suppressed warnings or coerced logits.
4. Sine/cosine of the clean **pre-cutoff EEG** phase at its dominant 4–30 Hz Fourier bin and pre-cutoff log mean-square power. The shared task metadata records the maximum declared history used for these measurements. These are retrospective causal measurement diagnostics, not the latent instantaneous phase and not clinical labels.
5. Mechanism tasks only when the panel explicitly declares them and supplies their arrays. Absent diffusion, burst or kinetics contrasts are not invented.

Label budgets count selected training contexts. Selection cycles across circuit/class strata, and development circuits select regularization. Evaluation circuits remain untouched by tuning. The supervised decoder audit additionally uses outer training-circuit holdouts with inner training-circuit tuning. It needs at least three training circuits and adequate class support; otherwise its unavailability is recorded. This audit is **conditional on frozen upstream training representations/quantizers**. It is not end-to-end cross-validation of encoder retraining, particularly when the forecast encoder has already used training future labels.

Classification output includes probabilities, normalized log-probability logits, classes, warning records, CE in bits, empirical held-out task entropy and `H_hat(T)-CE`. The latter is a decoder-dependent, finite-sample estimate of a task-MI lower bound. Negative estimates are retained. Differences of estimates are not certified MI losses. Weak decoding can reflect approximation or fitting error, not absent information. Marginal task readouts do not certify joint sufficiency for the entire task family.

Numeric predictions and per-context losses permit score recomputation. Per-circuit scores separate the independent circuit unit from within-circuit contexts/futures. Small numbers of evaluation circuits do not justify narrow population claims.

## What scale analysis actually does

For every executed panel, `scale.py` reports the actual input rate, input sample count, history, acquisition precision, future lead/duration, target dimension/grid and target fingerprints. Acquisition bits, floating-array storage precision and latent code bits are distinct fields. Stored input timestamps must be causal and match the claimed rate. Target dimensions are checked against the shared specification when present.

Each target receives its own training-mean reference risk `R0`, causal-state fork variance estimate when there are at least two futures, physical/standardized risks and per-circuit summaries. M=1 variance is unavailable, never silently zero. Coarsened, RMS, log-power and magnitude targets are distinct estimands; smaller absolute MSE in their units is not an information gain. The legacy delayed-stream target is not silently equated with a new physical-time target. Physical scalar units and conventions come from the shared registry; old pilot RMS/log-total-power results with incompatible conventions cannot be subtracted from the new target risks.

An input-rate/precision/history comparison qualifies as fixed-target only when physical target fingerprints, contexts, circuits, scoring/reference future IDs and masks, and physical target arrays agree. A same-fingerprint pairing failure is reported explicitly. A waveform/full-complex-FFT comparison additionally verifies the shared registry transform against the saved physical arrays and its loss-preserving scaling; input changes remain a separate axis. All other target comparisons retain their distinct reference risks.

When `benchmarks/<job>/summary.json` and scored NPZ artifacts exist, scale analysis checks their schema, checksums, target fingerprints, context/future pairing and actual targets. It recomputes MSE from the saved predictions and reports model risk/R0/skill, per-circuit proper scores, physical risk, training accounting and inference time. It links existing compression task-retention rows. It does not refit models or manufacture a model comparison from data-only jobs.

DFT-bin spacing is reported as a descriptive window quantity: 128 ms corresponds to 7.8125 Hz. This is not a universal impossibility theorem for parametric sinusoidal estimation. Token counts, actual target dimensions, saved fit accounting and input-filter support/delay allow declared latency/compute comparisons; configuration lists alone are never counted as completed experiments.

## Recommended bounded profile and cost

For the first finite-compression run, use dimensions `[8,32]`, budgets `[32,128]`, label budgets `[8,'all']`, the five existing representation types and the two declared readout families. Keep `[8,32,128]` / `[32,128,512]` / `[8,32,'all']` as an explicit optional extension after inspecting measured runtime/storage.

With sufficient rank and all seven tasks available, the bounded profile has 18 feasible latent points (two feature points plus four points for each of the other four banks), plus the direct-history comparator. That gives at most **532 main task/readout/budget fits per panel** before class/support skips. The full suggested grid has 30 feasible latent points plus direct history, giving **1,302 main fits**. Each main fit also performs regularization candidates and, where supported, nested supervised cross-fitting. These counts exclude the small representation bank fits and are not wall-time measurements.

The bounded profile therefore reduces the principal readout count by about 59% and avoids the largest latent/code extension while retaining two dimensions, two feasible rates and a label-learning comparison. Direct-history reconstruction heads and saved forecast draws can still consume substantial storage. Measure this stage on the planned development recordings; do not extrapolate the tiny analytical smoke into a GPU or full-network runtime promise.

## Executed local evidence

The dedicated compression/scale test suite passed **19 tests** after implementation. It covers exact bits/Fano/Gaussian controls and a 100,000-sample Gaussian distortion check; exact code lengths and round trips for every supported requested pair; train-only quantizers; split/class-support guards; numeric CE rescoring; nested decoder folds; checksum-aware resume/tampering; FFT covariance trace and unit scaling; shared-registry FFT equivalence; physical target pairing; causal availability/rate checks; and unavailable M=1 variance.

Two additional **analytical engineering** runs were executed locally while implementation was in progress:

* `/tmp/eegrepr-h1-p3-p4-analytic-smoke`: 52 analytical continuations from the smoke's disjoint circuit-ID splits, one derived job, three diagnostic benchmark models, then compression/scale. Generation took 1.632 s, derivation 0.281 s, benchmark fitting/scoring 1.344 s, compression 4.061 s and scale 0.363 s. The directory contained 16,256,438 bytes at measurement. Compression produced 60 completed readouts and ten explicit unsupported carrier-class entries because the two-regime smoke has only one rhythmic carrier class. Scale independently reaggregated three model score files and linked 60 task-retention entries. This run used five-step representation/model fits and is pipeline evidence, not a competent-model or information-limit result.
* `/tmp/eegrepr-h1-scale-pair-smoke`: 52 analytical continuations reused across three derived jobs (500-Hz floating input, 250-Hz/4-bit input, and full-complex-FFT target). Generation/derivation/scale took 3.434 s and recorded 14,109,042 bytes. The acquisition comparison had exactly matching physical future arrays and context/future IDs; waveform/FFT pairs matched the shared registry's loss-preserving transform. No benchmark was fit for this three-job run.

These measurements describe the recorded development snapshots and machines, not the final full campaign. Later code/provenance changes intentionally invalidate compatible resume of older outputs. The root implementation report records the final frozen-source checks separately. No full 40-cell compression study, population-level conclusion, confirmatory task-retention claim or multi-day campaign was run by this work.

## Artifacts and resume

The top-level CLI holds the output lock. Analysis calls additionally validate existing H1 provenance/config/source compatibility. Stage identities include configuration, actual panel manifests and H1 code. Completed manifests checksum every saved artifact, and a compatible resume verifies those hashes. Existing completed output without explicit resume, changed source/config/data identity, or a missing/corrupt artifact is rejected. JSON/NPZ writes are atomic through the shared artifact helpers.

Useful outputs include `compression/controls`, `compression/<job>/representations`, each point's `codes.npz`, `codebook.npz/json`, numeric readout weights and `predictions.npz`, `compression/surface.json`, `scale/<job>/context_scores.npz`, `scale/executed_scale_table.json`, figures and completion manifests. The parent workstation script supplies the real commands and ordering; large launches remain explicit user actions after the measured cost gate.
