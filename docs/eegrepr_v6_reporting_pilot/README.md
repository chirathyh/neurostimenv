# V6 eight-worker reporting pilot

This separate experiment develops the H1–H4 reporting templates on **already inspected historical data**. It is not confirmation. It never enters the N192/N257, 768-episode numerical, or automatic precision-expansion execution paths. Original V4/V6 configurations and native recordings remain unchanged.

Entry point: `scripts/eegrepr/run_h1_h4_v6_reporting_pilot.sh`.
Configuration: `configs/eeg_representation/v6_reporting_pilot/pilot.yaml`.
Implementation: `experiments/eeg_representation/h1/v6_pilot/`.
Commands: [WORKSTATION.md](WORKSTATION.md).

## Fixed allocation and budget

The source is the 832-episode bounded-prospective panel. Its standard archive SHA256 is `6818befa7ca70df3a0272bf69328f9e2d2316817493e49a13bac6e4b74a5817f`. Native HDF5 recordings must also be available: the standard archive alone cannot derive the entire matrix. The importer validates the original manifest, configuration and each native file before reuse. Old generation provenance is retained separately from new analysis and state-export identities.

| Component | Additional workstation NEURON calls |
|---|---:|
| Eight-worker measured concurrency check; captures reused | 8 |
| Remaining complete-state cutoff recoveries | 48 |
| Two exact-forcing full-prefix replay controls | 2 |
| Two original contexts: 24 additional scoring futures each | 48 |
| Default workstation total | **106** |
| Already spent local engineering calls, including failures | **21** |
| Combined default | **127 of 160** |

The immutable engineering debit is documented in `engineering_invocations.json`. Every new workstation attempt is charged **before** entering NEURON, including failed/interrupted calls and retries. The remaining reserve is 33 calls. If the two historical H2 continuations are incompatible, a separately registered one-context M32/L32 alternative needs 64 futures rather than 48, bringing the intended combined allocation to **143**, leaving 17 calls. Existing failed calls can consume that reserve. The runner refuses to exceed 160; it never treats retries as free. It does not blend a fresh context with the old context.

The selected historical H2 contexts are `s36012-p0000-R0` and `s36013-p0000-Ralpha`, with original M8 scoring and L32 reference pools. Their complete original prefixes, physical states, caused queues and drive histories must match before adding scoring draws 8–31. The fallback circuit is separately registered as 37700, outside the old panel. H2 uses nested M=1,2,4,8,16,32 and finite L32; it does not use the later L128 protocol.

The matched complete-state cohort contains 32 training contexts (eight circuits), eight development contexts (two circuits), eight separate calibration contexts (two circuits) and eight evaluation contexts (two circuits). Every split has all four regimes. Development circuits 36008/36009 and calibration circuits 36010/36011 are separated. Evaluation circuits are 36012/36013. Ordinary eight-circuit historical evaluation is reported separately. Replay has its own two-context population and recomputed comparator risks.

## State information and fitted models

The versioned export and its per-field inventory include physical units, segment/contact identities, shapes and source fields. The declared model is restricted to the implemented fixed-step HH/passive/Exp2Syn circuit at 6.3 C; unknown mechanisms or unhandled state cause an error. It is not a claim of biological completeness for arbitrary neurons.

Captured inputs include:

- Every segment voltage, both extracellular layers, all live HH/passive/ion state and parameters, and the actual spike-source detector state and preceding-step voltage.
- Exact external/recurrent Exp2Syn A/B states, kinetics, contact identities, weights and connectivity.
- Every causally pending recurrent delivery, with emission/arrival/contact/weight attributes; actual NEURON queue entries are cross-checked against caused events.
- Current shared-drive phase/rate/envelope state, geometry, forward transforms, acquisition states and all relevant physical parameters.

Unknown future external schedules, RNG seeds/states, and presampled future drive paths are excluded. Hashes and seed identifiers remain binding/audit metadata, never predictor coordinates. Changes to only the future draw must leave the complete-state input identical. Recovery uses full-prefix reruns, not a restorable checkpoint.

The adapter preserves all coordinates and presence masks. Structural paths/shapes may be unioned across the registered cohort, but numerical scaling is fitted only on training values with physical scale floors. There is no event-count substitution, queue truncation, or hidden deletion of constant/rare fields. Ridge is the fitted model's own compression. Exporting the state alone does not satisfy the completion gate.

Each of the 60 settings fits ridge alpha `{0.1,10,1000}` plus a training-mean fallback. Development MSE chooses the selected policy. The predeclared, actually fitted **alpha=10 state-dependent candidate** is retained and scored even if the selected policy is the mean. Complete-state fitted errors are never replaced by the state-variance reference. Model schemas, coefficients, predictions, losses and regularization paths are saved.

EEG fits use the same bounded ridge grid with the last 64 samples (128 ms) and all 2000 samples (four seconds). Four additional one-seed TCN fits cover A1 h0/h10 W128 crossed with the two histories, at most 500 updates each, with development checkpoints/shrinkage/mean fallback. The source's compatible forecast representation is reused after exact physical history/target/ID checks. Its original 256-context encoder pretraining differs from the 32-context matched readout training, and is explicitly disclosed. If unavailable, a single bounded 200-update forecast encoder per cohort is permitted; this is not an objective comparison.

Feature18, PCA and forecast codes use d8 projected unquantized and d8/B32 controls. B32 means four bits per coordinate and 32 packed payload bits, with model/codebook storage separate. Fixed training-only clipping bounds are audited. Future readouts use A0 to keep the acquisition constant; A0 is therefore the main code/state/EEG information-ladder panel. A1 is shown separately for immediate-history and TCN comparisons.

## Targets and interpretation

All 60 acquisition-target settings reuse native episode support: A0/A1/A2 crossed with six native waveform targets (h0/2/10, W32/128), two one-second Fourier targets (h0/10), and six direct band-energy plus six log-energy targets. At h0 the cutoff sample is excluded; the first bin is `(t0,t0+2 ms]`. Block targets are available only at their end. Acquisition timing is never backdated.

Normalization and the mean baseline are target-specific and training-only, shared across acquisition/history/method comparisons on the same cohort. Physical waveform/Fourier MSE has units V²; direct-energy MSE has units V⁴; log-energy MSE has squared dimensionless log units. These are distinct from past reconstruction NMSE (baseline 1), the error display scale (baseline 100), relative MSE reduction in percentage points, and carrier cross entropy in bits. Carrier decoding is conditional on rhythmic membership; rhythm balanced accuracy is also retained.

H1 retains unbiased repeated-future variance, measured fitted errors, unknown ideal EEG error, and negative finite-M excess estimates. Fitted-model differences do not identify pure observation/compression/fitting gaps. Privileged replay error and theoretical zero are separate rows on explicitly identified populations. H2 corrects for finite-reference mean noise and reports reference uncertainty; M1 variance is unavailable. Cross-regime conditional Monte Carlo uses synchronized future deletions. Circuit-level intervals are descriptive, with the two-circuit population intervals deferred. Offset/duration comparisons use distinct target-specific denominators and are not described as the same prediction problem.

## Runtime and completion

Eight means eight independent spawned simulation processes, each pinned to a distinct allowed CPU with one BLAS/OpenMP thread. Preflight requires at least eight CPU equivalents, 8 GB available RAM and 20 GB free destination storage. No worker-count fallback is allowed. Eight-worker timing measures actual overlap, RSS and native bytes before projecting the remaining allocation. CUDA is used only for fitting.

Using the historical single-call timing, 106 full-length calls would be about 3.0 worker-hours; cutoff-only recoveries are shorter. This is a provisional planning calculation, not measured eight-way elapsed time. An ideal division by eight omits contention, queueing, replay/recovery differences, derivation, fitting, PDF rendering and export. The actual-host plan records these distinctions. Device-specific fitting elapsed time is recorded; it is not claimed to be GPU kernel utilization time.

The complete command derives, validates, generates, fits, scores, creates all table formats and vector/PNG figures, compiles `report/pilot_findings.pdf`, renders every page, checks layout, and exports lossless numeric evidence. A completion receipt additionally requires all 60 measured complete-state candidate/selected rows and their verified artifacts. A PDF containing incomplete evidence never produces a completed experimental receipt. Automated layout/render checks do not replace the reader's scientific or aesthetic review.

The export includes complete-state fitted weights and numeric evidence; large native files and ordinary readout weights have explicit omission manifests and hashes. NPZ contents are deduplicated without changing NPY bytes, precision or dtype. Models are never inspected by unpickling supplied archives.

Review the pilot's templates before deciding whether to run any later confirmation. Modest or negative outcomes remain valid pilot results.
