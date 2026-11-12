# H1 implementation and validation report — 12 November 2026

The new H1 implementation is complete for workstation development, with engineering validation and explicit scientific qualifications. The eight-continuation full-network timing gate and every full H1 campaign remain **not run**. No new biophysical H1 result or H2 conclusion is claimed.

Branch: `codex/eegrepr-h1-information-v2`. Base: `4532e19f29cfcf318f8afbe7c25850f29981803e`. Tested implementation commit: `a5247e69a89c8bc5bf6c588898108bb292f9c768`. The delivery includes a subsequent documentation/evidence commit; `git rev-parse HEAD` gives that delivery commit. The implementation is pushed normally to `origin`; it is not merged. The original COSYNE checkout and all pilot outputs were preserved.

Start with [the exact workstation commands](eegrepr_h1_workstation.md). The runnable launcher is `scripts/eegrepr/run_h1.sh`; no-argument invocation only prints help. [The protocol](eegrepr_h1_protocol.md) defines targets, conditioning, statistical units, phase meanings and paper mappings. [P3/P4 notes](eegrepr_h1_p3_p4_notes.md) document quantizers and decoder limits. [The numeric validation ledger](eegrepr_h1_validation.json) records source hashes, timings, counts, phase qualifications and log hashes.

## Implemented and actually exercised

| Phase | Implementation | Local evidence | Remaining workstation evidence |
|---|---|---|---|
| P0 | Frozen legacy protocol, shared native TargetSpec, explicit jobs, immutable manifests/resumes, train-only acquisition/normalization | Timing/causality/isolation tests; actual two-rate/FFT pairing check; compatible full-pipeline resume | Inspect accessible pilot raw data; no claim those remote files are currently present |
| P1 | Exact Kalman/Poisson controls; matched simple/neural objectives; empirical and unit-Gaussian references; numeric proper scores | Exact controls pass; eight-model analytical run; four-baseline 2E/1I run; native replay/early-stop equivalence | Full 32E/8I gate, bounded fit competence, population variation and matched state floors |
| P2 | Finite-M curves/corrections, independent pools, pooled one-future uncertainty, real fixed-budget allocation execution | Analytical reference/scoring pools; pooled held-out scoring; exact/negative/M=1 tests | High-M reference precision; 768-training-future allocation extension; circuit-level precision |
| P3 | Exact bits/Fano/Gaussian/variance-only controls; five representation types; fixed codes and readouts | Packing tests; analytical compression 60 completed/10 unsupported rows; all-bank helper tested | Bounded finite-rate task retention on full-network data |
| P4 | Seven-job core/25-job extension, strict same-target tests, FFT invariance, operator-specific R0 | Actual three-job analytical pairing check; single-job final smoke reports; FFT tests | Seven-job scale campaign, input-noise sensitivity, qualified later mechanism extensions |

The legacy scientific source hash remains exactly `9a1fa6f48f04125143569c062b839859c820c61df17ba90df3487ee3389ae3d0`. The local prior review ZIP has the expected pilot provenance and 640 trial records but no native pilot HDF5 or fitted checkpoints. It does not supply new raw-data rescoring or establish that step 5 already ran.

## Tests and measured validation

- **65 new tests passed**, 15.265 s, real NEURON integration enabled, no skips. Coverage includes exact risks, finite-M correction, negative excess, one-future pooling, bits/code lengths, native timing, future perturbation, float/unit operators, splits, source/config/checksum resume, fixed allocations, trusted timing audits, actual job execution, raw score recomputation and portable export.
- **70 existing tests passed**, 18.125 s, real NEURON tests enabled, no skips. These include actual 40-cell pending-event checks, independent phase futures, synapse kinetics, background streams and online lifecycle.
- Isolated-cell field polarization passed all five criteria. Axial DC gain was -0.104357 mV/(V/m), linearity R² 0.999979, transverse/axial ratio 1.51e-9.
- Online/legacy comparison passed: 8,000 online samples against 8,001 legacy samples including t=0; correlation 1.0, relative RMS error 2.55e-11, configured/effective temperature both 6.3 C.
- The documented reduced A/B and stimulation smokes completed. A/B held-out accuracy was 0.333 on this tiny check. Stimulation reachability remained **NOT PASSED** (distance gain -0.422152, alignment -0.768771). These are preserved execution/regression outcomes, not new efficacy evidence.
- A nonzero-diffusion 2E/1I early-stop test matched a longer episode exactly through the required final sample, including causal checkpoint and event-queue state. No checkpoint-restoration cache is claimed.
- `bash -n` passed. The actual launcher completed analytical and simulator modes and a compatible analytical resume. An extracted ZIP outside the checkout independently recomputed all eight analytical benchmark MSEs and reaggregated the saved circuit scores without Git, NEURON or model pickles. Figures were visually checked, including a rendered PDF.

| Functional qualification smoke | Native episodes | Evaluation circuits / contexts | Models | End-to-end wall time | Peak process RSS | Output bytes before extra exports |
|---|---:|---:|---:|---:|---:|---:|
| Analytical | 64 | 2 / 4 | 8 | 22.02 s | 643,473,408 | 32,037,738 |
| NEURON 2E/1I | 12 | 1 / 2 | 4 | 25.37 s | 555,958,272 | 14,337,953 |

Both qualification roots record the clean tested implementation commit above and backend/engineering status. After the PDF-label correction at `9534b68ee9ef49b3b9d33506c36254fe8daee410`, both end-to-end runs were repeated (20.80 s analytical; 23.12 s simulator). All saved truth, mean, variance and risk arrays matched the qualification runs exactly for all twelve model records. The corrected PDF was rendered and visually checked, and eight benchmark scores were recomputed from its extracted package outside the checkout. Final delivered evidence uses this corrected source snapshot. Timings include CLI startup, exact controls, generation, fitting/analysis and export. These validations ran concurrently on the CPU laptop (CUDA unavailable), so these times do not estimate isolated GPU training or full-network simulation speed. The two engineering runs include one extra numerical replay per generation invocation, outside the counted saved episodes. Measured generation timings and native byte counts are in their timing artifacts.

## Qualifications and deviations requiring attention

1. The bounded primary target is a **64-value native future box-average projection**, not the legacy delayed filtered waveform and not native 16-kHz waveform prediction. Lead/support, finite FIR response/delay and units are explicit. Outputs retain float64 for state-floor scoring. This target decision is reflected in the protocol and plans.
2. The final analytical smoke's tiny encoder adds **23.94%** parameters for the Gaussian variance head, above its declared 15% tolerance. This diagnostic is visibly failed; the run cannot establish a parameter-matched objective comparison. The overfit diagnostic itself passed (training risk ratio 0.00895), as did the exact learnable-signal control. Full-size parameter/compute accounting must be inspected on the workstation. No simple baseline is padded to fake capacity parity.
3. Carrier readouts are unsupported in the two-regime smoke because it has only one rhythmic carrier class. Ten such rows are retained. The actual bounded design has three rhythmic carriers. Missing-class diagnostics and warning records are saved; warnings are not globally suppressed.
4. M=1 direct variance, absent high-M sample sizes, absent independent references and insufficient candidate allocations remain explicitly unavailable/pending. The tiny simulator has too few training circuits for pooled uncertainty fitting; the analytical run exercises it. The bounded full-network config has no independent reference pool, so finite-reference RMSE/containment requires the expanded design.
5. The bounded compression grid is reduced to dimensions 8/32, rates 32/128 and label budgets 8/all. The optional compression config retains 128 dimensions, 512 bits and label budget 32. Coding/rank failures remain unsupported, rather than padded. Decoder cross-fitting is conditional on frozen upstream representations; it is not encoder-retraining cross-validation.
6. The default core is D=0, tonic modulation, default kinetics and clean EEG. Input-noise plans are implemented separately. Nonzero-D, burst, kinetics and gain/nuisance mechanism studies need their own development qualification before new task claims. No automatic factorial or invented clinical tasks are supplied.
7. The fitted EEG state-space baseline is a delay-coordinate approximation, with its discrete propagation approximation disclosed. It is not the exact physical-state Kalman oracle. The true nonlinear EEG-history floor and separate observation/compression/fitting gaps remain unidentified.
8. Only prefix replay with early stopping is qualified. More efficient checkpoint caching and low-rank Gaussian covariance are not implemented; current distribution models use disclosed diagonal covariance. Source/RSS/compute records preserve these limitations.

The independent code review also corrected aggregate risk/R0 weighting, secured the timing gate against mutable-alias measurements, ensured the first-future training budget matches across baselines, added the requested prospective N(mean_train,I) reference, and made exports independently inspectable. These are H1 implementation changes; correct legacy fork corrections were retained, not claimed as new inventions.

## Next action and decision sequence

Run the workstation doctor/tests/smokes, then the eight-continuation timing gate. Inspect its actual cost plan before explicitly launching bounded development. Review model competence, proper-score calibration, paired circuit scores and compression class/rank diagnostics before expanding. Use expanded references to quantify finite-M uncertainty; use the separate fixed-budget extension only if context-versus-replication allocation is the decision of interest. The seven-job scale campaign follows its own measured plan. Freeze any revised protocol and precision-based confirmation design before touching new confirmation outcomes.

No multi-day job, merge, force-push, remote deployment or unattended automation was launched. Full-network H1 results, CUDA validation and confirmatory claims remain pending user-run workstation evidence. Implementation is not blocked by those deliberately deferred experiments.
