# NeuroStimEnv v6: bounded H1–H4 evaluation

This implementation preserves the v4 generation law and historical data. Python remains under `experiments.eeg_representation.h1` for compatibility. H1 is prediction-risk accounting; H2 is finite-future estimation; H3 is task retention; H4 is target/measurement scale. H5/H6 are deferred. The 40-cell, 6.3 °C simulator is a toy test bed, not calibrated human EEG.

## Status and design

The default candidate has 32 train, 8 development, 24 independent calibration and 192 fresh test circuits. It contains 13,056 core episodes, of which the original 320 train/development episodes are verified and reused. The separate coupled-integration panel has 768 episodes. Each new episode uses full-prefix replay and 7.010-s support. No saved EEG prefix is treated as a runnable NEURON checkpoint.

The historical 1.575-pp A1 influence SD used **eight** futures; the primary v6 protocol uses four. Recomputing on the first four gives SD 1.975962 pp and conservative upper SD 3.551102 pp, yielding a 0.57899-pp half-width at N192. The historical-only minimum is N257. This is provisional: full-history Phase0 may require more. The runner recomputes all relevant primary precision components from inspected evidence, selects the registered minimum N before any fresh test generation, writes `selected-config.json`, and recomputes the measured resource plan. It never extends N from a new test effect or p-value. N above the registered extension or any resource cap is a saved blocker, not permission to truncate the design. The historical feature-versus-forecast contrast remains descriptive and does not consume another primary precision family.

Default N192 remains visible in the checked-in config. The proposed N257 design has 15,136 core episodes / 14,816 new episodes, plus the separate 768 numerical episodes. It is not an already-authorized executable plan: full-history qualification, numerical confidence bounds and actual-host cost gates must pass.

## Scientific operators and learners

Phase0 reuses the exact saved physical arrays and produces nine fixed-target comparisons (A0/A1/A2 × short native/long native/T3), plus 18 separately named direct energy/log-energy targets. A0 is the original imperfect FIR, A1 is completed 2-ms box bins, and A2 is the causal 4–30 Hz SOS. No operator is silently repaired. Physical target arrays are verified identical across acquisitions.

Retained history is four seconds (2000 coordinates at 500 Hz). The complete 14 tail-ridge grid remains a diagnostic; eligible full-history candidates use raw history, train-PCA64, a fixed real Fourier summary, Feature18, and a small residual TCN. The TCN has ten dilated kernel-3 layers, width16, receptive field2047, and fewer than50k parameters. The same architecture runs a64-sample ablation. Five seed-specific risks are averaged inside circuit; this is not an ensemble prediction. A train-mean fallback is always eligible. Optimization is bounded at10,000 updates with development checkpointing, early stopping and residual shrinkage. Every candidate, learning curve, selected step, actual count and elapsed time is retained.

T3 uses 54 real orthonormal coefficients of the16,000 native samples in (6010,7010]ms; these are frequency/component axes, not a time trace. Energies sum squared physical coefficients/16000 over [8,12), [12,20), [20,30]Hz. Energy has unitsV² and squared errorV⁴. Logs are natural logs relative to1V² with a1e−40V² floor. Floor activations are counted. Targets are transformed per future before moment estimation. A Gaussian energy calculation includes mean² plus variance; it is not obtained by squaring the predicted Fourier mean alone.

`linear_audit.yaml` is an explicitly incomplete local CPU check. It runs linear models only and cannot qualify the prospective study. `existing.yaml` runs the full Phase0 candidate grid. `timing_gate.yaml` runs exactly eight full-network episodes. CUDA is for fitting only.

## Inference and controls

Family A uses A0 Feature18 versus train-PCA, d8/B32, fixed .005/.995 quantile ranges, all-label dev-selected linear readouts. It requires both carrier CE improvement >.20bits and reconstruction disagreement >.02 baseline NMSE. The reconstruction ratio uses the same fitted-mean history baseline and includes denominator uncertainty. The conjunction uses the larger one-sided component p-value at .025 and both lower97.5% bounds. Carrier is conditional on rhythmic membership. Rhythm balanced accuracy and full four-class logits/confusion matrices are secondary.

Family B uses the dev-selected full-history A1 short-native predictor. G=100(1−mean circuit risk/mean circuit train-mean baseline risk), with a97.5% paired-circuit t/influence interval. Entirely above+1pp means meaningful improvement; entirely within±1pp means practical equivalence for this fitted policy; entirely below−1pp means meaningful worsening; otherwise inconclusive. Each primary family receives alpha.025. Bootstrap and sign diagnostics are sensitivities. Sign-flip interpretation requires symmetry; binomial sign inference concerns direction/median. Inference is conditional on the training set and selection policy.

Scoring uses the first four registered futures everywhere, including H2 contexts. H2 alone uses M1/2/4/8/16/32 and disjoint L128 references at32 registered contexts on eight test circuits. Reference variance/L is subtracted, uncertainty retained, negatives kept. Shared-reference groups remain dependent. Resampling preserves paired regimes and independent prefixes. More futures and optimizer repeats do not create more circuits.

Mean selection uses train/development only. The runner freezes all candidate weights before generating calibration circuits, fits the same constant diagonal residual variance family on24 separate circuits, and freezes that calibration before test generation. Retention fits/PCA/ranges/readout selection use only train/development and save selections before scoring test outcomes. No readout or test score chooses a scientific endpoint. This is not conformal or joint calibration.

The numerical study uses identical external schedules/physical parameters at .0625/.03125/.015625ms, with common16kHz output. Circuit and paired-future uncertainty must fit within .1pp gain and2% relative-floor tolerances. Inconclusive confidence bounds block the core. No change of dt, biological driver, or reuse law is automatically substituted after failure.

## Provenance, recovery and export

Roots are immutable by resolved configuration. Generation records bind physical dependencies and original provenance; analysis and export have separate fingerprints. New analysis code does not relabel or regenerate valid imported native recordings. Imported recordings are checked against their original physics commit and byte hashes. Resume checks candidate device/dtype, train/dev bytes, means, selected policies, calibration, test IDs and operator contracts. Staged derivation accepts only a strict native-record superset with unchanged existing bytes and preserves original train/dev NPZ files. A mismatch requires a new versioned root, not deletion of frozen evidence.

Progress is in `progress/`, `campaign_progress.json`, generation manifests and per-trial records. Failures are JSON records under `failures/`. No `all` mode exists. `run --execute` is the single gated finite workflow, stopping with a nonzero status and saved report on a blocker. It does not stop on a negative scientific effect after the protocol is frozen.

Reports produce unrounded circuit tables and publication-size vector PDF/SVG figures only for existing outcomes. Standard ZIPs deduplicate exact NPY members, retain every target/prediction/variance/logit/code array, and record original NPZ container hashes. The included v4 standard-library restore tool verifies every payload and restores exact array bytes; reconstructed ZIP container bytes can differ. Native data and numeric model/readout weights have explicit omission manifests; `--full-native` includes available originals, including reused external files. Pickle checkpoints are never loaded. A requested size cap fails rather than silently dropping evidence.

See `WORKSTATION.md` for exact commands. Full network generation is deliberately not a laptop validation task.
