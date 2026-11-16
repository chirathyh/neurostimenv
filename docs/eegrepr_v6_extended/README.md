# V6 immediate forecasts and information-based error comparisons

This is a secondary-analysis amendment to the implemented V6 H1–H4 campaign. The complete entry point is `scripts/eegrepr/run_h1_h4_v6_extended.sh`; it includes the original study, rather than assuming it has run. The original wrapper and configurations remain available. The extension is opt-in through `extension.enabled: true` in `configs/eeg_representation/v6_extended/`.

## Preserved, added and unavailable

| Scope | Status and interpretation |
|---|---|
| Original 27 acquisition/target jobs | Preserved. All physical and normalized target/input arrays and operator fingerprints were exactly reproduced from the local V4 evidence against the previous V6 analysis. |
| Full-history models, separate calibration, two primary families, numerical and resource gates | Preserved. Training/development selection remains separate from 24 calibration circuits and fresh test circuits. |
| Matched immediate/delayed target matrix | Implemented: 60 jobs, with identical physical targets and target normalization across the three acquisitions. |
| Information-condition tables and matched future-code decoders | Implemented secondary analyses; fitted model differences do not identify pure information losses. |
| Actual downloaded archive | Supports 33 of the 60 jobs. The 27 zero-gap jobs require omitted native recordings. No missing waveform or Fourier samples are reconstructed from a lossy signal. |
| Native workstation recordings | Required for complete old-data qualification. A retained native manifest alone does not establish that its H5 payloads are available. |
| Full EEG-history Bayes error and a fitted complete-state oracle | Scientifically unidentified/unavailable; neither is inserted as zero or replaced by a fitted EEG/partial-state score. |
| New confirmation campaign | Runnable through measured gates; implementation, synthetic controls and old-data analysis are not fresh confirmation results. |

The default remains the recorded N192 candidate. The existing first-four-future precision audit found that N192 does not meet the target precision and proposed at least N257 under the inspected historical evidence. The runner recomputes precision after complete Phase0, freezes the minimum registered eligible N **before fresh evaluation outcomes**, and never resets an existing selection to N192. Full qualification can require more circuits; exceeding a registered range or resource bound is a blocker, not a reason to reduce N or relax the protocol.

## Target conventions

`lead_ms` has always meant the left-open **target-start offset**, not the first bin's end. With observation cutoff t0=6000 ms, h=0 predicts bins `(6000,6002]`, `(6002,6004]`, and so on. The first bin averages exactly 32 native samples at 0.0625 ms spacing; the sample at t0 is excluded. At h=2 the first bin is `(6002,6004]`; at h=10 it is `(6010,6012]`. There is no timestamp backdating or changed filter.

The matrix contains, for each A0/A1/A2 acquisition:

- Six native waveform targets: h∈{0,2,10} ms × duration∈{32,128} ms.
- Two one-second Fourier targets: h∈{0,10} ms, with the same 54 real orthonormal coordinates at integer frequencies 4–30 Hz.
- Twelve direct energy/log-energy targets: three bands × two transforms × h∈{0,10} ms. Bands remain [8,12), [12,20), [20,30] Hz; energy is coefficient square-sum/16000 in V² and its squared prediction error has units V⁴. Logs are natural logarithms relative to 1 V² with the frozen 1e−40 V² floor.

Zero gap makes the target start immediately after the cutoff; a one-second block is still available only after the entire future second. Targets retain their own training-only baseline, offset and scale, shared across acquisitions. Explicit native support, sample indices, first-bin end, target availability and input availability are exported. Energies are transformed separately for each future before means and variances are estimated; expected energy includes variability and cannot generally be obtained by squaring a waveform mean.

## Information rows and inference

For each fixed setting, the readable table rescales the same training-mean risk to 100 and retains unrounded physical MSE, baseline risk, ratio, relative reduction and uncertainty. Rows distinguish theoretical complete-state-plus-future-input zero error; measured deterministic replay, where actually available; unbiased repeated-future state variability; fitted partial-state features; unknown best-possible EEG predictor; fitted short/four-second history; matched code decoders; and the training mean.

The partial-state learner uses exported membrane/HH features, stable physical units and training-only scaling. It omits synapses, drive phase, event queues, parameters and acquisition state, and is explicitly not a complete-state predictor. Future schedules/RNG encodings never enter an ordinary learner. Privileged replay uses exact external schedules and full-prefix deterministic reconstruction, not a saved EEG prefix as a simulator checkpoint. The eight timing-gate contexts form their own common-subset table. Where the frozen Phase0 predictor artifacts are verified, training-mean, short-history and full-history predictions are recomputed on that same subset and baseline. Code/partial-state rows without a supported deployment path and direct M=1 state variance remain unavailable.

Future readouts compare A0 Feature18, train-PCA and three fitting repeats of the legacy 200-update forecast representation, each at d8 projected-unquantized and d8/B32 payload. Every readout uses the same seven ridge alphas plus the training-mean fallback and development-only selection. The three forecast repeats are optimizer repeats, not extra circuits. At 60 jobs the added upper grid is 1,820 ridge fits plus 260 explicit mean fallbacks: 200 code readouts on 20 A0 targets and 60 partial-state readouts. Three legacy forecast encoders each use 200 updates once and are reused across targets. Original H3 retention training remains separately accounted for.

Both original primary families are unchanged: H3's Feature18/PCA conjunction has 0.20-bit carrier and 0.02-baseline-NMSE reconstruction margins; H1 uses A1 h=2 ms/W=32 ms and the existing ±1-percentage-point/97.5%-interval rule. Each receives alpha 0.025. New zero-gap/state-feature/code-forecast contrasts are secondary/descriptive with paired circuit-level intervals. Their inspected-cohort intervals are not multiplicity-adjusted confirmation. All primary forecast comparisons use the first four registered scoring futures. H2 extends the same contexts to nested M=1/2/4/8/16/32 with disjoint L128 references, including zero-gap targets. At M1, direct sample variance about an unknown conditional mean is unavailable. Finite-reference uncertainty and negative corrected excess remain visible.

MSE minus estimated state-conditioned variance combines observation ambiguity, any representation restriction and imperfect fitting. It is not an identified separation into those components. A fitted state-feature score is not the state floor, and a fitted EEG score is not the EEG Bayes floor. A 100-unit baseline is presentation scaling, not V².

## Execution, resources and outputs

The wrapper offers `preflight`, `dry-run`, `smoke`, `complete`, `resume`, `status`, and `report-only`, plus the original explicit stage modes. `complete` and `resume` require `--execute` and both the verified old archive and old native source. No generic automatic factorial or biological robustness run is added.

The new targets reuse the same 7.010-second recordings. Default N192 contains 13,056 core episodes with 320 old train/development episodes reused; N257 contains 15,136 core episodes with 14,816 new. A separate 768-episode numerical panel remains unchanged. The new replay repeats eight timing-gate episodes; the original gate itself also contains eight episodes. The planner measures actual-host generation/RSS/storage and fitting throughput, includes added derivation/readout/replay costs, and enforces the unchanged 500 summed generation-hour, 120 GB native and 250 GB minimum free-space floor. The working/output/export estimate imposes a larger dynamic reserve: currently about 573 GB at N192 and 722 GB at N257, including two native and two non-native working/export copies; non-native evidence alone is projected at about 272.05 GB for N257. These are planning estimates to verify on the actual host, not an assurance that 250 GB is enough. GPU accelerates fitting only. Expanded target fitting is 60/27≈2.22 times the original target matrix before the additional readouts; generation support is unchanged.

Reports under the selected `confirmation-n<N>/report/` contain `MAIN_ANALYSIS.md`, `main_analysis.json`, manuscript paragraphs, tables and vector PDF/SVG figures. Detailed grids and failed diagnostics remain supplementary. Lossless exports preserve numeric arrays, IDs, operator/normalization fingerprints and omitted-native/weight manifests. `--full-native` provides the larger available-native export. Neither an export-size cap nor a failed scientific gate silently drops endpoints.

See `WORKSTATION.md` for exact setup, smoke, complete execution, resume and report-only commands. The remote paths there are checked by the commands; they were not authenticated or verified remotely during this local implementation.
