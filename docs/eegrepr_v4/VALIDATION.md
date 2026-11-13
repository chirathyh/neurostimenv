# Implementation validation record

Implementation and cheap verification ran on `chirath-Latitude-7440` using `/home/chirath/Documents/depression-simulator/bin/python`, Python 3.10.12, NEURON 8.2.3, LFPy 2.3, NumPy 1.26.3, SciPy 1.15.3 and PyTorch 2.4.1. CUDA hardware was unavailable. Actual workstation SSH authentication was unavailable through the gateway. No remote runtime, disk measurement, GPU training or full-network prospective generation is claimed.

The isolated worktree is `/tmp/neurostimenv-evaluation-v4`, branch `codex/eeg-evaluation-h1-h4-v4`, based on `061fe31`. The original dirty worktree and frozen pilot/COSYNE/H1 configuration files were preserved. New source/configuration files, source hashes and numeric evidence are included in the expert export. The lossless upload-sized packaging trial passed as recorded below. Verification artifacts explicitly record the pre-commit dirty tree and its content hash; a Git commit does not change that scientific source fingerprint.

The completed final existing-data analysis has scientific source fingerprint `971b1a57feedbbb4cfec19d2f52d6648fa77cf1840a45921dfc8f8efb40d8c78`. Older engineering runs retain their own fingerprints rather than being relabelled as generation by this final source.

## Meaningful checks

- V4 scientific/integrity suite: 60 tests passed in the final release-source run (10.959 seconds), including matched-family calibration and lossless numeric export/restoration.
- Existing H1 suite: 65 tests: 64 passed in the regular run, then the opt-in reduced NEURON early-stop test passed separately (3.595 seconds).
- Existing online stimulation suite: seven tests passed.

Six additional lossless transport-packaging tests passed (0.187 seconds), covering exact IEEE-bit restoration, endian/nonfinite fallback, full archive restoration, corruption/unsafe-path rejection and upload-cap enforcement.

These are **138 distinct passing tests**: 60 v4 scientific/integrity + 6 packaging + 65 legacy H1 + 7 online stimulation. The separately executed scientific smokes below are additional checks, not extra unit-test counts.
- Isolated-cell polarization validation passed.
- Online/legacy smoke: 8000 online samples over two 250-ms windows; correlation 1.0, relative RMS error 2.54635e-11; configured/effective temperatures both 6.3 C.
- Documented reduced A/B smoke: three circuits, three 500-ms steps, one burn-in step; completed. Its tiny-sample held-out accuracy was .333, not evidence for A/B discriminability.
- Documented reduced stimulation-reachability smoke: one discovery and one validation seed; completed, rate-safe, reachability **NOT PASSED** (fractional improvement −.422152, alignment −.768771). Negative findings were preserved.

The legacy Hydra smokes originally attempted their default relative `../../results` path, which is unwritable from a `/tmp` worktree. They were rerun with explicit `experiment.dir` and `hydra.run.dir` under `/tmp/eegrepr-v4-validation`. No permissions or scientific configurations were weakened.

Tests cover exact frozen A0/T0 reproduction, acquisition response/startup/causality, no future-to-X leakage, future-only T3/Parseval, centered-power identities, disjoint pools and pending-event forks, finite-M/L correction including negative estimates, paired-regime jackknife and circuit inference, neural learnability/mean fallback, complete tuning grids, train/dev-only fitting isolation, numeric checkpoint/class-order handling, code packing/clipping/projection attribution, known geometry/rank ceilings, optional common-event moments, and corrupted/stale artifact rejection.

## Existing-data evidence

All 576 original native files passed SHA256 verification. Exact original target arrays and physical-state floor/baseline values reproduced. All ten original models' saved MSE/NLL/CRPS/coverage reproduce to at most 2.22e-16 absolute difference. Native evidence supports 36 new acquisition/target panels; three one-second T3 panels are unavailable because the old recording support is too short. Saved-array-only imports support eight panels and explicitly label other derivations unavailable.

The final root `/tmp/eegrepr-v4-completed-release-20261113/existing` completed original-evidence auditing, all six CPU small-model qualification jobs (25 candidates each; 150 fits), scoring, H2 replication, retention, geometry and the report. Scoring saves 450 candidate/calibration rows across uncalibrated, multiplicative-calibrated and exactly matched context-independent variance families. The completed H2 analysis correctly reports zero independent reference futures in these old recordings; finite-reference empirical assessments remain unavailable.

The frozen-bank retention grid has 1177 rows: **1092 complete, 84 unavailable and one unsupported**. All 84 unavailable cases lack the label budget needed to cover every required class-by-circuit cell. The unsupported row exceeds native width or the K−1 sample rank ceiling; padding is refused. All five banks, projection/quantization controls and forecast pooling diagnostics are represented. A completed stage does not imply every requested endpoint is supported. Geometry audits use actual native cutoff states and saved histories; no nonexistent neuron trajectories are inferred. Nonlinear intrinsic-dimension estimation remains explicitly unavailable. See OBSERVED_FINDINGS.md and observed_h3_geometry.json for scoped numerical conclusions.

The final prospective protocol was frozen against that qualification evidence and its provisional plan records 832 episodes. An explicit generation attempt without a measured timing gate was correctly rejected. No full-network new generation followed.

## Runnable work not executed

The eight-episode full 32E/8I timing gate and 832-episode prospective generation remain **unrun**. The 23.416-hour and 4.75-GB figures are provisional extrapolations, not measured resource claims for this workstation. A saved qualified protocol, measured same-host gate, and explicit `--execute` are mandatory. The old expanded-reference and confirmation plans were not launched. CUDA fitting commands were checked structurally and fail explicitly without CUDA; GPU execution remains for the workstation.

Full neuron-voltage/current instrumentation and nonlinear intrinsic-dimension estimation remain explicitly unavailable. Optional common events and phase diffusion are separately named, default-off sensitivity mechanisms. They are not added to the 832-episode default design.

## Integration provenance during implementation

Completed engineering evidence has separate, preserved provenance:

| Run | Evidence root | Recorded source fingerprint |
|---|---|---|
| Analytical pipeline, 80 episodes; compact/full export and stage-resume checks | `/tmp/eegrepr-v4-verified-release-20261113/analytic_smoke` | `e4a537e05ea153ba6e63a4e6a3570555e4891f5791cf2dc1696d196a3f330a89` |
| Default reduced NEURON smoke, seven episodes plus one identical-input replay | `/tmp/eegrepr-v4-final-validation-20261113-r3/simulator_smoke` | `1592fe6a4673fb6753c95733d9d77d3a90a0a43f565e41ef9187bffe14d95e5a` |
| Optional common-event/phase-diffusion reduced NEURON smoke, seven episodes plus one replay | `/tmp/eegrepr-v4-verified-release-20261113/optional_common_smoke` | `e4a537e05ea153ba6e63a4e6a3570555e4891f5791cf2dc1696d196a3f330a89` |
| Final optional-driver geometry reanalysis | `/tmp/eegrepr-v4-completed-release-20261113/optional-geometry` | `971b1a57feedbbb4cfec19d2f52d6648fa77cf1840a45921dfc8f8efb40d8c78` |

All three generated engineering panels support all 39 acquisition/target derivations. Both reduced native runs passed the independent-future prefix/state/pending-event audit. They use 2E/1I and eight contacts per cell and do not qualify the full-network resource gate. Variance-comparison and compact-storage refinements changed the global source hash; resume correctly rejected attempts to relabel older roots. Later analysis imports the original recordings with their original generation provenance.

The final optional-driver test also verified corresponding common RNG streams across regimes while keeping E/I rate groups independent. Its reduced native forks passed. A sparse-state geometry case exposed overflow in squaring already-large finite eigenvalues; participation ratio now normalizes eigenvalues before squaring, preserving the exact scale-invariant statistic. The original failing data and a large finite-covariance regression both pass.


## Expert export status

The small analytical compact and full-native exports completed, including lossless numeric restoration checks: 3445 payload files and 315 transformed NPZ containers were verified with exact `.npy` member bytes. Reconstructed ZIP/NPZ container bytes may differ without changing their numeric members; the original container hashes remain recorded.

The ordinary compact export passed verification of 21,596 payload members and 2,350 transformed NPZ files, but was 620,526,617 bytes. A separate, reversible transport packer reduces this full package to **496,443,632 bytes**, below the **512,000,000-byte** limit. Its complete post-write verification restored all **21,597 original members byte for byte**, including the original export manifest. Reports remain directly readable. The packer used 829 byte-shuffled arrays, 517 fourth-order predictor/XOR arrays and 20,251 unchanged members. It discards no precision or numeric evidence. The trial SHA256 is `012cbe87d6d79fcf085e18a9feaca07b61fa8248a025956ee88e6a5442f968cb`. Final delivery adds the release record and updated documentation; use its accompanying checksum sidecar for that final ZIP.

The outer packing/unpacking helper requires NumPy. It verifies every restored original SHA256; unsupported predictor layouts fall back to byte shuffling or unchanged bytes. The subsequent NPZ-restoration helper uses only the standard library. Full ZIP/NPZ container byte identity is not claimed. Remote commands, remote timing and CUDA execution remain unverified because SSH authentication and local CUDA hardware were unavailable.
