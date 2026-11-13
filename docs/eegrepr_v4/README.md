# Bounded EEG evaluation campaign, manuscript v4

The wrapper prints the exact command, root, backend and source/configuration fingerprints. There is **no `all` mode**. Generation always needs a saved plan and `--execute`; prospective generation additionally needs a frozen qualified protocol and a verified same-host timing gate. A NEURON error never triggers analytical substitution.

Run from the new worktree with the repository-parent Python environment activated. Each root is immutable with respect to source/configuration. Use `--resume` to continue the same root. A changed source, configuration, import, panel, operator or model fingerprint requires a new root. Do not point a v4 root at frozen old results. For an interrupted compatible stage, resume verifies completed artifacts; failed simulations retain their failure records.

`--config` may follow wrapper arguments to override its profile. The wrapper defaults to bounded.yaml, except existing-data modes use existing.yaml and timing-gate uses timing_gate.yaml. All subsequent modes on a root must use the SAME config. Device is `cpu` or `cuda`; the wrapper sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python starts.

| Mode | Purpose |
|---|---|
| doctor | Read-only current host, executable, dependency, CUDA, output parent and registry checks |
| controls | Measurement, same neural training path, known geometry and optional-event generator controls; no neural circuit simulation |
| audit-existing | Explicit read-only content-addressed import, supported derivation, original arrays/scores reproduction |
| qualify-existing | Supported existing-data derivation, complete small-model fitting and separate scoring |
| geometry-existing | Runtime inventory, covariance/innovation diagnostics from actual available arrays |
| plan | Save provisional or measured cost plan; optionally freeze protocol using `--qualification-root` and bind `--timing-root` |
| timing-gate | Explicit eight-episode full-network NEURON timing run using timing_gate.yaml; requires plan and `--execute` |
| generate | Explicit configured generation; requires eligible saved plan and `--execute` |
| derive | Derive all supported A0/A1/A2 × T0/T1/T2/T3 panels with separate hashes and train-only normalization |
| fit | Fit configured selected jobs; retain every candidate and train/dev/evaluation prediction |
| score | Risk accounting, Gaussian diagnostics and paired/circuit uncertainty; never select on evaluation |
| retention | Native/projected/packed representation attribution and task readouts |
| replication | Nested M=1/2/4/8 and finite independent-reference analyses; unavailable references remain labelled |
| report | Plain-language H1–H4 findings, statuses, target-specific skill plot and machine-readable findings |
| export | Compact ZIP with numeric evidence, reports, source and omitted-file hashes; `--full-native` adds raw bytes |

`--source` accepts the historical evidence directory, original native ZIP, or previous expert ZIP with an `evidence/` prefix. Imports keep their original generation provenance. Changed targets have a separate analysis/operator fingerprint. No pickle models are loaded to inspect evidence. Compact export retains all current numeric evidence, original provenance/configuration/manifests and hashes of omitted original payload. Use `--include-imported` to include the original nonraw payload as well; `--full-native` includes all available native source files. The first imported source path is recorded for later modes. Moving it requires an explicit new `--source` with matching evidence identities.

The default prospective fit evaluates A2/T3 plus the original A0/T0 10/128 and A1/T0 2/32 anchors. All 39 acquisition/target combinations are derived. Existing evidence supports 36 with native files, or eight with saved primary arrays only; the old recordings do not support T3. `--jobs` can select exploratory analyses, but prospective fitting is constrained to the protocol's frozen selection.

Retention is more expensive than the prediction qualification: endpoints multiply across banks, dimensions, bits, ranges, tasks and label budgets. It remains a separately invoked mode. Numerical prediction files and packed codes are saved; shared model/codebook storage is separate from the per-example payload. Nonlinear ID estimation and layer instrumentation are unavailable rather than silently substituted.

See [WORKSTATION.md](WORKSTATION.md), [SCIENTIFIC_CONTRACT.md](SCIENTIFIC_CONTRACT.md) and [VALIDATION.md](VALIDATION.md). Local and remote execution are distinguished in the validation record. No expensive prospective workstation simulation was launched during implementation.

## Compact numeric storage

By default, compact exports deduplicate exact NPY array members across NPZ files. `EXPORT_MANIFEST.json` maps every original logical NPZ path to an ordered descriptor and retains its original container SHA256. Targets, predictions, variances, logits, IDs and packed code payloads are retained without numeric rounding. The included standard-library `restore_export.py` verifies every payload and reconstructs ordinary NPZ files for inspection. Restored NPY bytes are identical; ZIP container hashes can differ, and the restore report records both hashes. The original scientific manifests are never rewritten to hide that distinction. Use `--plain-npz` for untouched original NPZ containers, or `--full-native` for original containers plus available native files. A compact inspection package is not a replacement for the untouched source needed by strict original-container resume checks.

For a 512 MB upload cap, first create a standard compact export, then run `python scripts/eegrepr/pack_v4_evidence.py pack SOURCE.zip SHARE.zip --max-bytes 512000000`. This additional NumPy-based transport layer chooses reversible byte shuffling or a causal fourth-order float predictor with bitwise XOR residuals, or leaves each array unchanged. No precision is discarded. All reconstructed original member hashes are checked; reports remain ordinary ZIP files. `PACKING_README.md` and the included helper explain verification and unpacking before the standard-library NPZ restoration above. Unsupported predictor layouts fall back to reversible byte shuffling or original storage. A failed size cap remains an explicit error.
