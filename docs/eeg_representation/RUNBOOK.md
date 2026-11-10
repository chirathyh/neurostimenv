# Workstation runbook

Run every command from the **repository root**. The simulation environment is
the existing repository-parent virtual environment; ML may use that same
environment or a separate compatible environment. The local validation host
has Python 3.10.12, NEURON 8.2.3, LFPy 2.3, NumPy 1.26.3, SciPy 1.15.3,
PyTorch 2.4.1 and no accessible CUDA device. No workstation/CUDA execution is
claimed. See `IMPLEMENTATION_STATUS.md` for the exact executed checks.

## Checkout and environments

These are workstation commands, syntax checked locally. They do not merge the
feature branch into COSYNE or main. Preserve any existing dirty checkout by
using another worktree:

```bash
# cwd: your existing clone; shell only; no data/model inputs
git fetch origin
git worktree add ../neurostimenv-eegrepr --track -b feature/eeg-representation-position2027 origin/feature/eeg-representation-position2027
cd ../neurostimenv-eegrepr
```

If the local feature branch already exists, inspect it and instead use
`git worktree add ../neurostimenv-eegrepr feature/eeg-representation-position2027`.
Do not reset an existing branch. Confirm the pushed commit against the handoff.

Activate the validated simulation environment (the path is user-specific):

```bash
# cwd: repository root; SIM_ENV is the existing validated environment
export SIM_ENV=/path/to/existing/depression-simulator
source "$SIM_ENV/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
bash scripts/eegrepr/workstation_setup.sh check
bash scripts/eegrepr/workstation_setup.sh compile-check
python -m experiments.eeg_representation.cli audit-base
python -m experiments.eeg_representation.cli --help
```

`doctor`, `audit-base`, help and `compile-check` were executed locally. They
produce terminal diagnostics. BallAndStick has **no repository `.mod` files
to compile**. It uses built-in HH/passive/Exp2Syn/extracellular mechanisms;
LFPy schedules supplied spikes directly with `NetCon.event(t)`. A VecStim
build is not required. Do not run `nrnivmodl` in unrelated L23Net/FS0 paths.
If the validated stack is absent, restore the project's existing simulation
environment using its requirements; do not upgrade/downgrade a working stack
to fix ML dependencies. A real simulator smoke, not package presence alone,
is the integration check.

Optional separate ML environment, **not created or CUDA-validated locally**:

```bash
# cwd: repository root; creates only a NEW destination
bash scripts/eegrepr/workstation_setup.sh create-ml /path/to/new/eegrepr-ml-env
source /path/to/new/eegrepr-ml-env/bin/activate
# Install the workstation-compatible PyTorch build in this new environment.
# CPU-reproduction route matching the tested version (installation not run here):
python -m pip install torch==2.4.1
python -m experiments.eeg_representation.cli doctor
```

For CUDA, use the already validated workstation PyTorch environment/build and
run `doctor` there. The suite does not hard-code a CUDA version, GPU model or
device count. Do not run the CPU-reproduction install against an unrelated
existing CUDA environment. `--device cuda` fails explicitly without CUDA.
Mixed precision is disabled until separately validated. Data exchange is
versioned HDF5+JSON; the ML environment does not import a live simulator.

## Small reproducible checks

Use a fresh writable location outside frozen COSYNE results. Paths below are
the only user-specific placeholders. Data, checkpoints and generated tables
must stay outside git. All commands use the repository root as cwd.

```bash
export OUT_BASE=/path/to/new/eegrepr-results
python -m experiments.eeg_representation.cli validate --profile smoke --output-root "$OUT_BASE/analytical-validation"
python -m experiments.eeg_representation.cli validate --profile simulator --output-root "$OUT_BASE/simulator-validation"
```

These commands run analytical laws and unit tests; `simulator` additionally
runs real tiny circuits and inherited background/kinetics/online tests.
They write genuine control and validation JSON, not a fabricated dataset.

Analytical end-to-end check (CPU; exact interface executed locally):

```bash
export CFG=configs/eeg_representation/smoke.yaml
export OUT="$OUT_BASE/analytical-smoke"
python -m experiments.eeg_representation.cli plan --config "$CFG" --output-root "$OUT"
python -m experiments.eeg_representation.cli generate --config "$CFG" --output-root "$OUT" --resume
python -m experiments.eeg_representation.cli derive --config "$CFG" --output-root "$OUT" --resume
python -m experiments.eeg_representation.cli forks --config "$CFG" --output-root "$OUT" --resume
python -m experiments.eeg_representation.cli fit-suite --config "$CFG" --output-root "$OUT" --device cpu --resume
python -m experiments.eeg_representation.cli evaluate --config "$CFG" --output-root "$OUT" --device cpu --resume
python -m experiments.eeg_representation.cli summarize --config "$CFG" --output-root "$OUT" --resume
```

Repeat that sequence with `CFG=configs/eeg_representation/simulator_smoke.yaml`
and `OUT="$OUT_BASE/neuron-smoke"` for actual NEURON/LFPy → stored traces →
causal stream/noise → every minimum recipe → probes/forecasts → tables.
This deliberately uses 2 E/1 I cells and short records for engineering checks.
It is not the 40-cell pilot and cannot qualify observability or transfer.
Default-size 32 E/8 I adapter behavior is checked separately.

## Core pilot generation and future ensembles

The following full-pilot commands were **syntax/planning checked only**;
the 640-trial campaign was not launched. Activate the simulation environment.

```bash
export CFG=configs/eeg_representation/pilot.yaml
export OUT="$OUT_BASE/core-pilot"
python -m experiments.eeg_representation.cli plan --config "$CFG" --output-root "$OUT"
# Starts the requested CPU campaign; one serial worker by default:
python -m experiments.eeg_representation.cli generate --config "$CFG" --output-root "$OUT" --workers 1 --resume
python -m experiments.eeg_representation.cli derive --config "$CFG" --output-root "$OUT" --resume
```

Equivalent generation wrapper: `bash scripts/eegrepr/run_pilot_cpu.sh "$OUT" 1`.
Choose a small worker count using actual free CPUs/RAM. Each worker is a
separate serial NEURON process, writes a separate shard, and limits numerical
threads. Do not use `mpirun` for this adapter or start competing coordinators.
The planner reports trial/sample counts, uncompressed array arithmetic,
fork/recipe job counts, label caps and any measured generation summary. Raw
pilot signal arrays alone are 5,734,400,000 bytes before compression, events,
metadata and checkpoints; this is storage arithmetic, not measured pilot size.
No full-run runtime is inferred from the tiny circuit smoke.

The separate future ensemble is substantial: 10 states/regime × 64 futures,
with 128 futures on the nested convergence subset. It is **not** launched by
the pilot wrapper. Start deliberately after checking costs and prefixes:

```bash
# cwd: repository root; simulation env; CPU; input resolved pilot config
python -m experiments.eeg_representation.cli forks --config "$CFG" --output-root "$OUT" --resume
```

Outputs include `forks/manifest.json`, native future shards,
`prefix_validation.json`, `state_variance.json`, and `summary.json`.
A mismatched state/history/pending-event prefix raises an error. All future
targets use explicit time support and sampling grids. Default `vary: both`
resamples all exogenous sources; source-specific modes are diagnostics.
Pilot fork jobs never inspect final test structures.

Targeted factor development uses the same commands with
`configs/eeg_representation/extensions.yaml` and a fresh output root. It is
opt-in and was not run as a campaign. Evaluate burst input/EEG power and rate
confounds; the input-power-matched tonic candidate is not a guaranteed
output-power match. A bounded modulation-depth development sweep must use
new registered development seeds and separate frozen YAML files/output roots;
do not tune using confirmation data.

## CUDA fitting and transfer

Activate the chosen ML environment, retain the same repository revision,
resolved config and output root. GPU commands below were syntax checked only;
their CPU equivalents were executed on the tiny datasets. One fit runs at a
time. Each command reads the derived manifests and writes separate model
checkpoints under `models/PROTOCOL/RECIPE/seed_N/budget_*`.

```bash
# One recipe; source labels follow the declared task registry:
python -m experiments.eeg_representation.cli fit --recipe forecast_lowrank --config "$CFG" --output-root "$OUT" --seed 0 --device cuda --resume
# A selected suite and paired optimizer seeds:
python -m experiments.eeg_representation.cli fit-suite --recipes features masked_mse spectral_hybrid forecast_lowrank joint progressive reverse random_order reconstruction_supervised --seeds 0 1 --config "$CFG" --output-root "$OUT" --device cuda --resume
# The complete minimum recipe family (omit --recipes to use the YAML selection):
python -m experiments.eeg_representation.cli fit-suite --recipes features specialist masked_mse masked_l1 spectral_hybrid momentum_contrastive forecast_lowrank forecast_diagonal forecast_ar joint progressive reverse random_order reconstruction_supervised --seeds 0 1 --config "$CFG" --output-root "$OUT" --device cuda --resume
python -m experiments.eeg_representation.cli evaluate --config "$CFG" --output-root "$OUT" --protocol leave-task-label-out --device cuda --resume
```

Recipe names in the complete command are individually selectable with `fit`.
`fit-suite` expands the configured random-order repetitions independently from
optimizer seeds. `--budget N` on `fit` or `--budgets N ...` on `fit-suite`
selects independent source-label group budgets. `evaluate --budgets N ...`
selects the downstream label-budget axis; freeze it before confirmation.
Exceeding available independent families is explicitly capped and cannot
silently contribute to a complete primary area.

Stronger transfer trains a separate model pool; it is not a relabelled result:

```bash
python -m experiments.eeg_representation.cli fit-suite --config "$CFG" --output-root "$OUT" --protocol leave-generator-family-out --device cuda --resume
python -m experiments.eeg_representation.cli evaluate --config "$CFG" --output-root "$OUT" --protocol leave-generator-family-out --device cuda --resume
python -m experiments.eeg_representation.cli summarize --config "$CFG" --output-root "$OUT" --resume
```

Evaluation writes frozen linear/shallow nonlinear heads, per-task and
per-structure metrics, capped/missing labels, forecast scores and transfer
areas. `evaluation.adaptation_steps > 0` enables separate limited-label
fine-tuning and a random-initialized downstream specialist; choose that in
the YAML **before generation** to preserve the complete config hash. This
uses only training/development structure labels, not test-circuit support.
Likewise declare `evaluation.variants: [clean, snr_20, snr_10, snr_0]` and
`evaluation.gains: [0.5, 1.0, 2.0]` before generation for frozen-probe stress
evaluation. Alternative history, lead, masks or architecture use separate
resolved configurations and output roots; they do not overwrite a frozen run.

`summarize` creates real-result CSV/JSON, LaTeX and PNG/PDF figures. It lists
missing expected jobs and leaves incomplete aggregates Pending. Smoke
diagnostics are excluded from empirical primary conclusions. Review
`summary/missing_jobs.csv`, individual tasks and efficiency records alongside
any aggregate; no global average substitutes for a failed family.

## Resume and deliberate confirmation

Repeat the same command with `--resume` to verify and reuse committed shards
or continue an interrupted checkpoint. Incompatible schema, config, source or
content hashes are errors. Do not edit provenance to bypass them. Use a new
output root after code/config changes. A crash between atomic shard rename and
manifest update regenerates only that uncommitted deterministic shard.

The provided confirmation YAML is provisional and generation is blocked.
After the pilot, copy it to a new versioned file, choose the final precision/
sample-size plan and task family, record the development-only selection,
set `evaluation.frozen: true` and `evaluation.allow_test: true`, and commit
that protocol **before** generating confirmation circuits. Keep the registered
disjoint structure seeds. This freeze is a scientific action, not an automated
success gate. Then plan and deliberately run:

```bash
# User-created and committed frozen config; no confirmation campaign run locally
export CFG=configs/eeg_representation/confirmation_frozen.yaml
export OUT="$OUT_BASE/confirmation-frozen"
python -m experiments.eeg_representation.cli plan --config "$CFG" --output-root "$OUT"
python -m experiments.eeg_representation.cli generate --config "$CFG" --output-root "$OUT" --workers 1 --resume
python -m experiments.eeg_representation.cli derive --config "$CFG" --output-root "$OUT" --resume
```

Follow the separate fork, CUDA fit, both-protocol evaluation and aggregation
commands above. `confirmation_frozen.yaml` is intentionally not supplied with
invented pilot decisions; create it only after the actual pilot assessment.

## Expected-output checklist

- `provenance.json`, `resolved_config.json`, `splits.json`, `task_registry.json`.
- `manifest.json` plus independently hashed `trials/*.h5`; native physical units.
- `derived/*/normalization.json` and manifests with all four observation variants.
- Exact prefix/state validation and per-state/structure future-risk summaries.
- Encoder/head checkpoints, learning/retention curves and exposure accounting.
- Probe/forecast results for both explicitly named holdout protocols.
- Summary tables/figures, missing-job inventory and Pending scientific claims.
- No edits to frozen COSYNE protocols/results, no generated arrays in git,
  no claims of a CUDA test or full pilot/confirmation run from CPU smokes.
