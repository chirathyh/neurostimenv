# Workstation execution and recovery

The commands below validate paths before use. The workstation paths came from the prior v4 handoff; this laptop could not verify them because the SSH gateway required interactive authentication. The code paths and CLI commands were tested locally; full workstation execution is not claimed. Use the existing shell environment; never install packages or change NEURON versions halfway through a frozen run.

## Isolated checkout

On `neuroinformaticsgpu01`:

```bash
set -euo pipefail
parent=/home/u7041472/Documents/chirath/depression-simulator
v4_checkout="$parent/neurostimenv-evaluation-v4"
v6_checkout="$parent/neurostimenv-evaluation-v6"
activation="$parent/bin/activate"
source_archive=/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective-standard.zip
native_source=/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective
campaign=/data/u7041472/eegrepr-results/evaluation-v6

test "$(hostname -s)" = neuroinformaticsgpu01
test -f "$activation"
test -d "$v4_checkout/.git" || test -f "$v4_checkout/.git"
test -f "$source_archive"
test -f "$native_source/generation/native/manifest.json"
printf '%s  %s\n' 6818befa7ca70df3a0272bf69328f9e2d2316817493e49a13bac6e4b74a5817f "$source_archive" | sha256sum -c -

git -C "$v4_checkout" fetch origin codex/eeg-evaluation-h1-h4-v6-confirmation
if [ ! -e "$v6_checkout" ]; then
  git -C "$v4_checkout" worktree add --detach "$v6_checkout" origin/codex/eeg-evaluation-h1-h4-v6-confirmation
fi
cd "$v6_checkout"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/codex/eeg-evaluation-h1-h4-v6-confirmation)"
test -z "$(git status --porcelain)"
source "$activation"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import sys,torch,neuron,LFPy; print(sys.executable); print(torch.__version__,neuron.__version__,LFPy.__version__); print("CUDA",torch.cuda.is_available())'
mkdir -p "$campaign/logs"
df -h "$campaign"
```

If the historical v4 checkout path fails, locate the actual v4 worktree with `git worktree list` from your known checkout, then set `v4_checkout` to that verified path. Do not create an empty substitute for a missing result directory. If the new checkout exists at a different commit, preserve it and use a new versioned path; do not reset it.

## One bounded entry point

Use `workers=1` for the conservative serial gate. If doctor shows sufficient CPU/RAM, choose a small explicit worker count such as 4 **before** the timing gate and keep it fixed. Eight full-prefix trials measure the chosen concurrency, actual RSS and disk size. The plan checks worker identity and resource bounds; GPU does not speed NEURON. Increasing workers after the gate requires a fresh gate/root, not a silent speedup assumption.

```bash
workers=1
bash scripts/eegrepr/run_h1_h4_v6.sh doctor \
  --root "$campaign/preflight" --device cuda --workers "$workers" --resume
```

Then use tmux so the run survives a disconnected terminal:

```bash
tmux new-session -s eeg-v6
```

Inside tmux, initialize the new shell explicitly. If you verified different paths or chose a different worker count above, use those same values here:

```bash
set +e  # keep the interactive tmux pane open when a gate blocks the run
set -o pipefail
parent=/home/u7041472/Documents/chirath/depression-simulator
v6_checkout="$parent/neurostimenv-evaluation-v6"
activation="$parent/bin/activate"
source_archive=/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective-standard.zip
native_source=/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective
campaign=/data/u7041472/eegrepr-results/evaluation-v6
workers=1
cd "$v6_checkout"
source "$activation"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$campaign/logs"
```

Then launch the bounded runner:

```bash
bash scripts/eegrepr/run_h1_h4_v6.sh run \
  --root "$campaign" \
  --config configs/eeg_representation/v6/confirmation.yaml \
  --source-archive "$source_archive" --native-source "$native_source" \
  --device cuda --workers "$workers" --execute --resume \
  2>&1 | tee -a "$campaign/logs/run.log"
run_status=${PIPESTATUS[0]}
printf '%s\n' "$run_status" > "$campaign/logs/exit-status.txt"
printf 'Runner exit status: %s\n' "$run_status"
```

Detach with Ctrl+b, then d. The wrapper runs in a child shell; its exit leaves the interactive tmux shell open with the settings above. A gate failure exits 2 and preserves a concrete blocker report. Inspect it rather than restarting until it turns green. Completion of a plan is not completion of generation. The runner updates `campaign_progress.json` and writes `campaign_result.json` when it returns a completed or blocked result. An exception can leave no new result file; inspect the progress, stage failure records and log together. Fresh test generation is never selected by an effect direction.

The runner performs full inspected-data qualification, selects the conservative minimum registered N, records `selected-config.json`, runs the eight-episode timing/fit/calibration gate, then the separate 768-episode numerical panel if eligible. That numerical panel is a substantial simulation stage, not a cheap analytical check. It replans before any core generation. It generates train/development, freezes means, generates/calibrates the 24 calibration circuits, and only then generates fresh evaluation. Retention/readout fitting uses train/development only. It scores both primary families, H2 and the full descriptive grid, then exports. N257 is only a historical provisional proposal; the actual root is `confirmation-n<N>` and is reported in the progress file.

## Explicit stages

These stages use the checked-in default scientific configuration. Run one block at a time and inspect its status before continuing. For recovery after `run`, use its saved `phase0-config.json` and `timing-config.json` when present. If you changed prospective models, retention or simulation settings, use the runner to construct matching qualification/gate configurations instead of mixing them with defaults. Keep `workers` and fitting device unchanged on resume. Set the core configuration only after the pretest precision choice; never replace it after evaluation exists.

```bash
phase0="$campaign/phase0"
timing="$campaign/timing-gate"
phase0_config=configs/eeg_representation/v6/existing.yaml
timing_config=configs/eeg_representation/v6/timing_gate.yaml
if [ -f "$campaign/phase0-config.json" ]; then phase0_config="$campaign/phase0-config.json"; fi
if [ -f "$campaign/timing-config.json" ]; then timing_config="$campaign/timing-config.json"; fi

(
set -euo pipefail
bash scripts/eegrepr/run_h1_h4_v6.sh controls --root "$phase0" \
  --config "$phase0_config" --device cuda --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh import --root "$phase0" \
  --config "$phase0_config" --source "$source_archive" --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh audit-existing --root "$phase0" \
  --config "$phase0_config" --source "$source_archive" --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh explore-existing --root "$phase0" \
  --config "$phase0_config" --source "$source_archive" --device cuda --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh geometry-existing --root "$phase0" \
  --config "$phase0_config" --source "$source_archive" --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh timing-gate --root "$timing" \
  --config "$timing_config" --device cuda --workers "$workers" --execute --resume
)
```

Use the runner to make and save the pretest N choice, or use `plan` to inspect a candidate. `plan` always produces a hashable report; its exit0 means the report was written, not that execution is eligible. It saves `pretest_design_proposal.json` if a precision certificate is available. The next line deliberately reviews the default candidate and may be blocked:

```bash
bash scripts/eegrepr/run_h1_h4_v6.sh plan --root "$campaign/default-n192-review" \
  --phase0-root "$phase0" --timing-root "$timing" \
  --source-archive "$source_archive" --native-source "$native_source" \
  --device cuda --workers "$workers" --resume
```

The runner writes `selected-config.json` automatically. For a fully staged default run, save the registered proposal after Phase 0 is complete using this block, which refuses to replace a different frozen configuration. This selects N from pretest evidence only; the subsequent plan must recompute precision and resources at that N before generation is eligible.

```bash
python - "$campaign" <<'PYSELECT'
import json, sys
from pathlib import Path
from experiments.eeg_representation.h1.v6.artifacts import atomic_json
base = Path(sys.argv[1])
proposal = json.loads((base / 'default-n192-review/pretest_design_proposal.json').read_text())
if proposal['status'] not in ('proposed', 'unchanged_candidate') or not proposal.get('phase0_complete'):
    raise SystemExit('STOP: complete Phase 0 and inspect the pretest proposal before choosing N')
chosen = proposal['resolved_config']
path = base / 'selected-config.json'
if path.exists() and json.loads(path.read_text()) != chosen:
    raise SystemExit('STOP: selected configuration differs; preserve it and use a new campaign root')
atomic_json(path, chosen)
print(path, 'test circuits:', chosen['protocol']['test_circuits'])
PYSELECT
```

After selection succeeds, or when resuming a runner-selected configuration, resolve the actual N without guessing:

```bash
frozen_config="$campaign/selected-config.json"
test -f "$frozen_config"
chosen_n=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol"]["test_circuits"])' "$frozen_config")
core="$campaign/confirmation-n$chosen_n"
```

All downstream stages use the same configuration and root. This helper prints plan blockers and prevents an expensive command when its eligibility flag is false:

```bash
require_plan_flag() {
  python - "$core/plan.json" "$1" <<'PYPLAN'
import json, sys
plan = json.load(open(sys.argv[1]))
flag = sys.argv[2]
print(flag, plan.get(flag, False))
if not plan.get(flag, False):
    print(json.dumps(plan.get('blocked_reasons', []), indent=2))
    raise SystemExit('STOP: this execution stage is not eligible')
PYPLAN
}
```

First plan and, only if eligible, run the numerical panel:

```bash
bash scripts/eegrepr/run_h1_h4_v6.sh plan --root "$core" --config "$frozen_config" \
  --phase0-root "$phase0" --timing-root "$timing" --numerical-root "$core" \
  --source-archive "$source_archive" --native-source "$native_source" --workers "$workers" --resume
if require_plan_flag eligible_numerical_execution; then
  bash scripts/eegrepr/run_h1_h4_v6.sh numerical-check --root "$core" --config "$frozen_config" \
    --phase0-root "$phase0" --workers "$workers" --execute --resume
fi
bash scripts/eegrepr/run_h1_h4_v6.sh plan --root "$core" --config "$frozen_config" \
  --phase0-root "$phase0" --timing-root "$timing" --numerical-root "$core" \
  --source-archive "$source_archive" --native-source "$native_source" --workers "$workers" --resume

require_plan_flag eligible_for_generation
```

Continue only when the final line reports `True`. A failed or inconclusive numerical result stops here. The following sequence runs in a subshell so any failed stage stops downstream work while leaving your interactive tmux shell open:

```bash
(
set -euo pipefail
require_plan_flag eligible_for_generation
for split in train development; do
  bash scripts/eegrepr/run_h1_h4_v6.sh generate --root "$core" --config "$frozen_config" \
    --split "$split" --native-source "$native_source" --workers "$workers" --execute --resume
done
bash scripts/eegrepr/run_h1_h4_v6.sh derive --root "$core" --config "$frozen_config" --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh fit --root "$core" --config "$frozen_config" --device cuda --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh generate --root "$core" --config "$frozen_config" \
  --split calibration --native-source "$native_source" --workers "$workers" --execute --resume
bash scripts/eegrepr/run_h1_h4_v6.sh derive --root "$core" --config "$frozen_config" --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh predict --root "$core" --config "$frozen_config" --device cuda --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh calibrate --root "$core" --config "$frozen_config" --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh generate --root "$core" --config "$frozen_config" \
  --split evaluation --native-source "$native_source" --workers "$workers" --execute --resume
bash scripts/eegrepr/run_h1_h4_v6.sh derive --root "$core" --config "$frozen_config" --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh predict --root "$core" --config "$frozen_config" --device cuda --workers "$workers" --resume
bash scripts/eegrepr/run_h1_h4_v6.sh retention --root "$core" --config "$frozen_config" --device cuda --workers "$workers" --resume
for mode in score replication confirm-statistics report; do
  bash scripts/eegrepr/run_h1_h4_v6.sh "$mode" --root "$core" --config "$frozen_config" --workers "$workers" --resume
done
bash scripts/eegrepr/run_h1_h4_v6.sh export --root "$core" --config "$frozen_config" \
  --destination "$campaign/confirmation-standard.zip" --workers "$workers" --resume
(cd "$campaign" && sha256sum -c confirmation-standard.zip.sha256)
)
```

The SHA sidecar uses the ZIP basename; run its check from `cd "$campaign"` if currently elsewhere. An explicit `export` always requires a new filename, even with `--resume`; verify an existing completed ZIP instead of repeating that stage. To preserve all native files and numeric weights in a separate larger export, choose a new destination and add `--full-native`. `--max-bytes 536870912` enforces a 512 MiB cap without dropping payloads; a larger evidence package fails that cap and needs an explicitly smaller *scope*, not silent numeric omission.

## Recovery and status

```bash
tmux list-sessions
pgrep -afu "$USER" '[e]xperiments.eeg_representation.h1.v6.cli|[r]un_h1_h4_v6.sh'
tail -n 60 "$campaign/logs/run.log"
cat "$campaign/campaign_progress.json"
cat "$campaign/campaign_result.json" 2>/dev/null || true
cat "$campaign/logs/exit-status.txt" 2>/dev/null || true
find "$campaign" -path '*/failures/*.json' -type f -print
```

Resume the identical `run` command. A completed run can return its verified completion receipt without rerunning stages; an existing ZIP without a valid receipt requires a new `--destination` for another export. Do not change dtype/device/worker count, overwrite native files, or remove failed scientific results. If the numerical CI or precision/resource plan fails, retain the report and formulate a new bounded protocol. A numerical refinement is not implemented as an automatic escape route. The old 96 h/927 h plans and optional biological/instrumentation factorials are never launched.

## Local/cheap verification

```bash
source /home/chirath/Documents/depression-simulator/bin/activate
python -m unittest -v tests.test_eegrepr_v6_statistics tests.test_eegrepr_v6_models tests.test_eegrepr_v6_numerical tests.test_eegrepr_v6_integration tests.test_eegrepr_v6_orchestration
bash scripts/eegrepr/run_h1_h4_v6.sh controls --root /tmp/eegrepr-v6-local-controls --device cpu --resume
```

For bounded CPU reuse of the actual old archive, `explore-existing --config configs/eeg_representation/v6/linear_audit.yaml --source <verified archive>` runs all linear candidates and explicitly leaves neural qualification unrun. The full scientific grid uses `existing.yaml`, normally on workstation CUDA. Engineering tests include a small 2E/1I, 60 ms real NEURON common-forcing comparison; it does not replace the full-network convergence panel.
