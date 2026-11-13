# Workstation commands

These commands use the historical workstation account and directories reported by the user. Local code and smoke runs were verified on `chirath-Latitude-7440`; this task's read-only SSH check could not authenticate through the gateway. The commands therefore **check existence** before relying on the remote paths. They do not claim remote execution already occurred. See VALIDATION.md for what actually ran.

## Check out the isolated implementation

On `neuroinformaticsgpu01`:

```bash
set -euo pipefail
export EEGREPR_ENV=/home/u7041472/Documents/chirath/depression-simulator/bin/activate
export EEGREPR_BASE=/home/u7041472/Documents/chirath/depression-simulator/neurostimenv-eegrepr-h1
export EEGREPR_REPO=/home/u7041472/Documents/chirath/depression-simulator/neurostimenv-evaluation-v4
test -f "$EEGREPR_ENV"
test -d "$EEGREPR_BASE/.git" || test -f "$EEGREPR_BASE/.git"
source "$EEGREPR_ENV"
git -C "$EEGREPR_BASE" fetch origin codex/eeg-evaluation-h1-h4-v4
if [ ! -e "$EEGREPR_REPO" ]; then
  git -C "$EEGREPR_BASE" worktree add -b codex/eeg-evaluation-h1-h4-v4 \
    "$EEGREPR_REPO" origin/codex/eeg-evaluation-h1-h4-v4
fi
cd "$EEGREPR_REPO"
test "$(git branch --show-current)" = codex/eeg-evaluation-h1-h4-v4
git status --short
git rev-parse HEAD
export EEGREPR_ROOT=/data/u7041472/eegrepr-results/evaluation-v4
export EEGREPR_OLD=/data/u7041472/eegrepr-results/h1-information-v2/bounded_development
test -f "$EEGREPR_OLD/resolved_config.json"
test -f "$EEGREPR_OLD/native/manifest.json"
mkdir -p "$EEGREPR_ROOT"
bash scripts/eegrepr/run_h1_h4_v4.sh doctor "$EEGREPR_ROOT/qualified-existing" cpu \
  --config configs/eeg_representation/v4/existing.yaml
df -h "$EEGREPR_ROOT"
```

If that local branch already exists in another worktree, Git will refuse to add it. Use `git worktree list` and the existing **v4** checkout; do not reset, overwrite, or repurpose the historical H1 checkout. If these output subdirectories already contain results from another source hash, choose a new versioned parent root. The expected prospective native footprint is provisionally 4.75 GB before fitting/exports; compact and full exports require additional disk. The measured gate replaces the provisional estimate.

## Reuse old evidence on CPU, with no simulation

```bash
cd "$EEGREPR_REPO"
Q="$EEGREPR_ROOT/qualified-existing"
CFG=configs/eeg_representation/v4/existing.yaml
bash scripts/eegrepr/run_h1_h4_v4.sh controls "$Q" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh audit-existing "$Q" cpu --config "$CFG" \
  --source "$EEGREPR_OLD" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh qualify-existing "$Q" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh replication "$Q" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh retention "$Q" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh geometry-existing "$Q" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh report "$Q" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh export "$Q" cpu --config "$CFG" --resume \
  --destination "$EEGREPR_ROOT/qualified-existing-standard.zip"
python scripts/eegrepr/pack_v4_evidence.py pack \
  "$EEGREPR_ROOT/qualified-existing-standard.zip" \
  "$EEGREPR_ROOT/qualified-existing-compact.zip" --max-bytes 512000000
```

`--resume` is also valid for the first call on an empty root. Repeated exports require a new destination filename. The standard compact export retains every current numeric array and can exceed 512 MB. The separate packing command applies reversible byte shuffling/prediction residuals, verifies every restored member against its original SHA256 and keeps reports directly readable. It requires NumPy and changes no scientific values. Both ZIP containers temporarily need disk space. The packer fails explicitly if the resulting package still exceeds the upload cap; it never deletes evidence to fit. The default original inputs are never regenerated. A source without native files completes only supported saved-array derivations.

## Analytical and reduced native engineering checks

These explicit commands use separate roots and cannot stand in for the full-network timing gate.

```bash
S="$EEGREPR_ROOT/analytical-smoke"
CFG=configs/eeg_representation/v4/analytic_smoke.yaml
bash scripts/eegrepr/run_h1_h4_v4.sh plan "$S" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh generate "$S" cpu --config "$CFG" --resume --execute
bash scripts/eegrepr/run_h1_h4_v4.sh derive "$S" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh fit "$S" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh score "$S" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh replication "$S" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh retention "$S" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh report "$S" cpu --config "$CFG" --resume

N="$EEGREPR_ROOT/reduced-native-smoke"
CFG=configs/eeg_representation/v4/simulator_smoke.yaml
bash scripts/eegrepr/run_h1_h4_v4.sh plan "$N" cpu --config "$CFG" --resume
bash scripts/eegrepr/run_h1_h4_v4.sh generate "$N" cpu --config "$CFG" --resume --execute
bash scripts/eegrepr/run_h1_h4_v4.sh derive "$N" cpu --config "$CFG" --resume
```

The optional common-event sensitivity has a separate `optional_common_smoke.yaml`, with c=.02 and D=.5 rad²/s on a reduced network. It is off in all default campaign configurations. Running it is a separate decision; it is not included in the 832-episode quote or evidence for human validity. No layer-instrumentation implementation is claimed.

## Freeze and measure before prospective generation

First review the qualification report and record scientific decisions. The `plan --qualification-root` step freezes target/operator, learner grid, calibration policy, new circuit namespace and one matched-code task contrast. Positive evaluated skill is not an eligibility condition.

```bash
P="$EEGREPR_ROOT/bounded-prospective"
G="$EEGREPR_ROOT/timing-gate"
Q="$EEGREPR_ROOT/qualified-existing"
bash scripts/eegrepr/run_h1_h4_v4.sh plan "$P" cpu --resume --qualification-root "$Q"
bash scripts/eegrepr/run_h1_h4_v4.sh plan "$G" cpu --resume \
  --config configs/eeg_representation/v4/timing_gate.yaml
bash scripts/eegrepr/run_h1_h4_v4.sh timing-gate "$G" cpu --resume --execute
bash scripts/eegrepr/run_h1_h4_v4.sh plan "$P" cpu --resume \
  --qualification-root "$Q" --timing-root "$G"
cat "$P/plan.json"
```

The eight-episode gate is the first full-network new simulation the user must run. It measures host-specific runtime, native bytes and RSS. The provisional 832-episode cost is **23.416 summed serial hours**, excluding fitting, derivation, replay controls and exports. CPU topology, library versions and full prefix replay affect this cost; CUDA does not accelerate NEURON. The gate's timing and checksums must match the current host/software and 7.010-s episode support.

## Execute the frozen bounded design, then fit on CUDA

Only after checking the measured plan:

```bash
P="$EEGREPR_ROOT/bounded-prospective"
bash scripts/eegrepr/run_h1_h4_v4.sh generate "$P" cpu --resume --execute
bash scripts/eegrepr/run_h1_h4_v4.sh derive "$P" cpu --resume
bash scripts/eegrepr/run_h1_h4_v4.sh doctor "$P" cuda
bash scripts/eegrepr/run_h1_h4_v4.sh fit "$P" cuda --resume
bash scripts/eegrepr/run_h1_h4_v4.sh score "$P" cpu --resume
bash scripts/eegrepr/run_h1_h4_v4.sh replication "$P" cpu --resume
bash scripts/eegrepr/run_h1_h4_v4.sh retention "$P" cuda --resume
bash scripts/eegrepr/run_h1_h4_v4.sh geometry-existing "$P" cpu --resume \
  --config configs/eeg_representation/v4/bounded.yaml
bash scripts/eegrepr/run_h1_h4_v4.sh report "$P" cpu --resume
bash scripts/eegrepr/run_h1_h4_v4.sh export "$P" cpu --resume \
  --destination "$EEGREPR_ROOT/bounded-prospective-standard.zip"
python scripts/eegrepr/pack_v4_evidence.py pack \
  "$EEGREPR_ROOT/bounded-prospective-standard.zip" \
  "$EEGREPR_ROOT/bounded-prospective-compact.zip" --max-bytes 512000000
# Optional full-native archive; provision enough additional disk first.
bash scripts/eegrepr/run_h1_h4_v4.sh export "$P" cpu --resume --full-native \
  --destination "$EEGREPR_ROOT/bounded-prospective-full-native.zip"
cd "$EEGREPR_ROOT"
sha256sum -c bounded-prospective-compact.zip.sha256
```

The CPU implementation and the existing-data and engineering workflows were executed locally. The full-network timing-gate and 832-episode prospective-generation commands above remain unrun. CUDA hardware was unavailable locally, so actual GPU training is a workstation check, not a claimed local test. The deterministic CuBLAS environment is set before application startup. Do not switch device halfway through a fitted control resume: its signature records the device.

No command automatically launches old expanded-reference or confirmation plans. Keep zero/negative results. A later powered confirmation needs a new protocol based on circuit-level heterogeneity, practical effect or CI precision and multiplicity; futures and optimizer seeds cannot replace independent circuits.

## Inspect the upload-sized package

The package contains its own `pack_v4_evidence.py` helper and `PACKING_README.md`. Extract that helper first. NumPy is required for this outer lossless layer:

```bash
python pack_v4_evidence.py verify qualified-existing-compact.zip
python pack_v4_evidence.py unpack qualified-existing-compact.zip restored-standard.zip
```

Then extract `implementation/experiments/eeg_representation/h1/v4/restore_export.py` from `restored-standard.zip` and use its standard-library verification/NPZ restoration instructions. Every original member and NPY array is byte exact; ZIP/NPZ container metadata can differ. Preserve the supplied `.sha256` sidecar when sharing.
