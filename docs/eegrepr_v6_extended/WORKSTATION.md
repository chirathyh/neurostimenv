# Complete extended V6 workstation campaign

Run these commands on `neuroinformaticsgpu01`, using the existing environment. The historical workstation paths below must pass the explicit checks. They have not been verified from this laptop because the SSH gateway requires interactive authentication. A path check failure is a reason to locate the actual worktree/data, not to create an empty substitute. Never reset an existing checkout or modify historical result roots.

## 1. Isolated code, environment and data preflight

```bash
set -euo pipefail
parent=/home/u7041472/Documents/chirath/depression-simulator
old_checkout="$parent/neurostimenv-evaluation-v4"
checkout="$parent/neurostimenv-evaluation-v6-extended"
activation="$parent/bin/activate"
source_archive=/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective-standard.zip
native_source=/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective
campaign=/data/u7041472/eegrepr-results/evaluation-v6-extended
branch=codex/eeg-evaluation-v6-immediate-information
workers=1

test "$(hostname -s)" = neuroinformaticsgpu01
test -f "$activation"
test -d "$old_checkout/.git" || test -f "$old_checkout/.git"
test -f "$source_archive"
test -f "$native_source/generation/native/manifest.json" || test -f "$native_source/native/manifest.json"
printf '%s  %s\n' 6818befa7ca70df3a0272bf69328f9e2d2316817493e49a13bac6e4b74a5817f "$source_archive" | sha256sum -c -
git -C "$old_checkout" fetch origin "$branch"
if [ ! -e "$checkout" ]; then
  git -C "$old_checkout" worktree add --detach "$checkout" "origin/$branch"
fi
cd "$checkout"
test "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$branch")"
test -z "$(git status --porcelain)"
git rev-parse HEAD
source "$activation"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import sys,torch,neuron,LFPy; print(sys.executable); print(torch.__version__,neuron.__version__,LFPy.__version__); print("CUDA",torch.cuda.is_available())'
mkdir -p "$campaign/logs"
df -h "$campaign"

bash scripts/eegrepr/run_h1_h4_v6_extended.sh preflight \
  --root "$campaign/preflight" --device cuda --workers "$workers" --resume \
  2>&1 | tee "$campaign/logs/preflight.log"
```

If the old checkout path differs, run `git worktree list` from your known checkout and change `old_checkout` accordingly. If the new checkout already exists at another commit, preserve it and select a new versioned checkout path. The clean-commit and environment records bind the actual implementation used.

The archive alone supports 33 of 60 jobs. The H5 native payloads, rather than only their manifest, are needed for the 27 immediate-target jobs and full Phase0. The following read-only inventory verifies all native payload hashes before fitting:

```bash
python - "$native_source" <<'PYNATIVE'
import sys
from experiments.eeg_representation.h1.v6.evidence import EvidenceSource
with EvidenceSource(sys.argv[1]) as source:
    if not source.records:
        raise SystemExit('STOP: no native manifest records')
    missing = [r['path'] for r in source.records.values() if not source.exists(r['path'])]
    if missing:
        raise SystemExit('STOP: native recordings missing: ' + ', '.join(missing[:8]))
    for name in source.records:
        source.verify(name)
    print('Verified native recordings:', len(source.records))
PYNATIVE
```

Keep `workers=1` unless the actual-host CPU/RAM assessment justifies another small fixed value. Choose it before the timing gate and retain it on resume. GPU does not accelerate NEURON.

## 2. Dry-run and meaningful smoke

The initial dry-run describes the full 60-target matrix, prerequisites and provisional costs. It does not generate episodes or fit the scientific campaign. Before Phase0, precision and numerical gates will be unresolved. Exit zero means the plan was written; inspect its eligibility fields and blocked reasons.

```bash
bash scripts/eegrepr/run_h1_h4_v6_extended.sh dry-run \
  --root "$campaign/design-preview" \
  --source-archive "$source_archive" --native-source "$native_source" \
  --phase0-root "$campaign/phase0" --timing-root "$campaign/timing-gate" \
  --device cuda --workers "$workers" --resume \
  2>&1 | tee "$campaign/logs/dry-run.log"

bash scripts/eegrepr/run_h1_h4_v6_extended.sh smoke \
  --root "$campaign/smoke" --device cpu --resume \
  2>&1 | tee "$campaign/logs/smoke.log"
cat "$campaign/smoke/smoke_result.json"
```

The smoke runs the same small neural path on a known learnable control, exact target/bin/causality and saved-array tests, finite-future/statistical controls, information accounting, replay-audit semantics and staged code/state readout checks. It uses synthetic/analytical data and does **not** launch the 40-cell numerical panel. Synthetic pass results are not biophysical findings. The test log is `smoke/smoke_tests.log`.

## 3. Complete campaign in a recoverable tmux shell

Start an interactive shell so a failed gate leaves the pane open:

```bash
tmux new-session -s eeg-v6-extended
```

Inside tmux, set the same verified values explicitly:

```bash
set +e
set -o pipefail
parent=/home/u7041472/Documents/chirath/depression-simulator
checkout="$parent/neurostimenv-evaluation-v6-extended"
activation="$parent/bin/activate"
source_archive=/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective-standard.zip
native_source=/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective
campaign=/data/u7041472/eegrepr-results/evaluation-v6-extended
workers=1
cd "$checkout"
source "$activation"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$campaign/logs"

bash scripts/eegrepr/run_h1_h4_v6_extended.sh complete \
  --root "$campaign" \
  --config configs/eeg_representation/v6_extended/confirmation.yaml \
  --source-archive "$source_archive" --native-source "$native_source" \
  --destination "$campaign/confirmation-extended-standard.zip" \
  --device cuda --workers "$workers" --execute --resume \
  2>&1 | tee -a "$campaign/logs/complete.log"
run_status=${PIPESTATUS[0]}
printf '%s\n' "$run_status" > "$campaign/logs/exit-status.txt"
printf 'Extended runner exit status: %s\n' "$run_status"
```

Detach with Ctrl+b, then d. Reattach with `tmux attach -t eeg-v6-extended`. The child wrapper exits without closing this interactive pane. Exit 2 indicates a saved blocker/failure; inspect the stage report and log before proceeding. A plan or timing gate finishing does not mean the full campaign is complete.

The complete command performs the following stages in order:

| Stage | Reuse/new work | Eligibility and expected outputs |
|---|---|---|
| Doctor, controls and historical audit | Verify the old archive and reproduce original metrics | `phase0/doctor.json`, `controls.json`, audit records |
| Expanded Phase0 | Reuse all 832 historical native episodes; derive all 60 jobs, full model grid, added readouts and descriptive information/H2 reports | `phase0/derived`, `models`, `information_models`, `information`, `replication`, `report`; all inspected/exploratory |
| Precision selection | Use inspected development/circuit variation, including first-four-future accounting | `design-review/precision_certificate.json`, `selected-config.json`; no fresh-test significance-based N changes |
| Timing and throughput | Eight registered 40-cell episodes; actual CPU/RSS/native bytes and fitting throughput | `timing-gate/generation`, `fit_throughput.json`; device/worker-bound |
| Replay diagnostic | Repeat the same eight gate episodes under exact realised forcing | Selected core `replay/summary.json`; separate common-subset table with verified Phase0-frozen mean/short/full predictions; no full-panel measured zero |
| Numerical convergence | Separate 768 coupled-integration episodes | Selected core `numerical/`; inconclusive or failed tolerance blocks fresh core generation |
| Final measured plan | Recheck precision, native reuse, replay, numerical and resource gates | Selected core `plan.json`; all required eligibility must pass |
| Train/development | Verify/reuse 320 old episodes and generate the additional registered circuits; derive all targets and freeze means/readouts | Selected core `generation`, `derived`, `models`, `information_models/freeze.json` |
| Calibration | Generate 24 separate calibration circuits, predict with frozen means, calibrate variance | `calibration`; no mean/readout retuning |
| Fresh evaluation and H2 | Generate frozen test cohort with primary first-four futures and registered M32/L128 extensions | `generation`, unchanged models/readouts, new inference arrays |
| Retention, scores and inference | Original H3 conjunction/H1 primary plus labelled secondary waveform/code/state contrasts and H2 analyses | `retention`, `scores`, `information`, `replication`, `confirmation_statistics.json` |
| Readable report and export | Completed evidence only; unknown quantities remain unknown | `report/MAIN_ANALYSIS.md`, vector figures, lossless standard ZIP plus SHA sidecar |

The checked-in N192 candidate has 13,056 core episodes, 320 reused. Historical first-four-future precision requires at least a provisional N257 (15,136 core / 14,816 new), plus the unchanged 768 numerical episodes, the eight-episode gate and eight additional replay episodes. Previous same-host planning rates put N257 generation near 467 summed CPU hours before small gate/replay overhead; this is provisional, not a current measurement or wall-time promise. The actual plan must pass the 500-hour generation cap, 120 GB native cap and 250 GB minimum free-space floor **and** the larger dynamic working/export reserve. Current planning estimates are approximately 573 GB at N192 and 722 GB at N257 (two native plus two non-native working/export copies; projected non-native evidence is about 272.05 GB at N257). These provisional estimates must be checked on the actual storage volume; 250 GB alone is not sufficient for this expanded plan. Parallel workers change wall time, not summed trial hours. All 60 derived targets reuse each native recording. The 60/27 target-grid expansion increases fitting substantially; actual-host throughput and the additional 1,820 ridge fits/260 mean fallbacks plus three 200-update forecast encoders are costed separately. No automatic old 96-hour/927-hour or biological factorial is launched.

## 4. Status, recovery and resume

```bash
bash scripts/eegrepr/run_h1_h4_v6_extended.sh status --root "$campaign"
tmux list-sessions
pgrep -afu "$USER" '[e]xperiments.eeg_representation.h1.v6.(extended_cli|cli)|[r]un_h1_h4_v6'
tail -n 80 "$campaign/logs/complete.log"
cat "$campaign/logs/exit-status.txt" 2>/dev/null || true
```

`status` only reads artifacts. A saved `running` entry is not proof that a process is still alive; check `pgrep`, tmux and the log together. It reports the frozen N, latest stage records, missing targets and failures without rewriting them.

Resume in the same tmux shell with the same paths, configuration, worker count, device, dtype and output destination:

```bash
bash scripts/eegrepr/run_h1_h4_v6_extended.sh resume \
  --root "$campaign" \
  --config configs/eeg_representation/v6_extended/confirmation.yaml \
  --source-archive "$source_archive" --native-source "$native_source" \
  --destination "$campaign/confirmation-extended-standard.zip" \
  --device cuda --workers "$workers" --execute \
  2>&1 | tee -a "$campaign/logs/complete.log"
run_status=${PIPESTATUS[0]}
printf '%s\n' "$run_status" > "$campaign/logs/exit-status.txt"
```

Resume verifies stage/operator/protocol/data/model fingerprints and reuses valid completed work. Native physical provenance remains independent of analysis-only source additions. Mismatches and incomplete scientific gates are preserved as blockers; do not delete a gate result until it passes. A completed run verifies its receipt and ZIP payloads rather than refitting. An orphan export without a completed receipt needs a new explicit export filename. Changing a scientific protocol or a frozen precision choice requires a new versioned result root.

## 5. Report-only regeneration and lossless export

Report-only automatically resolves the saved `confirmation-n<N>` from `selected-config.json`; it never guesses N192 or fits a model. If the campaign stopped before core initialization, it reports the existing inspected Phase0 evidence. An explicit initialized stage root is also accepted.

```bash
bash scripts/eegrepr/run_h1_h4_v6_extended.sh report-only --root "$campaign"
bash scripts/eegrepr/run_h1_h4_v6_extended.sh status --root "$campaign"
```

Resolve the actual selected core for inspection or a separate new export:

```bash
frozen_config="$campaign/selected-config.json"
test -f "$frozen_config"
chosen_n=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["protocol"]["test_circuits"])' "$frozen_config")
core="$campaign/confirmation-n$chosen_n"
cat "$core/report/MAIN_ANALYSIS.md"
(cd "$campaign" && sha256sum -c confirmation-extended-standard.zip.sha256)
```

Expected main outputs:

- `report/MAIN_ANALYSIS.md`, `report/FINDINGS.md`, `report/main_analysis.json`: completion overview and H1–H4 findings with methods, numbers, uncertainty and limitations.
- `report/manuscript_results.md`: paragraphs grounded in completed outcomes only.
- `report/tables/`: information-condition, gap/history/acquisition, H2 and H3/H4 numeric tables in JSON/CSV.
- `report/figures/`: target timing, matched gap/history performance and information-versus-error figures in PDF/SVG/PNG; panels with absent data are marked unavailable, never fabricated.
- `information/<job>/summary.json` and `loss_components.npz`: unrounded physical/model/baseline/floor/excess data and per-circuit components; `information/paired_contrasts.json`: dependent gap/acquisition/history comparisons.
- `information_models/manifest.json`, `freeze.json`, `timing.json`; per-readout `selection.json`, `state.npz`, `predictions.npz`, `inference.json`; bank projections, quantizer metadata and code payloads.
- `replay/summary.json`, replay native/record artifacts: separately identified privileged checks.
- `derived/<job>/manifest.json`: target-start/first-bin/end/native support, sample availability, operator identity, target-specific train-only normalization and source provenance.
- `confirmation-extended-standard.zip` and `.sha256` at the campaign base: lossless target/prediction/variance/logit/code evidence, source closure and omitted-native/readout-weight manifests.

For a separate larger full-native export after the campaign, use a new filename:

```bash
bash scripts/eegrepr/run_h1_h4_v6_extended.sh export \
  --root "$core" --config "$frozen_config" --resume \
  --destination "$campaign/confirmation-extended-full-native.zip" --full-native
(cd "$campaign" && sha256sum -c confirmation-extended-full-native.zip.sha256)
```

A new standard export can enforce `--max-bytes 536870912` if required; exceeding the cap fails rather than silently dropping arrays. The standard archive intentionally omits native recordings and numeric weights with hashes; use the full-native option when those are needed and available. Supplied archives are never unpickled.

## Local verification scope

The target amendment has been tested on synthetic/native-schema fixtures and the actual downloaded V4 arrays. Exact preservation was checked for all 27 historical V6 jobs; 33 expanded saved-array jobs are available and 27 immediate jobs are missing native payloads locally. This is implementation/exploratory evidence, not completion of the workstation campaign. Run the wrapper smoke on the actual host and retain its log. Full neural qualification, measured resource gates, full-network numerical convergence, prospective replay and fresh confirmation remain distinct workstation stages.
