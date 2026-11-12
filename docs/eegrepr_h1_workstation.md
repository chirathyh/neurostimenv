# Run H1 v2 on the GPU workstation

Use the new implementation branch and new `/data` output roots below. These commands preserve the current COSYNE and legacy EEG checkouts. No full campaign was run locally. The launcher is serial for NEURON generation; CUDA is used only for supported learning stages. The full-network gate is mandatory for development/expanded plans.

## One-time checkout and environment

Run on `neuroinformaticsgpu01` after connecting with your existing SSH configuration. The worktree command is for first setup; if that exact worktree already exists, enter it and verify the branch instead of recreating it.

```bash
cd /home/u7041472/Documents/chirath/depression-simulator/neurostimenv-eegrepr
git fetch origin
git worktree add --track -b codex/eegrepr-h1-information-v2 \
  /home/u7041472/Documents/chirath/depression-simulator/neurostimenv-eegrepr-h1 \
  origin/codex/eegrepr-h1-information-v2

cd /home/u7041472/Documents/chirath/depression-simulator/neurostimenv-eegrepr-h1
source /home/u7041472/Documents/chirath/depression-simulator/bin/activate
git branch --show-current
git rev-parse HEAD
git status --short
python -m experiments.eeg_representation.h1.cli doctor
```

The branch must be `codex/eegrepr-h1-information-v2`, and the source should be clean. Use the pushed commit reported with this delivery. No new Python package is required beyond the working EEG pilot environment: NumPy, SciPy, h5py, PyYAML, matplotlib, scikit-learn, PyTorch, NEURON and LFPy (plus existing repository dependencies). Do not replace working CUDA PyTorch or upgrade NEURON just to run this campaign. If `doctor` reports a missing analysis package, this command installs only missing packages and leaves existing versions alone:

```bash
python - <<'PY'
import importlib.util, subprocess, sys
packages={'numpy':'numpy','scipy':'scipy','h5py':'h5py','yaml':'PyYAML',
          'matplotlib':'matplotlib','sklearn':'scikit-learn'}
missing=[package for module,package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Installed missing analysis packages:', missing)
PY
```

A missing NEURON/LFPy or CUDA runtime requires repairing the existing simulator environment before proceeding; the doctor and simulator smoke expose that problem. It is not resolved by silently substituting analytical data. Confirm space on the **output filesystem**, then run the checks:

```bash
mkdir -p /data/u7041472/eegrepr-results/h1-information-v2
df -h /data/u7041472/eegrepr-results/h1-information-v2
bash scripts/eegrepr/run_h1.sh doctor /data/u7041472/eegrepr-results/h1-information-v2 cpu
bash scripts/eegrepr/run_h1.sh tests /data/u7041472/eegrepr-results/h1-information-v2 cpu
bash scripts/eegrepr/run_h1.sh smoke /data/u7041472/eegrepr-results/h1-information-v2 cpu
bash scripts/eegrepr/run_h1.sh simulator-smoke /data/u7041472/eegrepr-results/h1-information-v2 cpu
```

`doctor` audits `/data/u7041472/eegrepr-results/core-pilot` read-only and verifies accessible native/derived checksums. Missing paths are reported, not inferred as present. `smoke` is analytical; `simulator-smoke` is 2E/1I. Both are explicitly engineering evidence. The latter is not the full-network cost gate. The scripted smoke runs plan → exact controls → training/development generation → evaluation forks → derive → fit/score → replication → compression (analytical smoke only) → scale → summarize → export. It includes explicit numeric replay and prefix checks. The controls-only command is also available:

```bash
bash scripts/eegrepr/run_h1.sh controls /data/u7041472/eegrepr-results/h1-information-v2 cpu
```

## Full-network timing gate, then bounded development

This is the first local-to-workstation boundary. The following command runs **eight** full-network continuations (two new development circuits, two regimes, two futures) with early stopping. It prints the measured resources and bounded/expanded projections afterwards. It does not launch those larger stages.

```bash
bash scripts/eegrepr/run_h1.sh gate /data/u7041472/eegrepr-results/h1-information-v2 cpu
```

Read the saved `timing_gate/timing_audit.json` and the printed plans. Projections assume one serial worker and include replayed histories; the fit/derive overhead is additional. The historical 209.93 trial-hour figure is not this measured gate. Changing source/config/hardware invalidates the rationale for reusing a timing audit. The source/config checks are enforced; the operator must also remeasure after changing machines.

```bash
bash scripts/eegrepr/run_h1.sh plan-bounded /data/u7041472/eegrepr-results/h1-information-v2 cuda
```

After inspecting the cost, explicitly run the bounded campaign:

```bash
bash scripts/eegrepr/run_h1.sh bounded /data/u7041472/eegrepr-results/h1-information-v2 cuda
```

This executes `bounded_development.yaml`: 256 separate training histories/outcomes, 64 development histories/outcomes, then 16 evaluation contexts with 16 futures each, **576** native episodes total. It fits all declared baselines and paired neural objectives, scores identical held-out futures, computes available finite-M analyses and the pooled one-future model, runs bounded compression, summarizes, and creates `bounded_development-evidence.zip` plus SHA-256. Reference RMSE/coverage remains pending here because this bounded panel has no independent high-M reference pool. Higher M entries and unsupported fixed-budget allocations are explicitly pending.

The script always uses `--resume`, which safely creates an empty new root on its first invocation. Rerunning it checks and reuses compatible completed stages. Do not edit a configuration/source halfway through a run, delete provenance, or overwrite old data to bypass an incompatibility. Choose a new versioned output base after a scientific change. An existing archive is retained; choose a new archive filename for a later export. The per-command logs are outside the immutable scientific roots.

## Seven-job scale comparison

This is a **separate** planned dataset root, because the longer maximum target lead changes episode duration/cutoff requirements and job metadata. It is not an assertion that the bounded single-job data already executed a sweep.

```bash
bash scripts/eegrepr/run_h1.sh plan-scale /data/u7041472/eegrepr-results/h1-information-v2 cuda
bash scripts/eegrepr/run_h1.sh scale /data/u7041472/eegrepr-results/h1-information-v2 cuda
```

`scale_core.yaml` runs primary, 250-Hz input, 100-Hz input, 8-bit acquisition, 1-s input history, 100-ms lead and RMS-target comparisons. Native episodes are shared across jobs within this run. Fitting and task readouts repeat per job, so generation count alone understates its compute/storage cost. Inspect the plan and measured bounded compression cost before this launch. Source-model counts, latent-code feasibility and skips are recorded separately.

## Explicit optional extensions

Each `plan-*` command requires a completed compatible gate and prints a plan without generating the campaign. Each following run is a separate explicit action. Run only the chosen extension after reviewing development diagnostics; these are not a recommended automatic sequence.

```bash
# Independent 256/128-future references at eight preselected contexts; 32/32 elsewhere.
bash scripts/eegrepr/run_h1.sh plan-expanded /data/u7041472/eegrepr-results/h1-information-v2 cuda
bash scripts/eegrepr/run_h1.sh expanded /data/u7041472/eegrepr-results/h1-information-v2 cuda

# B_sim=256 nested allocations; 768 unique training futures +64 dev +3584 evaluation.
bash scripts/eegrepr/run_h1.sh plan-fixed-budget /data/u7041472/eegrepr-results/h1-information-v2 cuda
bash scripts/eegrepr/run_h1.sh fixed-budget /data/u7041472/eegrepr-results/h1-information-v2 cuda

# Optional larger latent/code/label grid on its own immutable data root.
bash scripts/eegrepr/run_h1.sh plan-compression /data/u7041472/eegrepr-results/h1-information-v2 cuda
bash scripts/eegrepr/run_h1.sh compression /data/u7041472/eegrepr-results/h1-information-v2 cuda

# Twenty-five explicit target/input jobs, not a Cartesian product.
bash scripts/eegrepr/run_h1.sh plan-targeted-scale /data/u7041472/eegrepr-results/h1-information-v2 cuda
bash scripts/eegrepr/run_h1.sh targeted-scale /data/u7041472/eegrepr-results/h1-information-v2 cuda

# Independent new-circuit 20/10/0-dB input-noise extension; future remains clean.
bash scripts/eegrepr/run_h1.sh plan-noise /data/u7041472/eegrepr-results/h1-information-v2 cuda
bash scripts/eegrepr/run_h1.sh noise /data/u7041472/eegrepr-results/h1-information-v2 cuda
```

The provisional confirmation configuration has 12 new evaluation circuits and five optimizer seeds, **37,632 native episodes** including training/development. This is a costed design, not a powered sample-size commitment. Refine and freeze it only after development precision and model diagnostics; avoid inspecting confirmation outcomes to tune it. The implemented explicit commands are:

```bash
bash scripts/eegrepr/run_h1.sh plan-confirmation /data/u7041472/eegrepr-results/h1-information-v2 cuda
# Run only after the final scientific protocol and measured budget have been accepted:
bash scripts/eegrepr/run_h1.sh confirmation /data/u7041472/eegrepr-results/h1-information-v2 cuda
```

The launcher passes `--execute-large` only for the explicitly requested expanded, fixed-budget or confirmation mode. Plain help, plans, smoke and gate do not trigger these runs. No scheduler, background automation or automatic multi-day launch is installed.

## Inspect and share evidence

For each root, inspect `summary/summary.json`, `summary/per_circuit_scores.csv`, model diagnostics, pooled-estimator qualifications, pending/unsupported entries, `compression/surface.json`, and `scale/executed_scale_table.json`. Figures are saved as PNG and PDF. The summary recomputes saved numerical scores and rejects a disagreement with the original summary.

Compact archives include all derived targets, masks/IDs, model predictions/draws, codebooks/codes, warning/class sets, score components, plots, source snapshot and checksums. They explicitly omit native HDF5 simulation arrays to keep transfer practical. To include those raw native recordings for a complete independent analysis, use an **additional**, new archive filename:

```bash
python -m experiments.eeg_representation.h1.cli h1-export \
  --config configs/eeg_representation/h1/bounded_development.yaml \
  --output-root /data/u7041472/eegrepr-results/h1-information-v2/bounded_development \
  --resume --include-native \
  --archive /data/u7041472/eegrepr-results/h1-information-v2/bounded_development-with-native.zip
```

On your laptop, copy the compact evidence for review using your existing SSH alias:

```bash
mkdir -p "$HOME/Downloads/eegrepr-analysis/h1-information-v2"
scp \
  neuroinformatics:/data/u7041472/eegrepr-results/h1-information-v2/bounded_development-evidence.zip \
  neuroinformatics:/data/u7041472/eegrepr-results/h1-information-v2/bounded_development-evidence.zip.sha256 \
  "$HOME/Downloads/eegrepr-analysis/h1-information-v2/"
cd "$HOME/Downloads/eegrepr-analysis/h1-information-v2"
sha256sum -c bounded_development-evidence.zip.sha256
```

The same naming convention applies to `scale_core`, `expanded_reference` and other completed roots. Do not combine original pilot and new engineering/full-network scores into one unlabelled result table.
