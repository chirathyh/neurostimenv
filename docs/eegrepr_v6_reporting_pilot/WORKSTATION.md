# Run the bounded reporting pilot on neuroinformaticsgpu01

These commands use the historical workstation paths reported in this project and explicitly check them. The implementation machine could not authenticate through the SSH gateways noninteractively; **no workstation pilot has been launched**. The native 832-episode source is needed on the workstation. The downloaded standard ZIP alone is insufficient.

## 1. Connect and create the isolated checkout

From the laptop:

```bash
ssh neuroinformatics
```

On the workstation:

```bash
hostname -s
PARENT=/home/u7041472/Documents/chirath/depression-simulator
PILOT_REPO="$PARENT/neurostimenv-v6-reporting-pilot"
test -f "$PARENT/bin/activate" || { echo 'Missing historical Python environment'; exit 1; }
test ! -e "$PILOT_REPO" || { echo 'Pilot checkout already exists: inspect it; do not overwrite'; exit 1; }
git clone --single-branch --branch codex/eeg-v6-eight-worker-reporting-pilot \
  https://github.com/chirathyh/neurostimenv.git "$PILOT_REPO"
cd "$PILOT_REPO"
source "$PARENT/bin/activate"
git status --short
git rev-parse HEAD
python --version
```

For an existing pilot checkout, inspect `git status --short` and the recorded commit before updating. Do not alter a running or partially completed experiment's implementation; resume with its original checkout/config/device. Do not reuse a different experiment's output directory.

Check PDF dependencies before starting. Installing these packages does not launch a simulation:

```bash
python -m pip install reportlab pypdf pillow matplotlib
command -v pdftoppm
command -v pdfinfo
```

If either binary is absent, install the OS `poppler-utils` package through the host's approved package-management procedure, then rerun preflight. Do not upgrade NEURON/LFPy/CUDA as part of this pilot without a separate compatibility check.

## 2. Establish a persistent session and verify source paths

```bash
tmux new-session -s eegrepr-v6-reporting-pilot
```

Inside tmux, run:

```bash
PARENT=/home/u7041472/Documents/chirath/depression-simulator
PILOT_REPO="$PARENT/neurostimenv-v6-reporting-pilot"
PILOT_ROOT=/data/u7041472/eegrepr-results/v6-reporting-pilot-v1
SOURCE_ROOT=/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective
SOURCE_ZIP=/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective-standard.zip
cd "$PILOT_REPO"
source "$PARENT/bin/activate"
test -d "$SOURCE_ROOT" || { echo 'Native source directory absent'; false; }
test -f "$SOURCE_ZIP" || { echo 'Source archive absent'; false; }
python - "$SOURCE_ZIP" <<'PY'
import hashlib, pathlib, sys
p=pathlib.Path(sys.argv[1]); h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(1<<20),b''): h.update(b)
assert h.hexdigest()=='6818befa7ca70df3a0272bf69328f9e2d2316817493e49a13bac6e4b74a5817f', 'Wrong historical archive'
print('Historical archive verified:', p)
PY
```

The command below creates a new pilot root. If preflight was already run, add `--resume`; do not delete its budget ledger. A blocked preflight returns the concrete dependency/RAM/storage/CPU issue. Resolve that issue before proceeding.

```bash
bash scripts/eegrepr/run_h1_h4_v6_reporting_pilot.sh preflight \
  --root "$PILOT_ROOT" --device cuda
bash scripts/eegrepr/run_h1_h4_v6_reporting_pilot.sh plan \
  --root "$PILOT_ROOT" --device cuda --resume \
  --source-archive "$SOURCE_ZIP" --native-source "$SOURCE_ROOT"
bash scripts/eegrepr/run_h1_h4_v6_reporting_pilot.sh smoke \
  --root "$PILOT_ROOT" --device cuda --resume
```

Plan and smoke do not invoke NEURON. The actual eight-process concurrency measurement runs inside the complete command, after source/target validation and before the remaining simulation allocation. Its measured plan must pass before execution continues. The plan includes the 21-call local engineering debit; do not remove it.

CPU fitting is supported by using `--device cpu` consistently from the first preflight on a **different new root**, for example `/data/u7041472/eegrepr-results/v6-reporting-pilot-cpu-v1`. Do not switch devices midway through one root.

## 3. One complete command: simulations through automatic PDF and export

```bash
mkdir -p "$PILOT_ROOT/logs"
set -o pipefail
LOG="$PILOT_ROOT/logs/complete-$(date -u +%Y%m%dT%H%M%SZ).log"
bash scripts/eegrepr/run_h1_h4_v6_reporting_pilot.sh complete-pilot \
  --root "$PILOT_ROOT" --device cuda --resume --execute \
  --source-archive "$SOURCE_ZIP" --native-source "$SOURCE_ROOT" \
  2>&1 | tee "$LOG"
PILOT_EXIT=${PIPESTATUS[0]}
printf 'Pilot exit code: %s\nLog: %s\n' "$PILOT_EXIT" "$LOG"
```

Leave the shell open after the command returns. Detach while it runs with Ctrl+B, then D. Reattach using `tmux attach -t eegrepr-v6-reporting-pilot`. This is one coordinator, not an MPI launch; do not wrap it in `mpirun`, `srun -n8`, or launch eight copies of the wrapper. The wrapper itself manages eight workers and thread limits.

The hard cap is 160 combined actual NEURON calls including development, failures and retries. Default intended combined count is 127; the separate H2 fallback is 143 before any other failures. There is no confirmation or automatic power expansion after the pilot finishes.

## 4. Status, recovery, report-only

Read recorded progress without launching work:

```bash
bash scripts/eegrepr/run_h1_h4_v6_reporting_pilot.sh status --root "$PILOT_ROOT"
pgrep -afu "$USER" '[v]6_pilot.cli'
tmux list-sessions
```

A saved running stage is not proof of a live process. Check the process listing and the most recent log. Failures are retained under `failures/`; completed/failed/interrupted native attempts remain in `invocation_ledger/`.

After a terminated process, resume with the **same source, config, code and device**:

```bash
set -o pipefail
LOG="$PILOT_ROOT/logs/resume-$(date -u +%Y%m%dT%H%M%SZ).log"
bash scripts/eegrepr/run_h1_h4_v6_reporting_pilot.sh resume \
  --root "$PILOT_ROOT" --device cuda --execute \
  --source-archive "$SOURCE_ZIP" --native-source "$SOURCE_ROOT" \
  2>&1 | tee "$LOG"
PILOT_EXIT=${PIPESTATUS[0]}
printf 'Pilot resume exit code: %s\nLog: %s\n' "$PILOT_EXIT" "$LOG"
```

The runner verifies frozen identities and checksums. Complete native/state artifacts can be reused; incomplete attempts remain charged. It does not silently overwrite mismatched evidence. If a final evidence ZIP exists but no completion receipt was written, retain it and use `--destination "$PILOT_ROOT-recovered-evidence.zip"` on resume; this avoids overwriting the earlier export.

Rebuild tables, figures and the PDF without simulation or fitting:

```bash
bash scripts/eegrepr/run_h1_h4_v6_reporting_pilot.sh report-only --root "$PILOT_ROOT"
```

Use report-only after `pilot_results.json` exists. It honestly marks incomplete evidence and cannot create missing fits. A later PDF rebuild does not silently update the already frozen evidence ZIP or original completion receipt.

## 5. Outputs and copy back

Expected when complete:

- `$PILOT_ROOT/report/pilot_findings.pdf` — automatic self-contained findings PDF.
- `$PILOT_ROOT/report/pilot_findings.md` — editable narrative (see actual report receipt for source filename).
- `$PILOT_ROOT/report/tables/` — CSV, unrounded JSON/NPZ, Markdown and LaTeX.
- `$PILOT_ROOT/report/figures/` — PDF/SVG/PNG figures.
- `$PILOT_ROOT/report/` — every-page render and layout QA artifacts, receipt and input evidence.
- `$PILOT_ROOT/pilot_completed.json` — substantive completion receipt; absence means no completed-pilot claim.
- `/data/u7041472/eegrepr-results/v6-reporting-pilot-v1-evidence.zip` and `.sha256` — lossless compact evidence and omission manifests.

From the laptop after success:

```bash
mkdir -p "$HOME/Downloads/eegrepr-analysis/v6-reporting-pilot-v1"
scp \
  neuroinformatics:/data/u7041472/eegrepr-results/v6-reporting-pilot-v1-evidence.zip \
  neuroinformatics:/data/u7041472/eegrepr-results/v6-reporting-pilot-v1-evidence.zip.sha256 \
  neuroinformatics:/data/u7041472/eegrepr-results/v6-reporting-pilot-v1/report/pilot_findings.pdf \
  neuroinformatics:/data/u7041472/eegrepr-results/v6-reporting-pilot-v1/pilot_completed.json \
  "$HOME/Downloads/eegrepr-analysis/v6-reporting-pilot-v1/"
cd "$HOME/Downloads/eegrepr-analysis/v6-reporting-pilot-v1"
sha256sum -c v6-reporting-pilot-v1-evidence.zip.sha256
```

Review whether the figures distinguish fitted full-state error from state variance, whether populations and baseline scales are clear, whether uncertainty/deferred fields are honest, whether h0/bin-end/block availability is understandable, and whether H3 reconstruction versus task differences are readable. Then decide separately on the later full study.
