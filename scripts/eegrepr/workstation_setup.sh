#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
MODE="${1:-check}"
case "$MODE" in
  check)
    python -m experiments.eeg_representation.cli doctor
    ;;
  create-ml)
    TARGET_ENV="${2:?Usage: workstation_setup.sh create-ml /path/to/new/ml-env}"
    if [[ -e "$TARGET_ENV" ]]; then
      echo "Refusing an existing environment: $TARGET_ENV" >&2
      exit 2
    fi
    python3 -m venv "$TARGET_ENV"
    "$TARGET_ENV/bin/python" -m pip install -r requirements/eegrepr-ml.txt
    echo "Base ML dependencies installed. Install a compatible PyTorch build in this NEW environment, then run doctor."
    ;;
  compile-check)
    # BallAndStick uses built-in hh/pas/Exp2Syn/extracellular and direct NetCon.event schedules;
    # no repository NMODL sources exist in setup/circuits/ballnstick.
    python - <<'PY'
from pathlib import Path
import LFPy
from neuron import h
mods = list(Path('setup/circuits/ballnstick').glob('*.mod'))
if mods:
    raise SystemExit('Unexpected BallAndStick NMODL sources: inspect before compiling')
for mechanism in ('Exp2Syn', 'NetStim'):
    if not hasattr(h, mechanism):
        raise SystemExit(f'Missing {mechanism}; check the validated NEURON installation')
print('BallAndStick needs no repository nrnivmodl step; built-in NEURON mechanisms are available.')
print('LFPy installation:', Path(LFPy.__file__).parent)
PY
    ;;
  *) echo "Usage: $0 [check|compile-check|create-ml NEW_ENV]" >&2; exit 2 ;;
esac
