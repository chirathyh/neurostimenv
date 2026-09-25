"""Compare two independently constructed L23Net no-field replay artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.l23net_analysis.replay_validation import (  # noqa: E402
    compare_replay_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = compare_replay_artifacts(args.first, args.second)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)
    print(f"Saved {args.output.resolve()}", flush=True)
    if comparison["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
