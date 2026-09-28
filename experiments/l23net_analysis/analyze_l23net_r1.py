"""Reaudit and summarize an R1 suite; reports technical failures, not silent exclusions."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.l23net_analysis.r1_analysis import analyze_suite, summary_markdown
from experiments.l23net_analysis.r1_protocol import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--core", type=Path, help="Relocated core directory for extension reanalysis")
    parser.add_argument("--output-dir", type=Path, help="Separate reanalysis directory; preserves original NCI artifacts")
    args = parser.parse_args()
    output = args.output_dir or args.suite
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = analyze_suite(args.suite, core_override=args.core, output_directory=output)
    except Exception as exc:
        result = {"status": "failed", "technical_passed": False,
                  "errors": [f"{type(exc).__name__}: {exc}"], "interpretation": "No valid inference; inspect error."}
    # A failed later audit must not erase an existing scientific result.
    stem = "r1_summary"
    if not result["technical_passed"] and (output/"r1_summary.json").exists():
        stem = "r1_reanalysis_failure"
    save_json(output/f"{stem}.json", result)
    markdown = summary_markdown(result)
    (output/f"{stem}.md").write_text(markdown)
    print(markdown, flush=True)
    return 0 if result["technical_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
