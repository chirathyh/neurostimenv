"""Analyze a complete transferred S1 suite; does not submit simulations."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.l23net_analysis.s1_analysis import analyze_suite,latest_suite

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',type=Path,nargs='?')
    parser.add_argument('--latest',choices=['qualification','discovery'])
    args=parser.parse_args()
    if (args.suite is None)==(args.latest is None):parser.error('Provide a suite path OR --latest STAGE')
    directory=args.suite if args.suite is not None else latest_suite(args.latest)
    print(f'ANALYZING_SUITE={directory}',flush=True)
    result=analyze_suite(directory)
    print(json.dumps(result,indent=2))
    raise SystemExit(1 if result['errors'] else 0)
