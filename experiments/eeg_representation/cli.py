"""Stable command line for an opt-in, resumable EEG representation study."""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from .schema import load_config, config_hash
from .storage import atomic_json, jsonable

ROOT = Path(__file__).resolve().parents[2]


def doctor():
    packages = {}
    for name in ["numpy", "scipy", "h5py", "PyYAML", "torch", "scikit-learn", "NEURON", "LFPy", "mpi4py", "hydra-core"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = dict(python=sys.version, executable=sys.executable, packages=packages,
        logical_cpus=os.cpu_count(), cuda_available=False,
        simulator_imported=False, analytical_available=all(packages[x] for x in ["numpy", "scipy", "h5py", "PyYAML"]),
        simulator_packages_available=all(packages[x] for x in ["NEURON", "LFPy", "mpi4py", "hydra-core"]),
        note="Package presence is not simulator/compilation or workstation CUDA validation")
    if packages["torch"]:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_build"] = torch.version.cuda
        result["gpu_devices"] = [dict(name=torch.cuda.get_device_name(i), total_memory=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]
    return result


def plan(config, output_root=None):
    from .data import trial_specs
    specs = trial_specs(config)
    s = config["simulation"]
    samples = round((s["burn_in_s"] + s["duration_s"]) * 1000 / s["native_dt_ms"])
    native = len(specs) * samples * 5 * 8
    model = len(specs) * round(s["duration_s"] * 500) * (3 + len(config["observation"]["snr_db"])) * 8 * 2
    counts = {k: len(v) for k, v in config["data"]["structures"].items()}
    r = dict(name=config["name"], claim_status=config["claim_status"], config_hash=config_hash(config),
        backend=config["data"]["backend"], trials=len(specs), structure_counts=counts,
        population={"E": s["population_e"], "I": s["population_i"]},
        native_samples_per_trial=samples, native_uncompressed_signal_bytes=native,
        derived_uncompressed_signal_bytes=model,
        storage_note="Array arithmetic only; excludes events, full decision states, HDF5 overhead and checkpoints; compression unmeasured",
        cpu_time="Pending an actual matching-profile benchmark; no full campaign estimate invented",
        workers=config["data"]["workers"], training_device="chosen explicitly at fit time",
        recipe_seed_jobs=(len(config["training"]["recipes"]) + (min(config["training"]["random_orders"], math.factorial(len(config["training"]["source_tasks"]))) - 1 if "random_order" in config["training"]["recipes"] else 0)) * len(config["training"]["seeds"]) * len(config["training"]["pretraining_budgets"]),
        fork_episodes=len(config["data"]["regimes"]) * (config["forks"]["states_per_regime"] * config["forks"]["futures"] + min(config["forks"]["convergence_states"], config["forks"]["states_per_regime"]) * max(0, config["forks"]["convergence_futures"] - config["forks"]["futures"])),
        forecast_leads_ms=config["forecast"]["leads_ms"],
        grid_note="25 ms is not on a 500-Hz grid: primary config explicitly uses 26 ms",
        label_budget_caps={k: counts["train"] * config["data"]["trials_per_regime"] for k in [x["name"] for x in config["data"]["regimes"]]},
        final_test_locked=config["claim_status"] == "confirmation" and not config["evaluation"]["allow_test"],
        source_order_selection=config["training"]["order_selected_on"])
    if output_root:
        measured = Path(output_root) / "generation_summary.json"
        if measured.exists():
            r["measured_generation"] = json.loads(measured.read_text())
    return r


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect dependencies/resources without importing the simulator")
    sub.add_parser("audit-base", help="Print the source-audited base and current git state")
    v = sub.add_parser("validate", help="Run independent controls and meaningful unit tests")
    v.add_argument("--profile", choices=["smoke", "simulator"], default="smoke")
    v.add_argument("--output-root", type=Path, required=True)
    for cmd in ["plan", "generate", "derive", "forks", "fit", "fit-suite", "evaluate", "summarize"]:
        q = sub.add_parser(cmd)
        q.add_argument("--config", type=Path, required=True)
        q.add_argument("--output-root", type=Path, required=True)
        if cmd != "plan":
            q.add_argument("--resume", action="store_true")
        if cmd == "generate":
            q.add_argument("--workers", type=int)
        if cmd in {"fit", "fit-suite", "evaluate"}:
            q.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
            q.add_argument("--protocol", default="leave-task-label-out", choices=["leave-task-label-out", "leave-generator-family-out"])
            if cmd == "fit":
                q.add_argument("--recipe", required=True)
                q.add_argument("--seed", type=int, default=0)
                q.add_argument("--budget", type=int)
            else:
                q.add_argument("--recipes", nargs="+")
                q.add_argument("--seeds", nargs="+", type=int)
                q.add_argument("--budgets", nargs="+", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        result = doctor()
    elif args.command == "audit-base":
        result = dict(audited_base="research/ballnstick-h1-h4@855a1c33623573aeae71682dc7740658bdeac401",
            current_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            audit=(ROOT / "docs/eeg_representation/BASELINE_AUDIT.md").read_text())
    elif args.command == "validate":
        from .analytical import run_controls
        result = run_controls(args.output_root)
        tests = ["tests.test_eegrepr_analytical", "tests.test_eegrepr_aggregation", "tests.test_eegrepr_data", "tests.test_eegrepr_learning"]
        if args.profile == "simulator":
            tests += ["tests.test_eegrepr_simulator", "tests.test_ballnstick_background", "tests.test_ballnstick_synapse_kinetics", "tests.test_online_stimulation"]
        env = dict(os.environ, MAIN_PATH=str(ROOT), EEGREPR_RUN_NEURON="1" if args.profile == "simulator" else "0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        proc = subprocess.run([sys.executable, "-m", "unittest", "-v", *tests], cwd=ROOT, env=env)
        result["unit_tests_returncode"] = proc.returncode
        atomic_json(args.output_root / "validation.json", result)
        if proc.returncode:
            print(json.dumps(jsonable(result), indent=2))
            return proc.returncode
    else:
        config = load_config(args.config)
        from .tasks import validate_tasks
        validate_tasks(config)
        if args.command == "plan":
            result = plan(config, args.output_root)
        elif args.command == "generate":
            from .data import generate
            result = generate(config, args.output_root, args.resume, args.workers)
        elif args.command == "derive":
            from .data import derive
            result = derive(config, args.output_root, args.resume)
        elif args.command == "forks":
            from .forks import run_forks
            result = run_forks(config, args.output_root, args.resume)
        elif args.command in {"fit", "fit-suite", "evaluate"}:
            from .storage import require_compatible, output_lock
            require_compatible(args.output_root, config)
            with output_lock(args.output_root):
                if args.command == "fit":
                    from .training import fit
                    result = fit(config, args.output_root, args.recipe, args.seed, args.device, args.budget, args.protocol, args.resume)
                elif args.command == "fit-suite":
                    from .training import fit_suite
                    result = fit_suite(config, args.output_root, args.recipes, args.seeds, args.budgets, args.device, args.protocol, args.resume)
                else:
                    from .evaluation import evaluate
                    result = evaluate(config, args.output_root, args.protocol, args.device, args.recipes, args.seeds, args.budgets, args.resume)
                    result = {k: v for k, v in result.items() if k not in {"rows", "pending"}} | {"rows": len(result["rows"]), "pending": len(result["pending"])}
        else:
            from .aggregation import summarize
            from .storage import require_compatible, output_lock
            require_compatible(args.output_root, config)
            with output_lock(args.output_root):
                result = summarize(config, args.output_root)
    print(json.dumps(jsonable(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
