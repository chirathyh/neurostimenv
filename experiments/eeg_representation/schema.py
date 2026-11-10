"""Strict, versioned configuration; structural splits exist before any windows."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math

from . import SCHEMA_VERSION

REGIME_DEFAULTS = dict(name="R0", family="R0", frequency_hz=0.0,
    modulation_depth=0.0, diffusion=0.0, shared_fraction=1.0,
    kinetics_scale=1.0, burst=False)
DEFAULTS = {
    "schema_version": SCHEMA_VERSION, "name": "smoke", "claim_status": "smoke",
    "simulation": {"native_dt_ms": 0.0625, "burn_in_s": 2.0, "duration_s": 12.0,
        "model_sample_rate_hz": 500, "celsius": 6.3, "population_e": 32,
        "population_i": 8, "phase_grid_ms": 1.0, "save_native": True},
    "data": {"backend": "analytical", "analytical_kind": "oscillator",
        "structures": {"train": [31000], "development": [31001], "test": [31002]},
        "trials_per_regime": 2, "regimes": [], "namespace": "eegrepr-smoke-v1",
        "workers": 1},
    "analytical": {},
    "observation": {"snr_db": [20, 10, 0], "reference": "training_pool_channel_power",
        "variant": "clean", "gain": 1.0},
    "forecast": {"history_s": 1.0, "histories_s": [0.25, 1.0, 4.0],
        "lead_ms": 10.0, "leads_ms": [2, 10, 26, 50, 100, 250, 500, 1000],
        "block_ms": 128.0, "target_scales_ms": [32, 128, 512], "stride_s": 1.0,
        "target": "clean"},
    "forks": {"states_per_regime": 10, "futures": 64, "convergence_futures": 128,
        "convergence_states": 1, "decision_s": 4.0, "vary": "both",
        "bootstrap_samples": 1000},
    "model": {"architecture": "transformer", "embedding_dim": 128, "layers": 4,
        "heads": 4, "patch_ms": 64.0, "max_parameters": 2000000,
        "covariance_rank": 4},
    "training": {"recipes": ["features", "joint", "masked_mse"], "seeds": [0, 1],
        "steps": 1000, "batch_size": 16, "lr": 0.001, "source_tasks": ["rhythm", "carrier"],
        "heldout_tasks": ["baseline_alpha", "baseline_beta_low", "baseline_beta_high"],
        "specialist_task": "rhythm", "mask_fraction": 0.5, "mask_gap_ms": 128.0,
        "label_budgets": [10, 50, 250, 500], "replay_capacity": "full_source_pool",
        "random_orders": 3, "source_order": ["rhythm", "carrier"],
        "order_selected_on": "prespecified_candidate", "auxiliary_weight": 1.0,
        "threads": 1, "mixed_precision": False, "weight_decay": 0.01, "replay_fraction": 0.25, "checkpoint_every": 50, "teacher_momentum": 0.99, "pretraining_budgets": [None]},
    "evaluation": {"tasks": ["baseline_alpha", "baseline_beta_low", "baseline_beta_high"],
        "heldout_families": ["Rbeta_high"], "probe_steps": 100,
        "probe_hidden": 32, "bootstrap_samples": 2000, "minimum_gain": 0.02,
        "frozen": False, "allow_test": False, "probe_types": ["linear", "nonlinear"],
        "variants": ["clean"], "gains": [1.0], "adaptation_steps": 0, "linear_grid": [0.1, 1, 10], "nonlinear_grid": [0.001, 0.01, 0.1], "shuffle_diagnostic": True, "forecast_draws": 64, "batch_size": 32},
}


def _merge(base, update, path=""):
    out = deepcopy(base)
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration field: {path}{key}")
        if isinstance(base[key], dict) and base[key]:
            if not isinstance(value, dict):
                raise ValueError(f"{path}{key} must be a mapping")
            out[key] = _merge(base[key], value, f"{path}{key}.")
        else:
            out[key] = deepcopy(value)
    return out


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated resolved schema. as_dict is the adapter interoperability boundary."""
    values: dict

    @classmethod
    def from_dict(cls, raw: dict) -> "ExperimentConfig":
        c = _merge(DEFAULTS, raw)
        if c["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported schema version")
        if c["claim_status"] not in {"smoke", "pilot", "confirmation"}:
            raise ValueError("Unknown claim status")
        if c["data"]["backend"] not in {"analytical", "neuron"}:
            raise ValueError("backend must be analytical or neuron")
        if c["data"]["analytical_kind"] not in {"oscillator", "poisson", "variance"}:
            raise ValueError("Unknown analytical control")
        s = c["simulation"]
        if not s["save_native"]:
            raise ValueError("Version 1 requires native trace preservation; save_native=False is unsupported")
        if c["claim_status"] != "smoke" and (s["population_e"], s["population_i"]) != (32, 8):
            raise ValueError("Reduced populations are functional smoke settings only")
        if any(not isinstance(s[k], int) or s[k] < 1 for k in ["population_e", "population_i"]):
            raise ValueError("Population sizes must be positive integers")
        for key in ["native_dt_ms", "duration_s", "model_sample_rate_hz"]:
            if not isinstance(s[key], (int, float)) or not math.isfinite(s[key]) or s[key] <= 0:
                raise ValueError(f"simulation.{key} must be positive and finite")
        if s["burn_in_s"] < 0 or s["celsius"] != 6.3:
            raise ValueError("Nonnegative burn-in and canonical HH 6.3 C required")
        ratio = 1000 / s["native_dt_ms"] / s["model_sample_rate_hz"]
        if abs(ratio - round(ratio)) > 1e-9 or ratio < 1:
            raise ValueError("Native rate must be an integer multiple of model rate")
        if s["model_sample_rate_hz"] != 500:
            raise ValueError("Version 1 model stream is frozen at 500 Hz")
        for key in ["duration_s", "burn_in_s"]:
            if abs(s[key] * 1000 / s["native_dt_ms"] - round(s[key] * 1000 / s["native_dt_ms"])) > 1e-7:
                raise ValueError(f"{key} must align with native grid")
        groups = c["data"]["structures"]
        all_seeds = [seed for ids in groups.values() for seed in ids]
        if any(not isinstance(seed, int) or isinstance(seed, bool) or not 1 <= seed <= 429495 for seed in all_seeds):
            raise ValueError("Structure seeds must be integers in 1..429495 (LFPy seed*10000 bound)")
        if len(set(all_seeds)) != len(all_seeds) or any(not ids for ids in groups.values()):
            raise ValueError("Nonempty, disjoint structural splits required")
        if c["data"]["trials_per_regime"] < 1 or c["data"]["workers"] < 1:
            raise ValueError("Positive trial count and workers required")
        regimes = []
        for raw_regime in c["data"]["regimes"]:
            r = _merge(REGIME_DEFAULTS, raw_regime, "regime.")
            import re
            if not re.fullmatch(r"[A-Za-z0-9_-]+", r["name"]):
                raise ValueError("Regime names must be safe alphanumeric shard identifiers")
            if "family" not in raw_regime:
                r["family"] = r["name"]
            if not 0 <= r["modulation_depth"] <= 1 or r["diffusion"] < 0 or not 0 <= r["shared_fraction"] <= 1:
                raise ValueError("Invalid modulation, diffusion or shared fraction")
            if r["kinetics_scale"] <= 0 or r["frequency_hz"] < 0:
                raise ValueError("Invalid kinetics/carrier")
            if r["modulation_depth"] == 0 and (r["diffusion"] != 0 or r["burst"]):
                raise ValueError("Do not duplicate homogeneous trials under meaningless diffusion/burst labels")
            if r["modulation_depth"] > 0 and r["frequency_hz"] <= 0:
                raise ValueError("Rhythmic regimes require a positive carrier")
            regimes.append(r)
        if not regimes or len({r["name"] for r in regimes}) != len(regimes):
            raise ValueError("At least one uniquely named regime is required")
        c["data"]["regimes"] = regimes
        f = c["forecast"]
        for ms in [f["lead_ms"], f["block_ms"], *f["leads_ms"], *f["target_scales_ms"]]:
            n = ms * s["model_sample_rate_hz"] / 1000
            if n <= 0 or abs(n - round(n)) > 1e-8:
                raise ValueError(f"Target/lead {ms} ms does not align with 500-Hz grid; use explicit aligned values")
        if f["history_s"] + (f["lead_ms"] + f["block_ms"]) / 1000 > s["duration_s"]:
            raise ValueError("History and future block exceed retained duration")
        if f["target"] not in {"clean", "noisy"}:
            raise ValueError("forecast.target must explicitly select clean or noisy recordings")
        if c["observation"]["reference"] != "training_pool_channel_power":
            raise ValueError("Primary noise requires label-blind training-pool power")
        if c["observation"]["gain"] <= 0 or not math.isfinite(c["observation"]["gain"]):
            raise ValueError("Gain must be finite and positive")
        t = c["training"]
        if min(t["steps"], t["batch_size"], t["threads"]) < 1:
            raise ValueError("Training counts must be positive")
        if not isinstance(t["random_orders"], int) or t["random_orders"] < 1:
            raise ValueError("random_orders must be a positive integer")
        if t["replay_capacity"] != "full_source_pool":
            raise ValueError("This exposure-matched protocol retains the full source pool for replay")
        if not 0 < t["mask_fraction"] < 1:
            raise ValueError("Mask fraction must lie strictly between 0 and 1")
        if sorted(t["source_order"]) != sorted(t["source_tasks"]):
            raise ValueError("source_order must contain each source task once")
        if set(t["source_tasks"]) & set(t["heldout_tasks"]):
            raise ValueError("Downstream held-out labels cannot be source labels")
        if t["mixed_precision"]:
            raise ValueError("Mixed precision is disabled until workstation numerical validation")
        if any(not isinstance(b, int) or b < 1 for b in t["label_budgets"]) or not t["label_budgets"]:
            raise ValueError("Positive independent-trial label budgets required")
        if c["evaluation"]["allow_test"] and c["claim_status"] == "confirmation" and not c["evaluation"]["frozen"]:
            raise ValueError("Confirmation test access requires a frozen protocol")
        if c["forks"]["vary"] not in {"events", "phase", "both"} or c["forks"]["futures"] < 2:
            raise ValueError("Forks require >=2 futures and an explicit innovation source")
        grid = s["phase_grid_ms"]
        if abs((s["burn_in_s"] + c["forks"]["decision_s"]) * 1000 / grid - round((s["burn_in_s"] + c["forks"]["decision_s"]) * 1000 / grid)) > 1e-8:
            raise ValueError("Decision time must align with phase grid")
        return cls(c)

    def as_dict(self):
        return deepcopy(self.values)

    @property
    def hash(self):
        return config_hash(self.values)


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config(path):
    import yaml
    raw = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig.from_dict(raw).as_dict()
