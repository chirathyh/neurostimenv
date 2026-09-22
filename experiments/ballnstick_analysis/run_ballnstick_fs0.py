"""FS0: isolated-cell qualification of an opt-in tonic-conductance mechanism."""
from __future__ import annotations

import sys
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neuron
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.qualification import common, field_sensitivity as fs


def validate(cfg):
    a = cfg.analysis
    if cfg.env.name != "ballnstick":
        raise ValueError("FS0 is BallAndStick only; L23Net is read-only calibration")
    if cfg.env.network.celsius != 6.3 or cfg.env.network.dt <= 0:
        raise ValueError("Canonical HH requires 6.3 C and positive dt")
    if a.baseline_ms < 1000 or a.washout_ms < 500 or a.stimulation_ms-2*a.ramp_ms < 500:
        raise ValueError("Insufficient settling, measurement, or washout duration")
    if a.tonic.low_fraction != .6 or a.tonic.reversal_mV != -75:
        raise ValueError("Keep the predeclared tonic sensitivity perturbation")
    if not a.smoke and (list(a.frequencies_hz) != [8., 10., 12.] or
            list(a.amplitudes_v_per_m) != [.1, .4] or not a.diagnostics):
        raise ValueError("Full FS0 requires the complete frequency/dose grid and diagnostics")
    if not a.smoke and (cfg.env.network.dt != .0625 or
            [a.baseline_ms, a.stimulation_ms, a.washout_ms, a.ramp_ms] != [2000., 3000., 1000., 500.]):
        raise ValueError("Full FS0 requires the frozen 2+3+1-s protocol, ramps and base dt")
    if max(a.amplitudes_v_per_m) > .4 or min(a.amplitudes_v_per_m) <= 0:
        raise ValueError("FS0 active fields must be in (0, 0.4] V/m")
    # Audit the actual source rather than silently relying on a copied number.
    source = common.REPO / "setup/circuits/L23Net/Circuit_param.xls"
    params = pd.read_excel(source, sheet_name="SING_CELL_PARAM", index_col=0)
    if not np.isclose(float(params.loc["apic_tonic", "HL23PYR"]), a.tonic.l23_apical_gbar_S_per_cm2, rtol=0, atol=1e-12):
        raise ValueError("L23Net tonic parameter changed; revisit the prespecified calibration")
    biophysics = (common.REPO / "setup/circuits/L23Net/models/biophys_HL23PYR.hoc").read_text()
    leak = float(re.search(r"g_pas\s*=\s*([0-9.eE+-]+)", biophysics).group(1))
    if not np.isclose(leak, a.tonic.l23_gpas_S_per_cm2, rtol=0, atol=1e-12):
        raise ValueError("L23Net leak normalization changed")
    return source


def infer(frame, cfg, root):
    fields = frame[frame.kind == "field"]
    checks = {"finite_traces": bool(frame.finite.all()),
        "nonzero_axial_field_response": bool((fields.soma_field_gain > 1e-5).all() and
                                             (fields.distal_field_gain > 1e-5).all()),
        "all_responses_subthreshold": bool(frame.subthreshold.all()),
        "canonical_temperature_preserved": bool((frame.celsius == 6.3).all()),
        "exact_field_removal": bool((frame.field_final_residual_mV == 0).all()),
        "physiological_washout": bool((frame.washout_soma_difference_mV.abs() <= cfg.analysis.criteria.maximum_washout_difference_mV).all() and
                                      (frame.washout_distal_difference_mV.abs() <= cfg.analysis.criteria.maximum_washout_difference_mV).all())}
    numerical = []
    for row in frame[frame.kind.isin(["dt_refinement", "space_refinement"])].itertuples():
        original = fields[(fields.condition == row.condition) & (fields.frequency_hz == row.frequency_hz) &
                          (fields.amplitude_v_per_m == row.amplitude_v_per_m)]
        if len(original) != 1:
            continue  # smoke can deliberately omit the matched field case
        base = original.iloc[0]
        for site in ("soma", "distal"):
            x = complex(base[f"{site}_field_transfer_real"], base[f"{site}_field_transfer_imag"])
            y = complex(getattr(row, f"{site}_field_transfer_real"), getattr(row, f"{site}_field_transfer_imag"))
            numerical.append({"condition": row.condition, "audit": row.kind, "site": site,
                "relative_complex_error": abs(x-y)/max(abs(y), 1e-12)})
    pd.DataFrame(numerical).to_csv(root / "numerical_convergence.csv", index=False)
    checks["time_and_space_convergence"] = len(numerical) == 8 and all(r["relative_complex_error"] <=
        cfg.analysis.criteria.maximum_numerical_relative_complex_error for r in numerical)
    zero = frame[frame.condition == "zero_inserted"]
    if len(zero):
        baseline = frame[(frame.condition == "legacy") & (frame.kind == "sham")].iloc[0]
        p = np.load(root / "traces" / f"{baseline.id}.npz")
        q = np.load(root / "traces" / f"{zero.iloc[0].id}.npz")
        zero_error = max(np.max(np.abs(p[k]-q[k])) for k in ("soma_mV", "distal_mV", "dipole_nA_um"))
    else:
        zero_error = np.nan
    checks["zero_conductance_preserves_legacy_cell"] = bool(zero_error <= 1e-10)
    transverse = frame[frame.kind == "transverse"]
    transverse_ratios = []
    for row in transverse.itertuples():
        axial = fields[(fields.condition == row.condition) & (fields.frequency_hz == row.frequency_hz) &
                       (fields.amplitude_v_per_m == row.amplitude_v_per_m)]
        if len(axial) == 1:
            for site in ("soma", "distal"):
                transverse_ratios.append(getattr(row, f"{site}_response_amplitude") /
                    max(float(axial.iloc[0][f"{site}_response_amplitude"]), 1e-12))
    checks["transverse_field_null"] = len(transverse_ratios) == 6 and max(transverse_ratios) <= cfg.analysis.criteria.maximum_transverse_gain_fraction
    contrasts = []
    for (frequency, amplitude), group in fields.groupby(["frequency_hz", "amplitude_v_per_m"]):
        low = group[group.condition == "low"].iloc[0]
        reference = group[group.condition == "reference"].iloc[0]
        contrasts.append({"frequency_hz": frequency, "amplitude_v_per_m": amplitude,
            **{f"{site}_low_over_reference_gain": low[f"{site}_field_gain"]/reference[f"{site}_field_gain"] for site in ("soma", "distal")}})
    pd.DataFrame(contrasts).to_csv(root / "field_sensitivity_contrasts.csv", index=False)
    gains = [abs(r[f"{site}_low_over_reference_gain"]-1) for r in contrasts for site in ("soma", "distal")]
    sensitivity = bool(gains and max(gains) >= cfg.analysis.criteria.minimum_relative_gain_difference)
    comparison = []
    for condition in ("legacy", "low", "reference"):
        probe = frame[(frame.condition == condition) & (frame.kind == "synaptic_probe")].iloc[0]
        for row in fields[(fields.condition == condition) & (fields.frequency_hz == 10)].itertuples():
            comparison.append({"condition": condition, "amplitude_v_per_m": row.amplitude_v_per_m,
                "field_soma_gain_mV_per_V_per_m": row.soma_field_gain,
                "conductance_soma_gain_mV_per_uS": probe.soma_synaptic_gain,
                "field_to_synaptic_transfer_ratio": row.soma_field_gain/probe.soma_synaptic_gain})
    pd.DataFrame(comparison).to_csv(root / "field_vs_synaptic_transfer.csv", index=False)
    # No p-values: this is a deterministic cell/numerical assay, not n=35 subjects.
    return {"integrity_checks": checks, "zero_inserted_max_trace_error": zero_error,
        "maximum_transverse_to_axial_response_ratio": max(transverse_ratios) if transverse_ratios else None,
        "prespecified_5_percent_field_sensitivity_observed": sensitivity,
        "minimum_relative_gain_change": min(gains), "maximum_relative_gain_change": max(gains),
        "qualified_for_network_pilot": bool(all(checks.values()) and sensitivity and not cfg.analysis.smoke),
        "statistical_unit": "No population inference: deterministic isolated-cell transfer functions",
        "H5": "NOT TESTED", "preferred_dose_crossover": "NOT TESTED"}


def plots(frame, cfg, cal, root):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    v = np.linspace(-100, 40, 701)
    for label, factor in (("legacy", 0.), ("low", .6), ("reference", 1.)):
        current = factor*cal["reference_gbar_S_per_cm2"]*fs.activation(v)*(v-cfg.analysis.tonic.reversal_mV)
        axes[0, 0].plot(v, current, label=label)
        f = frame[(frame.kind == "field") & (frame.condition == label)]
        for dose, group in f.groupby("amplitude_v_per_m"):
            axes[0, 1].plot(group.frequency_hz, group.soma_field_gain, "o-", label=f"{label}, {dose:g} V/m")
        sham = frame[(frame.condition == label) & (frame.kind == "sham")].iloc[0]
        axes[1, 0].scatter([sham.soma_rest_mV], [sham.distal_rest_mV], label=label)
    example = frame[(frame.condition == "reference") & (frame.kind == "field")].iloc[-1]
    trace = np.load(root / "traces" / f"{example.id}.npz")
    mask = (trace["t_ms"] >= cfg.analysis.baseline_ms) & (trace["t_ms"] <= cfg.analysis.baseline_ms+1000)
    axes[1, 1].plot(trace["t_ms"][mask]/1000, trace["distal_mV"][mask], label="Distal V")
    axes[0, 0].set(xlabel="Membrane voltage (mV)", ylabel="Tonic current (mA/cm²)", title="Reduced equilibrium I–V relation")
    axes[0, 1].set(xlabel="Frequency (Hz)", ylabel="Somatic gain (mV per V/m)", title="Field response, not clinical dose")
    axes[1, 0].set(xlabel="Resting soma (mV)", ylabel="Resting distal cable (mV)", title="Operating-point changes are retained")
    axes[1, 1].set(xlabel="Time (s)", ylabel="Distal voltage (mV)", title=f"{example.frequency_hz:g}-Hz field onset")
    for ax in axes.flat:
        ax.legend(fontsize=7)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(root / f"cellular_sensitivity.{extension}", dpi=160)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    started = time.perf_counter()
    comm = MPI.COMM_WORLD
    source = validate(cfg)
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "fs0"
    cal = fs.calibration(OmegaConf.to_container(cfg.analysis.tonic))
    cases = fs.cases(OmegaConf.to_container(cfg.analysis))
    if comm.rank == 0:
        common.begin(root, cfg, __file__)
        (root / "traces").mkdir()
        common.write_json(root / "prespecified_cases.json", cases)
        common.write_json(root / "tonic_calibration.json", {**cal,
            "source_xls_sha256": common.sha256(source),
            "source_tonic_mod_sha256": common.sha256(common.REPO / "setup/circuits/L23Net/mod/tonic.mod"),
            "source_biophysics_sha256": common.sha256(common.REPO / "setup/circuits/L23Net/models/biophys_HL23PYR.hoc")})
        compiled = str(fs.compile_mechanisms(root))
    else:
        compiled = None
    compiled = comm.bcast(compiled, root=0)
    if not neuron.load_mechanisms(compiled):
        raise RuntimeError("Could not load FS0-only mechanisms")
    voltages = np.unique(np.r_[np.linspace(-100, 60, 401), -20., 10., -65.])
    actual = np.array([neuron.h.activation_bs_tonic_fs0(float(v)) for v in voltages])
    if not np.allclose(actual, fs.activation(voltages), atol=1e-12, rtol=1e-12):
        raise RuntimeError("Compiled tonic I--V differs from the audited analytic relation")
    if comm.rank == 0:
        pd.DataFrame({"v_mV": voltages, "open_fraction": actual,
            "inherited_relaxation_ms": fs.inherited_relaxation_ms(voltages),
            "reference_current_mA_per_cm2": cal["reference_gbar_S_per_cm2"]*actual*(voltages+75)
        }).to_csv(root / "tonic_IV_and_timescale_audit.csv", index=False)
    rows = []
    for index, case in enumerate(cases):
        if index % comm.size != comm.rank:
            continue
        print(f"FS0 rank {comm.rank}: {case['id']}, f={case['frequency_hz']:g}, E={case['amplitude_v_per_m']:g}", flush=True)
        row = fs.run_cell(case, cfg, cal, root)
        rows.append(row)
        common.write_json(root / "traces" / f"{case['id']}_metrics.json", row)
    gathered = comm.gather(rows, root=0)
    if comm.rank == 0:
        frame = pd.DataFrame([row for part in gathered for row in part]).sort_values("id")
        if len(frame) != len(cases) or frame.id.nunique() != len(cases):
            raise RuntimeError("Missing or duplicate FS0 cases")
        frame.to_csv(root / "cell_metrics.csv", index=False)
        summary = infer(frame, cfg, root)
        if cfg.experiment.plot:
            plots(frame, cfg, cal, root)
        common.finish(root, started, {"experiment": "FS0", "smoke": bool(cfg.analysis.smoke),
            "cell_simulations": len(cases), "mpi_ranks": comm.size, "calibration": cal, **summary})
    comm.barrier()


if __name__ == "__main__":
    common.guarded_main(main)
