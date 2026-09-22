"""FS0 math and isolated-cell simulation. Nothing here is imported by H1--H4."""
from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import numpy as np

from .common import REPO


def exprel(x):
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-4
    out = np.empty_like(x)
    out[small] = 1 + x[small]/2 + x[small]**2/12
    out[~small] = x[~small] / (-np.expm1(-x[~small]))
    return out


def activation(v):
    a, b = 50*exprel(.1*(np.asarray(v)+20)), 20*exprel(-.08*(np.asarray(v)-10))
    return a/(a+b)


def inherited_relaxation_ms(v):
    """Source rate-law timescale, for checking the equilibrium approximation.

    This does not assert that these are physiological mammalian kinetics.
    """
    return 1/(50*exprel(.1*(np.asarray(v)+20)) + 20*exprel(-.08*(np.asarray(v)-10)))


def unit_slope(v, reversal=-75.):
    h = 1e-3
    return ((v+h-reversal)*activation(v+h)-(v-h-reversal)*activation(v-h))/(2*h)


def calibration(a):
    slope = float(unit_slope(a["reference_voltage_mV"], a["reversal_mV"]))
    ratio = a["l23_apical_gbar_S_per_cm2"] * slope / a["l23_gpas_S_per_cm2"]
    gbar = ratio*a["toy_gpas_S_per_cm2"]/slope
    return {"reference_gbar_S_per_cm2": gbar, "unit_slope_at_reference": slope,
        "tonic_to_leak_slope_ratio_at_reference": ratio, "reference_voltage_mV": a["reference_voltage_mV"],
        "normalization": "Slope-conductance / passive-leak ratio at fixed prestimulation voltage",
        "not_a_calibrated_human_conductance": True,
        "equilibrium_gate_approximation": True}


def harmonic(values, times_ms, frequency_hz):
    phase = 2*np.pi*frequency_hz*np.asarray(times_ms)/1000
    design = np.column_stack([np.ones(len(phase)), np.cos(phase), np.sin(phase)])
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    # Convention y(t)=Re[complex_amplitude * exp(i omega t)].
    return complex(beta[1], -beta[2])


def cases(a):
    result = []
    def add(condition, kind, frequency=10., amplitude=0., **kwargs):
        row = dict(condition=condition, kind=kind, frequency_hz=float(frequency),
            amplitude_v_per_m=float(amplitude), dt_factor=1., d_lambda_factor=1.,
            direction=[0., 0., 1.], holding_nA=0.)
        row.update(kwargs)
        row["id"] = f"case_{len(result):03d}_{condition}_{kind}"
        result.append(row)
    for c in ("legacy", "low", "reference"):
        add(c, "sham")
        for f, amplitude in itertools.product(a["frequencies_hz"], a["amplitudes_v_per_m"]):
            add(c, "field", f, amplitude)
        add(c, "synaptic_probe")
        add(c, "resistance", holding_nA=a["resistance_probe_nA"])
    if a["diagnostics"]:
        add("zero_inserted", "sham")  # exact disabled-path equivalence
        for c in ("legacy", "low", "reference"):
            add(c, "transverse", 10, .4, direction=[1., 0., 0.])
        for c in ("legacy", "reference"):
            add(c, "dt_refinement", 10, .4, dt_factor=.5)
            add(c, "space_refinement", 10, .4, d_lambda_factor=.5)
    return result


def compile_mechanisms(root):
    source = Path(__file__).parent / "mod"
    compiled = root / "compiled_mechanisms"
    compiled.mkdir()
    result = subprocess.run(["nrnivmodl", str(source)], cwd=compiled,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (root / "mechanism_build.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"FS0 NMODL build failed; see {root / 'mechanism_build.log'}")
    return compiled


def run_cell(case, cfg, cal, root):
    import neuron
    from LFPy import NetworkCell
    from env.models.neuron.extracellular_online import OnlineExtracellularController
    from env.models.neuron.stimulation import make_sinusoidal_electric_field

    h = neuron.h
    a = cfg.analysis
    baseline, stimulation, washout = float(a.baseline_ms), float(a.stimulation_ms), float(a.washout_ms)
    dt = float(cfg.env.network.dt)*case["dt_factor"]
    total = baseline+stimulation+washout
    h("forall delete_section()")
    circuit = REPO / "setup/circuits/ballnstick"
    cell = NetworkCell(morphology=str(circuit / "BallAndStick.hoc"),
        templatefile=str(circuit / "BallAndStickTemplate.hoc"), templatename="BallAndStickTemplate",
        templateargs=None, delete_sections=False, dt=dt, tstop=total,
        nsegs_method="lambda100", d_lambda=.1*case["d_lambda_factor"], verbose=False)
    sections = list(cell.allseclist)
    segments = [s for sec in sections for s in sec]
    positions = np.column_stack([np.asarray(getattr(cell, axis)).mean(axis=1) for axis in ("x", "y", "z")])
    scale = {"legacy": 0., "zero_inserted": 0., "low": float(a.tonic.low_fraction), "reference": 1.}[case["condition"]]
    apical = [sec for sec in sections if "apic" in sec.name()]
    for sec in sections:
        sec.insert("extracellular")
    for sec in apical:
        if not np.isclose(sec.g_pas, a.tonic.toy_gpas_S_per_cm2):
            raise ValueError("Configured tonic normalization does not match the unmodified toy leak")
        if case["condition"] != "legacy":
            sec.insert("bs_tonic_fs0")
            for seg in sec:
                seg.gbar_bs_tonic_fs0 = scale*cal["reference_gbar_S_per_cm2"]
                seg.e_bs_tonic_fs0 = float(a.tonic.reversal_mV)
    probe = h.bs_synprobe_fs0(apical[0](.5))
    hold = h.IClamp(sections[0](.5))
    hold.delay, hold.dur, hold.amp = baseline, stimulation, case["holding_nA"]
    n = int(round(total/dt))
    # Field values are prescribed at integration-step left endpoints, matching
    # the original fixed-step uniform-field implementation.
    field = np.zeros(n)
    wave = make_sinusoidal_electric_field(amplitude_v_per_m=case["amplitude_v_per_m"],
        frequency_hz=case["frequency_hz"], start_ms=baseline, duration_ms=stimulation,
        dt_ms=dt, phase_rad=0., ramp_ms=float(a.ramp_ms), include_endpoint=True)
    begin = int(round(baseline/dt))
    field[begin:begin+len(wave.field_v_per_m)-1] = wave.field_v_per_m[:-1]
    unit_potential = OnlineExtracellularController.uniform_field_potential_mV(
        midpoints_um=positions, field_v_per_m=np.asarray([1.]), field_direction=case["direction"]).reshape(-1)
    h.dt, h.celsius = dt, float(cfg.env.network.celsius)
    cvode = h.CVode()
    cvode.active(0)
    cvode.use_fast_imem(1)
    h.finitialize(float(cfg.env.network.v_init))
    h.fcurrent()
    area = np.array([seg.area() for seg in segments])
    tonic_indices = [i for i, seg in enumerate(segments) if hasattr(seg, "i_bs_tonic_fs0")]
    t = np.empty(n)
    trace = np.empty((n, 8))
    try:
        for step in range(n):
            for seg, potential in zip(segments, unit_potential):
                seg.e_extracellular = float(field[step]*potential)
            left_ms = step*dt
            active = baseline <= left_ms < baseline+stimulation
            probe.g = float(a.synaptic_probe_mean_uS)*(1+.5*np.sin(2*np.pi*case["frequency_hz"]*left_ms/1000)) if active and case["kind"] == "synaptic_probe" else 0.
            h.fadvance()
            t[step] = h.t
            imem = np.array([seg.i_membrane_ for seg in segments])  # nA, fast-imem total
            tonic = sum(segments[i].i_bs_tonic_fs0*area[i]*.01 for i in tonic_indices)
            trace[step] = [segments[0].v, segments[-1].v, imem @ positions[:, 2],
                tonic, probe.i, probe.g, np.sum(imem), max(abs(seg.e_extracellular) for seg in segments)]
        measured = (t > baseline+a.ramp_ms) & (t <= baseline+stimulation-a.ramp_ms)
        before = (t > baseline-500) & (t <= baseline)
        after = t > total-500
        voltages = trace[:, :2]
        response = [harmonic(trace[measured, c], t[measured], case["frequency_hz"]) for c in range(3)]
        input_phasor = harmonic(field[measured], t[measured], case["frequency_hz"])
        syn_phasor = harmonic(trace[measured, 5], t[measured], case["frequency_hz"])
        row = {**case, "gbar_S_per_cm2": scale*cal["reference_gbar_S_per_cm2"],
            "dt_ms": dt, "n_segments": len(segments), "celsius": float(h.celsius),
            "field_peak_v_per_m": float(np.max(np.abs(field))),
            "field_final_residual_mV": float(trace[-1, 7]),
            "finite": bool(np.all(np.isfinite(trace))),
            "subthreshold": bool(np.max(voltages) < -40),
            "maximum_abs_net_transmembrane_current_nA": float(np.max(np.abs(trace[:, 6]))),
            "mean_tonic_current_nA": float(np.mean(trace[before, 3])),
            "soma_rest_mV": float(np.mean(trace[before, 0])),
            "distal_rest_mV": float(np.mean(trace[before, 1])),
            "washout_soma_difference_mV": float(np.mean(trace[after, 0])-np.mean(trace[before, 0])),
            "washout_distal_difference_mV": float(np.mean(trace[after, 1])-np.mean(trace[before, 1])),
            "soma_input_resistance_Mohm": float((np.mean(trace[measured, 0])-np.mean(trace[before, 0])) / case["holding_nA"]) if case["kind"] == "resistance" else np.nan}
        for label, phasor in zip(("soma", "distal", "dipole"), response):
            row[f"{label}_response_real"] = phasor.real
            row[f"{label}_response_imag"] = phasor.imag
            row[f"{label}_response_amplitude"] = abs(phasor)
            transfer = phasor/input_phasor if abs(input_phasor)>1e-12 else complex(np.nan, np.nan)
            row[f"{label}_field_gain"] = abs(transfer)
            row[f"{label}_field_phase_rad"] = np.angle(transfer)
            row[f"{label}_field_transfer_real"] = transfer.real
            row[f"{label}_field_transfer_imag"] = transfer.imag
            row[f"{label}_synaptic_gain"] = abs(phasor/syn_phasor) if abs(syn_phasor)>1e-12 else np.nan
        stride = max(1, round(1/(float(a.record_fs_hz)*dt/1000)))
        np.savez_compressed(root / "traces" / f"{case['id']}.npz", t_ms=t[::stride],
            soma_mV=trace[::stride, 0], distal_mV=trace[::stride, 1], dipole_nA_um=trace[::stride, 2],
            tonic_current_nA=trace[::stride, 3], probe_current_nA=trace[::stride, 4],
            probe_conductance_uS=trace[::stride, 5], field_v_per_m=field[::stride])
        return row
    finally:
        for seg in segments:
            seg.e_extracellular = 0.
        cell.__del__()
