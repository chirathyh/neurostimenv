"""Opt-in zero-field adapter around the frozen COSYNE persistent simulator.

NEURON/LFPy/MPI imports are local. This module never imports legacy folder
helpers and never modifies legacy source. Each worker must be a process;
NEURON and its global module hooks are not safe for concurrent threads.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import time
from typing import Any
from unittest.mock import patch
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_LOCK = threading.RLock()


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _seed(seed, namespace):
    return int(np.random.SeedSequence([int(seed), int(namespace)]).generate_state(1)[0])


def burst_envelope(*, stop_ms, seed, depth, on_mean_ms=500., off_mean_ms=1000.,
                   low_depth=0., future_start_ms=None, future_seed=None):
    """Stationary two-state CTMC amplitude envelope with exponential dwells.

    This is a RATE-envelope extension, independent of private event streams.
    Values are right-continuous. A fork preserves the current on/off state and
    resamples its exponential residual dwell, which is valid by memorylessness.
    """
    if not 0 <= low_depth <= depth <= 1 or min(on_mean_ms, off_mean_ms, stop_ms) <= 0:
        raise ValueError("Invalid bounded burst depths/dwell times.")
    if (future_start_ms is None) != (future_seed is None):
        raise ValueError("Burst split and seed must be supplied together.")
    rng = np.random.default_rng(_seed(seed, 0x42555253))
    active = bool(rng.random() < on_mean_ms/(on_mean_ms+off_mean_ms))
    times, values = [0.], [depth if active else low_depth]
    boundary = stop_ms if future_start_ms is None else float(future_start_ms)
    now = 0.
    while True:
        now += float(rng.exponential(on_mean_ms if active else off_mean_ms))
        if now >= boundary: break
        active = not active; times.append(now); values.append(depth if active else low_depth)
    if future_start_ms is not None:
        if not 0 < boundary < stop_ms: raise ValueError("Burst fork must be interior.")
        times.append(boundary); values.append(depth if active else low_depth)
        rng = np.random.default_rng(_seed(future_seed, 0x42555253)); now = boundary
        while True:
            now += float(rng.exponential(on_mean_ms if active else off_mean_ms))
            if now >= stop_ms: break
            active = not active; times.append(now); values.append(depth if active else low_depth)
    times.append(float(stop_ms)); values.append(values[-1])
    return np.asarray(times,dtype=np.float64), np.asarray(values,dtype=np.float64)


def _burst_events(*, start_ms, stop_ms, interval_ms, seed, modulation_depth,
                  frequency_hz, phase_rad, envelope_times, envelope_depths, envelope_max_depth,
                  phase_path_times_ms=None, phase_path_rad=None, **ignored):
    rng = np.random.default_rng(int(seed))
    maximum = float(envelope_max_depth)
    if not 0 <= maximum <= 1 or np.any(envelope_depths > maximum):
        raise ValueError("The configured thinning bound must dominate every envelope depth.")
    now, accepted = float(start_ms), []
    while True:
        now += float(rng.exponential(float(interval_ms)/(1+maximum)))
        if now >= stop_ms: break
        phase = (2*np.pi*frequency_hz*now/1000+phase_rad if phase_path_rad is None
                 else np.interp(now,phase_path_times_ms,phase_path_rad))
        index = np.searchsorted(envelope_times,now,side="right")-1
        depth = float(envelope_depths[index]) if modulation_depth > 0 else 0.
        if rng.random() < (1+depth*np.sin(phase))/(1+maximum): accepted.append(now)
    return np.asarray(accepted,dtype=np.float64)


def _validate(config, spec):
    sim = config.get("simulation", {})
    dt = float(sim.get("native_dt_ms", .0625))
    total = 1000*(float(sim.get("burn_in_s",2))+float(sim.get("duration_s",12)))
    phase_dt = float(sim.get("phase_grid_ms",sim.get("phase_grid_dt_ms",1.)))
    if float(sim.get("celsius",6.3)) != 6.3: raise ValueError("Canonical HH temperature is frozen at 6.3 C.")
    if dt != .0625: raise ValueError("The primary adapter freezes native dt at 0.0625 ms.")
    if not 0 <= int(spec["structure_seed"]) <= 429495:
        raise ValueError("LFPy structure_seed*10000 must fit uint32 (0..429495; conservative serial bound).")
    if total <= 0 or not np.isclose(total/dt,round(total/dt)):
        raise ValueError("Duration must contain a positive integer native sample count.")
    if phase_dt <= 0 or not np.isclose(total/phase_dt,round(total/phase_dt)):
        raise ValueError("Episode must end on the phase grid.")
    split = spec.get("future_start_ms")
    if split is not None:
        for grid in (dt, phase_dt):
            if not 0 < split < total or not np.isclose(split/grid,round(split/grid),rtol=0,atol=1e-9):
                raise ValueError("Fork cutoff must be interior and aligned to native AND phase grids.")
        if spec.get("future_seed") is None or spec.get("future_phase_seed") is None:
            raise ValueError("Forks require separately named future event and phase seeds.")
    regime = spec["regime"]
    depth = float(regime.get("modulation_depth",0.))
    diffusion = float(regime.get("diffusion",0.))
    if not 0 <= depth <= 1 or diffusion < 0: raise ValueError("Invalid modulation/depth.")
    if depth == 0 and diffusion != 0: raise ValueError("Homogeneous baseline has no diffusion label.")
    if depth > 0 and float(regime.get("frequency_hz",0)) <= 0: raise ValueError("Rhythm needs carrier.")
    return sim, dt, total, phase_dt


@contextmanager
def _construction_adapter(utils, spec, total, phase_dt, record):
    """Scope additive generator behavior to this one network construction."""
    original_phase = utils.generate_phase_diffusion_path
    original_events = utils.generate_split_background_spike_train
    regime = spec["regime"]
    burst = regime.get("burst",False)
    burst = {} if burst is True else burst
    envelope = None
    if burst is not False and burst is not None:
        envelope = burst_envelope(stop_ms=total, seed=spec.get("envelope_seed",spec.get("burst_seed",spec["phase_seed"])),
            depth=float(regime["modulation_depth"]),
            on_mean_ms=float(burst.get("on_mean_ms",500)),
            off_mean_ms=float(burst.get("off_mean_ms",1000)),
            low_depth=float(burst.get("low_depth",0)),
            future_start_ms=spec.get("future_start_ms"),
            future_seed=spec.get("future_burst_seed",spec.get("future_seed")))
        record["burst_times_ms"],record["burst_depth"] = envelope
    phase_cache = []
    def phase(**kwargs):
        kwargs["history_seed"] = _seed(spec["phase_seed"],0x50484446)
        if kwargs.get("future_seed") is not None:
            kwargs["future_seed"] = _seed(spec["future_phase_seed"],0x50484446)
        result = original_phase(**kwargs)
        phase_cache.append(result)
        return result
    def events(**kwargs):
        if envelope is None:
            result = original_events(**kwargs)
        else:
            split = kwargs.get("future_start_ms")
            arguments = {k:v for k,v in kwargs.items() if k not in
                         {"history_seed","future_seed","future_start_ms","start_ms","stop_ms"}}
            intervals = [(kwargs["start_ms"],kwargs["stop_ms"],kwargs["history_seed"])] if split is None else [
                (kwargs["start_ms"],split,kwargs["history_seed"]),
                (split,kwargs["stop_ms"],kwargs["future_seed"])]
            result = np.concatenate([_burst_events(start_ms=a,stop_ms=b,seed=s,
                envelope_times=envelope[0],envelope_depths=envelope[1],
                envelope_max_depth=float(regime["modulation_depth"]),**arguments) for a,b,s in intervals])
        record["event_trains"].append(result.copy())
        return result
    with patch.object(utils,"generate_phase_diffusion_path",phase), patch.object(utils,"generate_split_background_spike_train",events):
        yield
    if phase_cache:
        record["phase_time_ms"],record["phase_rad"] = phase_cache[0]
    else:
        t = np.arange(round(total/phase_dt)+1,dtype=np.float64)*phase_dt
        record["phase_time_ms"] = t
        record["phase_rad"] = record["initial_phase_rad"]+2*np.pi*float(regime.get("frequency_hz",0))*t/1000


def _network_audit(network):
    """Canonical structure description, excluding globally increasing HOC names."""
    cells = [cell for name in network.population_names for cell in network.populations[name].cells]
    cell_gids = {str(cell.template): int(cell.gid) for cell in cells}
    sections = {str(sec): (int(cell.gid),i) for cell in cells for i,sec in enumerate(cell.allseclist)}
    geometry, synapses, netcons = [], [], []
    for cell in cells:
        geometry.append(np.concatenate([np.asarray(getattr(cell,k)).ravel() for k in ("x","y","z","area","d","length")]))
        for sec_i,sec in enumerate(cell.allseclist):
            for seg in sec:
                for pp in seg.point_processes():
                    if hasattr(pp,"tau1"):
                        synapses.append([cell.gid,sec_i,pp.get_segment().x,pp.tau1,pp.tau2,pp.e])
    for nc in network._hoc_netconlist:
        pp = nc.syn(); seg = pp.get_segment()
        gid,sec_i = sections[str(seg.sec)]
        netcons.append([nc.srcgid(),gid,sec_i,seg.x,nc.delay,nc.threshold,*list(nc.weight)])
    # Background NetCons have a delivery delay too; record it for causal events.
    external = []
    for cell in cells:
        for syn_i,nc in enumerate(cell._hoc_netconlist):
            pp = nc.syn(); seg = pp.get_segment(); gid,sec_i = sections[str(seg.sec)]
            external.append([gid,sec_i,seg.x,nc.delay,*list(nc.weight)])
    geometry = np.concatenate(geometry).astype(np.float64)
    synapses = np.asarray(synapses,dtype=np.float64).reshape(-1,6)
    netcons = np.asarray(netcons,dtype=np.float64).reshape(-1,7)
    external = np.asarray(external,dtype=np.float64).reshape(-1,5)
    return {"geometry":geometry,"synapses":synapses,"recurrent_netcons":netcons,
            "external_netcons":external,"structure_hash":array_hash(geometry,synapses,netcons,external)}


def _physical_checkpoint(network):
    """All independent dynamic states of the inherited HH/passive/Exp2Syn model.

    Includes segment voltage, HH gates, extracellular potential and both Exp2Syn
    states. Input future schedules and RNG state are intentionally excluded.
    """
    values = []
    for name in network.population_names:
        for cell in network.populations[name].cells:
            for sec in cell.allseclist:
                for seg in sec:
                    values.extend([seg.v,float(seg.vext[0])])
                    if hasattr(seg,"hh"): values.extend([seg.hh.m,seg.hh.h,seg.hh.n])
                    for pp in seg.point_processes():
                        if hasattr(pp,"A"): values.extend([pp.A,pp.B])
    return np.asarray(values,dtype=np.float64)


def _pending_events(audit, events, spike_times, spike_gids, cutoff):
    pending_recurrent, pending_external = [], []
    for row in audit["recurrent_netcons"]:
        source, target, sec, pos, delay, threshold, weight = row
        times = spike_times[(spike_gids == source)&(spike_times <= cutoff)]
        for time_ms in times[times+delay > cutoff]:
            pending_recurrent.append([source,target,sec,pos,time_ms+delay,weight])
    # LFPy 2.3 Cell._load_spikes calls NetCon.event(tdeliver), NOT a
    # presynaptic source event. NEURON explicitly ignores NetCon.delay for
    # event(t). Thus all external events before cutoff have already arrived;
    # no extra delayed external deliveries are causally pending.
    return (np.asarray(pending_recurrent,dtype=np.float64).reshape(-1,6),
            np.asarray(pending_external,dtype=np.float64).reshape(-1,6))


def generate_trial(config: dict[str,Any], spec: dict[str,Any]) -> dict[str,Any]:
    """Run one real unstimulated persistent episode; returns time-major float64.

    `duration_s` is retained length, in addition to burn-in. The caller crops
    only after causal filtering. A rerun fork stops exactly at its cutoff to
    audit causal state before continuing; it never resets an ongoing episode.
    """
    sim,dt,total,phase_dt = _validate(config,spec)
    started = time.perf_counter()
    with _LOCK:
        old_numpy, old_python = np.random.get_state(), random.getstate()
        old_main = os.environ.get("MAIN_PATH")
        os.environ["MAIN_PATH"] = str(_ROOT)
        environment = None
        try:
            from mpi4py import MPI
            from omegaconf import OmegaConf
            from env.models.neuron.env_online import OnlineNeuronEnv
            from setup.circuits.ballnstick import utils
            import neuron, LFPy
            if MPI.COMM_WORLD.Get_size() != 1:
                raise ValueError("Use independent serial worker processes, not mpirun, for trial shards.")
            # Override stale MAIN_PATH loaded earlier without changing legacy code.
            old_utils_path = utils.MAIN_PATH
            utils.MAIN_PATH = str(_ROOT)
            phase0 = float(np.random.default_rng(_seed(spec["phase_seed"],0x50484930)).uniform(0,2*np.pi))
            record = {"event_trains":[],"initial_phase_rad":phase0}
            with tempfile.TemporaryDirectory(prefix="eegrepr-neuron-") as scratch:
                cfg = OmegaConf.create({"env":OmegaConf.load(_ROOT/"configs/env/ballnstick.yaml"),
                                        "experiment":{"dir":scratch,"plot":False,"tqdm":False}})
                cfg.env.simulation.duration = total
                cfg.env.simulation.obs_win_len = min(1000.,total)
                cfg.env.network.celsius = 6.3
                cfg.env.online.temperature_mode = "configured"
                cfg.env.ts.apply = True
                # Small-cell settings are permitted only in explicit functional smokes.
                if "population_sizes" in sim:
                    cfg.env.network.population.sizes = sim["population_sizes"]
                else:
                    cfg.env.network.population.sizes = {"E":int(sim.get("population_e",32)),"I":int(sim.get("population_i",8))}
                if "external_synapses_per_cell" in sim:
                    for population in ("E","I"):
                        cfg.env.network.background[population].n_synapses = int(sim["external_synapses_per_cell"])
                regime = spec["regime"]
                cfg.env.network.recurrent.i_to_e_tau2_multiplier = float(regime.get("kinetics_scale",1.))
                cfg.env.network.recurrent.i_to_e_preserve_conductance_time_area = True
                for population in ("E","I"):
                    rhythm = cfg.env.network.background[population].rhythm
                    depth = float(regime.get("modulation_depth",0))
                    rhythm.enabled = depth > 0
                    rhythm.modulation_depth = depth
                    rhythm.frequency_hz = float(regime.get("frequency_hz",0))
                    rhythm.phase_rad = phase0
                    rhythm.phase_diffusion_rad2_per_s = float(regime.get("diffusion",0))
                    rhythm.phase_diffusion_integration_dt_ms = phase_dt
                    rhythm.shared_modulated_fraction = float(regime.get("shared_fraction",1.))
                    rhythm.thinning_envelope_modulation_depth = depth
                structure = int(spec["structure_seed"])
                np.random.seed(structure*10000); random.seed(structure*10000)
                mpi = {"COMM":MPI.COMM_WORLD,"SIZE":1,"RANK":0,
                       "GLOBALSEED":int(spec["trial_seed"]),"SEED":structure*10000}
                cutoff = spec.get("future_start_ms")
                if cutoff is not None:
                    mpi.update(FUTUREGLOBALSEED=int(spec["future_seed"]),FUTURESTARTMS=float(cutoff))
                try:
                    with _construction_adapter(utils,spec,total,phase_dt,record):
                        environment = OnlineNeuronEnv(cfg,mpi,ENV_SEED=0)
                finally:
                    utils.MAIN_PATH = old_utils_path
                network = environment.network
                audit = _network_audit(network)
                # No stimulation waveform is needed: zero every extracellular segment.
                environment.stimulation_controller.clear(network)
                boundaries = list(np.arange(1000.,total,1000.))+[total]
                if cutoff is not None: boundaries.append(float(cutoff))
                boundaries = sorted(set(boundaries))
                times, eeg, dipoles, spikes_t, spikes_g = [],[],[],[],[]
                checkpoint = None
                for stop in boundaries:
                    output = network.advance_online(stop_ms=stop)
                    d = np.asarray(output["probe_data"][1],dtype=np.float64)
                    volts = environment.four_sphere_top.get_dipole_potential(d,
                        np.asarray(cfg.env.network.position,dtype=np.float64))*1e-3
                    times.append(output["time_ms"]); eeg.append(volts.T); dipoles.append(d.T)
                    for pop in ("E","I"):
                        for gid,spike_train in output["spikes"][pop]["per_cell"].items():
                            spikes_t.extend(spike_train); spikes_g.extend([int(gid)]*len(spike_train))
                    if cutoff is not None and stop == cutoff:
                        checkpoint = _physical_checkpoint(network)
                time_ms = np.concatenate(times).astype(np.float64)
                eeg_v = np.concatenate(eeg).astype(np.float64)
                dipole = np.concatenate(dipoles).astype(np.float64)
                spike_times = np.asarray(spikes_t,dtype=np.float64)
                spike_gids = np.asarray(spikes_g,dtype=np.int64)
                order = np.lexsort((spike_gids,spike_times)); spike_times,spike_gids = spike_times[order],spike_gids[order]
                diagnostics = network.online_diagnostics()
                residual = environment.stimulation_controller.max_abs_extracellular(network)
                if residual != 0 or diagnostics["effective_h_celsius"] != 6.3:
                    raise RuntimeError("Zero-field or configured-temperature invariant failed.")
                if len(time_ms) != round(total/dt) or not np.all(np.diff(time_ms)>0) or not np.isfinite(eeg_v).all():
                    raise RuntimeError("Finite EEG, monotonic time or exact native count failed.")
                offsets = np.concatenate([[0],np.cumsum([len(x) for x in record["event_trains"]])]).astype(np.int64)
                event_values = np.concatenate(record["event_trains"]) if offsets[-1] else np.empty(0)
                privileged = {"external_event_times_ms":event_values,"external_event_offsets":offsets,
                    "spike_times_ms":spike_times,"spike_gids":spike_gids,
                    "phase_time_ms":record["phase_time_ms"],"phase_rad":record["phase_rad"],
                    "structure_geometry":audit["geometry"],"recurrent_netcons":audit["recurrent_netcons"],
                    "external_netcons":audit["external_netcons"],"synapse_parameters":audit["synapses"]}
                for key in ("burst_times_ms","burst_depth"):
                    if key in record: privileged[key]=record[key]
                burn_ms = 1000*float(sim.get("burn_in_s",2))
                drive_diagnostics = {}
                retained = time_ms > burn_ms
                phase_native = np.interp(time_ms,record["phase_time_ms"],record["phase_rad"])
                depth_native = np.full(len(time_ms),float(regime.get("modulation_depth",0)))
                if "burst_times_ms" in record:
                    depth_native = record["burst_depth"][np.searchsorted(record["burst_times_ms"],time_ms,side="right")-1]
                modulation_mean = float(np.mean(depth_native[retained]*np.sin(phase_native[retained])))
                drive_diagnostics["envelope_mean_depth_retained"] = float(np.mean(depth_native[retained]))
                drive_diagnostics["envelope_rms_depth_retained"] = float(np.sqrt(np.mean(depth_native[retained]**2)))
                drive_diagnostics["conditional_rate_quadrature"] = "native-grid mean of 1+depth(t)*sin(phase(t)); audit only"
                n_synapses = int(sim.get("external_synapses_per_cell",64))
                event_index = 0
                for population in ("E","I"):
                    size = int(cfg.env.network.population.sizes[population])
                    trains = record["event_trains"][event_index:event_index+size*n_synapses]
                    event_index += size*n_synapses
                    count = sum(np.count_nonzero(train > burn_ms) for train in trains)
                    mean_rate = 1000./float(cfg.env.network.background[population].interval_ms)
                    drive_diagnostics[population] = {"baseline_afferent_rate_hz":mean_rate,
                        "realized_events_per_synapse_per_second":count/(size*n_synapses*(total-burn_ms)/1000) if n_synapses else None,
                        "retained_external_event_count":int(count),
                        "conditional_expected_afferent_rate_hz":mean_rate*(1+(np.floor(n_synapses*float(regime.get("shared_fraction",1.))+.5)/n_synapses if n_synapses else 0)*modulation_mean)}
                if "burst_times_ms" in record:
                    envelope_depth = depth_native
                    privileged["burst_occupancy"] = np.asarray(np.mean(envelope_depth[time_ms>burn_ms]==float(regime["modulation_depth"])))
                    privileged["burst_dwell_ms"] = np.diff(record["burst_times_ms"])
                    drive_diagnostics["burst"] = {"occupancy_retained":float(privileged["burst_occupancy"]),
                        "mean_rate_note":"Zero sinusoidal mean holds over phase randomization; individual finite-trial means may differ.",
                        "matched_power_control":"not imposed; assess/freeze only on development data"}
                elif float(regime.get("modulation_depth",0)) > 0:
                    privileged["burst_occupancy"] = np.asarray(1.)
                metadata = {"backend":"neuron_lfpy_real","native_dt_ms":dt,
                    "native_sample_count":len(time_ms),"burn_in_ms":1000*float(sim.get("burn_in_s",2)),
                    "total_duration_ms":total,"initial_phase_rad":phase0,
                    "structure_hash":audit["structure_hash"],"external_events_hash":array_hash(event_values,offsets),
                    "configured_celsius":6.3,"effective_celsius":diagnostics["effective_h_celsius"],
                    "applied_field_v_per_m":0.,"final_extracellular_mV":residual,
                    "initializations_per_episode":1,"sample_boundary_convention":"(start,stop]",
                    "units":{"eeg_v":"V","dipole_na_um":"nA um","time_ms":"ms"},
                    "software":{"neuron":neuron.__version__,"lfpy":LFPy.__version__,"numpy":np.__version__},
                    "rng_namespaces":{"structure":structure,"events":int(spec["trial_seed"]),
                        "phase":int(spec["phase_seed"]),"phase_initial_namespace":"PHI0",
                        "phase_innovation_namespace":"PHDF","burst_namespace":"BURS",
                        "future_events":spec.get("future_seed"),"future_phase":spec.get("future_phase_seed"),
                        "burst":int(spec.get("envelope_seed",spec.get("burst_seed",spec["phase_seed"]))),
                        "future_burst":spec.get("future_burst_seed",spec.get("future_seed"))},
                    "regime":dict(regime),"phase_grid_dt_ms":phase_dt,"drive_diagnostics":drive_diagnostics,
                    "population_sizes":dict(sim.get("population_sizes",{"E":int(sim.get("population_e",32)),"I":int(sim.get("population_i",8))})),
                    "scope":"controlled toy circuit; not human cortex or a clinical bound"}
                if cutoff is not None:
                    pending_r,pending_e = _pending_events(audit,record["event_trains"],spike_times,spike_gids,cutoff)
                    privileged.update(cutoff_physical_state=checkpoint,pending_recurrent_events=pending_r,
                                      pending_external_events=pending_e)
                    history_trains = [x[x<float(cutoff)] for x in record["event_trains"]]
                    history_offsets = np.concatenate([[0],np.cumsum([len(x) for x in history_trains])]).astype(np.int64)
                    history_events = np.concatenate(history_trains)
                    metadata["fork"] = {"cutoff_ms":float(cutoff),"backend":"identical_prefix_rerun",
                        "physical_state_hash":array_hash(checkpoint),
                        "external_history_hash":array_hash(history_events,history_offsets),
                        "pending_recurrent_hash":array_hash(pending_r),"pending_external_hash":array_hash(pending_e),
                        "prefix_spike_hash":array_hash(spike_times[spike_times<=cutoff],spike_gids[spike_times<=cutoff]),
                        "state_variables":"all segment v/vext, HH m/h/n, Exp2Syn A/B; drive phase at cutoff",
                        "pending_event_audit":"recurrent: exact presynaptic histories + NetCon delays/weights; external: direct NetCon.event delivery (delay ignored)",
                        "external_schedule_semantics":"LFPy2.3 NetCon.event(tdeliver); direct delivery, no extra delay",
                        "future_schedules_are_oracle_only":True}
                metadata["runtime_seconds"] = time.perf_counter()-started
                return {"time_ms":time_ms,"eeg_v":eeg_v,"dipole_na_um":dipole,
                        "privileged":privileged,"metadata":metadata}
        finally:
            if environment is not None:
                environment.close(); environment.close()
            np.random.set_state(old_numpy); random.setstate(old_python)
            if old_main is None: os.environ.pop("MAIN_PATH",None)
            else: os.environ["MAIN_PATH"]=old_main
