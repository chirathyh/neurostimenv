"""Deterministic-prefix fork audits and full-causal-state variance estimators.

Fork variance is a lower bound on EEG-only Bayes risk. Ensemble means are
oracle estimators, never learned-model inputs. All intervals here describe
Monte Carlo futures conditional on ONE state, not independent circuit units.
"""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from .simulator import array_hash


def validate_fork_prefixes(trials: Sequence[dict[str,Any]], cutoff_ms: float | None = None):
    if len(trials) < 2: raise ValueError("A fork validation needs at least two futures.")
    reference = trials[0]
    audit = reference["metadata"].get("fork")
    if not audit: raise ValueError("No causal-state fork audit is available.")
    cutoff = float(audit["cutoff_ms"] if cutoff_ms is None else cutoff_ms)
    if cutoff != float(audit["cutoff_ms"]): raise ValueError("Cutoff differs from checkpoint time.")
    reference_times = np.asarray(reference["time_ms"])
    mask = reference_times <= cutoff
    if not np.any(mask) or reference_times[mask][-1] != cutoff:
        raise ValueError("Cutoff is not an exact native sample boundary.")
    keys = ("physical_state_hash","external_history_hash","pending_recurrent_hash",
            "pending_external_hash","prefix_spike_hash")
    for candidate in trials[1:]:
        other = candidate["metadata"].get("fork",{})
        if other.get("cutoff_ms") != cutoff: raise ValueError("Mixed fork decision times.")
        if candidate["metadata"]["structure_hash"] != reference["metadata"]["structure_hash"]:
            raise ValueError("Fork structure/weights/delays differ.")
        for key in keys:
            if other.get(key) != audit[key]: raise ValueError(f"Fork causal-state audit failed: {key}")
        if not np.array_equal(candidate["time_ms"],reference_times): raise ValueError("Fork time grids differ.")
        for key in ("eeg_v","dipole_na_um"):
            if not np.array_equal(candidate[key][mask],reference[key][mask]):
                raise ValueError(f"Fork prefix differs in {key}.")
        for key in ("cutoff_physical_state","pending_recurrent_events","pending_external_events"):
            if not np.array_equal(candidate["privileged"][key],reference["privileged"][key]):
                raise ValueError(f"Checkpoint array differs: {key}")
        if "phase_time_ms" in reference["privileged"]:
            pt = np.asarray(reference["privileged"]["phase_time_ms"])
            if not np.array_equal(pt,candidate["privileged"]["phase_time_ms"]): raise ValueError("Phase grids differ.")
            pmask = pt <= cutoff
            if not np.array_equal(reference["privileged"]["phase_rad"][pmask],candidate["privileged"]["phase_rad"][pmask]):
                raise ValueError("Future phase innovation leaked into prefix.")
        if "burst_times_ms" in reference["privileged"]:
            def prefix_burst(trial):
                p=trial["privileged"]; keep=p["burst_times_ms"]<=cutoff
                return p["burst_times_ms"][keep],p["burst_depth"][keep]
            if any(not np.array_equal(a,b) for a,b in zip(prefix_burst(reference),prefix_burst(candidate))):
                raise ValueError("Burst-envelope history differs.")
    return {"passed":True,"n_futures":len(trials),"cutoff_ms":cutoff,
        "prefix_sample_count":int(mask.sum()),"structure_hash":reference["metadata"]["structure_hash"],
        "prefix_eeg_hash":array_hash(reference["eeg_v"][mask]),
        "pending_recurrent_count":len(reference["privileged"]["pending_recurrent_events"]),
        "pending_external_count":len(reference["privileged"]["pending_external_events"]),
        "checks":["exact EEG/dipole/latent-phase prefix","structure/locations/weights/delays",
                  "external event history",("all HH/passive/Exp2Syn dynamic states" if reference["metadata"].get("backend")=="neuron_lfpy_real" else "all analytical generator dynamic states"),
                  "pending caused recurrent and external events","precut recurrent spikes"],
        "interpretation":"deterministic-prefix replay; no SaveState event deletion"}


def target_block(values, times_ms, *, cutoff_ms, lead_ms, block_ms, sample_rate_hz=500.):
    """Causal target starts at cutoff+lead, with exactly block_ms*fs samples.

    Returns times in [cutoff+lead, cutoff+lead+block). A 2-ms lead at 500 Hz
    begins with the first new sample. No target sample may be in input history.
    """
    t = np.asarray(times_ms,dtype=np.float64)
    x = np.asarray(values,dtype=np.float64)
    step = 1000./float(sample_rate_hz)
    if lead_ms <= 0: raise ValueError("Forecast leads must be strictly positive.")
    for val in (lead_ms,block_ms):
        if val <= 0 or not np.isclose(val/step,round(val/step),atol=1e-9):
            raise ValueError("Lead and block lengths must align to the model grid.")
    start = float(cutoff_ms)+float(lead_ms)
    index = int(np.searchsorted(t,start))
    n = int(round(block_ms/step))
    if index+n > len(t) or not np.isclose(t[index],start,atol=1e-9):
        raise ValueError("Target block is unavailable at the required time support.")
    if not np.allclose(t[index:index+n],start+np.arange(n)*step,rtol=0,atol=1e-8):
        raise ValueError("Target time grid is not uniform.")
    return x[index:index+n]


def conditional_variance_summary(futures, *, bootstrap_samples=1000, seed=0):
    """Unbiased trace(sample covariance)/d and future-resampling uncertainty.

    Futures may be vectors or waveform blocks. Half squared differences use
    disjoint pairs only, reported as a check. Bootstrap intervals condition on
    this one causal state; aggregate across structure with a separate cluster
    analysis. The leave-one-out mean does not contain the scored realization.
    """
    x = np.asarray(futures,dtype=np.float64)
    if x.ndim < 2 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError("Expected at least two finite independent future targets.")
    x = x.reshape(len(x),-1)
    n,d = x.shape
    variance = float(np.var(x,axis=0,ddof=1).mean())
    m = n//2
    pair_values = np.mean((x[0:2*m:2]-x[1:2*m:2])**2,axis=1)/2
    loo = (x.sum(axis=0,keepdims=True)-x)/(n-1)
    loo_mse = float(np.mean((x-loo)**2))
    rng=np.random.default_rng(int(seed))
    estimates=[]
    for _ in range(int(bootstrap_samples)):
        draw=x[rng.integers(0,n,n)]
        estimates.append(float(np.var(draw,axis=0,ddof=1).mean()))
    ci = np.quantile(estimates,[.025,.975]).tolist() if estimates else [None,None]
    return {"n_futures":n,"target_dimension":d,"conditional_variance_trace_per_dimension":variance,
        "future_bootstrap_95_interval":ci,"monte_carlo_interval_scope":"futures conditional on one fixed causal state",
        "disjoint_pair_half_mse":float(pair_values.mean()),"n_independent_pairs":m,
        "leave_one_out_oracle_mean_mse":loo_mse,
        "loo_finite_ensemble_corrected_mse":loo_mse*(n-1)/n,
        "oracle_mean_includes_scored_future":False,
        "interpretation":"full-causal-state conditional risk; lower bound for EEG-history-only Bayes risk"}


def summarize_fork_ensemble(futures, **kwargs):
    """Stable alias for callers assembling already-extracted future targets."""
    return conditional_variance_summary(futures,**kwargs)


def _read_fork(path):
    import h5py, json
    with h5py.File(path,"r") as h:
        attrs=json.loads(h.attrs["metadata_json"])
        privileged={k:v[:] if v.ndim else v[()] for k,v in h["privileged"].items()}
        # Empty arrays are serialized as group attributes by the shared writer.
        privileged.update({k:np.asarray(json.loads(v)) for k,v in h["privileged"].attrs.items()})
        for key in ("pending_recurrent_events","pending_external_events"):
            if key in privileged: privileged[key]=privileged[key].reshape(-1,6)
        return {**{k:h[k][:] for k in ("time_ms","eeg_v","dipole_na_um")},
                "privileged":privileged,"metadata":attrs["metadata"]}


def run_forks(config, output_root, resume=False):
    """Generate resumable serial future shards on training structures only.

    Decision states use their own named history namespace. Structure identity
    is shared with the training set; test structures are never accessed. A
    convergence subset extends the same ensemble, so the first F futures are
    exactly nested within its 2F counterpart. Writes are one shard at a time.
    """
    from pathlib import Path
    from . import SCHEMA_VERSION
    from .storage import (atomic_json,read_json,initialize_output,output_lock,
                          require_compatible,write_trial,verify_shard,rng_seed)
    from .schema import config_hash
    from .preprocessing import causal_resample
    cfg=config["forks"]; root=Path(output_root)
    native=config["simulation"]; forecast=config["forecast"]
    cutoff=1000*(float(native["burn_in_s"])+float(cfg["decision_s"]))
    total=1000*(float(native["burn_in_s"])+float(native["duration_s"]))
    max_support=max(float(forecast["block_ms"]),*map(float,forecast["target_scales_ms"]))
    if cutoff+max(map(float,forecast["leads_ms"]))+max_support>total+1e-8:
        raise ValueError("Fork decision + maximum lead + target support exceeds episode; expand duration or reduce fork leads/scales.")
    if config["data"]["backend"]=="analytical" and cfg["vary"]=="phase":
        raise ValueError("Analytical controls have no separate exogenous phase source; use events or both.")
    if config["data"]["backend"]=="neuron":
        from .simulator import generate_trial
    else:
        from .analytical import generate_trial
    with output_lock(root):
        if (root/"provenance.json").exists(): require_compatible(root,config)
        else: initialize_output(root,config,resume=resume)
        provenance=read_json(root/"provenance.json")
        directory=root/"forks"; marker=directory/"manifest.json"
        if marker.exists() and not resume: raise FileExistsError("Fork artifacts exist; use --resume.")
        rows=read_json(marker)["futures"] if marker.exists() else []
        known={r["trial_id"]:r for r in rows}
        for row in rows: verify_shard(root,row)
        reports=[]; summaries=[]
        structures=config["data"]["structures"]["train"]
        namespace=config["data"]["namespace"]
        primary_n=int(cfg["futures"])
        for regime in config["data"]["regimes"]:
            for state in range(int(cfg["states_per_regime"])):
                structure=int(structures[state%len(structures)])
                history_id=f"fork-s{structure}-state{state:03d}-{regime['name']}"
                event_seed=rng_seed(namespace,"fork-history-events",structure,state)
                phase_seed=rng_seed(namespace,"fork-history-phase",structure,state)
                n=primary_n
                if state<int(cfg["convergence_states"]): n=max(n,int(cfg["convergence_futures"]))
                reference=None; all_targets={}; report=None
                for future in range(n):
                    trial_id=f"{history_id}-f{future:04d}"
                    event_future=future if cfg["vary"] in {"events","both"} else 0
                    phase_future=future if cfg["vary"] in {"phase","both"} else 0
                    spec={"structure_seed":structure,"trial_seed":event_seed,"phase_seed":phase_seed,
                        "envelope_seed":rng_seed(namespace,"fork-history-envelope",structure,state),
                        "future_seed":rng_seed(namespace,"fork-future-events",structure,state,event_future),
                        "future_phase_seed":rng_seed(namespace,"fork-future-phase",structure,state,phase_future),
                        "future_burst_seed":rng_seed(namespace,"fork-future-envelope",structure,state,future if cfg["vary"] in {"events","both"} else 0),
                        "future_start_ms":cutoff,"regime":regime,"history_id":history_id,
                        "trial_id":trial_id,"future_id":str(future),"structure_id":str(structure),"split":"train"}
                    if trial_id in known:
                        trial=_read_fork(root/known[trial_id]["path"])
                    else:
                        trial=generate_trial(config,spec)
                        row={"trial_id":trial_id,"parent_trial_id":history_id,"paired_family_id":history_id,
                            "history_id":history_id,"future_id":str(future),"structure_id":str(structure),
                            "split":"train","regime":regime,"spec":spec,
                            "schema_version":SCHEMA_VERSION,"config_hash":config_hash(config),
                            "commit":provenance["commit"],"source_hash":provenance["source_hash"],"software_versions":provenance["versions"],
                            "path":f"forks/trials/{trial_id}.h5","metadata":trial["metadata"]}
                        row["sha256"]=write_trial(root/row["path"],trial,{"spec":spec,"metadata":trial["metadata"]})
                        row["bytes"]=(root/row["path"]).stat().st_size
                        rows.append(row); known[trial_id]=row
                        atomic_json(marker,{"schema_version":SCHEMA_VERSION,"config_hash":config_hash(config),"futures":rows})
                    if reference is None: reference=trial
                    else: report=validate_fork_prefixes([reference,trial],cutoff)
                    values,times,filter_meta=causal_resample(trial["eeg_v"],trial["time_ms"],target_fs=native["model_sample_rate_hz"])
                    for lead in forecast["leads_ms"]:
                        block=target_block(values,times,cutoff_ms=cutoff,lead_ms=lead,
                                           block_ms=forecast["block_ms"],sample_rate_hz=native["model_sample_rate_hz"])
                        all_targets.setdefault((float(lead),"waveform",float(forecast["block_ms"])),[]).append(block.reshape(-1))
                        for scale in forecast["target_scales_ms"]:
                            block=target_block(values,times,cutoff_ms=cutoff,lead_ms=lead,
                                               block_ms=scale,sample_rate_hz=native["model_sample_rate_hz"])
                            power=np.mean(block**2,axis=0)
                            if np.any(power<=0): raise ValueError("Zero-power fork target cannot be logged.")
                            all_targets.setdefault((float(lead),"rms_envelope",float(scale)),[]).append(np.sqrt(power))
                            all_targets.setdefault((float(lead),"log_total_power",float(scale)),[]).append(np.log(power))
                    print(f"fork {history_id} future {future+1}/{n}",flush=True)
                report.update(history_id=history_id,structure_id=str(structure),n_futures=n)
                reports.append(report)
                for (lead,operator,scale),targets in all_targets.items():
                    counts=[primary_n] if n==primary_n else [primary_n,n]
                    for count in counts:
                        result=conditional_variance_summary(np.asarray(targets[:count]),
                            bootstrap_samples=int(cfg["bootstrap_samples"]),seed=rng_seed(namespace,"fork-mc",history_id,lead,operator,scale,count))
                        if cfg["vary"] != "both":
                            result["interpretation"] = "source-specific conditional variance with other future innovation path fixed; not the full-causal-state Bayes-risk floor"
                        summaries.append({**result,"history_id":history_id,"structure_id":str(structure),
                            "regime":regime["name"],"lead_ms":lead,"operator":operator,"target_scale_ms":scale,
                            "target_support":"[cutoff+lead,cutoff+lead+scale)","sample_rate_hz":native["model_sample_rate_hz"],
                            "target_units":"V" if operator in {"waveform","rms_envelope"} else "ln(V^2)",
                            "filter_delay_ms":filter_meta["group_delay_ms"]})
                atomic_json(directory/"prefix_validation.json",{"passed":True,"states":reports})
                atomic_json(directory/"state_variance.json",{"estimates":summaries,"claim_status":config["claim_status"]})
        # Equal state weights inside structure, equal structure weights overall.
        aggregate=[]
        groups=sorted({(s["regime"],s["lead_ms"],s["operator"],s["target_scale_ms"],s["n_futures"]) for s in summaries})
        for regime,lead,operator,scale,n in groups:
            subset=[s for s in summaries if (s["regime"],s["lead_ms"],s["operator"],s["target_scale_ms"],s["n_futures"])==(regime,lead,operator,scale,n)]
            ids=sorted({s["structure_id"] for s in subset})
            structural=[np.mean([s["conditional_variance_trace_per_dimension"] for s in subset if s["structure_id"]==sid]) for sid in ids]
            aggregate.append({"regime":regime,"lead_ms":lead,"operator":operator,"target_scale_ms":scale,"n_futures":n,
                "n_structures":len(ids),"n_states":len(subset),"structure_mean_conditional_variance":float(np.mean(structural)),
                "between_structure_sd":float(np.std(structural,ddof=1)) if len(ids)>1 else None,
                "structure_values":dict(zip(ids,map(float,structural))),
                "scope":("full-causal-state lower bound; Monte Carlo and structural variation are separate" if cfg["vary"]=="both" else "source-specific innovation variance; other future path held fixed; not full-causal-state Bayes risk")})
        result={"states":len(reports),"future_shards":len(rows),"bytes":sum(r["bytes"] for r in rows),
            "prefix_validation_passed":True,"claim_status":config["claim_status"],"aggregate":aggregate,
            "innovation_sources":cfg["vary"],
            "burst_innovation_policy":"varies with events/both; held fixed for phase-only",
            "risk_conditioning":"full causal state" if cfg["vary"]=="both" else "source-specific variance with the other future exogenous source held fixed; not the full innovation floor",
            "confirmation":"pending; smoke/pilot is not confirmation"}
        atomic_json(directory/"summary.json",result)
        return result
