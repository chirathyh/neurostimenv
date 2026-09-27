"""One G1B reference/MDD trajectory with read-only construction verification."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.l23net_analysis.run_l23net_no_field_replay import (
    _validate_configuration as validate_reference,
    run_replay,
)
from experiments.l23net_analysis.replay_validation import canonical_json_sha256
from experiments.l23net_analysis.g1b_analysis import REPORT_NAME, TRACE_NAME, PILOT_SEEDS


def validate_configuration(cfg, mpi_size):
    condition = str(cfg.analysis.condition)
    if condition not in ("reference", "mdd"):
        raise ValueError("G1B condition must be reference or mdd.")
    if bool(cfg.env.simulation.MDD) != (condition == "mdd"):
        raise ValueError("MDD flag and G1B condition disagree.")
    common = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    common.analysis.condition = "reference"
    common.env.simulation.MDD = False
    result = validate_reference(common, mpi_size)
    if bool(cfg.analysis.require_full_network):
        if (float(cfg.analysis.duration_ms), float(cfg.analysis.window_ms), mpi_size) != (28000., 1000., 624):
            raise ValueError("Full G1B is frozen to 28 s, 1-s windows, 624 ranks.")
        if int(cfg.experiment.seed) not in PILOT_SEEDS or int(cfg.analysis.env_seed) != 0:
            raise ValueError("Full G1B requires pilot seed 7101 or 7102 and env_seed=0.")
    if float(cfg.env.network.tstart) != 0 or float(cfg.env.network.v_init) != -80:
        raise ValueError("G1B requires tstart=0 ms and v_init=-80 mV.")
    return result


def expected_parameters(tables, mdd):
    """Literal existing L23Net intervention, including interneuron tonic terms."""
    names = list(tables["conn_probs"].index)
    syn = {}
    tonic = {}
    for post in names:
        for pre in names:
            syn[f"{pre}:{post}"] = float(tables["syn_cond"].at[pre, post]) * (0.6 if mdd and "SST" in pre else 1.)
        normal = float(tables["SING_CELL_PARAM"].at["norm_tonic", post])
        apical = float(tables["SING_CELL_PARAM"].at["apic_tonic", post])
        if "PYR" in post:
            apical *= 0.6 if mdd else 1.
        elif mdd:
            contributions = {
                pre: float(tables["syn_cond"].at[pre, post] * tables["n_cont"].at[pre, post] * tables["conn_probs"].at[pre, post])
                for pre in names if any(label in pre for label in ("SST", "PV", "VIP"))
            }
            normal *= 1. - .4 * sum(v for k, v in contributions.items() if "SST" in k) / sum(contributions.values())
        tonic[post] = {"somatic": normal, "basal": normal}
        if "PYR" in post:
            tonic[post]["apical"] = apical
    return {"synaptic_gmax": syn, "tonic_g_s_per_cm2": tonic,
            "connected_pairs": sorted(f"{pre}:{post}" for pre in names for post in names
                                      if tables["conn_probs"].at[pre, post] > 0)}


def construction_audit(environment, cfg):
    """Check actual values and hash topology/OU setup without advancing RNGs."""
    comm = environment.comm
    rank = comm.Get_rank()
    expected = None
    setup_error = None
    if rank == 0:
        try:
            expected = expected_parameters(
                pd.read_excel(ROOT / "setup/circuits/L23Net/Circuit_param.xls", sheet_name=None, index_col=0),
                bool(cfg.env.simulation.MDD),
            )
        except Exception as exc:
            setup_error = repr(exc)
    setup_error, expected = comm.bcast((setup_error, expected), root=0)
    if setup_error:
        raise RuntimeError(setup_error)
    network = environment.network
    topology = hashlib.sha256()
    background = hashlib.sha256()
    counts = {"synapses": 0, "tonic_segments": 0, "ou_processes": 0}
    errors = []
    checked_groups = {}

    def check(value, target, label):
        if not np.isclose(value, target, rtol=1e-12, atol=1e-15) and len(errors) < 10:
            errors.append(f"rank {rank} {label}: {value} != {target}")

    def update(digest, row):
        digest.update(canonical_json_sha256(row).encode("ascii"))

    try:
        section_owners = {}
        ranges = []
        for name in network.population_names:
            pop = network.populations[name]
            ranges.append((int(pop.first_gid), int(pop.first_gid + pop.POP_SIZE), name))
            for gid, cell in zip(pop.gids, pop.cells):
                for sec in cell.allseclist:
                    section_owners[sec.name()] = (int(gid), name, sec.name().split(".", 1)[-1])
                for group, target in expected["tonic_g_s_per_cm2"][name].items():
                    for sec in getattr(cell.template, group):
                        for seg in sec:
                            check(float(seg.g_tonic), target, f"{name}/{group} tonic")
                            check(float(seg.e_gaba_tonic), -75., f"{name}/{group} tonic reversal")
                            counts["tonic_segments"] += 1
                for index, ou in enumerate(cell.template.OUprocess):
                    values = [float(getattr(ou, key)) for key in ("E_e", "E_i", "g_e0", "g_i0", "tau_e", "tau_i", "std_e", "std_i")]
                    # Legacy Random uses a generator without seq() support.
                    # Never inspect it through methods that can reseed/draw.
                    update(background, [int(gid), index, values])
                    counts["ou_processes"] += 1
        for nc in network._hoc_netconlist:
            source = int(nc.srcgid())
            pre = next(name for first, stop, name in ranges if first <= source < stop)
            syn = nc.syn()
            segment = syn.get_segment()
            gid, post, section = section_owners[segment.sec.name()]
            key = f"{pre}:{post}"
            check(float(syn.gmax), expected["synaptic_gmax"][key], f"{key} synaptic gmax")
            kinetics = ("tau_r_AMPA", "tau_d_AMPA", "tau_r_NMDA", "tau_d_NMDA") if "PYR" in pre else ("tau_r", "tau_d")
            invariant = {k: float(getattr(syn, k)) for k in (*kinetics, "e", "Dep", "Fac", "Use", "u0")}
            update(topology, [source, gid, section, float(segment.x), float(nc.weight[0]), float(nc.delay), invariant])
            counts["synapses"] += 1
            checked_groups[key] = checked_groups.get(key, 0) + 1
    except Exception as exc:
        errors.append(f"rank {rank} build audit raised {exc!r}")
    rows = comm.gather({"rank": rank, "topology_sha256": topology.hexdigest(),
                        "background_setup_sha256": background.hexdigest(),
                        "counts": counts, "checked_synaptic_groups": checked_groups, "errors": errors}, root=0)
    if rank != 0:
        return None
    audit_errors = [error for row in rows for error in row["errors"]]
    if bool(cfg.analysis.require_full_network):
        checked = {key for row in rows for key in row["checked_synaptic_groups"]}
        if checked != set(expected["connected_pairs"]):
            audit_errors.append("Full-network audit did not observe every expected recurrent population pair.")
        if any(sum(row["counts"][key] for row in rows) <= 0 for key in counts):
            audit_errors.append("Full-network synapse/tonic/OU audit was empty.")
    return {"errors": audit_errors,
            "expected_parameters": expected, "by_rank": rows,
            "invariant_sha256": canonical_json_sha256([{k: v for k, v in row.items() if k != "errors"} for row in rows]),
            "interpretation": "Actual gmax/tonic values checked. Matching seeds, topology, OU parameters, and prescribed input events support common random inputs. Legacy OU generator state/paths are not inspected; G1A independently established deterministic reconstruction."}


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    run_replay(cfg, validator=validate_configuration, report_name=REPORT_NAME,
               trace_name=TRACE_NAME, build_audit=construction_audit, abort_on_failure=True,
               scope="G1B paired reference/MDD 28-s zero-field pilot; two structures, no confirmatory inference.",
               limitations=["Circuit seeds, not windows, are the independent units; this pilot has only two structures.",
                            "Fixed MPI decomposition is required for matching the rank-local random realization.",
                            "Ideal neural-only EEG; no extracellular stimulation in G1B.",
                            "Historical one-off subset stimulus after 4000 ms is retained in full mode."])


if __name__ == "__main__":
    main()
