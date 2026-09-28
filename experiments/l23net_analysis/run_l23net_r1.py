"""R1 replication: reuse the G1B-audited integrator and construction checks."""
from pathlib import Path
import sys

import hydra

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.l23net_analysis.r1_protocol import COHORTS, REPORT_NAME, TRACE_NAME
from experiments.l23net_analysis.run_l23net_g1b import construction_audit, validate_configuration as validate_g1b
from experiments.l23net_analysis.run_l23net_no_field_replay import run_replay


def validate_configuration(cfg, mpi_size):
    cohort = str(cfg.analysis.cohort)
    if cohort not in COHORTS or int(cfg.experiment.seed) not in COHORTS[cohort]:
        raise ValueError("R1 requires a prespecified seed in its declared cohort.")
    if cohort == "calibration" and str(cfg.analysis.condition) != "reference":
        raise ValueError("Calibration structures are reference-only.")
    return validate_g1b(cfg, mpi_size, allowed_seeds=COHORTS[cohort])


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg):
    run_replay(cfg, validator=validate_configuration, report_name=REPORT_NAME,
               trace_name=TRACE_NAME, build_audit=construction_audit, abort_on_failure=True,
               scope="R1 prospective reference/MDD replication; separate reference calibration; no tACS.",
               limitations=["Structure seed, not EEG window, is the inferential unit.",
                            "Fixed 624-rank decomposition and historical 4-s internal event retained.",
                            "Ideal neural-only EEG and model-specific inhibition intervention; not clinical evidence.",
                            "OU paths are not recorded; matched construction and deterministic replay support pairing."])


if __name__ == "__main__":
    main()
