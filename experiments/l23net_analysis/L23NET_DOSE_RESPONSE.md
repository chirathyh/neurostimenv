# L23Net tACS dose response on Gadi

This study tests whether varying amplitude at a fixed 14 Hz and +z direction
improves the prespecified three-band EEG outcome relative to matched sham. The
existing data audit supports screening for elevated theta, alpha and low-beta
power. The 14 Hz setting serves as a fixed probe motivated by the earlier
discovery and spike-timing results. The previous 0.4 V/m held-out pilot missed
its EEG efficacy criteria, and individual carrier and phase targeting remain
unqualified.

## Design and decision rules

| Stage | Frozen seeds | Arms (V/m, peak tissue field) | Runs |
|---|---|---|---:|
| Upper-dose qualification | 8601 | 0, 0.8 | 2 |
| Dose discovery | 8611–8616 | 0, 0.2, 0.4, 0.8 | 24 |

One sham trajectory serves all three dose comparisons within each discovery
seed. Each arm independently replays the same circuit and random-stream settings. The
audit checks exact prestimulation EEG, dipole and spike histories, seed and
construction fingerprints, raw field delivery, committed data and final PBS
accounting. The discovery phase uses the original S0 screen and baseline rate
guards, with ineligible arms mapped to sham and all six candidates recorded.
Qualification bypasses the phenotype screen to test actual 0.8 V/m delivery.
It retains the baseline rate guards and requires paired plateau/washout mean
rates within 20% of the sham rate (denominator floor 0.1 Hz) and within the
original population bounds.

The full network has 1,000 neurons at 34 °C with `MDD=true`, `DRUG=false`,
`dt=0.025 ms` and the original synaptic activity. Each trajectory runs 60 s on
624 ranks and retains the inherited input near 4 s.

| Interval (s) | Use |
|---|---|
| 0–8 | Burn-in and recovery excluded from EEG features |
| 8–28 | Stimulation-free phenotype and rate screen |
| 28–29 and 49–50 | Onset and offset ramps |
| 29–49 | Primary constant-amplitude plateau |
| 39–49 | Prespecified late secondary within the plateau |
| 50–60 | Zero-field washout |

The primary effect is sham minus active three-band distance to the frozen
Reference60 target. The late secondary uses independently calibrated Healthy
features from the same 16 Healthy structures over 39–49 s. Its scale is
window-specific. Record per-band changes, carrier-excluded distance, alignment
towards the reference, population rates, and washout separately. The timing
analysis compares matched-cell spike-to-field phase consistency across
separated one-second blocks, with two-second and circular interspike-interval
permutation checks. SST is the mechanistic readout of interest. The circuit
seed supplies the independent unit for inference.

Selection for independent confirmation requires at least five eligible seeds,
positive full-plateau benefit in at least five seeds, mean benefit of at least
0.2, positive means under every leave-one-out check, positive carrier-excluded
benefit and alignment, and all paired rate guards. Rank passing arms by mean
primary benefit, favoring the lower dose on an exact tie. The summary also
reports direct paired active-dose comparisons, confidence intervals, and Holm
correction within the reported test families. The six-seed experiment is dose
discovery. An independent cohort is required for confirmation.

## Quota and scheduling

Every worker requests `normal`, 624 CPUs, 256 GB and two hours on `fa32`.
The checkout, environment and output use `gdata/ny83` storage. The launcher
checks the live `fa32` balance before each submission and keeps 1 KSU
unreserved. Eight workers can run at once by default. Dependent workers become
eligible when the preceding job in their chain succeeds. Queue wait depends
on Gadi scheduling.

| Stage | Cost at historical 1.645 KSU/run | Maximum reservation |
|---|---:|---:|
| Qualification | 3.29 KSU | 4.992 KSU |
| Discovery | 39.48 KSU | 59.904 KSU |
| Total | 42.77 KSU | 64.896 KSU |

The estimates use the completed 60-second S2 jobs. New amplitudes may change
runtime. The maximum uses the `normal` rate of 2 SU per allocated CPU-hour.
At a starting balance of 200 KSU this leaves at least 135.104 KSU after both
stages, before any other spending. Use the discovery result and its uncertainty
to size any independent confirmation. [Gadi queue limits and charging](https://opus.nci.org.au/spaces/Help/pages/90308823/Queue+Limits)

## Run the qualification pair

From the Gadi login node, use the existing checkout and Reference60
environment. Check for jobs using the checkout before pulling.

```bash
cd /g/data/ny83/ch9972/NeuroStim/neurostimenv
qstat -u "$USER"
git switch feature/l23net-sinusoidal-uniform-field
git pull --ff-only origin feature/l23net-sinusoidal-uniform-field
module load python3/3.10.4 openmpi/5.0.5
source /g/data/ny83/ch9972/NeuroStim/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -m pip check
nci_account -P fa32
python3 experiments/l23net_analysis/nci/submit_l23net_dose_response.py --stage qualification
```

Review the dry-run plan. Submit the two workers from the same checkout.

```bash
python3 experiments/l23net_analysis/nci/submit_l23net_dose_response.py --stage qualification --submit
```

The submitter locates the completed Reference60 suite automatically. If it was
moved, pass `--reference` with the directory containing `reference60_target.json`,
`reference60_summary.json`, `s0_gate.json` and `qualification.json`. The completed
S2 pilot directory also carries those original frozen files. Job IDs and the
source reference directory are recorded in `submission.json`.

## Run discovery after qualification

After both qualification jobs exit, run the analyzer. Discovery requires the
signed `qualified` result and the same checkout. If a PBS epilogue is still
being copied, rerun the analyzer when the job files finish arriving.

```bash
python3 experiments/l23net_analysis/analyze_l23net_dose_response.py --stage qualification
```

After the report prints `qualified`, review the discovery reservation and submit.

```bash
python3 experiments/l23net_analysis/nci/submit_l23net_dose_response.py --stage discovery
python3 experiments/l23net_analysis/nci/submit_l23net_dose_response.py --stage discovery --submit
qstat -u "$USER"
```

Pass the same `--max-concurrent` value to both commands if modifying the
default of eight. Use the fixed suite directories under `results/` to resume
monitoring from a new shell. Keep the checkout and environment fixed through
both stages. A partial submission keeps the job IDs already recorded. Inspect
the manifest and PBS state before taking further submission actions.

## Collect the result

```bash
python3 experiments/l23net_analysis/analyze_l23net_dose_response.py --stage discovery
```

Copy both complete directories, `results/l23net_dose_response_v1_qualification`
and `results/l23net_dose_response_v1_discovery`. Include all worker HDF5 files,
`submission.json`, `pbs.out`, `pbs.err`, `job_live.log` and `heartbeat.tsv`.
The qualification report is `upper_dose_qualification.json`. Discovery writes
`dose_response_summary.json` after all 24 runs and comparisons pass technical
audit. A null `candidate_for_independent_confirmation` is a completed negative
discovery result. For an interrupted job, use the manifest's job ID with
`qstat -fx` and retain the final PBS accounting and any `failure_rank_*.json`
files alongside the worker logs.
