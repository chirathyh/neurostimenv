# Implementation validation and limits

Base: `codex/eeg-evaluation-v6-immediate-information`, commit `ac4b93c8fe924091418b7bb499566ce8d27a83a5`.
New isolated branch: `codex/eeg-v6-eight-worker-reporting-pilot`.
No original user checkout edits or historical output files were changed.

## Native engineering evidence

The development journal records **21 actual NEURON invocations**, including failed attempts; one additional import failure occurred before NEURON entry and is separately disclosed. These were 2E/1I engineering networks, not the full biological pilot. They are debited against the 160-call request. No default 106-call workstation allocation has run locally.

The final three-call proof passed same-state future invariance and exact-forcing full-prefix replay for h0/h2/h10 W32, with zero observed target replay MSE. Capture covered 66 segments, 16 contacts, both extracellular layers and the actual emitting spike detector. The final tiny network had zero pending deliveries at its cutoff; separate pure tests verify lossless nonempty event queues and rejection of mismatches. This is not substituted for the full-network workstation queue/compatibility gate.

Two earlier repeated-process attempts segfaulted during NEURON initialization because a temporary Python bound-method hook created a reference cycle. The final implementation deletes the instance hook before cleanup and successfully completed repeated native calls in one process. The failures remain charged and documented.

The production ridge learning path was tested on a deterministic target involving previously omitted state fields: Exp2Syn A/B, drive phase, pending arrival time and connection weight. With 48 train,16 development,16 calibration and16 evaluation synthetic contexts, selected held-out MSE was `1.0462194381917794e-05`, versus `2.277372136176216` for the training mean. The retained predeclared alpha10 state candidate scored `0.06897884025991487`. This validates code and does not establish simulator prediction performance.

## Actual archived numerical integration

A local one-target run used the actual original A0 h10 W128 arrays and exact archived forecast representations. It exercised `fit_ordinary` followed by `score_models` for both the registered matched cohort and the broader historical cohort, with zero NEURON calls. It produced 24 information rows, two paired-history rows, 12 H3 rows and two H4 rows. A duplicate metadata-key integration failure was found, fixed, and covered by a fit/readout/score regression test.

Representative integration numbers (not new prospective evidence):

| Population | d8/B32 bank | Past reconstruction NMSE | Carrier CE, bits | Future error, mean=100 |
|---|---|---:|---:|---:|
| Matched: 2 eval circuits / 8 contexts | Feature18 | 0.9949558309464875 | 0.8530536898387275 | 100.0 |
| Matched | PCA | 0.9125773282652387 | 1.5845668889993212 | 100.0 |
| Matched | Forecast | 0.9955929642884144 | 1.58771925267155 | 100.08213075503127 |
| Historical: 8 eval circuits / 32 contexts | Feature18 | 0.9798648892396107 | 0.4075174981133347 | 99.7338170420986 |
| Historical | PCA | 0.8838404404983788 | 1.5835987271574659 | 99.67315316325822 |
| Historical | Forecast | 0.9729070428891947 | 1.5399259568790518 | 100.0 |

These quantities measure different tasks. The encoder's historical pretraining exposure differs from matched readout fitting and remains disclosed. This integration does not fill the still-unrun complete-state biological learner or H2 extension rows.

## Automated verification

The final handoff includes full test logs. Checks cover:

- Future support, h0 cutoff exclusion, Fourier/energy identities, acquisition causality and original-schema compatibility.
- Lossless complete-state coverage, future exclusion, exact ragged queue handling, training-only scaling and the production omitted-field learnable control.
- Development-only selection, retained nonmean candidate, separate calibration, physical risk contracts and paired-history/acquisition/offset differences.
- All60 actual state-fit artifact requirements before H2, failed-gate stopping behavior, eight-process allocation, process-safe budget accounting, failure charging and resume at the cap.
- Frozen inputs/settings and artifact checksums, including forecast/code/state adapters; stale and corrupt artifacts fail closed.
- Report metrics, unavailable values, all-page render/layout checks, and refusal to label a PDF as completed experimental evidence without genuine full-state results.
- Lossless NPY-byte export/restoration and pilot-archive reading; native/ordinary-weight omissions have hashes, complete-state model weights remain included, object arrays are refused.

The final V6 suite, including the new pilot checks, passed **154 tests**. Together with66 V4 and7 online tests, **227 regression tests passed**. The full V6 suite and focused pilot suites are simulation-free; tests that print native progress use fake generators. The existing V4 suite passed66 tests and online-stimulation suite passed7 tests. No core actuator/lifecycle behavior was changed, so additional expensive native polarization/reachability experiments were not launched for this reporting-only extension.

## Report review and remaining execution

A 16-page real-archive preview was compiled and rendered. Every page was checked for clipping, typography and table/figure layout; it is explicitly **incomplete**. A separate complete-layout engineering fixture tested all60 state-row completion requirements without representing synthetic values as experimental findings. The eventual workstation PDF is automatically rendered and checked; its actual results still deserve human review.

The laptop preflight observed fewer than the required8 GB of available RAM and therefore blocked eight-worker native execution without reducing workers. Noninteractive SSH through the workstation gateways failed authentication. No successful workstation connection, full-network eight-worker timing, complete-state pilot cohort fit, 48-future extension or CUDA pilot run is claimed.

The remaining experiment is performed by the single documented complete-pilot command. It must validate all original prefixes, full-network state inputs, measured concurrency and the full fitted-state gate before spending the H2 extension allocation. A failed criterion leaves an explicit failure record and partial evidence. All confirmation, power and automatic expansion remain deferred.
