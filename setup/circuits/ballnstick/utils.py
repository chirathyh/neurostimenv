"""Construct the excitatory–inhibitory ball-and-stick network."""

from __future__ import annotations

import sys

import neuron
import numpy as np
import scipy.stats as st
from decouple import config
from LFPy import NetworkCell, Synapse


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)


def generate_poisson_spike_train(
    *,
    start_ms: float,
    stop_ms: float,
    interval_ms: float,
    seed: int,
) -> np.ndarray:
    """Generate a homogeneous Poisson spike train.

    Inter-spike intervals are sampled independently from an exponential
    distribution. The expected firing rate is:

        rate_hz = 1000 / interval_ms

    Parameters
    ----------
    start_ms:
        Beginning of the spike-generation interval, in milliseconds.
    stop_ms:
        End of the spike-generation interval, in milliseconds.
    interval_ms:
        Mean inter-spike interval, in milliseconds.
    seed:
        Seed for this individual synapse.

    Returns
    -------
    np.ndarray
        Sorted spike times in milliseconds within [start_ms, stop_ms).
    """
    if interval_ms <= 0:
        raise ValueError(
            f"interval_ms must be positive, received {interval_ms}."
        )

    if stop_ms <= start_ms:
        raise ValueError(
            "stop_ms must be greater than start_ms: "
            f"start_ms={start_ms}, stop_ms={stop_ms}."
        )

    # Local generator: this does not modify NumPy's global random state.
    rng = np.random.default_rng(seed)

    spike_times: list[float] = []
    current_time_ms = float(start_ms)

    while True:
        inter_spike_interval = rng.exponential(scale=interval_ms)
        current_time_ms += float(inter_spike_interval)

        if current_time_ms >= stop_ms:
            break

        spike_times.append(current_time_ms)

    return np.asarray(spike_times, dtype=np.float64)


def generate_sinusoidally_modulated_poisson_spike_train(
    *,
    start_ms: float,
    stop_ms: float,
    interval_ms: float,
    seed: int,
    modulation_depth: float,
    frequency_hz: float,
    phase_rad: float = 0.0,
    thinning_envelope_modulation_depth: float | None = None,
    phase_path_times_ms: np.ndarray | None = None,
    phase_path_rad: np.ndarray | None = None,
) -> np.ndarray:
    """Generate an inhomogeneous Poisson train with sinusoidal rate.

    The instantaneous rate is

    ``lambda(t) = lambda_0 * (1 + m * sin(2*pi*f*t + phase))``,

    where ``lambda_0 = 1000 / interval_ms`` in Hz and ``m`` is the
    dimensionless modulation depth.  Events remain stochastic and are
    generated with exact Poisson thinning; this function does not prescribe
    postsynaptic spike times.

    A caller may instead provide a continuous, unwrapped latent phase path.
    This is used for the shared phase-diffusion state: synapses share the
    instantaneous afferent rate phase, but retain independent candidate-event
    and acceptance draws through their private ``seed`` values.

    ``thinning_envelope_modulation_depth`` may be held fixed across two
    conditions.  With the same seed, this gives them a common candidate-event
    process and common acceptance uniforms while allowing their accepted event
    trains to differ according to modulation depth.  This common-random-number
    coupling reduces comparison noise without changing either marginal point
    process.
    """
    values = np.asarray(
        [
            start_ms,
            stop_ms,
            interval_ms,
            modulation_depth,
            frequency_hz,
            phase_rad,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Modulated Poisson parameters must be finite.")
    if interval_ms <= 0.0:
        raise ValueError(
            f"interval_ms must be positive, received {interval_ms}."
        )
    if stop_ms <= start_ms:
        raise ValueError(
            "stop_ms must be greater than start_ms: "
            f"start_ms={start_ms}, stop_ms={stop_ms}."
        )
    if modulation_depth < 0.0 or modulation_depth > 1.0:
        raise ValueError("modulation_depth must be in [0, 1].")
    if frequency_hz < 0.0 or (modulation_depth > 0.0 and frequency_hz == 0.0):
        raise ValueError(
            "frequency_hz must be positive when modulation_depth is nonzero."
        )

    envelope_depth = (
        float(modulation_depth)
        if thinning_envelope_modulation_depth is None
        else float(thinning_envelope_modulation_depth)
    )
    if not np.isfinite(envelope_depth):
        raise ValueError("thinning_envelope_modulation_depth must be finite.")
    if envelope_depth < modulation_depth or envelope_depth > 1.0:
        raise ValueError(
            "thinning_envelope_modulation_depth must be in "
            "[modulation_depth, 1]."
        )

    path_supplied = phase_path_times_ms is not None or phase_path_rad is not None
    if path_supplied:
        if phase_path_times_ms is None or phase_path_rad is None:
            raise ValueError(
                "phase_path_times_ms and phase_path_rad must be provided together."
            )
        path_times = np.asarray(phase_path_times_ms, dtype=np.float64)
        path_phase = np.asarray(phase_path_rad, dtype=np.float64)
        if (
            path_times.ndim != 1
            or path_phase.ndim != 1
            or path_times.size != path_phase.size
            or path_times.size < 2
            or not np.all(np.isfinite(path_times))
            or not np.all(np.isfinite(path_phase))
            or not np.all(np.diff(path_times) > 0.0)
        ):
            raise ValueError("The latent phase path must be finite and strictly ordered.")
        tolerance_ms = 1.0e-9
        if (
            path_times[0] > float(start_ms) + tolerance_ms
            or path_times[-1] < float(stop_ms) - tolerance_ms
        ):
            raise ValueError("The latent phase path must cover the event interval.")
        path_steps = np.diff(path_times)
        if not np.allclose(path_steps, path_steps[0], rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("The latent phase path must use a uniform time grid.")
        path_start_ms = float(path_times[0])
        path_step_ms = float(path_steps[0])
        path_last_interval = int(path_times.size - 2)
    else:
        path_times = None
        path_phase = None
        path_start_ms = 0.0
        path_step_ms = 1.0
        path_last_interval = 0

    # Preserve the legacy homogeneous generator exactly when no common
    # thinning envelope has been requested.
    if modulation_depth == 0.0 and envelope_depth == 0.0:
        return generate_poisson_spike_train(
            start_ms=start_ms,
            stop_ms=stop_ms,
            interval_ms=interval_ms,
            seed=seed,
        )

    rng = np.random.default_rng(seed)
    candidate_interval_ms = interval_ms / (1.0 + envelope_depth)
    denominator = 1.0 + envelope_depth
    omega_per_ms = 2.0 * np.pi * frequency_hz / 1000.0
    accepted_times: list[float] = []
    current_time_ms = float(start_ms)

    while True:
        current_time_ms += float(
            rng.exponential(scale=candidate_interval_ms)
        )
        if current_time_ms >= stop_ms:
            break
        if path_supplied:
            # The phase path is uniform. Direct scalar interpolation avoids the
            # large overhead of calling np.interp millions of times while
            # constructing all private afferent event trains.
            position = (current_time_ms - path_start_ms) / path_step_ms
            left = min(max(int(np.floor(position)), 0), path_last_interval)
            fraction = min(max(position - left, 0.0), 1.0)
            instantaneous_phase = float(
                path_phase[left]
                + fraction * (path_phase[left + 1] - path_phase[left])
            )
        else:
            instantaneous_phase = omega_per_ms * current_time_ms + phase_rad
        relative_rate = (
            1.0
            + modulation_depth
            * np.sin(instantaneous_phase)
        ) / denominator
        if rng.random() < relative_rate:
            accepted_times.append(current_time_ms)

    return np.asarray(accepted_times, dtype=np.float64)


def make_background_phase_seed(*, global_seed: int) -> int:
    """Derive the seed of the population-shared afferent phase process.

    This namespace is deliberately separate from all private synapse seeds.
    The same returned seed is used by E and I when their rhythm parameters
    match, making the nonstationary phase a shared afferent state rather than
    an accidental collection of unrelated per-synapse oscillators.
    """
    seed_sequence = np.random.SeedSequence(
        [int(global_seed), 0x50484446]  # ASCII-like namespace: ``PHDF``.
    )
    return int(seed_sequence.generate_state(n_words=1, dtype=np.uint32)[0])


def generate_phase_diffusion_path(
    *,
    start_ms: float,
    stop_ms: float,
    frequency_hz: float,
    phase_rad: float,
    diffusion_rad2_per_s: float,
    integration_dt_ms: float,
    history_seed: int,
    future_start_ms: float | None = None,
    future_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a continuous shared phase trajectory for afferent drive.

    The latent state follows the Euler-exact increments of a Wiener phase
    oscillator,

    ``d phi = 2*pi*f*dt + sqrt(2*D)*dW``,

    where ``D`` is in radian squared per second.  Thus phase-increment variance
    over a lag ``tau`` is ``2*D*tau`` and ideal phase coherence decays as
    ``exp(-D*tau)``.  The returned phase is unwrapped so interpolation is safe.

    When an independent future is requested, the value at the decision
    boundary is inherited continuously from the history and only subsequent
    Wiener increments use ``future_seed``.  This preserves identical observed
    histories across counterfactual actions.
    """
    values = np.asarray(
        [start_ms, stop_ms, frequency_hz, phase_rad, diffusion_rad2_per_s,
         integration_dt_ms],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Phase-diffusion parameters must be finite.")
    if stop_ms <= start_ms:
        raise ValueError("stop_ms must be greater than start_ms.")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive.")
    if diffusion_rad2_per_s < 0.0:
        raise ValueError("diffusion_rad2_per_s must be non-negative.")
    if integration_dt_ms <= 0.0:
        raise ValueError("integration_dt_ms must be positive.")
    if (future_start_ms is None) != (future_seed is None):
        raise ValueError("future_start_ms and future_seed must be provided together.")

    duration_steps = (float(stop_ms) - float(start_ms)) / float(integration_dt_ms)
    n_steps = int(round(duration_steps))
    if not np.isclose(duration_steps, n_steps, atol=1.0e-10):
        raise ValueError(
            "The phase-diffusion interval must contain an integer number of steps."
        )
    times_ms = float(start_ms) + np.arange(n_steps + 1) * float(integration_dt_ms)

    split_step: int | None = None
    if future_start_ms is not None:
        split_position = (
            (float(future_start_ms) - float(start_ms)) / float(integration_dt_ms)
        )
        split_step = int(round(split_position))
        if (
            not np.isclose(split_position, split_step, atol=1.0e-10)
            or split_step <= 0
            or split_step >= n_steps
        ):
            raise ValueError(
                "future_start_ms must be an interior phase-grid boundary."
            )

    dt_s = float(integration_dt_ms) / 1000.0
    deterministic_increment = 2.0 * np.pi * float(frequency_hz) * dt_s
    noise_scale = np.sqrt(2.0 * float(diffusion_rad2_per_s) * dt_s)
    increments = np.full(n_steps, deterministic_increment, dtype=np.float64)
    if noise_scale > 0.0:
        history_rng = np.random.default_rng(int(history_seed))
        history_count = n_steps if split_step is None else split_step
        increments[:history_count] += noise_scale * history_rng.standard_normal(
            history_count
        )
        if split_step is not None:
            future_rng = np.random.default_rng(int(future_seed))
            increments[split_step:] += noise_scale * future_rng.standard_normal(
                n_steps - split_step
            )

    phase = np.empty(n_steps + 1, dtype=np.float64)
    phase[0] = float(phase_rad) + 2.0 * np.pi * float(frequency_hz) * (
        float(start_ms) / 1000.0
    )
    phase[1:] = phase[0] + np.cumsum(increments)
    return times_ms, phase


def make_background_synapse_seed(
    *,
    global_seed: int,
    population_index: int,
    cell_identifier: int,
    synapse_index: int,
) -> int:
    """Derive a reproducible seed for one background synapse.

    The seed depends on the experiment seed, population, cell and synapse.
    Consequently, different synapses receive different event trains while
    an experiment remains reproducible.
    """
    seed_sequence = np.random.SeedSequence(
        [
            int(global_seed),
            int(population_index),
            int(cell_identifier),
            int(synapse_index),
        ]
    )

    return int(
        seed_sequence.generate_state(
            n_words=1,
            dtype=np.uint32,
        )[0]
    )


def generate_split_background_spike_train(
    *,
    start_ms: float,
    stop_ms: float,
    interval_ms: float,
    history_seed: int,
    future_start_ms: float | None = None,
    future_seed: int | None = None,
    rhythm_enabled: bool = False,
    modulation_depth: float = 0.0,
    frequency_hz: float = 0.0,
    phase_rad: float = 0.0,
    thinning_envelope_modulation_depth: float | None = None,
    phase_path_times_ms: np.ndarray | None = None,
    phase_path_rad: np.ndarray | None = None,
) -> np.ndarray:
    """Generate background events with an optional independent future stream.

    With no split, this dispatches to the historical generators exactly.  With
    a split, events before ``future_start_ms`` depend only on ``history_seed``
    and events at or after it depend only on ``future_seed``.  A Poisson
    process has independent increments, so restarting the generator at a fixed
    decision boundary preserves the intended marginal process while allowing
    repeated stochastic futures to share an identical observed history.
    """
    if (future_start_ms is None) != (future_seed is None):
        raise ValueError(
            "future_start_ms and future_seed must be provided together."
        )

    def generate(part_start_ms: float, part_stop_ms: float, seed: int) -> np.ndarray:
        if rhythm_enabled:
            return generate_sinusoidally_modulated_poisson_spike_train(
                start_ms=part_start_ms,
                stop_ms=part_stop_ms,
                interval_ms=interval_ms,
                seed=seed,
                modulation_depth=modulation_depth,
                frequency_hz=frequency_hz,
                phase_rad=phase_rad,
                thinning_envelope_modulation_depth=(
                    thinning_envelope_modulation_depth
                ),
                phase_path_times_ms=phase_path_times_ms,
                phase_path_rad=phase_path_rad,
            )
        return generate_poisson_spike_train(
            start_ms=part_start_ms,
            stop_ms=part_stop_ms,
            interval_ms=interval_ms,
            seed=seed,
        )

    if future_start_ms is None:
        return generate(float(start_ms), float(stop_ms), int(history_seed))

    split = float(future_start_ms)
    if not float(start_ms) < split < float(stop_ms):
        raise ValueError(
            "future_start_ms must lie strictly inside the generation interval."
        )
    history = generate(float(start_ms), split, int(history_seed))
    future = generate(split, float(stop_ms), int(future_seed))
    return np.concatenate((history, future))


def setup_network_ballnstick(network, args, MPI_VAR) -> None:
    """Create and connect the ball-and-stick E/I network."""
    global_seed = int(MPI_VAR["GLOBALSEED"])
    future_global_seed = MPI_VAR.get("FUTUREGLOBALSEED")
    future_start_ms = MPI_VAR.get("FUTURESTARTMS")
    if (future_global_seed is None) != (future_start_ms is None):
        raise ValueError(
            "FUTUREGLOBALSEED and FUTURESTARTMS must be provided together."
        )
    mpi_rank = int(MPI_VAR["RANK"])

    # ------------------------------------------------------------------
    # Cell and population configuration
    # ------------------------------------------------------------------

    cell_parameters = {
        "morphology": (
            f"{MAIN_PATH}/setup/circuits/ballnstick/BallAndStick.hoc"
        ),
        "templatefile": (
            f"{MAIN_PATH}/setup/circuits/ballnstick/"
            "BallAndStickTemplate.hoc"
        ),
        "templatename": "BallAndStickTemplate",
        "templateargs": None,
        "delete_sections": False,
        "dt": args.env.network.dt,
        "tstop": args.env.simulation.duration,
        "verbose": False,
    }

    population_cfg = args.env.network.population
    population_parameters = {
        "Cell": NetworkCell,
        "cell_args": cell_parameters,
        "pop_args": {
            "radius": float(population_cfg.radius_um),
            "loc": float(population_cfg.z_mean_um),
            "scale": float(population_cfg.z_sd_um),
        },
        "rotation_args": {
            "x": float(population_cfg.rotation_x_rad),
            "y": float(population_cfg.rotation_y_rad),
        },
    }

    population_names = ["E", "I"]
    population_sizes = [
        int(population_cfg.sizes.E),
        int(population_cfg.sizes.I),
    ]
    if any(size <= 0 for size in population_sizes):
        raise ValueError("BallAndStick E and I population sizes must be positive.")

    # Lazily populated below. Matching E/I rhythm configurations reuse the
    # same latent phase path while every synapse keeps a private event stream.
    shared_phase_paths: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}

    # Rows are presynaptic populations; columns are postsynaptic:
    #
    #     [[E -> E, E -> I],
    #      [I -> E, I -> I]]
    probability_cfg = args.env.network.connection_probability
    connection_probability = [
        [float(probability_cfg.ee), float(probability_cfg.ei)],
        [float(probability_cfg.ie), float(probability_cfg.ii)],
    ]
    if any(
        probability < 0.0 or probability > 1.0
        for row in connection_probability
        for probability in row
    ):
        raise ValueError("All recurrent connection probabilities must be in [0, 1].")

    # ------------------------------------------------------------------
    # Recurrent synapse configuration
    # ------------------------------------------------------------------

    synapse_model = neuron.h.Exp2Syn

    kinetics = args.env.network.synapse_kinetics
    excitatory_kinetics = {
        "tau1": float(kinetics.excitatory.tau1_ms),
        "tau2": float(kinetics.excitatory.tau2_ms),
        "e": float(kinetics.excitatory.reversal_mV),
    }
    inhibitory_kinetics = {
        "tau1": float(kinetics.inhibitory.tau1_ms),
        "tau2": float(kinetics.inhibitory.tau2_ms),
        "e": float(kinetics.inhibitory.reversal_mV),
    }
    synapse_parameters = [
        [
            # E -> E
            dict(excitatory_kinetics),
            # E -> I
            dict(excitatory_kinetics),
        ],
        [
            # I -> E
            dict(inhibitory_kinetics),
            # I -> I
            dict(inhibitory_kinetics),
        ],
    ]

    recurrent_weights = args.env.network.recurrent_weights
    recurrent_cfg = args.env.network.recurrent
    inhibition_scale = float(args.env.network.inhibition_scale)
    weight_cv = float(recurrent_cfg.weight_cv)

    if inhibition_scale < 0 or weight_cv < 0:
        raise ValueError(
            "network.inhibition_scale and recurrent.weight_cv must be "
            "non-negative."
        )

    # Scale only I -> E connections in the primary reduced-inhibition
    # experiment. I -> I remains unchanged.
    weight_arguments = [
        [
            {
                "loc": float(recurrent_weights.ee_mean),
                "scale": weight_cv * float(recurrent_weights.ee_mean),
            },
            {
                "loc": float(recurrent_weights.ei_mean),
                "scale": weight_cv * float(recurrent_weights.ei_mean),
            },
        ],
        [
            {
                "loc": (
                    float(recurrent_weights.ie_mean)
                    * inhibition_scale
                ),
                "scale": (
                    weight_cv
                    * float(recurrent_weights.ie_mean)
                    * inhibition_scale
                ),
            },
            {
                "loc": float(recurrent_weights.ii_mean),
                "scale": weight_cv * float(recurrent_weights.ii_mean),
            },
        ],
    ]

    weight_function = np.random.normal
    minimum_weight = float(recurrent_cfg.minimum_weight_uS)

    delay_function = np.random.normal
    delay_mean_ms = float(recurrent_cfg.delay_mean_ms)
    delay_sd_ms = float(recurrent_cfg.delay_sd_ms)
    delay_arguments = [
        [
            {"loc": delay_mean_ms, "scale": delay_sd_ms},
            {"loc": delay_mean_ms, "scale": delay_sd_ms},
        ],
        [
            {"loc": delay_mean_ms, "scale": delay_sd_ms},
            {"loc": delay_mean_ms, "scale": delay_sd_ms},
        ],
    ]
    minimum_delay = float(recurrent_cfg.minimum_delay_ms)

    multapse_function = np.random.normal
    multapse_mean = recurrent_cfg.multapse_mean
    multapse_sd = recurrent_cfg.multapse_sd
    multapse_arguments = [
        [
            {"loc": float(multapse_mean.ee), "scale": float(multapse_sd.ee)},
            {"loc": float(multapse_mean.ei), "scale": float(multapse_sd.ei)},
        ],
        [
            {"loc": float(multapse_mean.ie), "scale": float(multapse_sd.ie)},
            {"loc": float(multapse_mean.ii), "scale": float(multapse_sd.ii)},
        ],
    ]

    excitatory_location = recurrent_cfg.excitatory_location
    inhibitory_location = recurrent_cfg.inhibitory_location
    synapse_position_arguments = [
        [
            {
                "section": ["soma", "apic"],
                "fun": [st.norm, st.norm],
                "funargs": [
                    {
                        "loc": float(excitatory_location.soma_z_mean_um),
                        "scale": float(excitatory_location.z_sd_um),
                    },
                    {
                        "loc": float(excitatory_location.apic_z_mean_um),
                        "scale": float(excitatory_location.z_sd_um),
                    },
                ],
                "funweights": [
                    float(excitatory_location.soma_weight),
                    float(excitatory_location.apic_weight),
                ],
            }
            for _ in range(2)
        ],
        [
            {
                "section": ["soma", "apic"],
                "fun": [st.norm, st.norm],
                "funargs": [
                    {
                        "loc": float(inhibitory_location.soma_z_mean_um),
                        "scale": float(inhibitory_location.z_sd_um),
                    },
                    {
                        "loc": float(inhibitory_location.apic_z_mean_um),
                        "scale": float(inhibitory_location.z_sd_um),
                    },
                ],
                "funweights": [
                    float(inhibitory_location.soma_weight),
                    float(inhibitory_location.apic_weight),
                ],
            }
            for _ in range(2)
        ],
    ]

    # ------------------------------------------------------------------
    # Create populations and population-specific background activity
    # ------------------------------------------------------------------

    for population_index, (population_name, population_size) in enumerate(
        zip(population_names, population_sizes)
    ):
        network.create_population(
            name=population_name,
            POP_SIZE=population_size,
            **population_parameters,
        )

        if not args.env.network.syn_activity:
            continue

        # E and I may now receive different external-drive parameters.
        background = args.env.network.background[population_name]

        n_background_synapses = int(background.n_synapses)
        background_weight = float(background.weight)
        background_interval_ms = float(background.interval_ms)

        # Population-specific start and stop values are optional.
        background_start_ms = float(
            background.get(
                "start_ms",
                0.0,
            )
        )
        background_stop_ms = float(
            background.get(
                "stop_ms",
                args.env.simulation.duration,
            )
        )

        if n_background_synapses < 0:
            raise ValueError(
                f"background.{population_name}.n_synapses must be "
                f"non-negative, received {n_background_synapses}."
            )

        if background_weight < 0:
            raise ValueError(
                f"background.{population_name}.weight must be "
                f"non-negative, received {background_weight}."
            )

        for local_cell_index, cell in enumerate(
            network.populations[population_name].cells
        ):
            segment_indices = cell.get_rand_idx_area_norm(
                section="allsec",
                nidx=n_background_synapses,
            )

            # LFPy cells normally expose a global gid. The fallback remains
            # unique across MPI ranks if gid is unavailable.
            raw_gid = getattr(cell, "gid", None)

            if raw_gid is not None:
                cell_identifier = int(raw_gid)
            else:
                cell_identifier = (
                    mpi_rank * 1_000_000
                    + local_cell_index
                )

            for synapse_index, segment_index in enumerate(segment_indices):
                background_synapse = Synapse(
                    cell=cell,
                    idx=int(segment_index),
                    syntype="Exp2Syn",
                    weight=background_weight,
                    tau1=float(background.tau1_ms),
                    tau2=float(background.tau2_ms),
                    e=float(background.reversal_mV),
                )

                synapse_seed = make_background_synapse_seed(
                    global_seed=global_seed,
                    population_index=population_index,
                    cell_identifier=cell_identifier,
                    synapse_index=synapse_index,
                )

                future_synapse_seed = (
                    None
                    if future_global_seed is None
                    else make_background_synapse_seed(
                        global_seed=int(future_global_seed),
                        population_index=population_index,
                        cell_identifier=cell_identifier,
                        synapse_index=synapse_index,
                    )
                )

                rhythm = background.get("rhythm", None)
                rhythm_enabled = rhythm is not None and bool(
                    rhythm.get("enabled", False)
                )
                diffusion = (
                    float(rhythm.get("phase_diffusion_rad2_per_s", 0.0))
                    if rhythm_enabled else 0.0
                )
                if diffusion < 0.0:
                    raise ValueError(
                        "phase_diffusion_rad2_per_s must be non-negative."
                    )
                phase_path_times_ms = None
                phase_path_rad = None
                if diffusion > 0.0:
                    phase_dt_ms = float(
                        rhythm.get("phase_diffusion_integration_dt_ms", 1.0)
                    )
                    path_key = (
                        background_start_ms,
                        background_stop_ms,
                        float(rhythm.frequency_hz),
                        float(rhythm.get("phase_rad", 0.0)),
                        diffusion,
                        phase_dt_ms,
                        -1.0 if future_start_ms is None else float(future_start_ms),
                    )
                    if path_key not in shared_phase_paths:
                        phase_history_seed = make_background_phase_seed(
                            global_seed=global_seed
                        )
                        phase_future_seed = (
                            None if future_global_seed is None
                            else make_background_phase_seed(
                                global_seed=int(future_global_seed)
                            )
                        )
                        shared_phase_paths[path_key] = generate_phase_diffusion_path(
                            start_ms=background_start_ms,
                            stop_ms=background_stop_ms,
                            frequency_hz=float(rhythm.frequency_hz),
                            phase_rad=float(rhythm.get("phase_rad", 0.0)),
                            diffusion_rad2_per_s=diffusion,
                            integration_dt_ms=phase_dt_ms,
                            history_seed=phase_history_seed,
                            future_start_ms=(
                                None if future_start_ms is None
                                else float(future_start_ms)
                            ),
                            future_seed=phase_future_seed,
                        )
                    phase_path_times_ms, phase_path_rad = shared_phase_paths[path_key]
                spike_times = generate_split_background_spike_train(
                    start_ms=background_start_ms,
                    stop_ms=background_stop_ms,
                    interval_ms=background_interval_ms,
                    history_seed=synapse_seed,
                    future_start_ms=(
                        None if future_start_ms is None else float(future_start_ms)
                    ),
                    future_seed=future_synapse_seed,
                    rhythm_enabled=rhythm_enabled,
                    modulation_depth=(
                        float(rhythm.modulation_depth) if rhythm_enabled else 0.0
                    ),
                    frequency_hz=(
                        float(rhythm.frequency_hz) if rhythm_enabled else 0.0
                    ),
                    phase_rad=(
                        float(rhythm.get("phase_rad", 0.0))
                        if rhythm_enabled else 0.0
                    ),
                    thinning_envelope_modulation_depth=(
                        float(rhythm.get(
                            "thinning_envelope_modulation_depth",
                            rhythm.modulation_depth,
                        )) if rhythm_enabled else None
                    ),
                    phase_path_times_ms=phase_path_times_ms,
                    phase_path_rad=phase_path_rad,
                )

                background_synapse.set_spike_times(spike_times)

    # ------------------------------------------------------------------
    # Create recurrent network connections
    # ------------------------------------------------------------------

    for presynaptic_index, presynaptic_name in enumerate(population_names):
        for postsynaptic_index, postsynaptic_name in enumerate(
            population_names
        ):
            connectivity = network.get_connectivity_rand(
                pre=presynaptic_name,
                post=postsynaptic_name,
                connprob=connection_probability[
                    presynaptic_index
                ][postsynaptic_index],
            )

            network.connect(
                pre=presynaptic_name,
                post=postsynaptic_name,
                connectivity=connectivity,
                syntype=synapse_model,
                synparams=synapse_parameters[
                    presynaptic_index
                ][postsynaptic_index],
                weightfun=weight_function,
                weightargs=weight_arguments[
                    presynaptic_index
                ][postsynaptic_index],
                minweight=minimum_weight,
                delayfun=delay_function,
                delayargs=delay_arguments[
                    presynaptic_index
                ][postsynaptic_index],
                mindelay=minimum_delay,
                multapsefun=multapse_function,
                multapseargs=multapse_arguments[
                    presynaptic_index
                ][postsynaptic_index],
                syn_pos_args=synapse_position_arguments[
                    presynaptic_index
                ][postsynaptic_index],
                save_connections=False,
            )
