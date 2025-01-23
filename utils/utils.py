import os
import sys
import shutil
import numpy as np
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
import warnings
warnings.filterwarnings('ignore')


def setup_folders(args):
    # create the folder which will save experiment data.
    LOG_DIR = args.experiment.dir
    CHECK_FOLDER = os.path.isdir(LOG_DIR)
    if CHECK_FOLDER:
        shutil.rmtree(LOG_DIR)
    else:
        os.makedirs(LOG_DIR)
    return args


def run_experiment(args, MPI_VAR, tACS=None):
    from utils.env import NeuronEnv
    import time
    env = NeuronEnv(args, MPI_VAR)
    tic_0 = time.perf_counter()
    obs, reward, done, info = env.step(action=tACS)
    if MPI_VAR['RANK'] == 0:
        print('Time for simulation step: ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes')
        print('n_step: ', str(info['sim_t']))
    MPI_VAR['COMM'].Barrier()
    env.close()
    return obs


def generate_spike_train(start=0., interval=10., number=1E3, noise=1.0):
        """
        Generates a spike train with deterministic or stochastic intervals.

        Parameters:
            start (float): Start time of the first spike in ms.
            interval (float): Mean time between spikes in ms.
            number (int): Total number of spikes.
            noise (float): Noise factor (0 for deterministic, 1 for fully random).

        Returns:
            np.ndarray: Array of spike times in ms.
        """
        # Generate deterministic spike times
        number = int(number)
        spike_times = start + np.arange(number) * interval

        # Add stochastic noise if noise > 0
        if noise > 0.0:
            jitter = np.random.exponential(scale=interval * noise, size=number)
            spike_times += jitter - np.mean(jitter)  # Center jitter to preserve mean interval

        return np.round(spike_times, decimals=3)  # Return rounded spike times for readability


def prep_stim_seq(action, step_size, steps, dt):
    t_start_stim, t_stop_stim = 0., step_size
    I_stim, t_ext = np.array([]), np.array([])
    for i in range(0, steps):
        # calc parameters based on amplitude and freq.
        amplitude = action[i][0] * 1e6  # LFPy accepts units nA, convert mA -> nA
        freq = action[i][1]
        total_cycles = freq * (step_size/1000)
        total_pulses = 2 * total_cycles

        # print("debug")
        # print(step_size)
        # print(total_pulses)

        pulse_width = int(step_size/total_pulses)

        I_stim_cur, t_ext_cur = set_rollout_current_pulses(n_pulses=total_pulses, biphasic=True,
                                                           width1=pulse_width, amp1=amplitude,
                                                           width2=pulse_width, amp2=-amplitude,
                                                           dt=dt, interpulse=0,
                                                           t_stop=t_stop_stim, t_start=t_start_stim)

        I_stim = np.concatenate((I_stim, I_stim_cur))
        t_ext = np.concatenate((t_ext, t_ext_cur))
        t_start_stim += step_size
        t_stop_stim += step_size
    return I_stim, t_ext


def set_rollout_current_pulses(amp1, width1, interpulse, t_stop, dt,
                                 biphasic=False, interphase=None, n_pulses=None, n_bursts=None, interburst=None, amp2=None,
                                 width2=None, t_start=None):
            '''
            Computes and sets pulsed currents on specified electrode.

            Parameters
            ----------
            el_id: int
                Electrode index
            amp1: float
                Amplitude of stimulation. If 'biphasic', amplitude of the first phase.
            width1: float
                Duration of the pulse in ms. If 'biphasic', duration of the first phase.
            interpulse: float
                Interpulse interval in ms
            t_stop: float
                Stop time in ms
            dt: float
                Sampling period in ms
            biphasic: bool
                If True, biphasic pulses are used
            amp2: float
                Amplitude of second phase stimulation (when 'biphasic'). If None, this amplitude is the opposite of 'amp1'
            width2: float
                Duration of the pulse of the second phase in ms (when 'biphasic'). If None, it is the same as 'width1'
            interphase: float
                For biphasic stimulation, duration between two phases in ms
            n_pulses: int
                Number of pulses. If 'interburst' is given, this indicates the number of pulses in a burst
            interburst: float
                Inter burst interval in ms
            n_bursts: int
                Total number of burst (if None, no limit is used).
            t_start: float
                Start of the stimulation in ms

            Returns
            -------
            current: np.array
                Array of current values
            t_ext: np.array
                timestamps of computed currents
            '''
            if biphasic:
                if amp2 is None:
                    amp2 = -amp1
                if width2 is None:
                    width2 = width1
                if interphase is None:
                    interphase = 0
            if t_start is None:
                t_start = 0
            t_ext = np.arange(t_start, t_stop, dt)
            current_values = np.zeros(len(t_ext))

            if interburst is not None:
                assert n_pulses is not None, "For bursting pulses provide 'n_pulses' per burst"

            if n_pulses is None:
                n_pulses = np.inf
            if n_bursts is None:
                n_bursts = np.inf

            t_pulse = t_start
            n_p = 0
            n_b = 0
            while t_pulse < t_stop and n_p < n_pulses and n_b < n_bursts:
                pulse_idxs = np.where((t_ext > t_pulse) & (t_ext <= t_pulse + width1))
                current_values[pulse_idxs] = amp1
                if biphasic:
                    t_pulse2 = t_pulse + width1 + interphase
                    pulse2_idxs = np.where((t_ext > t_pulse2) & (t_ext <= t_pulse2 + width2))
                    current_values[pulse2_idxs] = amp2
                    t_pulse_end = t_pulse2 + width2
                else:
                    t_pulse_end = t_pulse + width1
                n_p += 1

                if interburst:
                    if n_p == n_pulses:
                        n_p = 0
                        t_pulse = t_pulse_end + interburst
                        n_b += 1
                    else:
                        t_pulse = t_pulse_end + interpulse
                else:
                    t_pulse = t_pulse_end + interpulse

            return current_values, t_ext
