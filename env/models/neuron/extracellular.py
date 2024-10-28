from LFPy import RecExtElectrode, CurrentDipoleMoment
import numpy as np


class ExtracellularModels:
    def __init__(self, args):
        # During tACS a weak oscillating current (1-4 mA) is applied.
        # Handle the EEG and tACS stimulation related aspects.
        self.obs_win_len = args.env.simulation.obs_win_len
        self.stim_electrode = 0  # Electrode position, based on the provided EEG configuration.
        self.stim_type = args.env.ts.type

        electrodeParameters=dict(x=[22.],
                                    y=[107.6],
                                    z=[-1312.],
                                    N=np.array([[0., 0., 1.] for _ in range(1)]),
            # x=np.array(args.env.network.position[0], dtype=float),  # µm
            #                      y=np.array(args.env.network.position[1], dtype=float),  # µm
            #                      z=np.array(args.env.network.position[2], dtype=float),  # µm
                                 r=args.env.ts.electrodeParameters.r,
                                 n=args.env.ts.electrodeParameters.n,
                                 sigma=args.env.ts.electrodeParameters.sigma,
                                 method=args.env.ts.electrodeParameters.method)
        # there is another parameter "N" normal vector

        # set up extracellular recording device.
        # Here `cell` is set to None as handles to cell geometry is handled internally
        self.electrode = RecExtElectrode(cell=None, **electrodeParameters)
        # set up recording of current dipole moments. Ditto in regard to `cell` being set to None
        self.current_dipole_moment = CurrentDipoleMoment(cell=None)

    def get_probes(self):
        return [self.electrode, self.current_dipole_moment]

    def set_pulse(self, network, _step, action):
        # AMPLITUTDE [0, 4] mA, FREQ = 1 - 20 Hz
        amplitude = action[0] * 1e6  # LFPy accepts units nA, convert mA -> nA
        freq = action[1]

        total_cycles = freq * (self.obs_win_len/1000)
        total_pulses = 2 * total_cycles
        tstart = _step * self.obs_win_len
        tstop = tstart + self.obs_win_len - 1

        if self.stim_type == 'tdcs':
            total_pulses = 1
            pulse_width = int(self.obs_win_len/total_pulses)
            I_stim, t_ext = self.electrode.probe.set_current_pulses(
                                 n_pulses=total_pulses,
                                 biphasic=False,  # width2=width1, amp2=-amp1
                                 width1=pulse_width,
                                 amp1=amplitude,  # 3000,  # nA
                                 dt=network.dt,
                                 t_stop=tstop,
                                 interpulse=0,
                                 el_id=self.stim_electrode,
                                 t_start=tstart)

        elif self.stim_type == 'pulse':
            pulse_width = int(self.obs_win_len/total_pulses)
            # I_stim, t_ext = self.electrode.probe.set_current_pulses(
            #                      n_pulses=total_pulses,
            #                      biphasic=True,  # width2=width1, amp2=-amp1
            #                      width1=pulse_width,
            #                      amp1=amplitude,  # 3000,  # nA
            #                      dt=network.dt,
            #                      t_stop=tstop,
            #                      interpulse=0,
            #                      el_id=self.stim_electrode,
            #                      t_start=tstart)

            # examples params in LFPy
            I_stim, t_ext = self.electrode.probe.set_current_pulses(
                                    n_pulses=20,
                                    biphasic=True,  # width2=width1, amp2=-amp1
                                    width1=5,
                                    amp1=3000000,  # nA
                                    dt=network.dt,
                                    t_stop=tstop,
                                    interpulse=200,
                                    el_id=self.stim_electrode,
                                    t_start=tstart)

        # TODO: handle stim_electrode !!! for above methods its given as int; below array ?
        elif self.stim_type == 'sin':
            t_ext = np.arange(tstart, tstop, network.dt)
            resolution = len(t_ext)
            length = np.pi * 2 * total_cycles
            I_stim = amplitude * np.sin(np.arange(0, length, length / resolution))
            self.electrode.probe.set_current(self.stim_electrode, I_stim)

        else:
            print('select corrects time type')
            I_stim, t_ext = [], []

        return I_stim, t_ext
