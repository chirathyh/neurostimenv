# A gym type environemnt for single cell.
import numpy as np
from LFPy import Cell, NetworkCell
import neuron
from neuron import units
from warnings import warn


class CellEnv(Cell):
    def __int__(self, **cellParameters):
        super().__init__(**cellParameters)

    def init_simulation(self, variable_dt=False, atol=0.001, rtol=0.,
                         probes=None,
                 rec_imem=False, rec_vmem=False,
                 rec_ipas=False, rec_icap=False,
                 rec_variables=[],
                 to_memory=True,
                 to_file=False, file_name=None,
                 **kwargs):

        """
        This is the main function running the simulation of the NEURON model.
        Start NEURON simulation and record variables specified by arguments.

        Parameters
        ----------
        probes: list of :obj:, optional
            None or list of LFPykit.RecExtElectrode like object instances that
            each have a public method `get_transformation_matrix` returning
            a matrix that linearly maps each segments' transmembrane
            current to corresponding measurement as

            .. math:: \\mathbf{P} = \\mathbf{M} \\mathbf{I}

        rec_imem: bool
            If true, segment membrane currents will be recorded
            If no electrode argument is given, it is necessary to
            set rec_imem=True in order to make predictions later on.
            Units of (nA).
        rec_vmem: bool
            Record segment membrane voltages (mV)
        rec_ipas: bool
            Record passive segment membrane currents (nA)
        rec_icap: bool
            Record capacitive segment membrane currents (nA)
        rec_variables: list
            List of segment state variables to record, e.g. arg=['cai', ]
        variable_dt: bool
            Use NEURON's variable timestep method
        atol: float
            Absolute local error tolerance for NEURON variable timestep method
        rtol: float
            Relative local error tolerance for NEURON variable timestep method
        to_memory: bool
            Only valid with probes=[:obj:], store measurements as `:obj:.data`
        to_file: bool
            Only valid with probes, save simulated data in hdf5 file format
        file_name: str
            Name of hdf5 file, '.h5' is appended if it doesnt exist
        """
        for key in kwargs.keys():
            if key in ['electrode', 'rec_current_dipole_moment',
                       'dotprodcoeffs', 'rec_isyn', 'rec_vmemsyn',
                       'rec_istim', 'rec_vmemstim']:
                warn('Cell.simulate parameter {} is deprecated.'.format(key))

        # set up integrator, use the CVode().fast_imem method by default
        # as it doesn't hurt sim speeds much if at all.
        cvode = neuron.h.CVode()
        try:
            cvode.use_fast_imem(1)
        except AttributeError:
            raise Exception('neuron.h.CVode().use_fast_imem() method not '
                            'found. Update NEURON to v.7.4 or newer')

        if not variable_dt:
            dt = self.dt
        else:
            dt = None
        self._set_soma_volt_recorder(dt)

        if rec_imem:
            self._set_imem_recorders(dt)
        if rec_vmem:
            self._set_voltage_recorders(dt)
        if rec_ipas:
            self._set_ipas_recorders(dt)
        if rec_icap:
            self._set_icap_recorders(dt)
        if len(rec_variables) > 0:
            self._set_variable_recorders(rec_variables, dt)
        if hasattr(self, '_stimitorecord'):
            if len(self._stimitorecord) > 0:
                self.__set_ipointprocess_recorders(dt)
        if hasattr(self, '_stimvtorecord'):
            if len(self._stimvtorecord) > 0:
                self.__set_vpointprocess_recorders(dt)
        if hasattr(self, '_synitorecord'):
            if len(self._synitorecord) > 0:
                self._Cell__set_isyn_recorders(dt)
        if hasattr(self, '_synvtorecord'):
            if len(self._synvtorecord) > 0:
                self.__set_vsyn_recorders(dt)

        # set time recorder from NEURON
        self._Cell__set_time_recorders(dt)

        neuron.h.dt = self.dt

        # variable dt method
        if variable_dt:
            cvode.active(1)
            cvode.atol(atol)
            cvode.rtol(rtol)
        else:
            cvode.active(0)

        # re-initialize state
        neuron.h.finitialize(self.v_init * units.mV)

        # initialize current- and record
        if cvode.active():
            cvode.re_init()
        else:
            neuron.h.fcurrent()
        neuron.h.frecord_init()

        # Starting simulation at t != 0
        neuron.h.t = self.tstart

        self._load_spikes()

        # have an if-else for probe or no-probe
        #self._step()

    def step(self, probes, variable_dt=False):
        # temporary vector to store membrane currents at each timestep
        imem = np.zeros(self.totnsegs)

        mini_step_time = 20

        # precompute linear transformation matrices for each probe
        transforms = []  # container
        for probe in probes:
            M = probe.get_transformation_matrix()
            assert M.shape[-1] == self.totnsegs, \
                'Linear transformation shape mismatch'
            transforms.append(M)
            if not variable_dt:
                probe.data = np.zeros((M.shape[0], int(mini_step_time / self.dt) + 1))
            else:
                # for variable_dt, data will be added to last axis each time step
                probe.data = np.zeros((M.shape[0], 0))

        def get_imem(imem):
            i = 0
            for sec in self.allseclist:
                for seg in sec:
                    imem[i] = seg.i_membrane_
                    i += 1
            return imem

        tstep = 0
        # run fadvance until time limit, and calculate LFPs for each timestep
        #while neuron.h.t < cell.tstop:
        for mini_step in range(0, int(mini_step_time / self.dt)):
            if neuron.h.t >= 0:
                imem = get_imem(imem)
                # print(imem)
                for j, (probe, transform) in enumerate(zip(probes, transforms)):
                    if not variable_dt:
                        probe.data[:, tstep] = transform @ imem
                    else:
                        probe.data = np.c_[probes[j].data, transform @ imem]
                tstep += 1
            neuron.h.fadvance()
            #print("Mini-step taken: {}".format(tstep))

        # calculate LFP after final fadvance() if needed
        # (may occur for certain values for dt)
        if tstep < len(self._neuron_tvec):
            imem = get_imem(imem)
            for j, (probe, transform) in enumerate(zip(probes, transforms)):
                if not variable_dt:
                    probe.data[:, tstep] = transform @ imem
                else:
                    probe.data = np.c_[probes[j].data, transform @ imem]
        #neuron.h.fadvance()
        print("Step taken at t = {}, point process current = ".format(list(self._neuron_tvec)[-1]))


class NetworkCellEnv(NetworkCell):
    def __int__(self, **cellParameters):
        super().__init__(**cellParameters)

    def init_simulation(self, variable_dt=False, atol=0.001, rtol=0.,
                         probes=None,
                 rec_imem=False, rec_vmem=False,
                 rec_ipas=False, rec_icap=False,
                 rec_variables=[],
                 to_memory=True,
                 to_file=False, file_name=None,
                 **kwargs):

        """
        This is the main function running the simulation of the NEURON model.
        Start NEURON simulation and record variables specified by arguments.

        Parameters
        ----------
        probes: list of :obj:, optional
            None or list of LFPykit.RecExtElectrode like object instances that
            each have a public method `get_transformation_matrix` returning
            a matrix that linearly maps each segments' transmembrane
            current to corresponding measurement as

            .. math:: \\mathbf{P} = \\mathbf{M} \\mathbf{I}

        rec_imem: bool
            If true, segment membrane currents will be recorded
            If no electrode argument is given, it is necessary to
            set rec_imem=True in order to make predictions later on.
            Units of (nA).
        rec_vmem: bool
            Record segment membrane voltages (mV)
        rec_ipas: bool
            Record passive segment membrane currents (nA)
        rec_icap: bool
            Record capacitive segment membrane currents (nA)
        rec_variables: list
            List of segment state variables to record, e.g. arg=['cai', ]
        variable_dt: bool
            Use NEURON's variable timestep method
        atol: float
            Absolute local error tolerance for NEURON variable timestep method
        rtol: float
            Relative local error tolerance for NEURON variable timestep method
        to_memory: bool
            Only valid with probes=[:obj:], store measurements as `:obj:.data`
        to_file: bool
            Only valid with probes, save simulated data in hdf5 file format
        file_name: str
            Name of hdf5 file, '.h5' is appended if it doesnt exist
        """
        for key in kwargs.keys():
            if key in ['electrode', 'rec_current_dipole_moment',
                       'dotprodcoeffs', 'rec_isyn', 'rec_vmemsyn',
                       'rec_istim', 'rec_vmemstim']:
                warn('Cell.simulate parameter {} is deprecated.'.format(key))

        # set up integrator, use the CVode().fast_imem method by default
        # as it doesn't hurt sim speeds much if at all.
        cvode = neuron.h.CVode()
        try:
            cvode.use_fast_imem(1)
        except AttributeError:
            raise Exception('neuron.h.CVode().use_fast_imem() method not '
                            'found. Update NEURON to v.7.4 or newer')

        if not variable_dt:
            dt = self.dt
        else:
            dt = None
        self._set_soma_volt_recorder(dt)

        if rec_imem:
            self._set_imem_recorders(dt)
        if rec_vmem:
            self._set_voltage_recorders(dt)
        if rec_ipas:
            self._set_ipas_recorders(dt)
        if rec_icap:
            self._set_icap_recorders(dt)
        if len(rec_variables) > 0:
            self._set_variable_recorders(rec_variables, dt)
        if hasattr(self, '_stimitorecord'):
            if len(self._stimitorecord) > 0:
                self.__set_ipointprocess_recorders(dt)
        if hasattr(self, '_stimvtorecord'):
            if len(self._stimvtorecord) > 0:
                self.__set_vpointprocess_recorders(dt)
        if hasattr(self, '_synitorecord'):
            if len(self._synitorecord) > 0:
                self._Cell__set_isyn_recorders(dt)
        if hasattr(self, '_synvtorecord'):
            if len(self._synvtorecord) > 0:
                self.__set_vsyn_recorders(dt)

        # set time recorder from NEURON
        self._Cell__set_time_recorders(dt)

        neuron.h.dt = self.dt

        # variable dt method
        if variable_dt:
            cvode.active(1)
            cvode.atol(atol)
            cvode.rtol(rtol)
        else:
            cvode.active(0)

        # re-initialize state
        neuron.h.finitialize(self.v_init * units.mV)

        # initialize current- and record
        if cvode.active():
            cvode.re_init()
        else:
            neuron.h.fcurrent()
        neuron.h.frecord_init()

        # Starting simulation at t != 0
        neuron.h.t = self.tstart

        self._load_spikes()

        # have an if-else for probe or no-probe
        #self._step()

    def step(self, probes, variable_dt=False):
        # temporary vector to store membrane currents at each timestep
        imem = np.zeros(self.totnsegs)

        mini_step_time = 20

        # precompute linear transformation matrices for each probe
        transforms = []  # container
        for probe in probes:
            M = probe.get_transformation_matrix()
            assert M.shape[-1] == self.totnsegs, \
                'Linear transformation shape mismatch'
            transforms.append(M)
            if not variable_dt:
                probe.data = np.zeros((M.shape[0], int(mini_step_time / self.dt) + 1))
            else:
                # for variable_dt, data will be added to last axis each time step
                probe.data = np.zeros((M.shape[0], 0))

        def get_imem(imem):
            i = 0
            for sec in self.allseclist:
                for seg in sec:
                    imem[i] = seg.i_membrane_
                    i += 1
            return imem

        tstep = 0
        # run fadvance until time limit, and calculate LFPs for each timestep
        #while neuron.h.t < cell.tstop:
        for mini_step in range(0, int(mini_step_time / self.dt)):
            if neuron.h.t >= 0:
                imem = get_imem(imem)
                # print(imem)
                for j, (probe, transform) in enumerate(zip(probes, transforms)):
                    if not variable_dt:
                        probe.data[:, tstep] = transform @ imem
                    else:
                        probe.data = np.c_[probes[j].data, transform @ imem]
                tstep += 1
            neuron.h.fadvance()
            #print("Mini-step taken: {}".format(tstep))

        # calculate LFP after final fadvance() if needed
        # (may occur for certain values for dt)
        if tstep < len(self._neuron_tvec):
            imem = get_imem(imem)
            for j, (probe, transform) in enumerate(zip(probes, transforms)):
                if not variable_dt:
                    probe.data[:, tstep] = transform @ imem
                else:
                    probe.data = np.c_[probes[j].data, transform @ imem]
        #neuron.h.fadvance()
        print("Step taken at t = {}, point process current = ".format(list(self._neuron_tvec)[-1]))
