import os
import numpy as np
import neuron
from LFPy import Network
from neuron import units
import h5py


def flattenlist(lst):
    return [item for sublist in lst for item in sublist]


def ReduceStructArray(sendbuf):
    """
    simplify MPI Reduce for structured ndarrays with floating point numbers

    Parameters
    ----------
    sendbuf: structured ndarray
        Array data to be reduced (default: summed)

    Returns
    -------
    recvbuf: structured ndarray or None
        Reduced array on RANK 0, None on all other RANKs
    """
    pc = neuron.h.ParallelContext()
    RANK = pc.id()

    if RANK == 0:
        shape = sendbuf.shape
        dtype_names = sendbuf.dtype.names
    else:
        shape = None
        dtype_names = None
    shape = pc.py_broadcast(shape, 0)
    dtype_names = pc.py_broadcast(dtype_names, 0)

    if RANK == 0:
        reduced = np.zeros(shape,
                           dtype=list(zip(dtype_names,
                                          ['f8' for i in range(len(dtype_names)
                                                               )])))
    else:
        reduced = None
    for name in dtype_names:
        recvbuf = neuron.h.Vector(sendbuf[name].flatten())
        pc.allreduce(recvbuf, 1)
        if RANK == 0:
            reduced[name] = np.array(recvbuf).reshape(shape)
    return reduced


class NetworkEnv(Network):
    def __int__(self, **networkParameters):
        super().__init__(**networkParameters)

    def init_simulation(self, probes=None, t_ext=None, obs_win_len=None,
                        rec_imem=False, rec_vmem=False,
                        rec_ipas=False, rec_icap=False,
                        rec_isyn=False, rec_vmemsyn=False, rec_istim=False,
                        rec_pop_contributions=False,
                        rec_variables=[], variable_dt=False, atol=0.001,
                        to_memory=True, to_file=False,
                        file_name='OUTPUT.h5', **kwargs):

        self.obs_win_len = obs_win_len

        # self.lfpy_state = neuron.h.SaveState()
        # network.simulate
        # self.setup_lfpy(probes=probes)  # TODO pass variables
        #
        # self.init_neuron(variable_dt=False, t_ext=t_ext, probes=probes)  # TODO pass variables
        # self.lfpy_state.save()

    def init_neuron(self, variable_dt=False, atol=0.001, t_ext=None, probes=None):

        cvode = neuron.h.CVode()
        try:
            cvode.use_fast_imem(1)
        except AttributeError:
            raise Exception('neuron.h.CVode().use_fast_imem() not found. '
                            'Please update NEURON to v.7.4 or newer')

        ####
        # set maximum integration step, it is necessary for communication of
        # spikes across RANKs to occur.
        # NOTE: Should this depend on the minimum delay in the network?
        self.pc.set_maxstep(10)
        # Initialize NEURON simulations of cell object
        neuron.h.dt = self.dt
        # needed for variable dt method
        if variable_dt:
            cvode.active(1)
            cvode.atol(atol)
        else:
            cvode.active(0)

        neuron.h.finitialize(self.v_init * units.mV)  # initialize state
        cvode.use_fast_imem(1)  # use fast calculation of transmembrane currents
        # initialize current- and record
        if cvode.active():
            cvode.re_init()
        else:
            neuron.h.fcurrent()
        neuron.h.frecord_init()
        neuron.h.t = self.tstart  # Starting simulation at tstart

    def setup_lfpy(self, probes=None,
             rec_imem=False, rec_vmem=False,
             rec_ipas=False, rec_icap=False,
             rec_isyn=False, rec_vmemsyn=False, rec_istim=False,
             rec_pop_contributions=False,
             rec_variables=[], variable_dt=False, atol=0.001,
             to_memory=True, to_file=False,
             use_ipas=False, use_icap=False, use_isyn=False,
             file_name='OUTPUT.h5',
             **kwargs):

        if probes is None:  # test some of the inputs
            assert rec_pop_contributions is False, \
                'rec_pop_contributions can not be True when probes is None'
        if not variable_dt:
            dt = self.dt
        else:
            dt = None
        for name in self.population_names:
            for cell in self.populations[name].cells:
                cell._set_soma_volt_recorder(dt)
                if rec_imem:
                    cell._set_imem_recorders(dt)
                if rec_vmem:
                    cell._set_voltage_recorders(dt)
                if rec_ipas:
                    cell._set_ipas_recorders(dt)
                if rec_icap:
                    cell._set_icap_recorders(dt)
                if len(rec_variables) > 0:
                    cell._set_variable_recorders(rec_variables)

        if to_memory and to_file:
            raise Exception('to_memory and to_file can not both be True')
        if to_file and file_name is None:
            raise Exception

        # create a dummycell object lumping together needed attributes
        # for calculation of extracellular potentials etc. The population_nsegs
        # array is used to slice indices such that single-population
        # contributions to the potential can be calculated.
        self.lfpy_population_nsegs, self.lfpy_network_dummycell = self._Network__create_network_dummycell()

        # {
        # set cell attribute on each probe, assuming that each probe was instantiated with argument cell=None
        for probe in probes:
            if probe.cell is None:
                probe.cell = self.lfpy_network_dummycell
            else:
                raise Exception('{}.cell!=None'.format(probe.__class__))
        # create list of transformation matrices; one for each probe
        self.lfpy_transforms = []
        if probes is not None:
            for probe in probes:
                self.lfpy_transforms.append(probe.get_transformation_matrix())
        # reset probe.cell to None, as it is no longer needed
        for probe in probes:
            probe.cell = None
        # }

        # create list of cells across all populations to simplify loops
        self.lfpy_cells = []
        for name in self.population_names:
            self.lfpy_cells += self.populations[name].cells
        # load spike times from NetCon, only needed if LFPy.Synapse class is used
        for cell in self.lfpy_cells:
            cell._load_spikes()

        # define data type for structured arrays dependent on the boolean arguments
        self.lfpy_dtype = [('imem', float)]
        if use_ipas:
            self.lfpy_dtype += [('ipas', float)]
        if use_icap:
            self.lfpy_dtype += [('icap', float)]
        if use_isyn:
            self.lfpy_dtype += [('isyn_e', float), ('isyn_i', float)]
        if rec_pop_contributions:
            self.lfpy_dtype += list(zip(self.population_names, [float] * len(self.population_names)))

        # setup list of structured arrays for all extracellular potentials
        # at each contact from different source terms and subpopulations
        if to_memory:
            for probe, M in zip(probes, self.lfpy_transforms):
                probe.data = np.zeros((M.shape[0], int(self.obs_win_len/ self.dt) + 1), dtype=self.lfpy_dtype)

    def step(self, probes=None,t_ext=None,
             rec_imem=False, rec_vmem=False,
             rec_ipas=False, rec_icap=False,
             rec_isyn=False, rec_vmemsyn=False, rec_istim=False,
             rec_pop_contributions=False,
             rec_variables=[], variable_dt=False, atol=0.001,
             to_memory=True, to_file=False,
             file_name='OUTPUT.h5',
             **kwargs):

        # self.lfpy_state.restore()

        # run fadvance until t >= tstop, and calculate LFP if asked for
        if probes is None and not rec_pop_contributions and not to_file:
            if not rec_imem:
                if self.verbose:
                    print("rec_imem==False, not recording membrane currents!")
            self.run_simulation_loop()
        else:
            self.run_simulation_with_probes_loop(
                probes=probes,
                t_ext=t_ext,
                variable_dt=variable_dt,
                atol=atol,
                to_memory=to_memory,
                to_file=to_file,
                file_name='tmp_output_RANK_{:03d}.h5',
                rec_pop_contributions=rec_pop_contributions,
                **kwargs)
        # self.lfpy_state.save()
        # self.lfpy_state.restore()

        # for name in self.population_names:
        #     for cell in self.populations[name].cells:
        #         # somatic trace
        #         cell.somav = np.array(cell.somav)
        #         if rec_imem:
        #             cell._calc_imem()
        #         if rec_ipas:
        #             cell._calc_ipas()
        #         if rec_icap:
        #             cell._calc_icap()
        #         if rec_vmem:
        #             cell._collect_vmem()
        #         if rec_isyn:
        #             cell._collect_isyn()
        #         if rec_vmemsyn:
        #             cell._collect_vsyn()
        #         if rec_istim:
        #             cell._collect_istim()
        #         if len(rec_variables) > 0:
        #             cell._collect_rec_variables(rec_variables)
        #         if hasattr(cell, '_hoc_netstimlist'):
        #             del cell._hoc_netstimlist

        # Collect spike trains across all RANKs to RANK 0
        for name in self.population_names:
            population = self.populations[name]
            population.spike_vectors = []
            for i in range(len(population._hoc_spike_vectors)):
                population.spike_vectors += \
                    [population._hoc_spike_vectors[i].as_numpy()]

        # collect spike times to RANK 0
        if self._RANK == 0:
            times = []
            gids = []
        else:
            times = None
            gids = None
        for i, name in enumerate(self.population_names):
            times_send = [x for x in self.populations[name].spike_vectors]
            gids_send = [x for x in self.populations[name].gids]
            if self._RANK == 0:
                times.append([])
                gids.append([])
                times[i] += flattenlist(self.pc.py_gather(times_send))
                gids[i] += flattenlist(self.pc.py_gather(gids_send))

                assert len(times[-1]) == len(gids[-1])
            else:
                _ = self.pc.py_gather(times_send)
                _ = self.pc.py_gather(gids_send)

        # create final output file, summing up single RANK output from
        # temporary files
        if to_file and probes is not None:
            # op = MPI.SUM
            fname = os.path.join(
                self.OUTPUTPATH,
                'tmp_output_RANK_{:03d}.h5'.format(
                    self._RANK))
            f0 = h5py.File(fname, 'r')
            if self._RANK == 0:
                f1 = h5py.File(os.path.join(self.OUTPUTPATH, file_name), 'w')
            dtype = []
            for key, value in f0[list(f0.keys())[0]].items():
                dtype.append((str(key), float))
            for grp in f0.keys():
                if self._RANK == 0:
                    # get shape from the first dataset
                    # (they should all be equal):
                    for value in f0[grp].values():
                        shape = value.shape
                        continue
                    f1[grp] = np.zeros(shape, dtype=dtype)
                for key, value in f0[grp].items():
                    recvbuf = neuron.h.Vector(
                        value[()].astype(float).flatten())
                    self.pc.allreduce(recvbuf, 1)
                    if self._RANK == 0:
                        f1[grp][key] = np.array(recvbuf).reshape(value.shape)
                    else:
                        recvbuf = None
            f0.close()
            if self._RANK == 0:
                f1.close()
            os.remove(fname)

        if probes is not None:
            if to_memory:
                # communicate and sum up measurements on each probe before returing spike times and corresponding gids:
                for probe in probes:
                    probe.data = ReduceStructArray(probe.data)


        return dict(times=times, gids=gids), neuron.h.t

    def run_simulation_loop(self):
        # only needed if LFPy.Synapse classes are used.
        # for name in self.population_names:
        #     for cell in self.populations[name].cells:
        #         cell._load_spikes()

        tstep = 0
        mini_step_time = self.obs_win_len

        for mini_step in range(0, int(mini_step_time / self.dt)):
            if neuron.h.t >= 0:
                tstep += 1
            neuron.h.fadvance()
            if neuron.h.t % 100. == 0.:
                if self._RANK == 0:
                    print('t = {} ms'.format(neuron.h.t))



    def run_simulation_with_probes_loop(self, t_ext,
                                        probes=None,
                                        variable_dt=False,
                                        atol=0.001,
                                        rtol=0.,
                                        to_memory=True,
                                        to_file=False,
                                        file_name=None,
                                        use_ipas=False,
                                        use_icap=False,
                                        use_isyn=False,
                                        rec_pop_contributions=False):

        tstep = 0
        # temporary vector to store membrane currents at each timestep:
        imem = np.zeros(self.lfpy_network_dummycell.totnsegs, dtype=self.lfpy_dtype)
        def get_imem(imem):
            '''helper function to gather currents across all cells
            on this RANK'''
            i = 0
            totnsegs = 0
            if use_isyn:
                imem['isyn_e'] = 0.  # must reset these for every iteration
                imem['isyn_i'] = 0.  # because we sum over synapses

            for cell in self.lfpy_cells:
                for sec in cell.allseclist:
                    for seg in sec:
                        imem['imem'][i] = seg.i_membrane_
                        if use_ipas:
                            imem['ipas'][i] = seg.i_pas
                        if use_icap:
                            imem['icap'][i] = seg.i_cap
                        i += 1

                if use_isyn:
                    for idx, syn in zip(cell.synidx, cell.netconsynapses):
                        if hasattr(syn, 'e') and syn.e > -50:
                            imem['isyn_e'][idx + totnsegs] += syn.i
                        else:
                            imem['isyn_i'][idx + totnsegs] += syn.i

                totnsegs += cell.totnsegs
            return imem

        # run fadvance until time limit, and calculate LFPs for each timestep
        for mini_step in range(0, int(self.obs_win_len / self.dt)):
            if neuron.h.t >= 0:
                imem = get_imem(imem)
                for j, (probe, M) in enumerate(zip(probes, self.lfpy_transforms)):
                    probe.data['imem'][:, tstep] = M @ imem['imem']
                tstep += 1
            neuron.h.fadvance()
            if neuron.h.t % 100. == 0.:
                if self._RANK == 0:
                    print('t = {} ms'.format(neuron.h.t))

    def enable_extracellular_stimulation_mpi(self, electrode, _step, t_ext=None, n=1, model='inf'):
        """
        Enable extracellular stimulation with NEURON's `extracellular`
        mechanism. Extracellular potentials are computed from electrode
        currents using the point-source approximation.
        If ``model`` is ``'inf'`` (default), potentials are computed as
        (:math:`r_i` is the position of a segment :math:`i`,
        :math:`r_n` is the position of an electrode :math:`n`,
        :math:`\\sigma` is the conductivity of the medium):

        .. math::
            V_e(r_i) = \\sum_n \\frac{I_n}{4 \\pi \\sigma |r_i - r_n|}

        If ``model`` is ``'semi'``, the method of images is used:

        .. math::
            V_e(r_i) = \\sum_n \\frac{I_n}{2 \\pi \\sigma |r_i - r_n|}

        Parameters
        ----------
        electrode: RecExtElectrode
            Electrode object with stimulating currents
        t_ext: np.ndarray or list
            Time in ms corresponding to step changes in the provided currents.
            If None, currents are assumed to have
            the same time steps as the NEURON simulation.
        n: int
            Points per electrode for spatial averaging
        model: str
            ``'inf'`` or ``'semi'``. If ``'inf'`` the medium is assumed to be
            infinite and homogeneous. If ``'semi'``, the method of
            images is used.

        Returns
        -------
        v_ext: dict of np.ndarrays
            Computed extracellular potentials at cell mid points
            for each cell of the network's populations. Formatted as
            ``v_ext = {'pop1': np.ndarray[cell, cell_seg,t_ext]}``
        """
        # self.lfpy_state.restore()
        print(self.tstop)
        v_ext = {}
        for popname in self.populations.keys():
            cells = self.populations[popname].cells
            if len(cells) != 0:  # TODO: <chirath> I'm skipping when cells are zero during multiple processes.
                v_ext[popname] = np.zeros((len(cells), cells[0].totnsegs, len(t_ext)))
                for id_cell, cell in enumerate(cells):
                    electrode.probe.electrodes[0].mapping = None  # this could be inefficient, mapping calculated every time.
                    v_ext[popname][id_cell] = cell.enable_extracellular_stimulation(electrode, t_ext, n, model)

        return v_ext
