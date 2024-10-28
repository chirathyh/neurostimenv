import sys
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
import numpy as np
import neuron
import scipy.stats as st
import LFPy
from LFPy import Synapse


def create_populations_connect(network, args, MPI_VAR):
    RANK = MPI_VAR['RANK']
    SEED = MPI_VAR['SEED']
    COMM = MPI_VAR['COMM']
    np.random.seed(SEED + RANK)
    local_state = np.random.RandomState(SEED + RANK)

    cellParameters = {
        'morphology': MAIN_PATH+'/setup/circuits/hl23pyrnet/models/HL23PYR.swc',
        'templatefile': MAIN_PATH+'/setup/circuits/hl23pyrnet/models/NeuronTemplate.hoc',
        'templatename': 'NeuronTemplate',
        'templateargs': MAIN_PATH+'/setup/circuits/hl23pyrnet/models/HL23PYR.swc',
        'v_init': -65.,
        'passive': False,  # initialize passive mechs, d=T, should be overwritten by btruncnormiophys
        'nsegs_method': None,
        'pt3d': True,  # use pt3d-info of the cell geometries switch, d=F
        'delete_sections': False,
        'verbose': False,
        'dt': args.env.network.dt, #0.512, #2**-4,
        'tstart': args.env.network.tstart,
        'tstop': args.env.simulation.duration,  # defaults to 100
    }

    populationParameters= {
        'Cell': LFPy.NetworkCell,
        'cell_args': cellParameters,
        'pop_args': {'radius': 100., 'loc': 0., 'scale': 20.},
        'rotation_args': dict(x=0., y=0.)
    }

    # create E and I populations:
    for name, size in zip(args.env.network.population_names, args.env.network.population_sizes):
        network.create_population(name=name, POP_SIZE=size, **populationParameters)

        # create excitatory background synaptic activity for each cell with Poisson statistics
        for cell in network.populations[name].cells:
            idx = cell.get_rand_idx_area_norm(section='allsec', nidx=64)
            for i in idx:
                syn = Synapse(cell=cell, idx=i, syntype='Exp2Syn', weight=0.001, **dict(tau1=0.2, tau2=1.8, e=0.))
                syn.set_spike_times_w_netstim(interval=1., seed=np.random.rand() * 2**32 - 1) #interval=50.0

    synapseParameters = [[dict(tau1=0.2, tau2=1.8, e=0.), dict(tau1=0.2, tau2=1.8, e=0.)],
                            [dict(tau1=0.1, tau2=9.0, e=-80.), dict(tau1=0.1, tau2=9.0, e=-80.)]]
    weightArguments = [[dict(loc=0.001, scale=0.0001), dict(loc=0.001, scale=0.0001)],
                            [dict(loc=0.01, scale=0.001), dict(loc=0.01, scale=0.001)]]
    delayArguments = [[dict(a=(0.3 - 1.5) / 0.3, b=np.inf, loc=1.5, scale=0.3),
                                dict(a=(0.3 - 1.5) / 0.3, b=np.inf, loc=1.5, scale=0.3)],
                            [dict(a=(0.3 - 1.5) / 0.3, b=np.inf, loc=1.5, scale=0.3),
                                dict(a=(0.3 - 1.5) / 0.3, b=np.inf, loc=1.5, scale=0.3)]]
    multapseArguments = [[dict(a=(1 - 2.) / .5, b=(10 - 2.) / .5, loc=2., scale=.5),
                                dict(a=(1 - 2.) / .5,b=(10 - 2.) / .5, loc=2., scale=.5)],
                               [dict(a=(1 - 5.) / 1.,b=(10 - 5.) / 1., loc=5., scale=1.),
                                dict(a=(1 - 5.) / 1.,b=(10 - 5.) / 1., loc=5., scale=1.)]]
    synapsePositionArguments = [[dict(section=['soma', 'apic'], fun=[st.norm, st.norm],
                                funargs=[dict(loc=0., scale=100.), dict(loc=500., scale=100.)], funweights=[0.5, 1.]) for _ in range(2)],
                                [dict(section=['soma', 'apic'], fun=[st.norm, st.norm],
                                funargs=[dict(loc=0., scale=100.), dict(loc=100., scale=100.)], funweights=[1., 0.5]) for _ in range(2)]]

    # create connectivity matrices and connect populations:
    for i, pre in enumerate(args.env.network.population_names):
        for j, post in enumerate(args.env.network.population_names):
            # boolean connectivity matrix between pre- and post-synaptic
            # neurons in each population (postsynaptic on this RANK)
            connectivity = network.get_connectivity_rand(
                pre=pre, post=post,
                connprob=args.env.network.connectionProbability[i][j]
            )

            # connect network:
            (conncount, syncount) = network.connect(
                pre=pre, post=post,
                connectivity=connectivity,
                syntype=neuron.h.Exp2Syn,
                synparams=synapseParameters[i][j],
                weightfun=np.random.normal,
                weightargs= weightArguments[i][j],
                minweight=0.,
                delayfun=st.truncnorm,
                delayargs=delayArguments[i][j],
                mindelay=None,
                multapsefun=st.truncnorm,
                multapseargs=multapseArguments[i][j],
                syn_pos_args=synapsePositionArguments[i][j] ,
                save_connections=False,
            )
