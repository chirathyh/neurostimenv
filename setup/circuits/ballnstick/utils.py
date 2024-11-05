import sys
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
import numpy as np
import scipy.signal as ss
import scipy.stats as st
import h5py
from mpi4py import MPI
import neuron
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, \
    CurrentDipoleMoment

def setup_network(network, args, MPI_VAR):
    OUTPUTPATH = 'example_network_stim_outputx'

    # class NetworkCell parameters:
    cellParameters = dict(
        morphology=MAIN_PATH+'/setup/circuits/ballnstick/BallAndStick.hoc',
        templatefile=MAIN_PATH+'/setup/circuits/ballnstick/BallAndStickTemplate.hoc',
        templatename='BallAndStickTemplate',
        templateargs=None,
        delete_sections=False,
        dt=2**-4,
        tstop=2000,
    )

    # class NetworkPopulation parameters:
    populationParameters = dict(
        Cell=NetworkCell,
        cell_args=cellParameters,
        pop_args=dict(
            radius=100.,
            loc=0.,
            scale=20.),
        rotation_args=dict(x=0., y=0.),
    )

    # class Network parameters:
    networkParameters = dict(
        dt=2**-4,
        tstop=1200.,
        v_init=-65.,
        celsius=36.5,
        OUTPUTPATH=OUTPUTPATH
    )

    # class RecExtElectrode parameters:
    electrodeParameters = dict(
        x=np.array([-100, 0, 100, -100, 0, 100, -100, 0, 100]),
        y=np.array([100, 100, 100, 0, 0, 0, -100, -100, -100]),
        z=np.zeros(9),
        N=np.array([[0., 0., 1.] for _ in range(9)]),
        r=20.,  # 5um radius
        n=50,  # nb of discrete point used to compute the potential
        sigma=1,  # conductivity S/m
        method="linesource"
    )

    # method Network.simulate() parameters:
    networkSimulationArguments = dict(
        rec_pop_contributions=True,
        to_memory=True,
        to_file=False
    )

    # population names, sizes and connection probability:
    population_names = ['E', 'I']
    population_sizes = [32, 8]
    connectionProbability = [[0.1, 0.1], [0.1, 0.1]]

    # synapse model. All corresponding parameters for weights,
    # connection delays, multapses and layerwise positions are
    # set up as shape (2, 2) nested lists for each possible
    # connection on the form:
    # [["E:E", "E:I"],
    #  ["I:E", "I:I"]].
    synapseModel = neuron.h.Exp2Syn
    # synapse parameters
    synapseParameters = [[dict(tau1=0.2, tau2=1.8, e=0.),
                          dict(tau1=0.2, tau2=1.8, e=0.)],
                         [dict(tau1=0.1, tau2=9.0, e=-80.),
                          dict(tau1=0.1, tau2=9.0, e=-80.)]]
    # synapse max. conductance (function, mean, st.dev., min.):
    weightFunction = np.random.normal
    weightArguments = [[dict(loc=0.001, scale=0.0001),
                        dict(loc=0.001, scale=0.0001)],
                       [dict(loc=0.01, scale=0.001),
                        dict(loc=0.01, scale=0.001)]]
    minweight = 0.
    # conduction delay (function, mean, st.dev., min.):
    delayFunction = np.random.normal
    delayArguments = [[dict(loc=1.5, scale=0.3),
                       dict(loc=1.5, scale=0.3)],
                      [dict(loc=1.5, scale=0.3),
                       dict(loc=1.5, scale=0.3)]]
    mindelay = 0.3
    multapseFunction = np.random.normal
    multapseArguments = [[dict(loc=2., scale=.5), dict(loc=2., scale=.5)],
                         [dict(loc=5., scale=1.), dict(loc=5., scale=1.)]]
    # method NetworkCell.get_rand_idx_area_and_distribution_norm
    # parameters for layerwise synapse positions:
    synapsePositionArguments = [[dict(section=['soma', 'apic'],
                                      fun=[st.norm, st.norm],
                                      funargs=[dict(loc=0., scale=100.),
                                               dict(loc=500., scale=100.)],
                                      funweights=[0.5, 1.]
                                      ) for _ in range(2)],
                                [dict(section=['soma', 'apic'],
                                      fun=[st.norm, st.norm],
                                      funargs=[dict(loc=0., scale=100.),
                                               dict(loc=100., scale=100.)],
                                      funweights=[1., 0.5]
                                      ) for _ in range(2)]]


    # create E and I populations:
    for name, size in zip(population_names, population_sizes):
        network.create_population(name=name, POP_SIZE=size,
                                  **populationParameters)

    # create connectivity matrices and connect populations:
    for i, pre in enumerate(population_names):
        for j, post in enumerate(population_names):
            # boolean connectivity matrix between pre- and post-synaptic
            # neurons in each population (postsynaptic on this RANK)
            connectivity = network.get_connectivity_rand(
                pre=pre, post=post,
                connprob=connectionProbability[i][j]
            )

            # connect network:
            (conncount, syncount) = network.connect(
                pre=pre, post=post,
                connectivity=connectivity,
                syntype=synapseModel,
                synparams=synapseParameters[i][j],
                weightfun=weightFunction,
                weightargs=weightArguments[i][j],
                minweight=minweight,
                delayfun=delayFunction,
                delayargs=delayArguments[i][j],
                mindelay=mindelay,
                multapsefun=multapseFunction,
                multapseargs=multapseArguments[i][j],
                syn_pos_args=synapsePositionArguments[i][j],
                save_connections=False,
            )
