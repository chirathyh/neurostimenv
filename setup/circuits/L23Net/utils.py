import sys
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
import time
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy import stats as st
import neuron
from neuron import h
import pandas as pd
import LFPy
from LFPy import Synapse
from utils.utils import generate_spike_train


def load_circuit_params(args, MPI_VAR, local_state):
    # MPI variables:
    COMM = MPI_VAR['COMM']
    RANK = MPI_VAR['RANK']

    halfnorm_rv = st.halfnorm
    halfnorm_rv.random_state = local_state
    uniform_rv = st.uniform
    uniform_rv.random_state = local_state

    TESTING = args.experiment.debug  # True # i.e.g generate 1 cell/pop, with 0.1 s runtime
    circuit_params = pd.read_excel(MAIN_PATH+'/setup/circuits/L23Net/Circuit_param.xls', sheet_name=None, index_col=0)

    # Get cell names and import biophys
    cell_names = [i for i in circuit_params['conn_probs'].axes[0]]
    for name in cell_names:
        h.load_file(MAIN_PATH+'/setup/circuits/L23Net/models/biophys_'+name+'.hoc')

    circuit_params["syn_params"] = {'none': {'tau_r_AMPA': 0, 'tau_d_AMPA': 0, 'tau_r_NMDA': 0,
                                    'tau_d_NMDA': 0, 'e': 0, 'Dep': 0, 'Fac': 0,'Use': 0, 'u0': 0, 'gmax': 0}}
    circuit_params["multi_syns"] = {'none': {'loc': 0, 'scale': 0}}

    # organizing dictionary for LFPY input
    for pre in cell_names:
        for post in cell_names:
            if "PYR" in pre:
                circuit_params["syn_params"][pre+post] = {'tau_r_AMPA': 0.3, 'tau_d_AMPA': 3, 'tau_r_NMDA': 2,
                                                          'tau_d_NMDA': 65, 'e': 0, 'u0':0,
                                                          'Dep': circuit_params["Depression"].at[pre, post],
                                                          'Fac': circuit_params["Facilitation"].at[pre, post],
                                                          'Use': circuit_params["Use"].at[pre, post],
                                                          'gmax': circuit_params["syn_cond"].at[pre, post]}
            else:
                circuit_params["syn_params"][pre+post] = {'tau_r': 1, 'tau_d': 10, 'e': -80, 'u0':0,
                                                          'Dep': circuit_params["Depression"].at[pre, post],
                                                          'Fac': circuit_params["Facilitation"].at[pre, post],
                                                          'Use': circuit_params["Use"].at[pre, post],
                                                          'gmax': circuit_params["syn_cond"].at[pre, post]}
            circuit_params["multi_syns"][pre+post] = {'loc': int(circuit_params["n_cont"].at[pre, post]),'scale':0}

    stimuli = []
    for stimulus in circuit_params['STIM_PARAM'].axes[0]:
        stimuli.append({})
        for param_name in circuit_params['STIM_PARAM'].axes[1]:
            stimuli[-1][param_name] = circuit_params['STIM_PARAM'].at[stimulus, param_name]
        new_param = circuit_params["syn_params"][stimuli[-1]['syn_params']].copy()
        new_param['gmax'] = stimuli[-1]['gmax']
        stimuli[-1]['new_param'] = new_param
    COMM.Barrier()

    if TESTING:
        # for name in cell_names:
        #     circuit_params['SING_CELL_PARAM'].at['cell_num', name] = 5  # change here to change number of cells. default was 1
        circuit_params['SING_CELL_PARAM'].at['cell_num', 'HL23PYR'] = args.env.debug_n_neurons['PYR']
        circuit_params['SING_CELL_PARAM'].at['cell_num', 'HL23SST'] = args.env.debug_n_neurons['SST']
        circuit_params['SING_CELL_PARAM'].at['cell_num', 'HL23PV'] = args.env.debug_n_neurons['PV']
        circuit_params['SING_CELL_PARAM'].at['cell_num', 'HL23VIP'] = args.env.debug_n_neurons['VIP']
        print('\n!!! This is the TEST version of L23Net!!!\n') if RANK==0 else None
    else:
        print('\nRunning full simulation...') if RANK==0 else None

    COMM.Barrier()

    #              L2/3   L4     L5
    PYRmaxApics = [550   ,1550  ,1900]
    uppers =      [-250  ,-1200 ,-1600]
    lowers =      [-1200 ,-1580 ,-2300]

    depths = []
    rangedepths = []
    minSynLocs = []
    syn_pos = []
    pop_args = {}

    for i in range(3):
        depths.append((lowers[i]-uppers[i])/2-PYRmaxApics[i])
        rangedepths.append(abs(lowers[i]-uppers[i])/2)
        minSynLocs.append((lowers[i]-uppers[i])/2*3-PYRmaxApics[i])

        syn_pos.append({'section': ['apic', 'dend'],
                        'fun': [uniform_rv, halfnorm_rv],
                        'funargs': [{'loc': minSynLocs[i], 'scale':abs(minSynLocs[i])},
                                    {'loc': minSynLocs[i], 'scale':abs(minSynLocs[i])}],
                        'funweights': [1, 1.]})
        syn_pos.append({'section': ['apic'],
                        'fun': [uniform_rv],
                        'funargs': [{'loc':minSynLocs[i], 'scale':abs(minSynLocs[i])}],
                        'funweights': [1.]})
        syn_pos.append({'section': ['dend'],
                        'fun': [uniform_rv],
                        'funargs': [{'loc':minSynLocs[i], 'scale':abs(minSynLocs[i])}],
                        'funweights': [1.]})
        syn_pos.append({'section': ['dend'],
                       'fun': [halfnorm_rv],
                       'funargs': [{'loc':minSynLocs[i], 'scale':abs(minSynLocs[i])}],
                       'funweights': [1.]})
        names = ['HL2', 'HL4', 'HL5']
        pop_args[names[i]] = {'radius': 250, 'loc': depths[i], 'scale': rangedepths[i]*4,'cap': rangedepths[i]}
        #pop_args[names[i]] = {'radius': 100, 'loc': 0, 'scale': 20.}
        # if RANK == 0:
        #     print(names[i])
        #     print(depths[i], rangedepths[i]*4, rangedepths[i])

    MDD = args.env.simulation.MDD
    reduce_inhibition = 0.4
    # decrease PN GtonicApic and MN2PN weight by 40%
    if MDD:
        # synaptic reduction
        for pre in cell_names:
            for post in cell_names:
                if 'SST' in pre:
                    circuit_params["syn_params"][pre+post]["gmax"] = circuit_params["syn_cond"].at[pre, post]*0.6
        # tonic reduction
        for post in cell_names:
            if 'PYR' in post:
                circuit_params['SING_CELL_PARAM'].at['apic_tonic',post] = circuit_params['SING_CELL_PARAM'].at['apic_tonic', post]*0.6
                circuit_params['SING_CELL_PARAM'].at["drug_apic_tonic",post] = circuit_params['SING_CELL_PARAM'].at["drug_apic_tonic", post]*0.6
            else:
                sst = 0
                total = 0
                for pre in cell_names:
                    if 'SST' in pre:
                        sst += circuit_params["syn_cond"].at[pre, post]*circuit_params["n_cont"].at[pre,post]*circuit_params["conn_probs"].at[pre, post]
                        total += circuit_params["syn_cond"].at[pre, post]*circuit_params["n_cont"].at[pre,post]*circuit_params["conn_probs"].at[pre, post]
                    elif 'PV' in pre or 'VIP' in pre:
                        total += circuit_params["syn_cond"].at[pre, post]*circuit_params["n_cont"].at[pre, post]*circuit_params["conn_probs"].at[pre, post]
                circuit_params['SING_CELL_PARAM'].at['norm_tonic', post] -= circuit_params['SING_CELL_PARAM'].at['norm_tonic', post]*sst/total*reduce_inhibition
                print(post, '_tonic reduced by: ', sst/total*100*reduce_inhibition, '%') if RANK == 0 else None

    return circuit_params, stimuli, cell_names, pop_args, syn_pos


def generateSubPop(args, circuit_params, network, popsize, mname, popargs, Gou, Gtonic, GtonicApic, MPI_VAR, local_state):
    RANK = MPI_VAR['RANK']
    SEED = MPI_VAR['SEED']

    print('Initiating ' + mname + ' population...') if RANK==0 else None
    morphpath = MAIN_PATH+'/setup/circuits/L23Net/morphologies/' + mname + '.swc'
    templatepath = MAIN_PATH+'/setup/circuits/L23Net/models/NeuronTemplate.hoc'
    templatename = 'NeuronTemplate'
    pt3d = True
    cellParams = {
            'morphology': morphpath,
            'templatefile': templatepath,
            'templatename': templatename,
            'templateargs': morphpath,
            'v_init': args.env.network.v_init,  # initial membrane potential, d=-65
            'passive': False,  # initialize passive mechs, d=T, should be overwritten by biophys
            'dt': args.env.network.dt,
            'tstart': args.env.network.tstart,
            'tstop': args.env.simulation.duration,  # defaults to 100
            'nsegs_method': None,
            'pt3d': pt3d,  # use pt3d-info of the cell geometries switch, d=F
            'delete_sections': False,
            'verbose': False}  # verbose output switch, for some reason doens't work, figure out why}

    rotation = {'x': circuit_params['SING_CELL_PARAM'].at['rotate_x', mname],
                'y': circuit_params['SING_CELL_PARAM'].at['rotate_y', mname]}

    popParams = {
            'CWD': None,
            'CELLPATH': None,
            'Cell': LFPy.NetworkCell,  # play around with this, maybe put popargs into here
            'POP_SIZE': int(popsize),
            'name': mname,
            'cell_args': {**cellParams},
            'pop_args': popargs,
            'rotation_args': rotation}

    network.create_population(**popParams)

    # Add biophys, OU processes, & tonic inhibition to cells
    for cellind in range(0,len(network.populations[mname].cells)):
        rseed = int(local_state.uniform()*SEED)
        biophys = 'h.biophys_' + mname + '(network.populations[\'' + mname + '\'].cells[' + str(cellind) + '].template)'
        exec(biophys)
        h.createArtificialSyn(rseed,network.populations[mname].cells[cellind].template,Gou)
        h.addTonicInhibition(network.populations[mname].cells[cellind].template,Gtonic,GtonicApic)


def addStimulus(args, network, circuit_params, stimuli, cell_names, MPI_VAR):
    GLOBALSEED = MPI_VAR['GLOBALSEED']
    cell_nums = [circuit_params['SING_CELL_PARAM'].at['cell_num', name] for name in cell_names]
    for stim in stimuli:
        stim_index = sum(cell_nums[:cell_names.index(stim['cell_name'])]) + stim['num_cells'] + stim['start_index']
        for gid, cell in zip(network.populations[stim['cell_name']].gids, network.populations[stim['cell_name']].cells):
            if gid < stim_index and gid >= sum(cell_nums[:cell_names.index(stim['cell_name'])]) + stim['start_index']:
                idx = cell.get_rand_idx_area_norm(section=stim['loc'], nidx=stim['loc_num'])
                for i in idx:
                    time_d = 0
                    syn = Synapse(cell=cell, idx=i, syntype=stim['stim_type'], weight=1, **stim['new_param'])
                    while time_d <= 0:
                        time_d = np.random.uniform(low=stim['delay'], high=stim['delay']+stim['delay_range'])

                    syn.set_spike_times(generate_spike_train(interval=10.))
                    # syn.set_spike_times(generate_spike_train(start=stim['start_time']+time_d, interval=stim['interval'],
                    #                                          number=stim['num_stim'], noise=0.0))
                    # syn.set_spike_times_w_netstim(noise=0, start=(stim['start_time']+time_d), number=stim['num_stim'],
                    #                             interval=stim['interval'], seed=GLOBALSEED)

                    # testing
                    # syn.set_spike_times_w_netstim(noise=0, start=(0), number=stim['num_stim'],
                    #                             interval=stim['interval'], seed=GLOBALSEED)
                    #
                    # syn.set_spike_times_w_netstim(noise=0, start=(stim['start_time']+time_d),
                    # number=stim['num_stim'], interval=stim['interval'], seed=GLOBALSEED)
                    # syn.set_spike_times_w_netstim(interval=1.,
                    #                           seed=np.random.rand() * 2**32 - 1
                    #                           )
    # population_names = ['HL23PYR', 'HL23SST', 'HL23PV', 'HL23VIP']
    # for name in population_names:
    #     for cell in network.populations[name].cells:
    #             idx = cell.get_rand_idx_area_norm(section='allsec', nidx=64)
    #             for i in idx:
    #                 syn = Synapse(cell=cell, idx=i, syntype='Exp2Syn',
    #                               weight=0.001,
    #                               **dict(tau1=0.2, tau2=1.8, e=0.))
    #                 syn.set_spike_times_w_netstim(interval=50.,
    #                                               seed=np.random.rand() * 2**32 - 1
    #                                               )


def setup_network(network, args, MPI_VAR):
    RANK = MPI_VAR['RANK']
    SEED = MPI_VAR['SEED']
    COMM = MPI_VAR['COMM']
    np.random.seed(SEED + RANK)
    local_state = np.random.RandomState(SEED + RANK)

    tic_0 = time.perf_counter()  # script runtime calculation value
    print('Mechanisms found: ', MAIN_PATH+'/setup/circuits/L23Net/mod/x86_64/special') if RANK==0 else None
    neuron.h('forall delete_section()')
    neuron.load_mechanisms(MAIN_PATH+'/setup/circuits/L23Net/mod/')
    h.load_file(MAIN_PATH+'/setup/circuits/L23Net/net_functions.hoc')

    no_connectivity = False
    DRUG = args.env.simulation.DRUG

    # setup args
    circuit_params, stimuli, cell_names, pop_args, syn_pos = load_circuit_params(args, MPI_VAR, local_state)

    # generate neural populations
    for cell_name in cell_names:
        if DRUG:
            generateSubPop(args, circuit_params, network, circuit_params['SING_CELL_PARAM'].at['cell_num', cell_name],
                           cell_name,pop_args[cell_name[:3]],
                           circuit_params['SING_CELL_PARAM'].at['GOU', cell_name],
                           circuit_params['SING_CELL_PARAM'].at['drug_tonic', cell_name],
                           circuit_params['SING_CELL_PARAM'].at['drug_apic_tonic', cell_name], MPI_VAR, local_state)
        else:
            generateSubPop(args, circuit_params, network, circuit_params['SING_CELL_PARAM'].at['cell_num', cell_name],
                           cell_name,pop_args[cell_name[:3]],
                           circuit_params['SING_CELL_PARAM'].at['GOU', cell_name],
                           circuit_params['SING_CELL_PARAM'].at['norm_tonic', cell_name],
                           circuit_params['SING_CELL_PARAM'].at['apic_tonic', cell_name], MPI_VAR, local_state)

    COMM.Barrier()

    # Synaptic Connection Parameters
    E_syn = neuron.h.ProbAMPANMDA
    I_syn = neuron.h.ProbUDFsyn
    for i, pre in enumerate(network.population_names):
        for j, post in enumerate(network.population_names):
            connectivity = network.get_connectivity_rand(
                            pre=pre,
                            post=post,
                            connprob=0 if no_connectivity else circuit_params["conn_probs"].at[pre, post])
            (conncount, syncount) = network.connect(
                            pre=pre, post=post,
                            connectivity=connectivity,
                            syntype=E_syn if "PYR" in pre else I_syn,
                            synparams=circuit_params["syn_params"][pre+post],
                            weightfun=local_state.normal,
                            weightargs={'loc':1, 'scale':0},
                            minweight=1,
                            delayfun=local_state.normal,
                            delayargs={'loc':0.5, 'scale':0},
                            mindelay=0.5,
                            multapsefun=local_state.normal,
                            multapseargs=circuit_params["multi_syns"][pre+post],
                            syn_pos_args=syn_pos[circuit_params["Syn_pos"].at[pre,post]])
    print('Setting up neural populations took ', str((time.perf_counter() - tic_0)/60)[:5], 'minutes') if RANK==0 else None
    COMM.Barrier()

    addStimulus(args, network, circuit_params, stimuli, cell_names, MPI_VAR)
    COMM.Barrier()
    #print('\n ####### not adding the stimulus ####### \n') if RANK == 0 else None

    print('### Done setting up the network.') if RANK == 0 else None
