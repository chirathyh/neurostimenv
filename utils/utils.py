import os
import sys
import shutil
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
