from scipy.stats import spearmanr

from fooof import FOOOF, FOOOFGroup
from fooof.plts.spectra import plot_spectra
from fooof.plts.annotate import plot_annotated_model
from fooof.plts.aperiodic import plot_aperiodic_params
from fooof.sim.params import Stepper, param_iter
from fooof.sim import gen_power_spectrum, gen_group_power_spectra
from fooof.utils.params import compute_time_constant, compute_knee_frequency

freqs, powers = gen_power_spectrum([1, 50], [0, 1], [10, 0.25, 2], freq_res=0.25)
print(freqs.shape)
print(powers.shape)
