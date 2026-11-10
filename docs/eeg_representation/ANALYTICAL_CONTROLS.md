# Analytical controls and inferential units

These controls import NumPy and SciPy without importing NEURON, LFPy,
SimNIBS, or the biophysical adapter. They test the measurement and inference
pipeline, and do not establish an EEG-only prediction limit for the circuit.
All streams use locally constructed generators with SHA-256 named namespaces;
the controls do not mutate NumPy's global random state.

The filtered Poisson process is
`S(t) = a sum_{t_i <= t} exp(-(t-t_i)/tau)`.
Events use cumulative exponential inter-arrivals at continuous times; sample
values include events exactly at their right endpoint. An independent 20-tau
prehistory precedes recording. Starting this prehistory at zero leaves relative
stationary-mean truncation `exp(-20)` and stationary-variance truncation
`exp(-40)`. The retained history is therefore a documented burn-in
approximation, not an exact infinite-past stationary draw. Conditional on the
cutoff state, future means and variances are exact:

```
E[S(t+h) | events <= t] = exp(-h/tau) S(t) + a lambda tau (1-exp(-h/tau))
Var[S(t+h) | events <= t] = a^2 lambda tau/2 (1-exp(-2h/tau)).
```

The damped oscillator follows `dX = A X dt + sqrt(q) dW`, with
`A=[[-gamma,-omega],[omega,-gamma]]`, `omega=2*pi*f`, and observes only
`H X`, `H=[1,0]`. The transition is the exact damped rotation, with process
covariance `q/(2*gamma)*(1-exp(-2*gamma*dt))*I`; its initial state is sampled
from the exact stationary Gaussian. `kalman_filter` conditions causally on
clean or independently corrupted observations of the same underlying history.
`oscillator_block_forecast` retains cross-time target covariance. All oracle,
clean-history, and noisy-history risks use the *same clean future targets*.
A full-state oracle receives neither future noise nor RNG state. Gaussian
state-space sufficiency makes these risks exact in this control only.

In the variance-class control, trial-level class chooses sigma 1 or 2 and
samples are independent zero-mean Gaussians. Both class-aware and class-blind
optimal MSE means are zero. The positive classifier uses the exact equal-prior
likelihood ratio of the unstandardized squared history. The probabilistic
predictor uses the posterior expected variance. Its Gaussian approximation
has the correct conditional mean and variance, although the exact conditional
law is a two-Gaussian mixture. There is no per-trial/window standardization.
This is an objective mismatch example, not proof of encoder collapse.

`generate_trial` supports deterministic prefix replay for independent future
innovations. The Poisson checkpoint contains the complete scalar filter state;
the oscillator checkpoint contains both state coordinates; the IID variance
control checkpoint contains its fixed variance parameter. Prefix physical
state and event/history hashes are checked separately from EEG equality.
These processes have no delayed recurrent events or separate latent afferent
phase innovations. Their metadata explicitly records these absences and a
phase-only fork request fails. Zero dipole arrays are compatibility placeholders
marked `dipole_available=false`, not physical source estimates.

Fork estimation uses the unbiased sample covariance trace divided by target
dimension for each fixed state. Disjoint independent-future pairs provide the
alternative half-squared-difference estimator. All pairwise differences are
not independent replicates. Each scored future's empirical oracle mean omits
that future; the expected finite-ensemble scored MSE is `n/(n-1)` times the
conditional variance. Future-bootstrap intervals are conditional Monte Carlo
uncertainty and are distinct from between-structure uncertainty. Circuit
inference averages repetitions within structure before paired cluster
resampling. The exact sign-flip helper refuses more than 20 structures rather
than silently switching to approximate inference. BH adjustment preserves the
declared secondary family. Primary transfer area is trapezoidal integration
over a frozen log independent-label-budget axis divided by its log range.

Executed implementation checks (10 November 2026, parent simulation Python):

```
python -m unittest -v tests/test_eegrepr_analytical.py
```

Thirteen tests passed, including event moments/ISIs, known conditional laws,
exact transition composition, joint oscillator forecast covariance, Kalman
information gaps, positive task/distribution controls, prefix/state replay,
finite-ensemble correction, equal structure weighting, and global RNG isolation.
The standalone `run_controls('/tmp/eegrepr-analytical-controls', seed=0)` smoke
also passed its declared checks. Its largest Poisson variance relative error
was 0.02533. The oscillator empirical/exact risks were respectively
0.14990/0.14786 (state), 0.18631/0.18424 (clean history), and
0.35229/0.33524 (noisy history), each within five Monte Carlo standard errors.
The variance classifier's balanced accuracy was 0.99756 and its conditional
Gaussian NLL gain was 0.11047 per scalar target. These are genuine analytical
implementation smokes, not biophysical, pilot, or confirmation results. Runtime
artifacts live outside version control; rerunning the same seed is idempotent
and an incompatible controls-file overwrite is refused.
