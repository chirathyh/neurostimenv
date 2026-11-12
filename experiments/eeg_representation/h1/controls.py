"""Exact H1 controls. Mathematical processes, never empirical circuit bounds.

Finite-history Gaussian conditioning separates innovation, observation,
compression and fitting terms. Poisson events retain continuous event times.
Existing finite-fork estimators are reused without changing their estimands.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..analytical import (exponential_shot_noise, kalman_filter, oscillator_transition,
                          poisson_events, poisson_conditional_mean, stream)
from ..statistics import independent_pair_variance, leave_one_out_oracle, per_state_fork_variance
from ..storage import atomic_json, jsonable


def _array(x, name):
    x = np.asarray(x, dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return x


def _psd_root(cov):
    cov = (cov + cov.T) / 2
    values, vectors = np.linalg.eigh(cov)
    scale = max(float(np.max(np.abs(values))), 1.0)
    if values.min() < -1e-10 * scale:
        raise ValueError("covariance is not positive semidefinite")
    return vectors * np.sqrt(np.maximum(values, 0))


def gaussian_information_model(F, Q, *, history_samples=8, target_steps=(1, 4, 10),
                               observation_noise=0.5, initial_covariance=None,
                               representation=None, fitted_weights=None, fitted_offset=None):
    """Joint Gaussian law of C, finite visible history V, linear r=L V and Y.

    C is the state at the LAST history observation, excluding random-generator
    state and future innovations. All target_steps are strictly after C.
    The initial covariance is at the FIRST history observation, not steady-state
    filtering: a finite Kalman recursion is independently checked against it.
    """
    F, Q = _array(F, "F"), _array(Q, "Q")
    if F.shape != (2, 2) or Q.shape != (2, 2) or max(abs(np.linalg.eigvals(F))) >= 1:
        raise ValueError("require stable two-dimensional dynamics")
    if history_samples < 1 or observation_noise < 0:
        raise ValueError("invalid history/noise")
    steps = np.asarray(target_steps, dtype=int)
    if (steps.ndim != 1 or not len(steps) or np.any(steps < 1)
            or not np.array_equal(steps, np.asarray(target_steps)) or np.any(np.diff(steps) <= 0)):
        raise ValueError("target steps must be increasing positive integers")
    initial = np.eye(2) if initial_covariance is None else _array(initial_covariance, "initial_covariance")
    _psd_root(initial); _psd_root(Q)
    last = history_samples - 1
    powers = [np.linalg.matrix_power(F, i) for i in range(last + int(steps[-1]) + 1)]
    covariance = [initial]
    for _ in range(last + int(steps[-1])):
        covariance.append(F @ covariance[-1] @ F.T + Q)
    def cross(t, s):
        return powers[t-s] @ covariance[s] if t >= s else covariance[t] @ powers[s-t].T
    vv = np.array([[cross(i, j)[0, 0] for j in range(history_samples)] for i in range(history_samples)])
    vv += observation_noise * np.eye(history_samples)
    cv = np.column_stack([cross(last, i)[:, 0] for i in range(history_samples)])
    A = np.stack([powers[int(step)][0] for step in steps])
    yy = np.array([[cross(last+int(i), last+int(j))[0, 0] for j in steps] for i in steps])
    yv = A @ cv
    state_floor_cov = yy - A @ covariance[last] @ A.T
    v_weights = yv @ np.linalg.pinv(vv, hermitian=True)
    posterior = covariance[last] - cv @ np.linalg.pinv(vv, hermitian=True) @ cv.T
    L = np.zeros((1, history_samples)) if representation is None else _array(representation, "representation")
    if representation is None:
        L[0, -1] = 1
    if L.ndim != 2 or L.shape[1] != history_samples:
        raise ValueError("representation must be a linear projection of this history")
    rr = L @ vv @ L.T
    r_weights = yv @ L.T @ np.linalg.pinv(rr, hermitian=True)
    fit = 0.8 * r_weights if fitted_weights is None else _array(fitted_weights, "fitted_weights")
    offset = np.full(len(steps), 0.1) if fitted_offset is None else np.asarray(fitted_offset, dtype=float)
    if fit.shape != r_weights.shape or offset.shape != (len(steps),) or not np.isfinite(offset).all():
        raise ValueError("fitted predictor dimensions differ from target/representation")
    obs_cov = A @ posterior @ A.T
    comp_cov = v_weights @ vv @ v_weights.T - r_weights @ rr @ r_weights.T
    fit_cov = (r_weights-fit) @ rr @ (r_weights-fit).T + np.outer(offset, offset)
    terms = dict(innovation=state_floor_cov, hidden_state=obs_cov,
                 compression=comp_cov, fitting=fit_cov)
    exact = {name: float(np.trace(value)/len(steps)) for name, value in terms.items()}
    # Numerical zero is not a Monte Carlo truncation of a measured negative gap.
    exact = {k: (0.0 if abs(v) < 1e-12 else v) for k, v in exact.items()}
    _, kalman_cov = kalman_filter(np.zeros(history_samples), F, Q,
                                 noise_variance=observation_noise, initial_covariance=initial)
    if not np.allclose(kalman_cov[-1], posterior, rtol=1e-8, atol=1e-10):
        raise AssertionError("finite Kalman posterior disagrees with joint conditioning")
    return dict(F=F, Q=Q, initial_covariance=initial, history_samples=history_samples,
                target_steps=steps, observation_noise=float(observation_noise),
                representation=L, fitted_weights=fit, fitted_offset=offset,
                v_weights=v_weights, r_weights=r_weights, state_weights=A,
                history_covariance=vv, state_posterior_covariance=posterior,
                innovation_covariance=state_floor_cov, hidden_state_covariance=obs_cov,
                exact_terms=exact, exact_model_risk=sum(exact.values()))


def gaussian_information_control(*, seed=0, n_trials=8192, history_samples=8,
                                 target_steps=(1, 4, 10), process_noise=4.0,
                                 observation_noise=0.5, representation=None,
                                 affine_scale=1.0, affine_offset=0.0, return_arrays=False):
    """Independent-path Monte Carlo validation with a fixed six-SE tolerance."""
    if n_trials < 4 or not np.isfinite([affine_scale, affine_offset]).all() or affine_scale == 0:
        raise ValueError("invalid Monte Carlo dimensions or affine transform")
    F, Q = oscillator_transition(.01, frequency_hz=9, damping_per_s=2,
                                diffusion_variance_per_s=process_noise)
    model = gaussian_information_model(F, Q, history_samples=history_samples,
        target_steps=target_steps, observation_noise=observation_noise, representation=representation)
    rng = stream(seed, "h1/gaussian/latent")
    x = rng.normal(size=(n_trials, 2)) @ _psd_root(model['initial_covariance']).T
    history = np.empty((n_trials, history_samples))
    target = np.empty((n_trials, len(target_steps)))
    steps = {history_samples-1+int(s): j for j, s in enumerate(target_steps)}
    qroot = _psd_root(Q)
    sensor_rng = stream(seed, "h1/gaussian/sensor")
    for time in range(history_samples + max(target_steps)):
        if time:
            x = x @ F.T + rng.normal(size=x.shape) @ qroot.T
        if time < history_samples:
            history[:, time] = x[:, 0] + np.sqrt(observation_noise)*sensor_rng.normal(size=n_trials)
        if time == history_samples-1:
            state = x.copy()
        if time in steps:
            target[:, steps[time]] = x[:, 0]
    r = history @ model['representation'].T
    predictions = dict(state=state @ model['state_weights'].T,
        history=history @ model['v_weights'].T, representation=r @ model['r_weights'].T,
        fitted=r @ model['fitted_weights'].T + model['fitted_offset'])
    target = affine_scale*target + affine_offset
    predictions = {k: affine_scale*v+affine_offset for k, v in predictions.items()}
    errors = dict(innovation=target-predictions['state'],
        hidden_state=predictions['state']-predictions['history'],
        compression=predictions['history']-predictions['representation'],
        fitting=predictions['representation']-predictions['fitted'],
        total=target-predictions['fitted'])
    exact = {k: v*affine_scale**2 for k,v in model['exact_terms'].items()}
    exact['total'] = sum(exact.values())
    result = {}
    for name, error in errors.items():
        values = np.mean(error**2, axis=1)
        se = float(values.std(ddof=1)/np.sqrt(n_trials))
        estimate = float(values.mean())
        tolerance = 6*se + 1e-10*max(1.0, exact[name])
        result[name] = dict(analytic_risk=exact[name], empirical_risk=estimate,
            mc_standard_error=se, tolerance=tolerance,
            passed=bool(abs(estimate-exact[name]) <= tolerance))
    output = dict(status="executed exact-process control; not a circuit bound", seed=int(seed),
        n_independent_trials=n_trials, target_steps=list(target_steps), dt_s=.01,
        history_samples=history_samples, process_noise=process_noise,
        observation_noise=observation_noise, affine_scale=affine_scale, affine_offset=affine_offset,
        terms=result, passed=all(row['passed'] for row in result.values()),
        decomposition="same Y: innovation + hidden-state ambiguity + linear compression + fixed fitting error",
        tolerance_policy="six estimated independent-path Monte Carlo standard errors plus numerical tolerance")
    if return_arrays:
        output['_arrays'] = dict(history=history, state=state, representation=r, target=target,
                                   **{f'prediction_{k}':v for k,v in predictions.items()})
    return output


def poisson_block_law(times_s, *, state=1.0, rate_hz=25., amplitude=1., tau_s=.05,
                      projection=None):
    """Exact event-filtered Poisson block moments conditional on its cutoff state.

    Cov(X_t,X_s|C)=a² lambda tau/2 exp(-|t-s|/tau)
    [1-exp(-2 min(t,s)/tau)]. A declared linear target projection is allowed;
    it changes the target and is applied to mean AND full covariance.
    """
    t = _array(times_s, "times_s")
    if t.ndim != 1 or np.any(t <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("future times must increase strictly after cutoff")
    mean = poisson_conditional_mean(t, state, amplitude=amplitude, rate_hz=rate_hz, tau_s=tau_s)
    covariance = amplitude**2*rate_hz*tau_s/2*np.exp(-abs(t[:,None]-t[None,:])/tau_s)
    covariance *= -np.expm1(-2*np.minimum(t[:,None], t[None,:])/tau_s)
    P = np.eye(len(t)) if projection is None else _array(projection, "projection")
    if P.ndim != 2 or P.shape[1] != len(t):
        raise ValueError("projection dimensions do not match future block")
    return P@mean, P@covariance@P.T


def poisson_block_control(*, seed=0, n_futures=8192, rate_hz=25., affine_scale=1.,
                          affine_offset=0., projection=None, return_arrays=False):
    times = np.asarray([.002, .01, .026, .05, .1, .25])
    P = np.eye(len(times)) if projection is None else np.asarray(projection, dtype=float)
    mean, cov = poisson_block_law(times, state=1.3, rate_hz=rate_hz, projection=P)
    rng = stream(seed, "h1/poisson/independent-futures")
    values = np.stack([P@exponential_shot_noise(times,
        poisson_events(rate_hz, 0, times[-1], rng), initial_state=1.3) for _ in range(n_futures)])
    values = affine_scale*values + affine_offset
    mean = affine_scale*mean + affine_offset
    cov = affine_scale**2*cov
    losses = np.mean((values-mean)**2, axis=1)
    exact = float(np.trace(cov)/len(mean))
    se = float(losses.std(ddof=1)/np.sqrt(n_futures))
    variance = per_state_fork_variance(values[None])[0]
    pair = independent_pair_variance(values[None])[0]
    loo = leave_one_out_oracle(values[None])[0]
    output = dict(n_futures=n_futures, target_times_s=times.tolist(), projection=P.tolist(),
        analytic_mean=mean.tolist(), empirical_mean=values.mean(axis=0).tolist(),
        analytic_covariance=cov.tolist(), empirical_covariance=np.atleast_2d(np.cov(values,rowvar=False)).tolist(),
        analytic_risk=exact, empirical_oracle_risk=float(losses.mean()), mc_standard_error=se,
        sample_variance=float(variance), independent_pair_half_mse=float(pair.mean()),
        loo_mse=float(np.mean((loo-values)**2)), loo_corrected_mse=float(np.mean((loo-values)**2)*(n_futures-1)/n_futures),
        passed=bool(abs(losses.mean()-exact) <= 6*se+1e-10),
        affine_scale=affine_scale, affine_offset=affine_offset,
        status="exact exponential-filtered continuous-time Poisson control; not neural EEG")
    if return_arrays:
        output['_arrays'] = dict(target=values, exact_mean=mean, exact_covariance=cov)
    return output


def repeated_future_control(*, seed=0, n_repeats=20000, m_values=(1,2,4,8,16,32,64,128,201)):
    """Hidden-state remainder and scalar-Gaussian finite-M identities.

    Conditional on fixed V, C~N(0,tau²); every context has independent
    innovations N(0,sigma²). This samples posterior C anew per context, then
    replicates futures at THAT same C. No trajectory time variance is used.
    """
    sigma2, tau2 = 1.5, .7
    rng = stream(seed, "h1/repeated")
    state = rng.normal(size=n_repeats)*np.sqrt(tau2)
    innovations = rng.normal(size=(n_repeats,max(m_values)))*np.sqrt(sigma2)
    rows=[]
    for m in m_values:
        means = state + innovations[:,:m].mean(axis=1)
        mean_noise = innovations[:,:m].mean(axis=1)
        row=dict(m=int(m), physical_single_future_variance=sigma2,
            analytic_mean_variance_given_state=sigma2/m,
            empirical_mean_variance_given_state=float(mean_noise.var(ddof=1)),
            analytic_mean_variance_given_visible_history=sigma2/m+tau2,
            empirical_mean_variance_given_visible_history=float(means.var(ddof=1)),
            hidden_state_remainder=tau2,
            sample_variance_status="unavailable" if m==1 else "available")
        noise_se=float(np.std(mean_noise**2,ddof=1)/np.sqrt(n_repeats))
        visible_se=float(np.std(means**2,ddof=1)/np.sqrt(n_repeats))
        row.update(mean_variance_given_state_mc_standard_error=noise_se,
            mean_variance_given_visible_mc_standard_error=visible_se,
            mean_identity_passed=bool(abs(mean_noise.var(ddof=1)-sigma2/m)<=6*noise_se and
                                     abs(means.var(ddof=1)-(sigma2/m+tau2))<=6*visible_se))
        if m>1:
            sample_var=innovations[:,:m].var(axis=1,ddof=1)
            row.update(analytic_sample_variance_variance=2*sigma2**2/(m-1),
                empirical_sample_variance_variance=float(sample_var.var(ddof=1)),
                sample_variance_relative_sd=float(sample_var.std(ddof=1)/sigma2),
                analytic_relative_sd=float(np.sqrt(2/(m-1))))
            variance_se=float(np.std((sample_var-sigma2)**2,ddof=1)/np.sqrt(n_repeats))
            row.update(variance_estimator_variance_mc_standard_error=variance_se,
                variance_identity_passed=bool(abs(sample_var.var(ddof=1)-2*sigma2**2/(m-1))<=6*variance_se))
        rows.append(row)
    return dict(rows=rows, gaussian_scalar_m_for_relative_sd_at_most_point_one=201,
                interpretation="201 is an exact IID Gaussian variance-estimation example, not a universal EEG cutoff",
                n_independent_context_repetitions=n_repeats,
                passed=all(r['mean_identity_passed'] and r.get('variance_identity_passed',True) for r in rows),
                tolerance_policy='six independent-context estimated Monte Carlo standard errors')


def run_controls(output_root, seed=0):
    """Write real small Monte Carlo checks and scientific PNG/PDF artifacts."""
    root=Path(output_root); root.mkdir(parents=True, exist_ok=True)
    if (root/'controls.json').exists(): raise FileExistsError('Control output exists; preserve it and choose a new root')
    from .artifacts import atomic_npz, file_hash, source_hash
    gaussian=[]
    for q, noise in ((0.,0.), (4.,0.), (4.,.5)):
        for steps in ((1,), (1,4,10)):
            gaussian.append(gaussian_information_control(seed=seed,process_noise=q,
                observation_noise=noise,target_steps=steps,return_arrays=True))
    poisson=[poisson_block_control(seed=seed,return_arrays=True), poisson_block_control(seed=seed,rate_hz=0.,return_arrays=True),
             poisson_block_control(seed=seed,projection=np.full((1,6),1/6),return_arrays=True)]
    scaled=gaussian_information_control(seed=seed,affine_scale=3.2,affine_offset=-7.,return_arrays=True)
    repeated=repeated_future_control(seed=seed)
    records=[]
    for name,group in (('gaussian',gaussian),('poisson',poisson),('affine',[scaled])):
        for i,control in enumerate(group):
            path=root/f'{name}_{i}_samples.npz'
            atomic_npz(path,**control.pop('_arrays'))
            records.append(dict(path=path.name,sha256=file_hash(path),bytes=path.stat().st_size))
    result=dict(schema_version="eegrepr-h1-controls-2.0", seed=seed,
        status="executed analytical controls; no empirical biophysical conclusion",
        source_hash=source_hash(),sample_artifacts=records,
        gaussian=gaussian, poisson=poisson, affine_gaussian=scaled, repeated=repeated,
        passed=all(x['passed'] for x in gaussian+poisson+[scaled]) and repeated['passed'],
        reused_estimators="statistics.per_state_fork_variance, independent_pair_variance, leave_one_out_oracle")
    atomic_json(root/'controls.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    terms=gaussian[-1]['terms']; names=['innovation','hidden_state','compression','fitting']
    axes[0].bar(names,[terms[k]['analytic_risk'] for k in names],label='Exact')
    axes[0].errorbar(names,[terms[k]['empirical_risk'] for k in names],
        yerr=[6*terms[k]['mc_standard_error'] for k in names],fmt='o',color='black',label='MC ±6 SE')
    axes[0].tick_params(axis='x',rotation=25); axes[0].set_ylabel('Same-target mean squared error')
    axes[0].legend()
    rows=repeated['rows']; ms=[r['m'] for r in rows]
    axes[1].plot(ms,[r['analytic_mean_variance_given_state'] for r in rows],label='Given full state')
    axes[1].plot(ms,[r['analytic_mean_variance_given_visible_history'] for r in rows],label='Given visible history')
    axes[1].scatter(ms,[r['empirical_mean_variance_given_visible_history'] for r in rows],s=14)
    axes[1].set(xscale='log',xlabel='Independent futures M',ylabel='Variance of future mean')
    axes[1].legend(); fig.suptitle('Exact-process validation — not a circuit or human EEG bound')
    for suffix in ('png','pdf'): fig.savefig(root/f'controls.{suffix}',dpi=180)
    plt.close(fig)
    return jsonable(result)
