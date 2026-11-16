"""Evidence-backed reports and publication-size figures; absent outcomes stay unrun."""
from pathlib import Path
import csv,json
import numpy as np
from .artifacts import read_json,atomic_json,file_hash,canonical_hash,ROOT,module_hash
from .config import counts
from .statistics import family_b

STAGES={'audit':'audit_existing.json','measurement':'measurement/qualification.json','derivation':'derived/manifest.json',
'models':'models/manifest.json','calibration':'calibration/manifest.json','scores':'scores/manifest.json','retention':'retention/summary.json',
'replication':'replication/manifest.json','geometry':'geometry/summary.json','numerical':'numerical/summary.json',
'precision':'precision_certificate.json','plan':'plan.json','protocol':'protocol.json','generation':'generation_result.json','primary_inference':'confirmation_statistics.json'}


def report(c,root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    stages={k:dict(status=read_json(root/v).get('status','complete'),path=v,sha256=file_hash(root/v)) if (root/v).exists() else dict(status='unrun',path=v) for k,v in STAGES.items()}
    findings=dict(status='complete',claim_status=c['claim_status'],stages=stages,config_hash=canonical_hash(c),counts=counts(c),selected_results=[])
    for p in sorted((root/'scores').glob('*/summary.json')):
        s=read_json(p);a=s['selected'];r=family_b(a['member_circuit_risks'] if a.get('selected_neural_policy') else a['model_circuit_risks'],a['baseline_circuit_risks'],neural=bool(a.get('selected_neural_policy')),bootstrap_draws=1000)
        r['descriptive_interval_location']=r.pop('decision')
        r['family']='descriptive_grid'
        r['formal_decision']='Only confirmation_statistics.json contains frozen primary family decisions'
        findings['selected_results'].append(dict(job=s['job_id'],candidate=a['candidate'],**r))
    if (root/'confirmation_statistics.json').exists():findings['primary_inference']=read_json(root/'confirmation_statistics.json')
    if (root/'plan.json').exists():findings['execution_blockers']=read_json(root/'plan.json')['blocked_reasons']
    if (root/'precision_certificate.json').exists():findings['precision']=read_json(root/'precision_certificate.json')
    text=['# NeuroStimEnv H1–H4 v6 evidence report','',
        f"Status: {c['claim_status']}. Each stage below is an observed artifact status, not a claim that the planned study completed.",'',
        'The simulator is a 32E/8I ball-and-stick toy circuit with deterministic canonical Hodgkin–Huxley somata at 6.3 °C, passive cables and stochastic external synaptic events. EEG is one ideal forward-model channel. Circuit seeds are simulator populations, not human subjects. New numerical controls and analytical tests are labelled separately from neural evidence.','',
        'H1 measures physical-state conditional future variance and the remaining combined observation/compression/fitting error. It does not identify the EEG-history Bayes floor or separate every error component. Risk and variance use the same target, physical support, normalization and population. Negative finite-M corrected excess estimates remain in the tables.','',
        'The prospective H2 protocol specifies eight registered circuits, 32 contexts, M=1/2/4/8/16/32 and a separate L=128 reference; these allocations are not observations unless replication is complete below. Every reference-mean squared error subtracts estimated reference variance/L and retains reference uncertainty. Extra futures and optimizer seeds do not increase circuit N. Primary forecast risk always uses the first four registered scoring futures.','',
        'The prospective H3 primary family A compares Feature18 and train-PCA after d8/B32 quantile .005/.995 coding, with development-selected linear readouts on the same A0 histories. Cross-entropy is mean negative log2 probability of the true carrier class, conditional on rhythmic membership. It measures decoder performance, not total mutual information. The conjunction requires PCA-minus-Feature18 carrier CE >0.20 bits and Feature18-minus-PCA reconstruction MSE divided by the common fitted-mean history baseline >0.02. Both one-sided 97.5% lower bounds must exceed their margins; the intersection-union p-value is the larger component p-value.','',
        'The prospective H1 primary family B selects one full-four-second A1 predictor using development only. Relative MSE improvement G=100×(1−mean circuit model risk/mean circuit training-mean baseline risk); it is a ratio of means, not a mean of per-circuit ratios. A 97.5% circuit t/influence interval entirely above +1 percentage point is meaningful improvement, entirely within [−1,+1] is equivalence for this fitted predictor, entirely below −1 is meaningful worsening; otherwise the result is inconclusive. The two families each receive alpha .025. Bootstrap and sign diagnostics are sensitivities; sign flipping requires symmetry.','',
        'Means, checkpoint selection and residual shrinkage use train/development only. The prospective protocol allocates twenty-four separate circuits to fit the same context-independent diagonal residual-variance family per frozen mean; this has not occurred unless calibration is complete below. Raw, development-calibrated and independently calibrated diagnostics are labelled separately. Coverage is marginal and parametric, not a conditional/joint or distribution-free guarantee. Five neural repeat risks are averaged inside circuit; predictions are not ensembled for primary risk.','',
        'H4 holds targets fixed across A0 legacy FIR, A1 completed 2-ms native bins and A2 causal 4–30 Hz SOS acquisition. Native targets use (6002,6034] ms and (6010,6138] ms. T3 is 54 orthonormal real Fourier coordinates of the 16,000 samples in (6010,7010] ms, available only after the full second. Direct future-only band energies are sum(c²)/16000 in [8,12), [12,20), [20,30] Hz; units V² and risk V⁴. Natural logs use a fixed 1e−40 V² floor and 1 V² reference. Every nonlinear target is applied to each future before estimating variance. Squaring a predicted mean is not predicting expected energy. Residual coordinate-lag diagnostics for T3 follow Fourier component order and are not temporal autocorrelations. Acquisition-row comparisons describe tuned predictor pipelines; they do not alone isolate an information-loss effect.','',
        'The required full-history qualification includes all 14 tail-ridge candidates, whole-history raw/PCA/Fourier/Feature18 regressions and a ≤50k-parameter TCN with receptive field ≥2000 samples, plus a 128-ms ablation. Maximum 10,000 updates, dev early stopping, five seeds and a training-mean fallback are fixed. Legacy 200-update transformer banks are secondary procedural context; H5/H6 objective/scheduling superiority remain deferred.','',
        'Inspected v4 circuits are exploratory design evidence. Fresh primary inference is conditional on this training set/model-selection policy, not variation across independently sampled training datasets. Geometry is scaling-sensitive descriptive covariance, not biological intrinsic dimension.','',
        '## Recorded stages','',
        f"This root enables neural fits: {c.get('models',{}).get('neural_enabled',True)}. A complete models stage means the configured grid finished; a linear-only grid does not complete the required neural qualification.",'']
    text += [f"- {k}: {v['status']} ({v['path']})." for k,v in stages.items()]
    if findings['selected_results']:
        text += ['', '## Observed development-selected results', '', 'All rows below are descriptive. Intervals are unadjusted across this grid; no row is promoted to fresh confirmation by appearing here. A zero-width interval for the training-mean fallback is an algebraic identity, not evidence against EEG predictability.', '', '| Acquisition / target | Selected candidate | G (percentage points) | 97.5% circuit interval | Circuits |', '|---|---|---:|---:|---:|']
        for r in findings['selected_results']:
            text.append(f"| {r['job']} | {r['candidate']} | {r['estimate']:.4f} | [{r['lower']:.4f}, {r['upper']:.4f}] | {r['n_circuits']} |")
    if findings.get('execution_blockers'):text+=['','## Execution blockers','']+[f'- {s}' for s in findings['execution_blockers']]
    text+=['','## Numeric evidence and figures','',
        'Unrounded JSON/NPZ files are authoritative. Figures are vector PDF/SVG plus 300-dpi PNG, with circuit-level intervals and explicit estimands. A figure is created only when the corresponding outcomes exist. Missing results are never drawn from plans. See figure_captions.txt for populations, units, support and uncertainty.','',
        'Compact exports retain target arrays, every saved prediction/variance/logit/code, IDs, normalization, coefficients, tables and protocol. Numeric NPY members are deduplicated without changing a byte. Native files and numeric fitted weights have checksummed omission records; request the separate full-native option to include them. No pickle loading is required.']
    (out/'FINDINGS.md').write_text('\n'.join(text)+'\n');atomic_json(out/'findings.json',findings)
    figures(root,c,findings)
    if c.get('extension',{}).get('enabled'):
        from .extended_reporting import report as extended_report
        findings['extension']=extended_report(c,root,findings)
    return findings


def figures(root,c,findings):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import t
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    out=Path(root)/'report';captions=[]
    rows=[]
    for p in sorted((Path(root)/'scores').glob('*/summary.json')):
        s=read_json(p);a=s['selected'];r=family_b(a['member_circuit_risks'] if a.get('selected_neural_policy') else a['model_circuit_risks'],a['baseline_circuit_risks'],neural=bool(a.get('selected_neural_policy')),bootstrap_draws=1000)
        rows.append((s['job_id'],r))
    if rows:
        # Read statistical schema explicitly to avoid guessing uncertainty from plots.
        fig,ax=plt.subplots(figsize=(6.75,max(2.8,.23*len(rows))))
        for i,(job,r) in enumerate(rows):
            estimate=r['estimate'];lo,hi=[r['lower'],r['upper']];color={'A0':'#0072B2','A1':'#D55E00','A2':'#009E73'}[job[:2]]
            ax.errorbar(estimate,i,xerr=[[estimate-lo],[hi-estimate]],fmt='o',markersize=3,capsize=2,color=color)
        ax.axvspan(-1,1,color='.92');ax.axvline(0,color='.4',lw=.6);ax.set_yticks(range(len(rows)),[s.replace('__',' / ') for s,_ in rows]);ax.invert_yaxis()
        ax.set_xlabel('Relative MSE improvement versus training mean (percentage points)')
        ax.set_title('Development-selected policies; 97.5% circuit intervals')
        fig.tight_layout()
        for ext in ('pdf','svg','png'):fig.savefig(out/('forecast_grid.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
        captions.append('forecast_grid: Each row is a different explicitly named acquisition/target contract. G=100(1−mean circuit model MSE/mean circuit train-mean MSE), using first four registered score futures and averaging neural repeat risks inside circuit. Whiskers are 97.5% paired-circuit t/influence intervals; these are descriptive across the grid, the A1 short target is designated for prospective family B, but inspected-data results remain exploratory. Gray region marks ±1pp. Raw standardized MSE values across targets are not compared. Claim status: '+c['claim_status']+'.')
        with (out/'forecast_grid.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['job','gain_pp','ci97_5_low_pp','ci97_5_high_pp','n_circuits'])
            for job,r in rows:w.writerow([job,r['estimate'],*[r['lower'],r['upper']],r['n_circuits']])
    pp=Path(root)/'retention/primary_components.json'
    if pp.exists():
        a=read_json(pp);rows=a['per_circuit'];base=np.mean([r['reconstruction_baseline_mse'] for r in rows])
        x=np.array([(r['reconstruction_mse_feature']-r['reconstruction_mse_pca'])/base for r in rows]);y=np.array([r['task_difference_bits'] for r in rows])
        fig,ax=plt.subplots(figsize=(3.25,2.8));ax.scatter(x,y,s=14,color='#0072B2',alpha=.7);ax.axvline(.02,color='#D55E00',ls='--',lw=.8);ax.axhline(.2,color='#D55E00',ls='--',lw=.8)
        ax.axvline(0,color='.7',lw=.5);ax.axhline(0,color='.7',lw=.5);ax.set_xlabel('Feature18 − PCA reconstruction NMSE');ax.set_ylabel('PCA − Feature18 carrier CE (bits)');ax.set_title('Matched d8/B32: one point per circuit');fig.tight_layout()
        for ext in ('pdf','svg','png'):fig.savefig(out/('task_reconstruction_contrast.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
        captions.append('task_reconstruction_contrast: A0 d8/B32, fixed train quantile .005/.995 ranges, all-label linear readouts selected on development. Carrier is conditional on rhythmic membership; cross entropy is decoder-dependent negative log2 probability. Reconstruction uses the same fitted-mean history denominator. Dashed lines show prespecified benchmark margins .02 baseline NMSE and .20 bits; the conjunction requires inference on both population means, not individual points. Points are circuits, not optimizer repeats. Formal confidence bounds and IUT appear only in confirmation_statistics.json after a frozen fresh test; status '+c['claim_status']+'.')
    (out/'figure_captions.txt').write_text('\n\n'.join(captions) if captions else 'No evaluated outcome artifacts are present. No outcome figures generated from plans.\n')
