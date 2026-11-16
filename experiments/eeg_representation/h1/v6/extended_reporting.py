"""Readable fixed-population H1--H4 results; planned rows never become observations."""
from pathlib import Path
import csv,json
import numpy as np
from .artifacts import atomic_json,read_json,file_hash,canonical_hash
from .targets import timing_metadata
from .models import settings as model_settings


def _read(path,default=None):
    return read_json(path) if path.exists() else default


def _number(x,digits=3):
    return 'unknown' if x is None else f'{float(x):.{digits}f}'


def _interval(row,key='error_baseline100_interval'):
    ci=row.get(key) or {}
    return f"[{_number(ci['lower'])}, {_number(ci['upper'])}]" if ci.get('status')=='estimated' else 'not estimated'


def _write_csv(path,rows,columns):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items() if k in columns})


def report(config,root,legacy=None):
    root=Path(root);out=root/'report';tables=out/'tables';tables.mkdir(parents=True,exist_ok=True)
    derived=_read(root/'derived/manifest.json',{});information=[]
    for p in sorted((root/'information').glob('*/summary.json')):
        s=read_json(p)
        if s.get('status')=='complete':information.append(s)
    contrasts=_read(root/'information/paired_contrasts.json',{'rows':[]})
    replay=_read(root/'replay/summary.json');retention=_read(root/'retention/summary.json');replication=_read(root/'replication/manifest.json',{})
    primary=_read(root/'confirmation_statistics.json');plan=_read(root/'plan.json',{})
    cohort=config['claim_status'];fresh=cohort=='prospective_confirmation'
    stagepaths={'Derivation':'derived/manifest.json','Ordinary model grid':'models/manifest.json','State/code future readouts':'information_models/manifest.json','Point forecast scores':'scores/manifest.json','Uncertainty calibration':'calibration/manifest.json','Information/error accounting':'information/manifest.json','H2 finite-reference analysis':'replication/manifest.json','H3 matched-code tasks':'retention/summary.json','Numerical convergence':'numerical/summary.json','Exact-forcing replay':'replay/summary.json','Primary confirmation decisions':'confirmation_statistics.json'}
    completion=[]
    for name,path in stagepaths.items():
        value=_read(root/path)
        completion.append(dict(stage=name,cohort=cohort,status=value.get('status','complete') if value else 'not yet run',source=path,sha256=file_hash(root/path) if value else None))
    legacy_scores=(legacy or {}).get('selected_results',[])
    result=dict(status='complete',claim_status=cohort,config_hash=canonical_hash(config),completion=completion,
        completed_information_jobs=len(information),configured_jobs=len(config['selected_job_ids']),
        unavailable_derivations=derived.get('unavailable_jobs',[]),information_tables=information,paired_contrasts=contrasts,
        original_primary_decisions=primary,replay=replay,primary_protocol='Original H3 conjunction and H1 A1/h2/W32, alpha .025 each, unchanged; all extension comparisons secondary/descriptive',
        ordinary_neural_enabled=model_settings(config)['neural_enabled'],source_scope='Only artifacts in this evidence root; historical outcomes are not promoted to fresh confirmation',
        unidentified=['ideal EEG-history Bayes risk','separate observation/compression/fitting gaps','complete-state fitted oracle when only membrane/gate features exist'])
    atomic_json(tables/'information_error.json',dict(tables=information,replay_separate_population=replay))
    flat=[dict(job_id=s['job_id'],physical_mse_units=s['physical_mse_units'],**r) for s in information for r in s['rows']]
    _write_csv(tables/'information_error.csv',flat,['job_id','id','information_available','method','status','physical_mse_units','physical_mse','reference_mse','risk_ratio','error_baseline100','error_baseline100_interval','relative_mse_reduction_pp','relative_mse_reduction_pp_interval','state_variance','combined_unresolved_excess','n_circuits','n_contexts','n_futures_per_context','sources'])
    atomic_json(tables/'paired_contrasts.json',contrasts)
    _write_csv(tables/'paired_contrasts.csv',contrasts['rows'],['comparison_axis','job_ids','method_id','gain_difference','physical_mse_difference','n_circuits','n_contexts','conditional_mc_gain_difference_se','held_fixed','selected_family_changed'])
    _write_csv(tables/'completion.csv',completion,['stage','cohort','status','source','sha256'])
    timing=[dict(job_id=j,**timing_metadata(j.split('__',1)[1])) for j in config['selected_job_ids']]
    atomic_json(tables/'target_timing.json',dict(rows=timing))
    h2,h3,h4=_quantitative_sections(root,tables,information,replication,retention)
    historical_h3=_historical_h3(root)
    atomic_json(tables/'historical_h3.json',historical_h3)
    result.update(h2_numeric=h2,h3_numeric=h3,h4_numeric=h4,historical_h3=historical_h3)
    text=['# NeuroStimEnv V6 extended: H1–H4 main analysis','',
        f"This report contains {len(information)} completed information/error tables from the **{cohort}** cohort. The configured design contains {len(config['selected_job_ids'])} acquisition–target jobs; this count is a plan, not a completed experiment count.",
        'The circuit is a 32-excitatory/8-inhibitory canonical 6.3 °C toy model with one ideal EEG channel. The results characterize this simulator and these fitted procedures; they do not establish human prediction thresholds or fractions of total information.',
        '', '## Completion and cohorts','',
        '| Cohort / component | Recorded status | Evidence |','|---|---|---|',
        '| Historical V4 recordings | Imported/exploratory only when listed in imports and audit_existing.json | Original generation provenance retained |',
        '| Fresh V6 confirmation | '+('Observed stages listed below; completeness requires the final campaign receipt' if fresh else 'Not run in this evidence root')+' | campaign_completed.json in campaign parent |']
    text += [f"| {r['cohort']} / {r['stage']} | {r['status']} | {r['source']} |" for r in completion]
    text += ['',f"The ordinary neural grid is {'enabled' if model_settings(config)['neural_enabled'] else '**disabled for this linear-only local analysis**'}. Completion of a linear grid does not qualify the full prospective neural campaign.",
        'Inspected circuits remain exploratory. Futurity, prefixes, conditions and optimizer repeats do not increase the number of independent circuits. A missing artifact is not evidence of a negative effect.',
        '', '## Metrics and timing','',
        'Mean squared error (MSE) averages squared target-coordinate error and then gives circuits, regimes, prefixes and futures equal weight at their respective levels. Physical waveform MSE is in V², energy MSE in V⁴ and log-energy MSE in squared natural-log units. Fourier coefficients retain the frozen orthonormal V scaling. Target-specific training normalization and the training-mean prediction are shared across acquisitions.',
        'For readability, error = 100 × model physical MSE / training-mean physical MSE. Relative reduction in MSE is G = 100 − error, in percentage points. These are ratios of equally circuit-weighted risks, not averages of circuit ratios. A baseline value of 100 is a display scale, not V². Unrounded physical values are in tables/information_error.json and CSV.',
        'The target start offset h is the left-open boundary: h=0 starts (0,2] ms; h=2 starts (2,4] ms; h=10 starts (10,12] ms, relative to cutoff. The cutoff sample is excluded. The first native future sample for h=0 is +0.0625 ms. Each waveform bin averages exactly 32 native samples. No acquisition timestamps or causal filter memory are shifted.',
        'Reported 97.5% circuit intervals are descriptive for the extension. Synchronized delete-one-future Monte Carlo diagnostics are recorded separately and are not added again to circuit variance. The original H3 conjunction and H1 A1/h2/W32 family each retain alpha .025; only confirmation_statistics.json may make their frozen fresh-cohort decisions.',
        '', '## H1 — Which future quantities can be predicted, from which information?','',
        'Question: for an identical physical target and identical held-out contexts/futures, how do achieved errors differ with available information? Ordinary learners receive only causal EEG. Partial-state learners receive verified membrane/gate summaries, omitting phase, synaptic state and pending events. The code comparison uses A0 four-second histories, Feature18/PCA/three frozen legacy-forecast repeats, d8/B32 plus projected-unquantized controls, the same seven ridge penalties and a mean fallback.',
        'At each context, unbiased sample variance across independent matched-state futures estimates future variability at the same present state. It is an estimated reference, not a trained hidden-state model. Model MSE minus that reference retains the finite-M correction and any negative estimates. It combines hidden-state ambiguity, compression and imperfect fitting; the separate ideal components are not identified.',
        'The theoretical deterministic error with complete causal state, parameters and the actual future external event schedule is zero. Measured full-prefix replay appears only in its separate registered diagnostic population. No theoretical zero or fitted EEG error is substituted for the unknown EEG-history Bayes floor.']
    mainjobs=['A0__T0_l10_d128','A0__T0_l0_d128','A1__T0_l10_d128','A1__T0_l0_d128']
    byjob={s['job_id']:s for s in information};paragraphs=[]
    for job in mainjobs:
        s=byjob.get(job);text+=['',f'### {job}','']
        if s is None:text+=['Unavailable or not yet run. This missing result is not plotted as zero.'];continue
        pop=s['population'];text += [f"Fixed target: {job.split('__')[1]}; acquisition {s['acquisition']}; {pop['n_circuits']} circuits, {pop['n_contexts']} contexts, the same first four registered futures ({pop['n_scoring_episodes']} episodes). Physical MSE units: {s['physical_mse_units']}. Equal circuit/regime/prefix weighting. Intervals below are 97.5% circuit intervals.",'',
            '| Information available | Ideal limit or fitted method | Error (baseline=100) | Circuit interval | Evidence status |','|---|---|---:|---|---|']
        for r in s['rows']:
            if r.get('main_table') is False:continue
            method=r['method']+((' ['+r['id']+']') if r.get('information_available')=='compressed EEG representation' else '')
            text.append(f"| {r['information_available']} | {method} | {_number(r.get('error_baseline100'))} | {_interval(r)} | {r['status']} |")
        row=next((r for r in s['rows'] if r['id']=='eeg_full_selected'),None)
        floor=next((r for r in s['rows'] if r['id']=='state_variance'),None)
        if row:
            sentence=f"On {job}, the development-selected four-second policy reduced MSE by {_number(row['relative_mse_reduction_pp'])} percentage points (97.5% circuit interval {_interval(row,'relative_mse_reduction_pp_interval')}; N={row['n_circuits']})."
            if floor:sentence+=f" Estimated same-state future variance was {_number(floor['error_baseline100'])} on the baseline-100 scale."
            sentence+=' This is an achieved fitted-policy result; the ideal EEG limit and separate information-loss components remain unresolved.'
            paragraphs.append(sentence);text+=['',sentence]
    text += ['', '### Matched comparisons and uncertainty models','',
        'All gap, duration, history and acquisition contrasts are paired on the same circuit/context/future IDs in tables/paired_contrasts.json. Family-specific policies keep the candidate family and tuning grid fixed. Overall selected-policy contrasts disclose when the winning family changes. Different target supports retain separate baseline denominators, and physical-MSE differences are also saved.',
        'Point-forecast risk, Gaussian uncertainty diagnostics and deterministic replay answer different questions. The original score summaries retain Gaussian negative log likelihood, marginal CRPS, 50/80/95% coverage, PIT, interval width and coordinate-lag diagnostics for uncalibrated, development-calibrated and separate-calibration variance families when available. Marginal coverage is not joint or conditional calibration. Fourier-coordinate lag is not temporal lag.',
        '', '## H2 — How precisely is uncertainty estimated?','',
        'Question: how much does finite sampling of futures affect the estimated conditional mean and variance? The registered design uses nested M=1,2,4,8,16,32 scoring futures and a disjoint L=128 reference pool on fixed H2 contexts. Repeated prefixes/futures improve conditional Monte Carlo precision, not circuit population size. All immediate and delayed targets reuse these same future IDs.',
        'The squared distance between a sample mean and a finite reference mean contains noise from both pools. The analysis subtracts estimated reference variance/L, retains negative corrected estimates, and carries both-pool/reference uncertainty. At M=1, within-context sample variance around an unknown mean is unavailable; a pooled variance learner requires additional modeling assumptions. New-future variability does not vanish when all M32 futures are reused.',
        f"Recorded H2 status: **{replication.get('status','not yet run')}**. Per-target outcomes and reasons are in replication/manifest.json. Numerical outputs are never filled from the planned M/L values.",
        '', '## H3 — Does reconstruction quality agree with task usefulness?','',
        'Question: under a matched A0/d8/B32 code budget, does the representation that reconstructs the past better also decode the carrier better? The original primary comparison is Feature18 versus PCA: PCA-minus-Feature18 carrier cross entropy must exceed 0.20 bits, and Feature18-minus-PCA reconstruction error divided by the shared history baseline must exceed 0.02. Both one-sided 97.5% lower bounds must exceed their margins (intersection-union rule). Carrier scoring is conditional on rhythmic membership.',
        f"Recorded retention status: **{retention.get('status','not yet run') if retention else 'not yet run'}**. Primary inference: **{primary.get('status','available; inspect decision record') if primary else 'not yet run'}**.",
        'Past reconstruction MSE, future-target MSE and carrier cross entropy are distinct endpoints with different denominators. The new future-readout table is separate from the original task/reconstruction table; no CE or reconstruction number substitutes for future prediction. Unquantized/projected/B32 results and clipping expose projection/quantization limitations where measured. Decoder error does not measure total mutual information, and differing fitted errors do not isolate pure compression loss. The legacy 200-update forecast banks are procedural comparisons, not H5/H6 objective-superiority evidence.',
        '', '## H4 — How do target definition, gap and duration change the answer?','',
        'The waveform grid is h={0,2,10} ms × W={32,128} ms. The selected real-Fourier block is h={0,10} ms × 1000 ms, with 54 orthonormal sine/cosine coefficients at integer frequencies 4 through 30 Hz. It drops DC and outside-band content. The target is future-only and available after the entire one-second block; zero gap does not make that block immediately observable.',
        'Direct energies sum squared selected Fourier coefficients divided by 16,000 in alpha [8,12), low beta [12,20), and high beta [20,30] Hz. Their units are V². Natural-log energy uses the fixed 10⁻⁴⁰ V² floor and 1 V² reference. Floor activations and training-only normalization are saved. Every nonlinear transform is applied to each individual future before computing its conditional mean/variance. Expected energy is not the square of an expected waveform.',
        'A0 remains the original causal FIR as realized; A1 is completed native two-millisecond box bins; A2 is the qualified causal 4–30 Hz SOS acquisition. These are different information conditions. Physical targets and target normalization remain identical across A0/A1/A2, with actual causal input availability and filter state retained.',
        '', '## Claims, limits and next evidence','',
        '| Claim | Supporting result / interval | Practical interpretation | Limitation | Additional evidence needed |','|---|---|---|---|---|',
        f"| Achieved EEG forecasting performance | {len(information)} fixed-setting tables; exact estimates and intervals above/supplement | Describes the permitted fitted policies | Does not identify ideal EEG risk | Complete frozen fresh campaign and neural qualification |",
        '| Same-state future variability | Unbiased within-context variance with circuit and conditional-MC diagnostics when M≥2 | Irreducible for a predictor given that complete state but no future inputs | Finite-future and finite-circuit uncertainty | Registered M32/L128 precision analysis |',
        '| Representation usefulness versus reconstruction | retention/primary_components.json and confirmation_statistics.json when present | Tests the frozen task/reconstruction conjunction | Readout-dependent; no total-information claim | Fresh primary inference and matched future-readout evidence |',
        '| Immediate versus delayed forecasts | Paired contrasts only for completed supports | Tests a changed prediction problem with fixed policy | Missing native arrays prevent archived h0 outcomes | Derive h0 from verified native recordings; run full models |',
        '| Deterministic future-input replay | replay/summary.json, separate population, when complete | Tests reproducibility given privileged realized forcing | Full-prefix replay is not a saved checkpoint or realistic EEG prediction | Measured eight-context diagnostic on actual host |',
        '', '## Files and reproduction','',
        'MAIN_ANALYSIS.md and FINDINGS.md are identical main reports. main_analysis.json retains unrounded records. tables/information_error.{json,csv}, tables/paired_contrasts.{json,csv} and tables/target_timing.json contain the full matched tables. figures/ contains publication-size PDF/SVG/PNG and captions.json. Original detailed grid and diagnostic files remain available. Compact exports retain float64 targets/predictions/variance/code arrays losslessly and list omitted native files and weights by hash. H5/H6 remain deferred.']
    sections={
        '## H3 — Does reconstruction quality agree with task usefulness?':_h2_text(h2),
        '## H4 — How do target definition, gap and duration change the answer?':_h3_text(h3)+historical_h3['text'],
        '## Claims, limits and next evidence':_h4_text(h4)}
    for heading,addition in sections.items():
        position=text.index(heading);text[position:position]=addition
    if plan.get('blocked_reasons'):text+=['','## Recorded execution blockers','']+['- '+str(x) for x in plan['blocked_reasons']]
    if result['unavailable_derivations']:text+=['',f"Unavailable derivations recorded: {len(result['unavailable_derivations'])}. See derived/manifest.json for precise native-support reasons."]
    atomic_json(out/'main_analysis.json',result)
    for name in ('MAIN_ANALYSIS.md','FINDINGS.md'):(out/name).write_text('\n'.join(text)+'\n')
    (out/'manuscript_results.md').write_text('# Results paragraphs — completed evidence only\n\n'+('\n\n'.join(paragraphs) if paragraphs else 'No measured forecast outcomes are present in this root. No results paragraph is inferred from a plan.')+'\n\nAll completed inspected-cohort statements remain exploratory; original primary inference and fresh-cohort status must be reported separately.\n')
    figures(config,root,information,contrasts,replication)
    return dict(status='complete',main_report='report/MAIN_ANALYSIS.md',numeric='report/main_analysis.json',completed_information_jobs=len(information))




def _historical_h3(root):
    from scipy.stats import t
    audit=_read(Path(root)/'audit_existing.json',{})
    values=audit.get('numeric_primary_component_reproduction',{});components={r['name']:r for r in audit.get('precision_components',[])}
    rows=[]
    for name,key,label in [('new_primary_PCA_minus_feature18_task','new_PCA_task_difference_bits','PCA minus Feature18 carrier CE (bits)'),('new_primary_feature18_minus_PCA_reconstruction','new_PCA_reconstruction_difference_nmse','Feature18 minus PCA past-reconstruction NMSE')]:
        if name not in components or key not in values:continue
        comp=components[name];n=comp['n_pilot_circuits'];point=values[key];half=t.ppf(.9875,n-1)*comp['sample_sd']/np.sqrt(n)
        rows.append(dict(endpoint=label,estimate=point,lower=float(point-half),upper=float(point+half),confidence=.975,n_circuits=n,status='reproduced historical exploratory evidence',source='audit_existing.json',source_archive_sha256=values['source_archive_sha256']))
    text=[]
    if rows:
        text=['','### Preserved historical H3 evidence','', 'These are reproduced inspected V4 results, distinct from the current future-readout analysis and from the unrun fresh V6 conjunction. Intervals are descriptive 97.5% circuit t/influence intervals; their computation does not make the cohort confirmatory.', '', '| Historical endpoint | Estimate | 97.5% circuit interval | N |','|---|---:|---|---:|']
        for r in rows:text.append(f"| {r['endpoint']} | {r['estimate']:.6f} | [{r['lower']:.6f}, {r['upper']:.6f}] | {r['n_circuits']} |")
        text+=['']
    return dict(status='complete' if rows else 'unavailable',rows=rows,text=text)

def _quantitative_sections(root,tables,information,replication,retention):
    from .statistics import ratio_of_means_interval
    h2=[]
    for job in replication.get('jobs',[]):
        if job['status']!='complete':continue
        source=root/job['path'];record=read_json(source);manifest=read_json(root/'derived'/job['id']/'manifest.json')
        scale=float(manifest['normalization']['y_scale']);groups={}
        # Conditional mean/variance diagnostics are identical across optimizer repeats.
        for context in record['records']:
            if context['fitting_repeat']!=0:continue
            for row in context['rows']:
                for key,power in [('reference_noise_corrected_mean_mse',2),('reference_noise_corrected_variance_mse',4),('scoring_variance',2),('reference_estimate_variance',2)]:
                    value=row['nested'].get(key)
                    if value is not None:groups.setdefault((row['M'],key,power),{}).setdefault(str(context['structure_id']),[]).append(value*scale**power)
        for (m,key,power),circuits in sorted(groups.items()):
            values=np.asarray([np.mean(x) for x in circuits.values()]);ci=ratio_of_means_interval(values,np.ones(len(values)),confidence=.975) if len(values)>1 else None
            h2.append(dict(job_id=job['id'],M=m,L=128,metric=key,physical_estimate=float(values.mean()),physical_circuit_interval=ci,n_circuits=len(values),n_contexts=record['n_contexts'],scale_power=power,target_units=manifest['target_spec']['units'],source=job['path'],source_sha256=file_hash(source),uncertainty='Circuit interval includes finite-future variation; both-pool conditional intervals/bootstrap remain in source, not added twice'))
    atomic_json(tables/'h2_precision.json',dict(status='complete' if h2 else 'unavailable',rows=h2))
    h3=[]
    if retention:
        for r in retention.get('rows',[]):
            if r.get('status')!='complete' or r.get('label_budget')!='all':continue
            if r.get('task') not in ('carrier','waveform_reconstruction'):continue
            if not r.get('variant','').startswith('d8_'):continue
            a=r['evaluation'];h3.append(dict(representation=r['representation'],variant=r['variant'],endpoint=r['task'],carrier_CE_bits=a.get('cross_entropy_bits'),past_reconstruction_NMSE=a.get('normalized_by_train_mean_risk'),future_MSE_baseline100=None,n_circuits=a.get('circuits'),n_contexts=a.get('contexts'),source='retention/summary.json'))
    for t in information:
        if t['job_id'] not in ('A0__T0_l10_d128','A0__T0_l0_d128'):continue
        for r in t['rows']:
            if r.get('information_available')!='compressed EEG representation' or r['status']!='measured':continue
            h3.append(dict(representation=r.get('representation',r['id']),variant=r.get('variant'),endpoint='future prediction '+t['job_id'],carrier_CE_bits=None,past_reconstruction_NMSE=None,future_MSE_baseline100=r['error_baseline100'],future_MSE_circuit_interval=r.get('error_baseline100_interval'),n_circuits=r['n_circuits'],n_contexts=r['n_contexts'],source='information/'+t['job_id']+'/summary.json'))
    atomic_json(tables/'h3_distinct_endpoints.json',dict(rows=h3,rule='Blank cells are distinct/absent endpoints, not zeros. Conditional carrier CE, past-history NMSE and future MSE use separate labels/losses.'))
    _write_csv(tables/'h3_distinct_endpoints.csv',h3,['representation','variant','endpoint','carrier_CE_bits','past_reconstruction_NMSE','future_MSE_baseline100','future_MSE_circuit_interval','n_circuits','n_contexts','source'])
    h4=[]
    for t in information:
        r=next((r for r in t['rows'] if r['id']=='eeg_full_selected'),None)
        if not r:continue
        h4.append(dict(job_id=t['job_id'],physical_mse=r['physical_mse'],physical_mse_units=t['physical_mse_units'],reference_mse=r['reference_mse'],G_pp=r['relative_mse_reduction_pp'],interval=r['relative_mse_reduction_pp_interval'],n_circuits=r['n_circuits'],n_contexts=r['n_contexts'],source='information/'+t['job_id']+'/summary.json'))
    atomic_json(tables/'h4_target_results.json',dict(rows=h4,comparability='Different target losses and target-specific denominators; matched paired contrasts are separate.'))
    return h2,h3,h4


def _h2_text(rows):
    if not rows:return ['', 'No completed M32/L128 reference outcomes are present. Older inspected M8/L32 arrays, when available, do not satisfy the registered H2 allocation and are not silently relabeled.', '']
    out=['', '### Observed H2 numerical summary', '', 'Physical corrected conditional-mean MSE below uses the target units squared; variance-estimation MSE uses the target units to the fourth power. Negative corrections are retained. Full target/M/metric tables and circuit intervals are in tables/h2_precision.json.', '', '| Target | M / L | Corrected conditional-mean MSE | Circuits |','|---|---:|---:|---:|']
    for r in rows:
        if r['job_id'] in ('A1__T0_l0_d32','A1__T0_l2_d32') and r['metric']=='reference_noise_corrected_mean_mse':out.append(f"| {r['job_id']} | {r['M']} / {r['L']} | {r['physical_estimate']:.8g} | {r['n_circuits']} |")
    return out+['']


def _h3_text(rows):
    if not rows:return ['', 'No completed matched-code endpoint rows are present in this root.', '']
    out=['','### Observed matched-code endpoints','', 'Separate columns prevent substituting past reconstruction or carrier decoding for future prediction. Blank cells mean that this row reports a different endpoint. The complete table includes every available projected-unquantized/B32 endpoint in tables/h3_distinct_endpoints.json.', '', '| Representation / variant | Endpoint | Carrier CE (bits) | Past reconstruction NMSE | Future error (baseline=100) |','|---|---|---:|---:|---:|']
    for r in rows:
        vals=[r.get(k) for k in ('carrier_CE_bits','past_reconstruction_NMSE','future_MSE_baseline100')]
        out.append(f"| {r['representation']} / {r['variant']} | {r['endpoint']} | "+' | '.join('—' if v is None else _number(v) for v in vals)+' |')
    return out+['']


def _h4_text(rows):
    wanted=[r for r in rows if any(x in r['job_id'] for x in ('__T3_','__T4_energy_alpha_','__T4_log_energy_alpha_'))]
    if not wanted:return ['', 'No completed Fourier/alpha-energy/log-alpha outcomes are available in this root.', '']
    out=['','### Observed target-definition results','', 'These predetermined Fourier/alpha-energy/log-alpha examples are descriptive, not a selection of significant outcomes. All other targets and physical baseline risks appear in tables/h4_target_results.json. G is relative reduction in MSE; a negative value is retained.', '', '| Acquisition / target | Physical MSE (units) | G (pp) | 97.5% circuit interval (pp) |','|---|---:|---:|---|']
    for r in wanted:out.append(f"| {r['job_id']} | {r['physical_mse']:.8g} ({r['physical_mse_units']}) | {_number(r['G_pp'])} | {_interval({'ci':r['interval']},'ci')} |")
    return out+['']

def figures(config,root,tables,contrasts,replication):
    import re
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    out=Path(root)/'report/figures';out.mkdir(exist_ok=True);captions={}
    def save(fig,name,caption,rect=None):
        fig.tight_layout(rect=rect)
        for ext in ('pdf','svg','png'):fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig);captions[name]=caption
    colors=['#0072B2','#D55E00','#009E73']
    def measured(value):return value is not None and np.isfinite(value)
    def whisker(ax,ci,y,color):
        if ci and measured(ci.get('lower')) and measured(ci.get('upper')):ax.plot([ci['lower'],ci['upper']],[y,y],color=color,lw=.8,zorder=1)
    def setting(job):
        acquisition,target=job.split('__',1);timing=timing_metadata(target)
        h=timing['target_start_offset_ms'];w=timing['target_end_offset_ms']-h
        return f'{acquisition}: h={h:g}, W={w:g} ms'
    fig,axs=plt.subplots(1,2,figsize=(6.75,2.55),gridspec_kw={'width_ratios':[1,1.45]})
    axs[0].axvspan(-2,0,color='.93',zorder=0)
    for i,h in enumerate((0,2,10)):
        ax=axs[0];ax.plot([h,h+2],[i,i],lw=5,color=colors[i],solid_capstyle='butt');ax.plot(h+2,i,'o',ms=3.2,color=colors[i]);ax.plot(h,i,'o',ms=3.2,mfc='white',mec=colors[i])
        axs[1].plot([h,h+128],[i,i],lw=2,color=colors[i],solid_capstyle='butt');axs[1].plot([h,h+32],[i,i],lw=6,color=colors[i],solid_capstyle='butt')
    axs[1].plot([0,1000],[3,3],lw=4,color=colors[0]);axs[1].plot([10,1010],[4,4],lw=4,color=colors[2])
    for ax in axs:
        ax.axvline(0,color='.3',ls='--',lw=.7);ax.set_xlabel('Time relative to cutoff (ms)');ax.invert_yaxis()
    axs[0].set_yticks(range(3),['h=0','h=2','h=10']);axs[0].set_title('First 2-ms bin: 32 native samples');axs[0].set_xlim(-2,14);axs[0].set_xticks([0,2,4,10,12])
    axs[1].set_yticks(range(5),['h=0 waveform','h=2 waveform','h=10 waveform','h=0 block','h=10 block']);axs[1].set_title('Waveforms: thick 32 ms / thin 128 ms\nFourier / energy blocks: 1000 ms');axs[1].set_xlim(-20,1040);axs[1].set_xticks([0,250,500,750,1000])
    save(fig,'target_timing','Protocol diagram, not experimental outcomes. Open left endpoint excludes cutoff; each waveform bin contains exactly 32 native samples. Thick/thin waveform support shows W32/W128. Fourier/energy blocks become available only at their right endpoint. h is target-start offset, not first-bin end.')
    byjob={s['job_id']:s for s in tables};mainids=('A0__T0_l0_d128','A0__T0_l10_d128')
    chosen=[byjob.get(j) for j in mainids]
    if any(chosen):
        def plotted(r):
            return r['id'] in ('state_variance','eeg_short_selected','eeg_full_selected','train_mean') or r.get('information_available')=='partial present-state features' or (r.get('information_available')=='compressed EEG representation' and 'B32' in r.get('variant',r['id']))
        def order(r):
            if r['id']=='state_variance':return (0,r['id'])
            if r.get('information_available')=='partial present-state features':return (1,r['id'])
            if r['id']=='eeg_short_selected':return (2,r['id'])
            if r['id']=='eeg_full_selected':return (3,r['id'])
            if r['id']=='train_mean':return (9,r['id'])
            return (4 if r['id'].startswith('feature18') else 5 if r['id'].startswith('pca') else 6,r['id'])
        unique={r['id']:r for s in chosen if s for r in s['rows'] if plotted(r)};layout=sorted(unique.values(),key=order)
        def label(r):
            labels={'state_variance':'Same-state future variance','eeg_short_selected':'128-ms EEG policy','eeg_full_selected':'Four-second EEG policy','train_mean':'Training mean'}
            if r['id'] in labels:return labels[r['id']]
            if r.get('information_available')=='partial present-state features':return 'Partial membrane/gate model'
            if r['id'].startswith('feature18'):return 'Feature18, d8/B32'
            if r['id'].startswith('pca'):return 'PCA, d8/B32'
            seed=re.search(r'seed_(\d+)',r['id'])
            return 'Forecast seed '+seed.group(1)+', d8/B32' if seed else r['id']
        fig,axs=plt.subplots(1,2,figsize=(6.75,max(3.35,.23*len(layout)+1.2)))
        for column,(ax,s,job) in enumerate(zip(axs,chosen,mainids)):
            ax.set_title(setting(job)+'\nNative waveform, MSE in V²')
            if s is None:
                ax.axis('off');ax.text(.5,.5,'Not available in this archive\n\nImmediate target needs\nomitted native recordings',ha='center',va='center',transform=ax.transAxes,color='.3');continue
            current={r['id']:r for r in s['rows']}
            for i,item in enumerate(layout):
                r=current.get(item['id']);value=r.get('error_baseline100') if r else None
                if not measured(value):
                    ax.text(.02,i,'unavailable',transform=ax.get_yaxis_transform(),color='.5',fontsize=7);continue
                reference=r['status']=='estimated reference';code=r.get('information_available')=='compressed EEG representation'
                color='#D55E00' if reference else '#009E73' if code else '#0072B2';marker='s' if reference else '^' if code else 'o'
                whisker(ax,r.get('error_baseline100_interval'),i,color);ax.plot(value,i,marker,ms=4,color=color,zorder=2)
            # If only the right panel has measurements, retain its full row labels.
            show_labels=column==0 or chosen[0] is None
            ax.set_yticks(range(len(layout)),[label(r) for r in layout] if show_labels else ['']*len(layout));ax.invert_yaxis();ax.axvline(100,color='.6',ls='--',lw=.7);ax.set_xlabel('Error (training mean = 100)')
        fig.text(.02,.012,'Ideal EEG error: unknown. Exact-forcing zero is theoretical; measured replay has a separate population.',fontsize=7)
        save(fig,'information_error','Within each panel: identical contexts, circuit weights and first four scoring futures. Squares denote estimated state-variance references, circles fitted uncompressed errors and triangles d8/B32 future-code decoders; forecast seeds are fitting repeats. Whiskers are 97.5% circuit intervals; conditional-MC diagnostics are separate. Missing h0 recordings and unknown ideal EEG error are never plotted as zero. Each target has its own train-mean denominator; physical V² errors and counts are in matching information tables.',rect=(0,.06,1,1))
    for axis in ('gap','history','acquisition','duration'):
        rows=[r for r in contrasts['rows'] if r['comparison_axis']==axis and r['method_id'] in ('eeg_full_ridge_policy','eeg_full_ridge_policy vs eeg_short_ridge_policy') and all('__T0_' in j for j in r['job_ids'])]
        rows=[r for r in rows if measured(r['gain_difference'].get('estimate',r['gain_difference'].get('difference_pp')))]
        if not rows:continue
        fig,ax=plt.subplots(figsize=(6.75,max(2.3,.24*len(rows)+.6)))
        for i,r in enumerate(rows):
            q=r['gain_difference'];value=q.get('estimate',q.get('difference_pp'));ax.plot(value,i,'o',ms=3,color='#0072B2');whisker(ax,q,i,'#0072B2')
        def contrast_label(r):
            if axis=='history':return setting(r['job_ids'][0])+'; 4 s − 128 ms'
            return setting(r['job_ids'][0])+' − '+setting(r['job_ids'][1])
        ax.set_yticks(range(len(rows)),[contrast_label(r) for r in rows]);ax.invert_yaxis();ax.axvline(0,color='.6',lw=.7);ax.set_xlabel('Paired difference in relative MSE reduction (percentage points)');ax.set_title('Matched '+axis+' comparison: fixed ridge-family policy')
        save(fig,'matched_'+axis,'Paired circuit contrasts, G(first)−G(second), with 97.5% descriptive intervals where estimable. Ridge family, candidate grid and mean fallback are fixed; each target retains its own denominator. History contrast compares 2000 vs 64 acquired samples. Physical-MSE contrasts and matched IDs are in tables/paired_contrasts.json.')
    energy=[]
    for s in tables:
        if 'T4_' not in s['job_id']:continue
        r=next((r for r in s['rows'] if r['id']=='eeg_full_selected'),None)
        if r and measured(r.get('relative_mse_reduction_pp')):energy.append((s,r))
    if energy:
        fig,ax=plt.subplots(figsize=(6.75,max(2.5,.23*len(energy)+.7)))
        for i,(s,r) in enumerate(energy):
            color=colors[int(s['acquisition'][1])];ax.plot(r['relative_mse_reduction_pp'],i,'o',ms=3,color=color);whisker(ax,r.get('relative_mse_reduction_pp_interval'),i,color)
        def energy_label(s):
            t=s['job_id'].split('__',1)[1];band=t.split('energy_',1)[1].rsplit('_l',1)[0].replace('_',' ');h=timing_metadata(t)['target_start_offset_ms'];kind='log energy' if '_log_' in t else 'energy'
            return f'{s["acquisition"]}: {band} {kind}, h={h:g} ms'
        ax.set_yticks(range(len(energy)),[energy_label(s) for s,r in energy]);ax.invert_yaxis();ax.axvline(0,color='.5',lw=.7);ax.set_xlabel('Relative MSE reduction (percentage points)');ax.set_title('Direct future energy / log energy; one-second blocks')
        save(fig,'h4_direct_energy','Development-selected full-history policies, first four registered futures, 97.5% descriptive circuit intervals where estimable. Energy and log-energy use different losses and target-specific denominators. Physical energy risks are V⁴; log-energy risks are squared natural-log units. No waveform-mean squaring is substituted.')
    h2=[]
    for job in replication.get('jobs',[]):
        if job['status']!='complete':continue
        s=read_json(Path(root)/job['path'])
        if job['id'] not in ('A1__T0_l0_d32','A1__T0_l2_d32','A1__T0_l10_d128'):continue
        group={}
        for r in s['records']:
            if r['fitting_repeat']!=0:continue
            for q in r['rows']:group.setdefault(q['M'],[]).append(q['nested']['reference_noise_corrected_mean_mse'])
        if group:h2.append((job['id'],group))
    if h2:
        fig,ax=plt.subplots(figsize=(3.25,2.5))
        for job,g in h2:ax.plot(sorted(g),[np.mean(g[m]) for m in sorted(g)],'o-',ms=3,lw=.8,label=setting(job).replace('A1: ',''))
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32]);ax.axhline(0,color='.6',lw=.6);ax.set_xlabel('Scoring futures M (nested)');ax.set_ylabel('Reference-corrected mean MSE\n(target-standardized coordinates)');ax.legend(frameon=False)
        save(fig,'h2_finite_reference','Recorded nested conditional-mean error after finite-reference correction, averaged over registered contexts. Actual M and L come from completed records, not the planned M32/L128 design. Lines are descriptive; dependence and both-pool uncertainty are in replication JSON. Negative corrected values are retained. Target-standardized errors across targets are not absolute-information rankings.')
    atomic_json(out/'captions.json',captions)
