"""Artifact-only reporting for the bounded exploratory pilot.

No simulator or fitter is imported here. Unknown values remain null, theoretical
limits never become measured rows, and successful PDF construction alone cannot
complete the experiment. JSON is authoritative; displayed tables are views.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

SCHEMA = 'eegrepr-v6-reporting-pilot-results-1'
TABLES = {
    'h1_information': ('H1: information available and prediction error', ['job','population','information','method','evidence_type','error_baseline100','risk_physical','reference_risk_physical','unresolved_excess_physical','uncertainty','circuits','contexts','futures','status']),
    'h1_comparisons': ('H1: immediate, delayed and history comparisons', ['job','acquisition','h_ms','duration_ms','history','error_baseline100','relative_mse_reduction_pp','paired_difference','uncertainty','population','status']),
    'h2_precision': ('H2: finite-future estimation', ['job','context_id','M','L','mean_error_corrected','variance_error','uncertainty','variance_uncertainty','status']),
    'h3_retention': ('H3: reconstruction, prediction and carrier decoding', ['representation','projection','budget_bits','reconstruction_nmse','future_error_baseline100','carrier_ce_bits','accuracy','balanced_accuracy','population','uncertainty','status']),
    'h4_targets': ('H4: target definition, support and predictability', ['job','target','h_ms','duration_ms','physical_units','risk_units','floor_reference_ratio','relative_mse_reduction_pp','uncertainty','status']),
    'findings': ('Findings and what remains unresolved', ['question','observed','interpretation','limitation','later_study','status']),
    'run_status': ('Run-status ledger', ['stage','category','status','count','details']),
    'full_grid': ('Complete acquisition-target grid', ['job','acquisition','target','h_ms','duration_ms','physical_units','method','error_baseline100','relative_mse_reduction_pp','population','status']),
}
GLOSSARY = [
 ('Physical future MSE', 'Mean squared coordinate error, averaged within context and then equally across circuits. Waveform and orthonormal Fourier coefficient risk: V^2. Energy target risk: V^4. Log-energy risk: dimensionless squared log units.'),
 ('Training-mean = 100', '100 times model risk divided by the risk of the training-only mean on the same target, contexts and futures. A smaller value is better. Physical risks remain in the numeric tables.'),
 ('Relative MSE reduction', '100 times (1 - model risk/reference risk), in percentage points. It equals 100 minus the preceding display scale. Negative values are retained.'),
 ('Past reconstruction NMSE', 'Reconstruction MSE divided by training-mean history reconstruction error on the same cohort. The reference is 1, not 100. This is a different task from future prediction.'),
 ('Carrier cross entropy', 'Mean negative log2 probability assigned to the true carrier class, in bits, conditional on rhythmic membership. 10/16/24 Hz denote input-rate carrier conditions, not guaranteed EEG peaks. Cross entropy depends on the fitted decoder and is not total mutual information.'),
 ('Estimated state variability', 'Unbiased within-context sample variance over independent futures; a physical-state reference for this population and target, not the ideal EEG predictor error. Model risk minus this estimate is a combined unresolved excess. Negative finite-sample excess is retained.'),
 ('Finite references', 'L is finite. Mean-estimation error subtracts reference variance/L. Variance error and interval diagnostics must retain reference uncertainty. M=1 has no direct within-context sample variance. Reused subsets/reference pools are dependent.'),
 ('Target timing', 'h is the left-open start offset from cutoff; h=0 begins after cutoff. A 2-ms bin at h=0 contains (t0,t0+2 ms], 32 native samples. One-second block targets are available only at the end of the block. Acquisition samples are never backdated.'),
 ('Complete-state learner', 'An actually fitted predictor of declared physical present state; no future external schedule, RNG state or seed is an input. It is not a Bayes oracle. The predeclared state-dependent candidate remains visible even when development selection chooses the mean.'),
 ('Population and uncertainty', 'Circuit is the population sampling unit; futures repeat a fixed context. Ordinary historical eight-circuit, matched state cohort, two-context H2 and replay populations are separate. Tiny-pilot intervals are descriptive; formal power/equivalence/confirmation is deferred.'),
]


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as stream:
        tmp = Path(stream.name); stream.write(value); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def _json(path, value):
    _atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def dependencies():
    """Check the exact runtime used by report(); never install during a study."""
    packages = {name: bool(importlib.util.find_spec(name)) for name in ('reportlab', 'pypdf', 'matplotlib', 'PIL')}
    binaries = {name: shutil.which(name) for name in ('pdftoppm', 'pdfinfo')}
    missing = [k for k,v in {**packages, **binaries}.items() if not v]
    return {'status': 'complete' if not missing else 'blocked', 'python_packages':packages,
            'binaries':binaries, 'missing':missing,
            'remedy':'Install reportlab pypdf matplotlib pillow in the experiment environment and Poppler (pdftoppm/pdfinfo) before launch.'}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _same(a, b):
    return math.isclose(float(a), float(b), rel_tol=5e-10, abs_tol=5e-10*max(abs(float(a)),abs(float(b)),1e-30))


def normalize_results(results):
    """Validate display identities and retain all supplied precision and fields."""
    r = json.loads(json.dumps(results, allow_nan=False)); r.setdefault('schema_version', SCHEMA)
    if r['schema_version'] != SCHEMA: raise ValueError('Unknown pilot report result schema')
    r.setdefault('purpose', 'exploratory_reporting_pilot'); r.setdefault('status','incomplete')
    r.setdefault('tables', {}); r.setdefault('runtime',{}); r.setdefault('completed_allocation',{})
    # Canonical aliases are explicit schema migration, not scientific recomputation.
    aliases={'relative_mse_reduction':'relative_mse_reduction_pp','combined_unresolved_excess':'unresolved_excess_physical','past_reconstruction_nmse':'reconstruction_nmse','carrier_cross_entropy_bits':'carrier_ce_bits','rhythm_balanced_accuracy':'balanced_accuracy','cohort_budget':'population','same_state_variance_ratio':'floor_reference_ratio','descriptive_uncertainty':'uncertainty','next_evidence':'later_study'}
    for name in TABLES:
        rows = r['tables'].setdefault(name, r.get(name, []))
        if not isinstance(rows,list) or any(not isinstance(row,dict) for row in rows):
            raise ValueError('Report table must contain records: '+name)
        for row in rows:
            row.setdefault('status','unavailable')
            for old,new in aliases.items():
                if new not in row and old in row:row[new]=row[old]
            if name=='h3_retention':row['payload_display']='B32 (4 bits x 8)' if row.get('budget_bits')==32 or row.get('variant')=='B32' else 'projected unquantized'
            if name=='h3_retention' and 'variant' in row:
                if 'budget_bits' not in row and 'B32' in str(row['variant']):row['budget_bits']=32
                if 'projection' not in row and 'd8' in str(row['variant']):row['projection']='d8'
            risk = row.get('risk_physical',row.get('physical_mse'))
            ref = row.get('reference_risk_physical',row.get('baseline_physical_mse'))
            if risk is not None: row['risk_physical']=risk
            if ref is not None: row['reference_risk_physical']=ref
            if _number(risk) and _number(ref) and ref>0:
                scaled = 100*float(risk)/float(ref)
                if _number(row.get('error_baseline100')) and not _same(row['error_baseline100'],scaled):
                    raise ValueError('Display/physical risk mismatch: '+str(row.get('job',name)))
                row['error_baseline100']=scaled
            if _number(row.get('error_baseline100')):
                gain=100-row['error_baseline100']
                if _number(row.get('relative_mse_reduction_pp')) and not _same(row['relative_mse_reduction_pp'],gain):
                    raise ValueError('Relative MSE reduction mismatch')
                row['relative_mse_reduction_pp']=gain
            floor=row.get('state_variance_physical')
            if _number(risk) and _number(floor): row['unresolved_excess_physical']=risk-floor
            if row.get('evidence_type') in ('unknown','theoretical') and row.get('status') in ('complete','measured'):
                raise ValueError('Unknown/theoretical evidence cannot have measured complete status')
            if name=='h2_precision' and row.get('M')==1 and row.get('variance_error') is not None:
                raise ValueError('Direct M=1 variance error must be unavailable, never zero')
    comparisons=[]
    for row in r['tables']['h1_comparisons']:
        if 'history' not in row and ('short_error' in row or 'full_error' in row):
            for history,key in (('short','short_error'),('full','full_error')):
                v=row.get(key);extra=dict(row,history=history,error_baseline100=v)
                extra['relative_mse_reduction_pp']=100-v if _number(v) else None
                comparisons.append(extra)
        else:comparisons.append(row)
    r['tables']['h1_comparisons']=comparisons
    return r


def completion_checks(root, results):
    """Substantive measurement gates, independent of whether a PDF was produced."""
    root=Path(root); issues=[]; s=results.get('complete_state_verification',{})
    if results.get('purpose') != 'exploratory_reporting_pilot': issues.append('Engineering/template evidence is not a completed biological pilot.')
    if results.get('status') not in ('complete','ready_for_report'): issues.append('Upstream pilot results are incomplete.')
    stages=results.get('required_stages',{})
    if not stages or any(v not in ('complete','reused') for v in stages.values()): issues.append('Required simulation/analysis stages are incomplete.')
    for key in ('input_coverage_complete','future_invariance_passed','fitted_state_dependent_candidate',
                'dev_selection_complete','held_out_predictions_complete','target_population_matching_passed'):
        if s.get(key) is not True: issues.append('Complete-state verification missing: '+key)
    if s.get('status')!='complete': issues.append('Complete-state adapter audit is not complete.')
    roles=set();role_counts={}
    for item in s.get('artifact_manifest',[]):
        p=(root/item.get('path','')).resolve()
        if root.resolve() not in p.parents or not p.is_file(): issues.append('Missing/unsafe state evidence artifact: '+str(item.get('path'))); continue
        if _hash(p)!=item.get('sha256'): issues.append('State evidence hash mismatch: '+str(item.get('path'))); continue
        roles.add(item.get('role'));role_counts[item.get('role')]=role_counts.get(item.get('role'),0)+1
    for role in ('schema','fitted_state','predictions','selection','losses','training_curve'):
        if role not in roles: issues.append('Verified complete-state artifact role missing: '+role)
        elif role!='schema' and role_counts[role]<60:issues.append('Complete-state artifact role does not cover all 60 target jobs: '+role)
    rows=results.get('tables',{}).get('h1_information',[])
    jobs={row.get('job') for row in rows if row.get('job')}
    if len(jobs)!=60: issues.append('Complete 60-setting information matrix not present.')
    for job in sorted(jobs):
        for method in ('complete_state_selected','complete_state_candidate'):
            candidates=[row for row in rows if row.get('job')==job and row.get('method')==method]
            good=[row for row in candidates if row.get('status') in ('complete','measured') and _number(row.get('risk_physical')) and _number(row.get('reference_risk_physical')) and row.get('reference_risk_physical',0)>0 and row.get('population') and row.get('contexts',0)>=8 and row.get('futures',0)>=2 and row.get('circuits',0)>=2 and row.get('evidence_type')=='fitted']
            if not good: issues.append('Missing measured '+method+' for '+job)
            elif method=='complete_state_candidate' and not any(row.get('actually_trained') is True for row in good): issues.append('Nonmean state candidate not actually trained: '+job)
    return {'status':'complete' if not issues else 'incomplete','experimental_completion_eligible':not issues,'issues':issues,
            'policy':'PDF generation/rendering alone cannot complete the experiment; theoretical zeros and mean-only artifacts do not satisfy the state learning gate.'}


def _display(value, digits=5):
    if value is None:return 'Unavailable'
    if isinstance(value,bool):return 'yes' if value else 'no'
    if isinstance(value,float):return f'{value:.{digits}g}'
    if isinstance(value,(dict,list)):return json.dumps(value,sort_keys=True,ensure_ascii=True)
    return str(value)


def _tex(value):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
    return ''.join(replacements.get(char,char) for char in _display(value))


def _preferred_population(rows):
    populations=sorted({str(r.get('population')) for r in rows if r.get('population')})
    return next((p for p in populations if 'matched' in p.lower()),populations[0] if populations else None)


def _anchor_acquisition(rows):
    if any(r.get('acquisition')=='A0' and str(r.get('method','')).startswith(('feature18','pca','forecast')) for r in rows):return 'A0'
    return 'A1' if any(r.get('acquisition')=='A1' for r in rows) else next((r.get('acquisition') for r in rows if r.get('acquisition')),'A1')


def _manuscript_view(name, rows):
    views={
      'h1_information':(['job','method','error_baseline100','uncertainty','population','status'],['Acquisition / target','Information / fitted method','MSE (mean=100)','Descriptive uncertainty','Population','Evidence status'],[.18,.21,.11,.19,.19,.12]),
      'h1_comparisons':(['job','history','error_baseline100','relative_mse_reduction_pp','paired_difference','uncertainty'],['Target','History','MSE (mean=100)','Reduction (pp)','Paired difference','Uncertainty'],[.22,.12,.13,.13,.16,.24]),
      'h2_precision':(['context_id','M','L','mean_error_corrected','variance_error','uncertainty'],['Context','M','L','Corrected mean error','Variance error','Uncertainty'],[.23,.06,.06,.18,.18,.29]),
      'h3_retention':(['representation','payload_display','reconstruction_nmse','future_error_baseline100','carrier_ce_bits','accuracy'],['Representation','Payload bits','Past NMSE','Future MSE (mean=100)','Carrier CE (bits)','Accuracy'],[.24,.12,.14,.20,.17,.13]),
      'h4_targets':(['job','physical_units','floor_reference_ratio','relative_mse_reduction_pp','uncertainty'],['Target and support','Target units','State variance / reference','MSE reduction (pp)','Uncertainty'],[.27,.18,.17,.17,.21]),
      'findings':(['question','observed','interpretation','limitation','later_study'],['Question','Observed value','Interpretation','Limitation','Later study'],[.18,.20,.22,.20,.20]),
      'run_status':(['stage','category','status','count','details'],['Stage','Evidence category','Status','Count','Details'],[.18,.18,.13,.08,.43]),
      'full_grid':(['job','method','error_baseline100','relative_mse_reduction_pp','status'],['Acquisition / target','Method','MSE (mean=100)','Reduction (pp)','Status'],[.30,.23,.16,.16,.15]),
    }
    keys,labels,widths=views[name]
    selected=rows
    anchor_acq=_anchor_acquisition(rows)
    if name=='h1_information':selected=[r for r in rows if r.get('acquisition')==anchor_acq and r.get('h_ms') in (0,10) and r.get('duration_ms')==128]
    if name=='h2_precision' and rows:
        job=next((r.get('job') for r in rows if str(r.get('job','')).startswith('A1') and 'l0_d128' in str(r.get('job'))),rows[0].get('job'))
        selected=[r for r in rows if r.get('job')==job]
    if name=='h4_targets':selected=[r for r in rows if r.get('acquisition','A1')==anchor_acq]
    def tex(rr):
        specs=''.join(r'>{\raggedright\arraybackslash}p{'+f'{w*.94:.4f}'+r'\linewidth}' for w in widths)
        lines=['% Requires array and longtable. Rounded presentation; unrounded JSON/NPZ is authoritative.',r'\begingroup',r'\footnotesize',r'\setlength{\tabcolsep}{2pt}',r'\begin{longtable}{'+specs+'}', ' & '.join(_tex(k) for k in labels)+r' \\ \hline',r'\endhead']
        lines += [' & '.join(_tex(row.get(k)) for k in keys)+r' \\' for row in rr]
        lines += [r'\end{longtable}',r'\endgroup'];return '\n'.join(lines)+'\n'
    return keys,labels,selected,tex(selected),tex(rows)


def write_tables(out, results):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); records={}
    for name,(title,preferred) in TABLES.items():
        rows=results['tables'][name]; columns=preferred+sorted(set().union(*(row.keys() for row in rows))-set(preferred)) if rows else preferred
        _json(out/(name+'.json'),rows)
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
        _atomic(out/(name+'.csv'),stream.getvalue())
        md=['# '+title,'','Unrounded JSON and CSV values are authoritative. Missing observations are unavailable, never zero.','', '| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
        md += ['| '+' | '.join(_display(row.get(k)).replace('|','\\|').replace('\n',' ') for k in columns)+' |' for row in rows]
        if not rows:md+=['','No measured records available.']
        _atomic(out/(name+'.md'),'\n'.join(md)+'\n')
        keys,labels,selected,tex,full_tex=_manuscript_view(name,rows)
        _atomic(out/(name+'.tex'),tex);_atomic(out/(name+'_supplement.tex'),full_tex)
        manuscript=['# '+title,'','Rounded presentation; unrounded JSON/NPZ/CSV is authoritative.','', '| '+' | '.join(labels)+' |', '| '+' | '.join(['---']*len(labels))+' |']
        manuscript += ['| '+' | '.join(_display(row.get(k)).replace('|','\\|').replace('\n',' ') for k in keys)+' |' for row in selected]
        _atomic(out/(name+'_manuscript.md'),'\n'.join(manuscript)+'\n')
        # Numeric mirrors keep binary float64 values, row alignment, and null masks.
        arrays={}
        for k in columns:
            if any(_number(row.get(k)) for row in rows):
                arrays[k]=np.asarray([row[k] if _number(row.get(k)) else np.nan for row in rows],dtype=np.float64)
                arrays[k+'__available']=np.asarray([_number(row.get(k)) for row in rows],dtype=bool)
        arrays['row_json']=np.asarray([json.dumps(row,sort_keys=True,allow_nan=False) for row in rows],dtype=str)
        tmp=out/(name+'.npz.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        os.replace(tmp,out/(name+'.npz'))
        records[name]={'rows':len(rows),'json_sha256':_hash(out/(name+'.json')),'csv_sha256':_hash(out/(name+'.csv')),'npz_sha256':_hash(out/(name+'.npz'))}
    return records


def report(root, config=None, results=None, *, render=True):
    """Generate tables, vector figures, editable prose, PDF, and every-page QA.

    ``render=False`` is only a development convenience and cannot issue an
    experimentally complete receipt. report-only calls never launch simulations.
    """
    root=Path(root); out=root/'report'; out.mkdir(parents=True,exist_ok=True)
    deps=dependencies()
    if deps['missing']: raise RuntimeError('Pilot PDF dependencies missing: '+', '.join(deps['missing'])+'; '+deps['remedy'])
    if results is None:
        p=root/'pilot_results.json'
        results=json.loads(p.read_text()) if p.exists() else {'status':'incomplete','tables':{}}
    results=normalize_results(results)
    results.setdefault('learner_protocol',{k:(config or {})[k] for k in ('ridge_alphas','state_candidate_alpha','neural_seed','neural_max_steps','neural_patience','histories','forecast_updates','score_futures') if k in (config or {})})
    checks=completion_checks(root,results)
    _json(out/'report_input.json',results); tables=write_tables(out/'tables',results)
    from .report_figures import create_figures
    from .report_pdf import create_pdf, verify_pdf
    figure_records=create_figures(out,results)
    source=write_narrative(out,results,checks,figure_records)
    layout=create_pdf(out,results,checks,figure_records)
    qa=verify_pdf(out,layout,render=render)
    receipt={'schema_version':'eegrepr-v6-pilot-report-1','status':'complete' if qa['status']=='complete' else 'incomplete',
             'experimental_completion_eligible': checks['experimental_completion_eligible'] and qa['status']=='complete',
             'experimental_checks':checks,'results_hash':_canonical(results),'config_hash':_canonical(config or {}),
             'report_source_hashes':{p.name:_hash(p) for p in (Path(__file__),Path(__file__).with_name('report_pdf.py'),Path(__file__).with_name('report_figures.py'))},
             'purpose':results['purpose'],'tables':tables,'figures':figure_records,'pdf':'report/pilot_findings.pdf','pdf_sha256':_hash(out/'pilot_findings.pdf'),
             'pdf_verification':qa,'dependencies':deps,'human_visual_review':'Not performed by the unattended runner. Every page has an automatic rendered/layout audit. Local engineering template visual review is recorded separately and is not a review of future outcomes.'}
    _json(out/'receipt.json',receipt)
    return receipt


def write_narrative(out,results,checks,figures):
    text=['# NeuroStimEnv reporting pilot','', '**Exploratory pilot / reporting-template development.**', '',
          'This report contains only supplied numeric artifacts. It does not authorize or complete the deferred N192/N257 confirmation study, numerical panel or power-driven expansion.', '',
          'Experimental completion: '+checks['status']+'. PDF compilation is a separate artifact check.','',
          '## Allocation and runtime','',_display(results.get('completed_allocation',{})), '',_display(results.get('runtime',{})),'',
          'Eight workers means eight independent simulation processes with one CPU thread each; it does not increase independent circuit N. Recorded runtimes are measurements only when explicitly supplied.','',
          '## Complete present-state learning','',_display(results.get('complete_state_verification',{})), '',
          'The predeclared trained candidate and development-selected policy are separate rows. Complete state is a physical input declaration, not a claim of a restorable checkpoint or optimal prediction. Unknown future external events and RNG encodings are excluded.','',
          '## Metric and timing glossary','']
    text += ['- **'+name+'**: '+body for name,body in GLOSSARY]
    for name,(title,_) in TABLES.items():
        text += ['', '## '+title, '', f'Numeric and editable tables: tables/{name}.json, .npz, .csv, .md, .tex. {len(results["tables"][name])} observed/status records.']
        if name=='findings':
            for row in results['tables'][name]:text += ['',_display(row.get('question'))+': '+_display(row.get('observed'))+'. '+_display(row.get('interpretation'))+' Limitation: '+_display(row.get('limitation'))]
    text+=['','## Completion blockers','']+(['- '+issue for issue in checks['issues']] or ['No recorded experimental completion blockers.'])
    text+=['','## Review before the full study','', '- Confirm the complete-state coverage inventory, future-invariance test and matched target IDs.', '- Review fitted state candidate and selected fallback separately; negative performance is retained.', '- Check populations, physical units and denominators before comparing rows.', '- Review every rendered PDF page, full 60-setting appendix, unavailable uncertainty and H2 L32 labels.', '- Decide on the later full study separately; this pilot does not trigger confirmation.']
    _atomic(Path(out)/'pilot_findings.md','\n'.join(text)+'\n');return text
