"""Audit raw recordings and analyze the complete, prespecified S2-P cohort."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.l23net_analysis import s2_pilot_protocol as p
from experiments.l23net_analysis import s1_analysis as s1
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.r1_analysis import paired_inference, bh_adjust, pbs_epilogue, frozen_json
from experiments.l23net_analysis.r1_protocol import save_json, sha256


def inference(values):
    if len(values) < 2:
        return {'n_structures':len(values), 'insufficient_for_inference':True}
    return paired_inference(values)


def audit_decision(report, gate, arm):
    """Recompute eligibility from pre-action features/rates, never outcomes."""
    w = report['window_protocol']; d = w['decision']
    screen = s0.scores(w['outcomes']['baseline']['log_powers'],gate['targets']['20s'])
    windows = [v for v in report['windows'] if v['start_ms'] >= 8000 and v['stop_ms'] <= 28000]
    if len(windows) != 20:
        raise ValueError('Missing baseline screening windows')
    safe = all(np.isfinite(v) and 0 <= v <= (20 if 'PYR' in pop else 100)
               for window in windows for pop,v in s1.population_rates(window['firing_rates']).items())
    eligible = bool(screen['eligible'] and safe)
    amplitude = .4 if eligible and arm != 'sham' else 0.
    expected = {'requested_arm':arm, 'delivered_arm':arm if eligible else 'sham',
                'amplitude_v_per_m':amplitude, 'frequency_hz':14. if arm != 'sham' else 0.,
                'phase_rad_at_onset':0., 'field_direction':[0.,0.,1.],
                'baseline_rates_safe':safe, 'qualification_screen_bypass':False,'decision_time_ms':28000.}
    if any(d.get(k) != v for k,v in expected.items()):
        raise ValueError('Decision/eligibility differs from the frozen prospective rule')
    if (d['screen']['eligible'] != screen['eligible'] or
            any(not np.isclose(d['screen'][k],screen[k],atol=1e-9,rtol=0) for k in ('distance','signed_score'))):
        raise ValueError('Saved screening score differs from raw baseline')
    if amplitude == 0 and w['nonzero_field_samples'] != 0:
        raise ValueError('Sham/rejected candidate received a field')
    return eligible


def contrast(seed, sham, active, target):
    a,b = sham['window_protocol'],active['window_protocol']; targets=target['targets']
    x=np.array(a['outcomes']['stimulation']['log_powers']); y=np.array(b['outcomes']['stimulation']['log_powers'])
    scale=np.array(targets['plateau']['scale_log10']); mu=np.array(targets['plateau']['mean_log10'])
    z=(x-mu)/scale; shift=(x-y)/scale
    sd=s0.scores(x,targets['plateau'])['distance']; ad=s0.scores(y,targets['plateau'])['distance']
    excluded=[s0.scores(w['fundamental_excluded']['low_beta']['log_powers'],
                        targets['fundamental_excluded_plateau']['low_beta'])['distance'] for w in (a,b)]
    wash=[s0.scores(w['outcomes']['washout']['log_powers'],targets['washout'])['distance'] for w in (a,b)]
    rates={}; safe=True
    for epoch,start,stop in [('plateau',29000,49000),('washout',50000,60000)]:
        ra,rb=s1.mean_rates(sham,start,stop),s1.mean_rates(active,start,stop)
        changes={pop:(rb[pop]-ra[pop])/max(ra[pop],.1) for pop in ra}
        safe &= all(np.isfinite(v) and abs(v)<=.2 for v in changes.values())
        safe &= all(np.isfinite(v) and 0<=v<=(20 if 'PYR' in pop else 100)
                    for values in (ra,rb) for pop,v in values.items())
        rates[epoch]={'sham_hz':ra,'active_hz':rb,'fractional_change':changes}
    return {'seed':seed,'eligible':bool(a['decision']['screen']['eligible'] and a['decision']['baseline_rates_safe']),
            'distance_sham':sd,'distance_active':ad,'benefit':sd-ad,
            'relative_benefit_percent':100*(sd-ad)/sd if sd>0 else None,
            'band_log10_sham_minus_active':(x-y).tolist(),
            'band_power_change_percent':(100*np.expm1(np.log(10)*(y-x))).tolist(),
            'excluded_benefit':excluded[0]-excluded[1], 'washout_benefit':wash[0]-wash[1],
            'shift_alignment':float(np.dot(z,shift)/np.linalg.norm(z)) if np.linalg.norm(z)>0 else 0.,
            'rate_safe':bool(safe),'rates':rates}


def cohort_result(rows):
    if sorted(r['seed'] for r in rows) != list(p.SEEDS):
        raise ValueError('Never infer on an incomplete, duplicate or unplanned pilot cohort')
    eligible=[r for r in rows if r['eligible']]
    primary=inference([r['benefit'] for r in eligible])
    bands={band:inference([r['band_log10_sham_minus_active'][i] for r in eligible])
           for i,band in enumerate(s0.BANDS)}
    if len(eligible)>=2:
        for band,q in zip(bands,bh_adjust([v['p_one_sided'] for v in bands.values()])):
            bands[band]['q_bh_three_bands']=q
    rule=p.PROTOCOL['pilot_support_rule']
    checks={'screen_coverage':len(eligible)>=rule['minimum_eligible'],
            'practical_mean':primary.get('mean',-np.inf)>=rule['minimum_mean_benefit'],
            'primary_sign_flip':primary.get('p_one_sided',1.)<=rule['one_sided_sign_flip_alpha'],
            'seed_consistency':bool(eligible) and sum(r['benefit']>0 for r in eligible)/len(eligible)>=rule['minimum_positive_fraction'],
            'excluded_direction':bool(eligible) and np.mean([r['excluded_benefit'] for r in eligible])>0,
            'alignment':bool(eligible) and np.mean([r['shift_alignment'] for r in eligible])>0,
            'rate_safety':bool(eligible) and all(r['rate_safe'] for r in eligible)}
    checks={k:bool(v) for k,v in checks.items()}
    return {'candidate_seeds':list(p.SEEDS),'eligible_seeds':[r['seed'] for r in eligible],
            'rejected_seeds':[r['seed'] for r in rows if not r['eligible']],
            'primary':primary,'secondary_bands':bands,
            'excluded_secondary':inference([r['excluded_benefit'] for r in eligible]),
            'washout_secondary':inference([r['washout_benefit'] for r in eligible]),
            'all_candidate_policy_audit':inference([r['benefit'] for r in rows]),
            'pilot_checks':checks,'pilot_supports_larger_confirmation':all(checks.values()),
            'interpretation':'Pilot only. Report negatives; no automatic bandit/phase/clinical claim. Do not append seeds after inspecting this result.'}


def analyze_suite(root, *, write=True):
    root=Path(root); manifest=json.loads((root/'submission.json').read_text())
    gate,target=p.check_manifest(manifest,root)
    errors=[]; reports={}; accounting=[]; hashes={}
    if manifest.get('submission_status')!='submitted':
        errors.append('Submission incomplete/partial; inspect the recorded jobs before retrying')
    for job in manifest['jobs']:
        folder=root/job['name']
        try:
            if (folder/'worker_exit_code.txt').read_text().strip()!='0':
                raise ValueError('Worker incomplete or failed')
            if (folder/'git_commit.txt').read_text().strip()!=manifest['commit']:
                raise ValueError('Worker checkout differs from submission')
            r=s1.audit_run(folder,gate)
            p.validate_contract(r['replay_contract'],gate)
            cfg=r['configuration']['analysis']; spec=job['run']
            if (r['seed_manifest']['experiment_seed'],cfg['condition'],cfg['arm'],cfg['mode'])!=(spec['seed'],'mdd',spec['arm'],'replication_pilot'):
                raise ValueError('Run does not match the planned seed/arm')
            meta=r['window_protocol']['s2_pilot']
            if (meta['protocol_sha256']!=p.reference.canonical_json_sha256(p.PROTOCOL) or
                    meta['code_sha256']!=p.code_hashes() or meta['target_sha256']!=p.TARGET_SHA):
                raise ValueError('Frozen target/protocol/code identity differs')
            audit_decision(r,gate,spec['arm'])
            for epoch,old in [('plateau','stimulation'),('washout','washout')]:
                expected=s0.scores(r['window_protocol']['outcomes'][old]['log_powers'],target['targets'][epoch])
                if any(not np.isclose(meta['outcomes'][epoch][k],v,atol=1e-9,rtol=0) for k,v in expected.items()):
                    raise ValueError('Saved Reference60 outcome differs from raw recomputation')
            expected=s0.scores(r['window_protocol']['fundamental_excluded']['low_beta']['log_powers'],
                               target['targets']['fundamental_excluded_plateau']['low_beta'])
            if any(not np.isclose(meta['excluded_plateau'][k],v,atol=1e-9,rtol=0) for k,v in expected.items()):
                raise ValueError('Excluded-space outcome differs')
            memory=[v['rss_gib']['sum'] for v in r['memory_snapshots'] if v['simulated_ms']>=8000]
            if not memory or max(memory)-min(memory)>max(2.,.05*memory[0]):
                raise ValueError('Persistent memory variation exceeds qualified tolerance')
            ep=pbs_epilogue(folder/'pbs.out')
            if not ep.get('final_accounting_available') or ep['exit_status']!=0 or ep['memory_gb_pbs']>230.4:
                raise ValueError('Final PBS accounting missing/failed or inadequate memory headroom')
            accounting.append({'job':job['name'],'project':job['project'],**ep})
            reports[job['name']]=r; hashes[job['name']]=sha256(folder/'l23net_s1_run.json')
        except Exception as exc:
            errors.append(f"{job['name']}: {exc}")
    rows=[]
    if not errors:
        for seed in p.SEEDS:
            try:
                an,bn=[f's{seed}_{arm}' for arm in p.ARMS]; a,b=reports[an],reports[bn]
                s1.compare_prefix(root/an,root/bn,a,b)
                if audit_decision(a,gate,'sham')!=audit_decision(b,gate,'fixed_low_beta'):
                    raise ValueError('Paired eligibility differs')
                rows.append(contrast(seed,a,b,target))
            except Exception as exc:
                errors.append(f'Pair {seed}: {exc}')
    result={'stage':'s2_pilot','status':'incomplete_or_failed','errors':errors,
            'completed_valid_runs':len(reports),'expected_runs':20,'rows':rows,'pbs':accounting,
            'target_sha256':target['sha256'],'code_sha256':p.code_hashes(),'run_report_hashes':hashes,
            'protocol':p.PROTOCOL,'technical_passed':not errors}
    if not errors:
        result.update(cohort_result(rows),status='completed_pilot',actual_ksu=sum(v['service_units'] for v in accounting)/1000)
        if write:
            frozen_json(root/'s2_pilot_summary.json',p.reference.signed(result))
    if write:
        save_json(root/'s2_pilot_status.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',type=Path,nargs='?'); parser.add_argument('--latest',action='store_true')
    parser.add_argument('--verify-only',action='store_true',help='Audit without writing; preserve frozen NCI outputs across analysis-library roundoff')
    args=parser.parse_args(argv)
    if args.latest:
        if args.suite: parser.error('Use a path OR --latest')
        paths=sorted(p for p in (ROOT/'results').glob('l23net_s2_pilot_*') if (p/'submission.json').is_file())
        if not paths: parser.error('No S2 pilot found; supply an explicit transferred directory')
        args.suite=paths[-1]
    if args.suite is None: parser.error('Supply a suite or --latest')
    result=analyze_suite(args.suite,write=not args.verify_only)
    print(json.dumps({k:result[k] for k in ('status','completed_valid_runs','expected_runs','errors')},indent=2))
    if result['technical_passed']:
        print(json.dumps({k:result[k] for k in ('eligible_seeds','primary','pilot_checks','actual_ksu')},indent=2))
    print('SUITE_DIRECTORY='+str(args.suite.resolve()))
    return 0 if result['technical_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
