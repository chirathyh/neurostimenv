"""Recompute the frozen v4 evidence; inspect actual native manifest fields.

A verified manifest attests the original hashes. When native payloads were omitted,
this audit does not claim to have recomputed physical-state hashes from raw arrays.
"""
from pathlib import Path
from collections import defaultdict, Counter
import itertools
import re
import numpy as np
from scipy.stats import binomtest
from .artifacts import atomic_json,canonical_hash,module_hash,resume_record
from .evidence import EvidenceSource,import_evidence
from .precision import archive_components,HISTORICAL_ARCHIVE_SHA256
from .statistics import stable_mean,validate_split_ids
from ..v4.scoring import gaussian_diagnostics


class _NumericView:
    """Adapt a verified source (directory or standard ZIP) to numeric readers."""
    def __init__(self,source,identity):
        self.source=source
        self.sha256=identity.get('archive_sha256')or identity['content_id']
    def json(self,path):return self.source.read_json(path.removeprefix('evidence/'))
    def arrays(self,path,keys=None):
        data=self.source.read_npz(path.removeprefix('evidence/'))
        return data if keys is None else {k:data[k]for k in keys}


def _close(observed,expected,label,rtol=1e-10,atol=1e-12):
    if not np.allclose(observed,expected,rtol=rtol,atol=atol):
        raise ValueError(f'Frozen numeric reproduction failed: {label}: {observed} versus {expected}')


def audit_native_metadata(rows):
    """Validate exact saved fork keys and every stream, not missing-key equality."""
    groups=defaultdict(list);paired=defaultdict(list);split_circuits=defaultdict(set)
    trial_ids=set();paths=set()
    hash_keys=('physical_state_hash','pending_recurrent_hash','pending_external_hash',
               'external_history_hash','prefix_spike_hash')
    seed_keys={'trial_seed':'events','phase_seed':'phase','envelope_seed':'burst',
               'future_seed':'future_events','future_phase_seed':'future_phase','future_burst_seed':'future_burst'}
    for row in rows:
        spec=row['spec'];meta=row['metadata'];fork=meta['fork'];rng=meta['rng_namespaces']
        cid=row['context_id'];match=re.fullmatch(r's(\d+)-p(\d+)-(.+)',cid)
        if match is None:raise ValueError('Malformed native context ID')
        sid,prefix,regime=match.groups();sid=int(sid);prefix=int(prefix)
        if row['trial_id']in trial_ids or row['path']in paths:raise ValueError('Duplicate native trial/path')
        trial_ids.add(row['trial_id']);paths.add(row['path'])
        if row['trial_id']!=spec['trial_id']or spec['history_id']!=cid:raise ValueError('Native trial/history identity mismatch')
        if row['structure_id']!=sid or int(spec['structure_id'])!=sid or spec['structure_seed']!=sid or rng['structure']!=sid:
            raise ValueError('Native circuit/structure identity mismatch')
        if row['pool']not in ('score','reference')or spec['future_id']!=f"{row['pool']}-{row['future_index']:04d}":
            raise ValueError('Native pool/future identity mismatch')
        if spec['split']!=row['split']or spec['regime']['name']!=regime or meta['regime']['name']!=regime:
            raise ValueError('Native split/regime mismatch')
        for key,rng_key in seed_keys.items():
            if spec[key]!=rng[rng_key]:raise ValueError('Saved RNG seed differs from actual runtime namespace: '+key)
        for key in hash_keys:
            value=fork.get(key)
            if not isinstance(value,str)or not re.fullmatch('[0-9a-f]{64}',value):
                raise ValueError('Required causal fork hash missing or malformed: '+key)
        if fork['cutoff_ms']!=6000 or spec['future_start_ms']!=6000 or not fork['future_schedules_are_oracle_only']:
            raise ValueError('Native causal cutoff/privileged future policy mismatch')
        if meta['native_dt_ms']!=.0625 or meta['total_duration_ms']!=7010 or meta['native_sample_count']!=112160:
            raise ValueError('Frozen native integration/support mismatch')
        if meta['configured_celsius']!=6.3 or meta['effective_celsius']!=6.3 or meta['initializations_per_episode']!=1:
            raise ValueError('Frozen temperature/lifecycle mismatch')
        if meta['sample_boundary_convention']!='(start,stop]':raise ValueError('Native timestamp convention mismatch')
        groups[cid].append(row);split_circuits[row['split']].add(sid)
        paired[(sid,prefix,row['pool'],row['future_index'])].append(row)
    validate_split_ids(split_circuits)
    repeated=0
    for cid,group in groups.items():
        repeated+=len(group)>1
        for key in hash_keys:
            if len({r['metadata']['fork'][key]for r in group})!=1:raise ValueError('Fork causal-state/history mismatch: '+cid+'/'+key)
        for key in ('trial_seed','phase_seed','envelope_seed'):
            if len({r['spec'][key]for r in group})!=1:raise ValueError('History seed changed across forks')
        for key in ('future_seed','future_phase_seed','future_burst_seed'):
            values=[r['spec'][key]for r in group]
            if len(set(values))!=len(values):raise ValueError('Repeated/overlapping score/reference future stream: '+cid+'/'+key)
        for pool in ('score','reference'):
            indices=[r['future_index']for r in group if r['pool']==pool]
            if sorted(indices)!=list(range(len(indices))):raise ValueError('Incomplete registered future ordinals')
    for group in paired.values():
        # Corresponding regimes share generators, not necessarily realized events.
        for key in seed_keys:
            if len({r['spec'][key]for r in group})!=1:raise ValueError('Regime-paired RNG namespace changed: '+key)
    return {'status':'complete','episodes':len(rows),'contexts':len(groups),'fork_contexts':int(repeated),
        'circuits_by_split':{k:sorted(v)for k,v in split_circuits.items()},
        'episode_counts_by_split':dict(Counter(r['split']for r in rows)),
        'episode_counts_by_pool':dict(Counter(r['pool']for r in rows)),
        'fork_hash_keys_verified':list(hash_keys),'rng_fields_verified':seed_keys,
        'regime_paired_stream_groups':len(paired),'score_reference_streams_disjoint':True,
        'physical_state_hash_recomputed_from_raw':False,
        'interpretation':'Verified saved manifest causal-state/pending-event hashes and seeds; raw state hashes are not reconstructed by this metadata audit.'}


def audit_existing(config,root,source,resume=False):
    owned=isinstance(source,(str,Path));s=EvidenceSource(source)if owned else source
    try:
        imp=import_evidence(s,root,verify_all=True);identity=imp['identity']
        if identity.get('archive_sha256')and identity['archive_sha256']!=HISTORICAL_ARCHIVE_SHA256:
            raise ValueError('Frozen standard source archive differs from the expected v4 evidence')
        signature=canonical_hash({'source':identity['content_id'],'config':config,'analysis':module_hash('audit','precision','statistics')})
        marker=Path(root)/'audit_existing.json';old=resume_record(marker,signature,resume)
        if old:return old
        results=[]
        for job in ['A0__T0_l10_d128','A1__T0_l2_d32','A2__T3_l10_d1000']:
            meta=s.read_json('derived/'+job+'/manifest.json');fit=s.read_json('models/'+job+'/manifest.json')
            p=s.read_npz('derived/'+job+'/evaluation.npz')
            pred=s.read_npz('models/'+job+'/'+fit['selected_candidate']+'/evaluation.npz')
            base=s.read_npz('models/'+job+'/train_mean/evaluation.npz')
            if not np.array_equal(p['context_id'],pred['context_id'])or not np.array_equal(p['context_id'],base['context_id']):
                raise ValueError('Prediction IDs/order differ from target panel')
            if str(pred['target_fingerprint'])!=meta['target_fingerprint']or canonical_hash(meta['risk_contract'])!=meta['target_fingerprint']:
                raise ValueError('Prediction risk/target fingerprint mismatch')
            y=np.asarray(p['Y'],np.float64);mask=np.asarray(p['future_mask'],bool)
            means=np.asarray(pred['mean'],np.float64);base_means=np.asarray(base['mean'],np.float64)
            risk=[];baseline=[];floor=[];corrected=[];identity_residual=[]
            for i in range(len(y)):
                yy=y[i,mask[i]];m=len(yy)
                if m!=8:raise ValueError('Frozen evaluation must have exactly eight scoring futures')
                rr=float(np.mean((yy-means[i])**2));vv=float(yy.var(0,ddof=1).mean())
                excess=float(np.mean((means[i]-yy.mean(0))**2)-vv/m)
                risk.append(rr);baseline.append(float(np.mean((yy-base_means[i])**2)));floor.append(vv);corrected.append(excess)
                identity_residual.append(rr-vv-excess)
            _close(identity_residual,0.,'unbiased finite-M excess identity',atol=2e-12)
            circuits=np.unique(p['structure_id']);equal=lambda x:stable_mean([stable_mean(np.asarray(x)[p['structure_id']==c])for c in circuits])
            rm=equal(risk);r0=equal(baseline);v=equal(floor)
            saved=s.read_json('scores/'+job+'/summary.json')
            selected_rows=[r for r in saved['rows']if r['candidate']==fit['selected_candidate']]
            if not selected_rows:raise ValueError('Selected saved score missing')
            calibration={}
            for row in selected_rows:
                diagnostics,_=gaussian_diagnostics(y,means,pred['variance_'+row['calibration']],mask,y_scale=meta['normalization']['y_scale'])
                probability={}
                for metric in ('coverage_50','coverage_80','coverage_95','marginal_crps','gaussian_nll','gaussian_nll_physical'):
                    value=equal(diagnostics[metric]);_close(value,row['summary'][metric]['mean'],job+'/'+row['calibration']+'/'+metric)
                    probability[metric]=value
                calibration[row['calibration']]=probability
                for metric,value in [('mean_risk',rm),('r0',r0),('state_floor',v),('corrected_excess',equal(corrected))]:
                    _close(value,row['summary'][metric]['mean'],job+'/'+metric)
                _close(rm/r0,row['mean_risk_over_mean_baseline'],job+'/ratio')
            results.append({'job_id':job,'selected':fit['selected_candidate'],'model_risk':rm,'baseline_risk':r0,
                'gain_pp':100*(1-rm/r0),'state_variance_standardized':v,'state_variance_physical_v2':v*meta['normalization']['y_scale']**2,
                'floor_baseline_ratio':v/r0,'corrected_excess':equal(corrected),'negative_context_excess_count':int(np.sum(np.asarray(corrected)<0)),
                'maximum_finite_M_identity_residual':float(np.max(np.abs(identity_residual))),'circuit_count':len(circuits),
                'original_score_rows_reproduced':len(selected_rows),'calibration_diagnostics':calibration,
                'calibration_scope':'Original train/development variance access; marginal diagonal Gaussian diagnostics, not separate-calibration v6 or joint coverage',
                'target_fingerprint':meta['target_fingerprint'],
                'acquisition_hash':meta['risk_contract']['acquisition_hash'],'normalization_id':meta['risk_contract']['normalization_id'],
                'target_support_ms':meta['risk_contract']['support'],'target_dimension':meta['risk_contract']['dimension'],
                'forecast_scope':'Original short-tail predictor; not a full-history Bayes limit'})
        components,reproduced=archive_components(_NumericView(s,identity))
        historical=next(r for r in components if r['family']=='secondary')
        difference=np.asarray(historical['circuit_values'],np.float64)
        match=s.read_json('retention/matched_contrast.json')
        saved=np.asarray([match['per_circuit_difference'][str(k)]for k in sorted(map(int,match['per_circuit_difference']))])
        _close(difference,saved,'H3 original per-circuit seed-average loss from logits')
        _close(stable_mean(difference),match['mean_difference'],'H3 original mean contrast')
        permutations=np.asarray(list(itertools.product([-1.,1.],repeat=len(difference))))
        signflip=float(np.mean(abs((permutations*difference).mean(1))>=abs(difference.mean())-1e-14))
        _close(signflip,match['signflip']['p_value'],'H3 original sign-flip diagnostic')
        native=s.read_json('native/manifest.json');native_audit=audit_native_metadata(native['trials'])
        if native_audit['episodes']!=832 or native_audit['contexts']!=352:raise ValueError('Frozen prospective episode/context counts differ')
        out={'status':'complete','signature':signature,'source':identity,'verification':imp['verification'],'forecasts':results,
            'frozen_h3':{'mean_difference_bits':stable_mean(difference),'circuit_differences':difference.tolist(),
                'all_positive':bool(np.all(difference>0)),'circuit_count':len(difference),'reproduced_from_saved_logits':True,
                'signflip_p':signflip,'binomial_sign_p':float(binomtest(int(np.sum(difference>0)),len(difference),.5).pvalue),
                'interpretation':'Original primary preserved; sign flips require sign symmetry for a mean null; binomial sign inference concerns direction probability.'},
            'numeric_primary_component_reproduction':reproduced,'precision_components':components,'native_metadata':native_audit,
            'episodes':native_audit['episodes'],'contexts':native_audit['contexts'],
            'native_verification':'Manifest metadata audited; raw native bytes absent'if not identity['native_present']else 'Available payload checksums verified; no simulation rerun',
            'scope':'Completed v4 evidence only; all v6 existing-data follow-up is exploratory.'}
        atomic_json(marker,out);return out
    finally:
        if owned:s.close()
