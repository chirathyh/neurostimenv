"""Independent calibration circuits fit only constant diagonal residual variances."""
from pathlib import Path
import json
import numpy as np
from ..artifacts import atomic_npz
from ..targets import canonical_hash
from ...storage import atomic_json,file_hash,read_json
from ..v4.models import job_ids
from .models import load_panel,matched_variance,settings,prediction_record


def calibrate(config,root,jobs=None,resume=False):
    root=Path(root);cfg=settings(config);outputs=[]
    for job in job_ids(root,jobs,config):
        panel,derived=load_panel(root,job)
        if 'calibration' not in panel:raise ValueError('Independent calibration split unavailable; no development substitution')
        calibration=panel['calibration'];circuits=np.unique(calibration['structure_id'])
        expected=int(config.get('calibration',{}).get('required_circuits',24))
        if config.get('claim_status')!='engineering' and len(circuits)!=expected:raise ValueError('Calibration circuit count differs from frozen protocol')
        fit=read_json(root/'models'/job/'manifest.json');selection=read_json(root/'models'/job/'selection.json')
        if selection['status']!='frozen' or (fit['selected_candidate']!=selection['selected_candidate'] or fit['selected_member_ids']!=selection['selected_member_ids']):raise ValueError('Means must be frozen before calibration')
        sig=canonical_hash(dict(source=file_hash(Path(__file__)),selection=file_hash(root/'models'/job/'selection.json'),
            fit=file_hash(root/'models'/job/'manifest.json'),calibration=file_hash(root/'derived'/job/'calibration.npz'),floor=cfg['variance_floor']))
        dest=root/'calibration'/job;marker=dest/'manifest.json'
        if marker.exists():
            old=read_json(marker)
            if not resume or old['signature']!=sig:raise ValueError('Calibration resume fingerprint mismatch')
            for rec in old['files']:
                if file_hash(root/rec['path'])!=rec['sha256']:raise ValueError('Calibration artifact checksum mismatch')
            outputs.append(dict(id=job,status='complete',resumed=True));continue
        records=[];files=[]
        for candidate in fit['candidates']:
            name=candidate['id'];predpath=root/'models'/job/name/'calibration.npz'
            expected=prediction_record(root,job,predpath)
            if file_hash(predpath)!=expected['sha256']:raise ValueError('Frozen calibration mean prediction checksum mismatch')
            with np.load(predpath,allow_pickle=False) as f:
                if not np.array_equal(f['context_id'],calibration['context_id']):raise ValueError('Calibration prediction IDs differ')
                if str(f['target_fingerprint'])!=derived['target_fingerprint']:raise ValueError('Calibration target fingerprint mismatch')
                if json.loads(str(f['risk_contract_json']))!=derived['risk_contract']:raise ValueError('Calibration risk contract mismatch')
                mean=np.asarray(f['mean'],float)
            variance=matched_variance(calibration,mean,cfg['variance_floor'])
            path=dest/name/'variance.npz';atomic_npz(path,variance=variance,calibration_context_id=calibration['context_id'],calibration_structure_id=calibration['structure_id'],
                target_fingerprint=np.asarray(derived['target_fingerprint']),risk_contract_json=np.asarray(json.dumps(derived['risk_contract'],sort_keys=True)))
            files.append(dict(path=str(path.relative_to(root)),sha256=file_hash(path),bytes=path.stat().st_size))
            records.append(dict(id=name,status='complete',path=str(path.relative_to(root)),mean_prediction_sha256=file_hash(predpath),
                variance_family='context-independent diagonal residual variance',fit_split='calibration',mean_frozen=True,
                circuit_count=len(circuits),context_count=len(mean),weighting='equal circuit, condition, prefix and valid future'))
        report=dict(status='complete',signature=sig,job_id=job,candidates=records,files=files,calibration_circuit_ids=circuits.tolist(),
            selected_candidate=fit['selected_candidate'],selected_member_ids=fit['selected_member_ids'],
            calibrated_means_unchanged=True,selection_uses_calibration=False,
            guarantee='parametric residual calibration; no conditional, joint or distribution-free conformal coverage guarantee')
        atomic_json(marker,report);outputs.append(dict(id=job,status='complete',resumed=False))
    result=dict(status='complete',jobs=outputs);atomic_json(root/'calibration/manifest.json',result);return result
