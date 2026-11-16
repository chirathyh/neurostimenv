"""Explicit descriptive amendment matrix and conservative resource accounting."""
from pathlib import Path
import math
from .artifacts import canonical_hash, file_hash
from .config import counts
from .information_models import cost as information_model_cost, settings as information_model_settings
from .models import settings
from .targets import target_spec, timing_metadata


def enabled(config):
    return bool(config.get('extension', {}).get('enabled'))


def target_matrix(config):
    rows=[]
    for job in config['selected_job_ids']:
        acquisition,target=job.split('__',1)
        spec=target_spec(target)
        physical=spec if isinstance(spec,dict) else spec.to_dict()
        rows.append(dict(job_id=job,acquisition=acquisition,target=target,
            timing=timing_metadata(target),units=physical['units'],risk_units=physical['risk_units'],
            dimension=physical.get('dimension',physical.get('target_dimension',1)),
            history_samples=[64,2000],primary=(job==config['protocol']['primary_job_id']),
            inference='original family B' if job==config['protocol']['primary_job_id'] else 'secondary/descriptive; no extra alpha'))
    return rows


def estimate(config, *, seconds_per_episode, bytes_per_episode, phase0_root=None):
    """Storage is an uncompressed working/export reserve, not a compression promise."""
    design=counts(config);cfg=settings(config);matrix=target_matrix(config)
    neural=2*len(cfg['neural_seeds']) if cfg['neural_enabled'] else 0
    contexts=design['contexts_by_split'];total_contexts=sum(contexts.values());evaluation=contexts['evaluation']
    derived=predictions=weights=scores=0;candidate_count=0
    # Observed schema stores future_id as U128 and contexts fit within U32.
    # Count repeated arrays per file, with a separate fixed header/JSON reserve.
    future_id_bytes=128*4;context_metadata_bytes=32*4+8;file_overhead=16384
    for row in matrix:
        d=row['dimension'];n=43+neural+(not row['target'].startswith('T3'))
        candidate_count+=n
        for split,k in contexts.items():
            m=32 if split=='evaluation' else 4 if split=='calibration' else 1
            reference=128 if split=='evaluation' else 0
            # Exact implementation pads evaluation pools; account for those zeros
            # before any lossless compression/deduplication savings.
            derived+=k*(2000*2*8+(m+reference)*d*2*8+(m+reference)*(future_id_bytes+1)+1024)+file_overhead+2000*8
        predictions+=n*(total_contexts*(5*d*8+context_metadata_bytes)+len(contexts)*(d*8+file_overhead))
        # Ridge/PCA maps, native numeric TCN states, metadata and optimization logs.
        weights+=n*(2000*64*8+2000*d*8+50000*8+262144)
        # Three variance families: 9 context/future/coordinate arrays, 3 widths,
        # 2 joint context/future arrays, IDs, plus one risk-component file.
        gaussian=evaluation*(9*4*d*8+3*d*8+2*4*8+4*future_id_bytes+context_metadata_bytes)+file_overhead
        components=evaluation*(12*8+2*4*8+4+4*future_id_bytes+context_metadata_bytes)+file_overhead
        scores+=n*(3*gaussian+components+1048576)  # JSON/MC summaries reserve per candidate
    # 20 A0 target readouts: two deterministic banks plus three forecast seeds,
    # projected/unquantized and B32, seven alphas and a mean fallback.
    readout_cost=information_model_cost(config)
    readouts=readout_cost['code_target_jobs']*readout_cost['code_methods_per_target']+len(matrix)
    secondary=readouts*(total_contexts*(64*8+2048)+4*1024*1024)
    nonnative=math.ceil(1.25*(derived+predictions+weights+scores+secondary))
    observed=None
    if phase0_root is not None and Path(phase0_root).exists():
        directories=['derived','models','scores','information_models','information','retention','replication','report']
        observed=sum(p.stat().st_size for name in directories for p in (Path(phase0_root)/name).rglob('*') if p.is_file())
    return dict(status='provisional_storage_upper_allowance',target_model_matrix=matrix,
        acquisition_target_jobs=len(matrix),ordinary_candidates=candidate_count,
        ordinary_neural_fits=len(matrix)*neural,maximum_selected_fp64_fits=len(matrix)*len(cfg['neural_seeds']) if neural and cfg['dtype']=='float32' and cfg['fp64_sensitivity'] else 0,
        code_forecast_readout_scope='A0, all20 targets; Feature18/PCA/three legacy forecast fits, d8 unquantized/B32; 7 ridge alphas plus mean fallback',
        state_model='bounded partial membrane/gate features when verified; synapses, phase, pending events, parameters and acquisition state omitted; no fitted complete-state oracle claim',
        information_models_operation_counts=readout_cost,possible_information_readout_methods=readouts,
        core_additional_episodes_for_targets=0,extra_replay_episodes=8,timing_gate_episodes=8,
        generation_overhead_hours=16*seconds_per_episode/3600,
        generation_overhead_wall_hours=8*seconds_per_episode/3600/config['data']['workers']+8*seconds_per_episode/3600,
        replay_timing_status='provisional serial replay; actual summed/wall time used once measured',
        replay_hours=8*seconds_per_episode/3600,generation_overhead_native_bytes=math.ceil(16*bytes_per_episode),
        working_nonnative_bytes=nonnative,lossless_nonnative_export_reserve_bytes=nonnative,
        storage_components_uncompressed=dict(derived=int(derived),predictions=int(predictions),weights=int(weights),scores=int(scores),secondary=int(secondary)),
        observed_phase0_nonnative_bytes=observed,
        storage_assumptions='Uncompressed padded arrays; future IDs U128 (512 bytes), context IDs <=U32, actual Gaussian array shapes plus 1MiB/candidate JSON/MC reserve; numeric weights; 25% overhead and another full numeric export. Compression savings are not spent before measurement.',
        fitting_cost='Main TCN50-update and fp64 throughput measured by same-host gate, scaled to60 jobs; Phase0 records actual linear and secondary readout elapsed times. H2/reference/report/export overhead is additional, not zero.',
        missing_data_policy='Archive supports33 delayed jobs.27 zero-gap jobs require actual native recordings; no additional core time support needed beyond7010ms.',
        primary_protocol_unchanged=True,secondary_only=True)


def protocol(config):
    directory=Path(__file__).parent
    names=('targets','information','information_models','replay','extension_planning','extended_reporting')
    return dict(version=config['extension']['version'],secondary_only=True,
        target_matrix=target_matrix(config),information_model_grid=information_model_settings(config),information_table_scale='100 × model physical MSE / target-specific train-mean physical MSE',
        unknown_EEG_ideal='scientifically unidentified; never filled by a fitted predictor',
        state_floor='unbiased within-context variance, first4registered scoring futures; finite-M uncertainty and negative excess preserved',
        comparisons='paired circuits; same acquisition/duration/history/model-family policy for gap comparisons',
        code_readout='A0 full-history Feature18/PCA/legacy forecast; d8/B32 and projected unquantized; equal dev-selected ridge grid and mean fallback',
        replay='eight registered timing contexts/futures replayed from full initial prefix with identical external schedule; no saved-state checkpoint claim',
        additional_alpha=0.,analysis_dependencies={n:file_hash(directory/(n+'.py')) for n in names})


def secondary_fit_cost(config,phase0_root,*,device=None):
    """Observed same-host timing plus explicitly uncertain scaling scenarios."""
    from .artifacts import read_json,artifact_record
    from .information_models import runtime_identity
    if phase0_root is None:return dict(status='unavailable',reason='Phase0 readout timing root not supplied')
    root=Path(phase0_root);path=root/'information_models/timing.json'
    if not path.exists():return dict(status='unavailable',reason='Phase0 persistent information-readout fitting timing missing')
    ledger=read_json(path);observed=ledger['summary']
    if observed.get('status')!='measured' or observed.get('measured_fit_elapsed_seconds',0)<=0:
        return dict(status='unavailable',reason='Inference-only resume is not an observed fitting cost')
    chosen_device=device or observed['runtime']['device'];current=runtime_identity(config,chosen_device)
    if current!=observed['runtime'] or canonical_hash(current)!=observed['runtime_hash']:
        return dict(status='unavailable',reason='Secondary fitting timing host/device/software/settings changed; remeasure on execution host')
    matched=[r for r in ledger['calls'] if r['runtime_hash']==observed['runtime_hash'] and r['includes_new_fitting']]
    if not matched or not math.isclose(sum(r['elapsed_seconds'] for r in matched),observed['measured_fit_elapsed_seconds'],rel_tol=1e-12):
        raise ValueError('Persistent secondary fitting timing sum does not reproduce')
    measured_methods=sum(len(r['new_readout_fits']) for r in matched)
    if not measured_methods or observed['training_contexts']<=0:raise ValueError('Readout timing lacks fitted methods/training contexts')
    design=counts(config);tr=design['contexts_by_split']['train'];total=sum(design['contexts_by_split'].values())
    measured_total=max(sum(r['split_context_counts'].values()) for r in matched)
    ops=information_model_cost(config);methods=ops['code_target_jobs']*ops['code_methods_per_target']+len(config['selected_job_ids'])
    fit_factor=max(1.,tr/observed['training_contexts']);context_factor=max(1.,total/measured_total);method_factor=max(1.,methods/measured_methods)
    elapsed=observed['measured_fit_elapsed_seconds'];linear=elapsed*max(context_factor,fit_factor)*method_factor
    nonlinear=elapsed*max(context_factor,fit_factor**3)*method_factor
    return dict(status='measured_extrapolation',source=artifact_record(root,path),phase0_root=str(root.resolve()),
        observed_fit_seconds=elapsed,observed_training_contexts=observed['training_contexts'],observed_readout_methods=measured_methods,
        operation_counts=ops,prospective_training_contexts=tr,prospective_readout_methods=methods,
        linear_context_projection_hours=linear/3600,cubic_training_design_scenario_hours=nonlinear/3600,
        scenarios_are_bounds=False,runtime_hash=observed['runtime_hash'],device=chosen_device,
        interpretation='Measured prior complete fitting calls; illustrative linear-context and cubic-training-design scaling scenarios. Different SVD dimensions, GPU kernels, caching and I/O can violate either scenario; these are not guaranteed wall-time bounds.',
        additional_unmeasured_cost='Original retention may fit three further 200-update forecast encoders; H2/bootstrap, report/figure/export I/O and other retention readouts remain additional. Main ordinary TCN/float64 timing is reported separately.')
