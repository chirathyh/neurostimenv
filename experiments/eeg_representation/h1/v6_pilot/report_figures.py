"""Publication-size vector figures from actual pilot result records only."""
from pathlib import Path
import os
import re
import numpy as np
from .reporting import _json, _hash, _number, _anchor_acquisition, _preferred_population

COLORS={'complete_state_candidate':'#CC79A7','complete_state_selected':'#7A3E78',
        'state_variance':'#777777','eeg_short':'#56B4E9','eeg_full':'#0072B2',
        'training_mean':'#999999','measured_replay':'#009E73'}


def _interval(row):
    v=row.get('error_interval',row.get('uncertainty'))
    if isinstance(v,(list,tuple)) and len(v)==2 and all(_number(x) for x in v):return v
    if isinstance(v,dict):
        if all(_number(v.get(k)) for k in ('lower','upper')):return [v['lower'],v['upper']]
    return None


def _label(row):
    return str(row.get('display_name',row.get('method','unknown'))).replace('_',' ')


def _unique(rows,keys):
    seen=set();out=[]
    for row in rows:
        key=tuple(str(row.get(k)) for k in keys)
        if key not in seen:out.append(row);seen.add(key)
    return out


def create_figures(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,
        'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    dest=Path(out)/'figures';dest.mkdir(parents=True,exist_ok=True);records=[]
    def save(fig,name,caption,category='measured'):
        fig.tight_layout()
        for ext in ('pdf','svg','png'):
            temporary=dest/(name+'.'+ext+'.tmp');fig.savefig(temporary,format=ext,dpi=300,bbox_inches='tight');os.replace(temporary,dest/(name+'.'+ext))
        plt.close(fig);records.append({'name':name,'caption':caption,'evidence_type':category,
            'paths':{ext:'figures/'+name+'.'+ext for ext in ('pdf','svg','png')},
            'sha256':{ext:_hash(dest/(name+'.'+ext)) for ext in ('pdf','svg','png')}})
    fig,ax=plt.subplots(figsize=(6.75,2.65))
    ax.axvspan(-45,0,color='#dbeaf2');ax.axvline(0,color='black',lw=.8)
    ax.text(-22,3.8,'Observed past\navailable by cutoff',ha='center',va='center',fontsize=8)
    for y,h in zip((3,2,1),(0,2,10)):
        ax.broken_barh([(h,32)],(y-.15,.3),facecolor='#0072B2',alpha=.75)
        ax.plot(h,y,'o',mfc='white',mec='#0072B2',ms=4);ax.plot(h+2,y,'|',color='black',ms=12)
        ax.plot(h+32,y,'o',color='#0072B2',ms=3)
        ax.text(h+33,y,f'h={h}: ({h}, {h+32}] ms',va='center',fontsize=8)
    ax.set_xlim(-45,103);ax.set_ylim(.4,4.3);ax.set_yticks([]);ax.set_xlabel('Time relative to cutoff (ms)')
    ax.text(55,3.65,'Open circle: excluded start\nBlack tick: first 2-ms bin end\nFilled circle: full-block availability',va='center',fontsize=7)
    ax.set_title('Target start, first bin and availability are different timestamps')
    save(fig,'target_timing','Design illustration: W32 waveform targets; W128 follows the same convention. A bin contains 32 native samples. Fourier/energy targets span one second after their stated h and become available only at block end. Causal acquisition timestamps remain unshifted.','design')
    sf=results.get('figures',{}).get('same_state_futures')
    if sf and sf.get('futures'):
        yy=np.asarray(sf['futures'],dtype=float);tt=np.asarray(sf['times_ms'],dtype=float)
        if yy.ndim!=2 or yy.shape[1]!=len(tt) or not np.isfinite(yy).all():raise ValueError('Invalid same-state figure arrays')
        fig,ax=plt.subplots(figsize=(6.75,2.6))
        past=np.asarray(sf.get('past',[]));pt=np.asarray(sf.get('past_times_ms',[]))
        if len(past):
            if len(past)!=len(pt):raise ValueError('Past figure length mismatch')
            ax.plot(pt,past,color='black',lw=.8,label='Identical observed past')
        lo,hi=np.quantile(yy,[.1,.9],axis=0)
        ax.fill_between(tt,lo,hi,color='#56B4E9',alpha=.25,label='Empirical 10-90% range')
        for i,row in enumerate(yy[:6]):ax.plot(tt,row,color='#0072B2',alpha=.35,lw=.55,label='Individual future' if i==0 else None)
        ax.plot(tt,yy.mean(axis=0),color='#D55E00',lw=1.2,label='Finite-ensemble mean')
        ax.axvline(0,color='.4',lw=.6);ax.set_xlabel('Time relative to cutoff (ms)');ax.set_ylabel(sf.get('units','V'))
        ax.set_title('One fixed physical context, several possible futures');ax.legend(ncol=2,loc='best')
        save(fig,'same_state_futures',f"Context {sf.get('context_id','unavailable')}; {len(yy)} sampled futures. Curves and empirical range describe this finite ensemble, not the exact conditional mean or a population confidence band. No future is selected for visual agreement.")
    info=results['tables']['h1_information']
    anchor_acq=_anchor_acquisition(info)
    anchors=[row for row in info if row.get('h_ms') in (0,10) and row.get('duration_ms')==128 and row.get('acquisition')==anchor_acq]
    anchors=[row for row in anchors if row.get('evidence_type') not in ('theoretical','unknown','privileged_replay') and row.get('method')!='measured_replay' and _number(row.get('error_baseline100'))]
    if anchors:
        populations=sorted({row.get('population','unspecified') for row in anchors})
        preferred=_preferred_population(anchors)
        anchors=[row for row in anchors if row.get('population','unspecified')==preferred]
        fig,axes=plt.subplots(1,2,figsize=(6.75,3.7),sharex=False)
        for ax,h in zip(axes,(0,10)):
            rows=_unique([r for r in anchors if r['h_ms']==h],['method'])[:12]
            if not rows:ax.text(.5,.5,'Unavailable: no measured target',ha='center',va='center',transform=ax.transAxes,fontsize=8)
            for i,row in enumerate(rows):
                v=row['error_baseline100']; interval=_interval(row);color=COLORS.get(row.get('method'),'#E69F00')
                ax.plot(v,i,'o',color=color,ms=4)
                if interval:ax.plot(interval,[i,i],color=color,lw=.9)
            ax.axvline(100,color='.6',lw=.6,ls='--');ax.set_yticks(range(len(rows)),[_label(r) for r in rows],fontsize=7);ax.invert_yaxis()
            ax.set_xlabel('MSE; training mean = 100');ax.set_title(f'{anchor_acq} / h{h} / W128')
        save(fig,'information_error','Matched population: '+preferred+'. Dots are measured fitted risks or estimated same-state variance, kept as different evidence types. The ideal EEG risk is unknown. Theoretical zero and measured replay are excluded from these fitted comparisons and reported separately. Whiskers appear only when a supplied descriptive interval is supported.')
    replay=[r for r in info if r.get('evidence_type')=='privileged_replay' or r.get('method')=='measured_replay']
    if replay and any(_number(r.get('error_baseline100')) for r in replay):
        pops={r.get('population') for r in replay}; rows=[r for r in info if r.get('population') in pops and _number(r.get('error_baseline100')) and r.get('evidence_type')!='theoretical']
        rows=_unique(rows,['method'])[:8];fig,ax=plt.subplots(figsize=(3.25,2.6))
        ax.scatter([r['error_baseline100'] for r in rows],range(len(rows)),s=16,color='#009E73');ax.set_yticks(range(len(rows)),[_label(r) for r in rows],fontsize=7);ax.invert_yaxis();ax.set_xlabel('Matched replay risk; mean = 100');ax.set_title('Measured privileged replay population')
        save(fig,'privileged_replay','Same replay population for all displayed comparators. Exact-future forcing is privileged and not available to state-only or EEG predictors. The theoretical zero is a separate mathematical row, not a substituted measurement.')
    comp=results['tables']['h1_comparisons']
    valid=[r for r in comp if _number(r.get('error_baseline100')) and r.get('population')==_preferred_population(comp)]
    if valid:
        fig,axes=plt.subplots(1,2,figsize=(6.75,2.8),sharey=True)
        for ax,w in zip(axes,(32,128)):
            rows=[r for r in valid if r.get('duration_ms')==w]
            for acq,col in zip(('A0','A1','A2'),('#0072B2','#D55E00','#009E73')):
                for hist,ls in (('short','--'),('full','-')):
                    rr=_unique([r for r in rows if r.get('acquisition')==acq and hist in str(r.get('history',''))],['h_ms']);rr=sorted(rr,key=lambda r:r.get('h_ms',0))
                    if rr:ax.plot([r['h_ms'] for r in rr],[r['error_baseline100'] for r in rr],ls+'o',ms=3,color=col,label=acq+' '+hist)
            ax.axhline(100,color='.7',ls=':',lw=.7);ax.set_xticks((0,2,10));ax.set_xlabel('Start offset h (ms)');ax.set_title(f'Native waveform / W{w}')
        axes[0].set_ylabel('MSE; same target mean = 100');axes[1].legend(ncol=2,fontsize=6)
        save(fig,'gap_history_acquisition','Same waveform definition is held fixed across acquisitions and histories at each h/W. Lines connect descriptive fitted-policy outcomes, not an isolated information-loss component. Changes across h or W change the target; denominator and population are recorded per row. Negative reductions remain visible.')
    h2=[r for r in results['tables']['h2_precision'] if _number(r.get('mean_error_corrected'))]
    if h2:
        job=next((r.get('job') for r in h2 if r.get('job','').startswith('A1') and 'l0_d128' in r.get('job','')),h2[0].get('job'))
        h2=[r for r in h2 if r.get('job')==job];fig,axes=plt.subplots(1,2,figsize=(6.75,2.8))
        for context in sorted({r.get('context_id','aggregate') for r in h2}):
            rows=sorted([r for r in h2 if r.get('context_id','aggregate')==context],key=lambda r:r['M'])
            axes[0].plot([r['M'] for r in rows],[r['mean_error_corrected'] for r in rows],'o-',ms=3,label=str(context))
            vr=[r for r in rows if _number(r.get('variance_error')) and r.get('M')!=1]
            if vr:axes[1].plot([r['M'] for r in vr],[r['variance_error'] for r in vr],'o-',ms=3,label=str(context))
        for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks((1,2,4,8,16,32),[str(x) for x in (1,2,4,8,16,32)]);ax.set_xlabel('Scoring futures M');ax.axhline(0,color='.6',lw=.6)
        mean_units=h2[0].get('mean_error_units','units unavailable');variance_units=h2[0].get('variance_error_units','units unavailable')
        axes[0].set_ylabel('Corrected mean error\n('+str(mean_units)+')');axes[1].set_ylabel('Corrected squared variance error\n('+str(variance_units)+')');axes[0].set_title('Finite-reference corrected mean');axes[1].set_title('M=1 variance unavailable');axes[0].legend(fontsize=6)
        save(fig,'h2_finite_futures','Target '+str(job)+'; actual L='+','.join(str(x) for x in sorted({r.get('L') for r in h2 if r.get('L') is not None}))+'. Two fixed contexts do not establish population generalization. Repeated subsets share outcomes and references. Negative finite-reference corrected errors are possible and retained. Variance error is squared variance-estimation error; its interval is unavailable unless explicitly supplied. A variance-difference interval is not substituted. Units, reference uncertainty and all targets appear in the numeric table.')
    h3=[r for r in results['tables']['h3_retention'] if r.get('population')==_preferred_population(results['tables']['h3_retention']) and _number(r.get('reconstruction_nmse')) and _number(r.get('carrier_ce_bits'))]
    if h3:
        fig,axes=plt.subplots(1,2,figsize=(6.75,2.8));styles={'Feature18':'#0072B2','PCA':'#D55E00','forecast':'#009E73'}
        for r in h3:
            name=str(r.get('representation','unknown'));col=next((v for k,v in styles.items() if k.lower() in name.lower()),'#777777')
            quant=r.get('budget_bits')==32
            axes[0].scatter(r['reconstruction_nmse'],r['carrier_ce_bits'],s=28,color=col,marker='o' if quant else 'x',label=name if quant else None)
            if _number(r.get('future_error_baseline100')):axes[1].scatter(r['reconstruction_nmse'],r['future_error_baseline100'],s=28,color=col,marker='o' if quant else 'x')
        axes[0].legend(loc='best',fontsize=7);axes[0].set_ylabel('Carrier cross entropy (bits)');axes[1].set_ylabel('Future MSE; mean = 100')
        for ax in axes:ax.set_xlabel('Past reconstruction NMSE; mean = 1')
        axes[0].set_title('Past reconstruction and task retention');axes[1].set_title('Past reconstruction and future prediction')
        save(fig,'h3_reconstruction_usefulness','Population: '+str(_preferred_population(h3))+'. Shared d8 codes: B32 means four bits per coordinate, eight coordinates, 32 payload bits. Crosses are projected-unquantized controls; circles are B32. Shared encoder/codebook storage is separate. A poorer reconstruction can coexist with lower task cross entropy; the decoder-dependent metrics are distinct and do not measure total information. Full clipping/projection controls remain in tables.')
    h4=[r for r in results['tables']['h4_targets'] if r.get('population')==_preferred_population(results['tables']['h4_targets']) and _number(r.get('relative_mse_reduction_pp'))]
    if h4:
        h4_acq='A1' if any(r.get('acquisition')=='A1' for r in h4) else next((r.get('acquisition') for r in h4 if r.get('acquisition')),'A1')
        rows=_unique([r for r in h4 if r.get('acquisition','A1')==h4_acq and r.get('h_ms') in (0,10)],['target','h_ms','duration_ms'])
        if rows:
            fig,ax=plt.subplots(figsize=(6.75,max(2.9,.22*len(rows))))
            ax.scatter([r['relative_mse_reduction_pp'] for r in rows],range(len(rows)),s=16,color='#0072B2');ax.axvline(0,color='.5',lw=.7)
            labels=[f"{r.get('target')} h{r.get('h_ms')} W{r.get('duration_ms')} [{r.get('physical_units','units unavailable')}]" for r in rows]
            ax.set_yticks(range(len(rows)),labels,fontsize=7);ax.invert_yaxis();ax.set_xlabel('MSE reduction vs target-specific mean\n(percentage points)');ax.set_title(h4_acq+' / full history: target-dependent fitted predictability')
            save(fig,'h4_target_outcomes','Each point uses its own target-specific physical reference risk. This dimensionless display is not a comparison of raw physical errors. Fourier coordinates use one-second future-only orthonormal block projections; energy is computed on each future before fitting. Expected energy is not generally the squared predicted mean waveform.')
    generated={'target_timing','same_state_futures','information_error','privileged_replay','gap_history_acquisition','h2_finite_futures','h3_reconstruction_usefulness','h4_target_outcomes'}
    present={r['name'] for r in records}
    for stale in generated-present:
        for ext in ('pdf','svg','png'):
            path=dest/(stale+'.'+ext)
            if path.exists():path.unlink()
    _json(dest/'captions.json',records);return records
