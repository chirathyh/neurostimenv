"""Future-only band energies from physical orthonormal real Fourier coordinates."""
from dataclasses import dataclass, replace
import re
import numpy as np
from ..v4.targets import TargetSpec as LegacyTargetSpec, extract_target, native_block, axes
from ..targets import canonical_hash


@dataclass(frozen=True)
class TargetSpec(LegacyTargetSpec):
 """V6 extension: lead is the left-open target-start offset, including zero.

 The historical serialization/fingerprint is inherited unchanged. Display timing
 metadata is recorded separately so old operator identities remain valid.
 """
 def __post_init__(self):
  if self.lead_ms==0:
   # Reuse all historical validation while permitting the requested zero gap.
   replace(self,lead_ms=2.)
  else:super().__post_init__()


def parse_target_id(target):
 match=re.fullmatch(r'(T0|T3|T4_energy_(?:alpha|beta_low|beta_high)|T4_log_energy_(?:alpha|beta_low|beta_high))_l(0|2|10)_d(32|128|1000)',target)
 if not match:raise ValueError('Unknown explicit physical target '+str(target))
 family,h,w=match.groups();h=float(h);w=float(w)
 if family=='T0' and (h not in (0,2,10) or w not in (32,128)):raise ValueError('Unsupported waveform support')
 if family!='T0' and (h not in (0,10) or w!=1000):raise ValueError('Unsupported Fourier/energy support')
 return family,h,w


def target_spec(target,cutoff_ms=6000.):
 family,h,w=parse_target_id(target)
 if family.startswith('T4_'):return spec(family.split('energy_',1)[1],family.startswith('T4_log_'),lead_ms=h,cutoff_ms=cutoff_ms)
 return TargetSpec(family=family,lead_ms=h,duration_ms=w,cutoff_ms=cutoff_ms)


def timing_metadata(target,*,cutoff_ms=6000.,input_availability_ms=None):
 """Native indices are one-based offsets after cutoff, never array positions.

 At h=0 the first sample is cutoff+.0625 ms and first 32-sample bin
 ends at cutoff+2 ms. The observation at cutoff belongs only to history.
 """
 if isinstance(target,str):family,h,w=parse_target_id(target)
 else:family,h,w=target.family,float(target.lead_ms),float(target.duration_ms);cutoff_ms=float(target.cutoff_ms)
 dt=.0625;wave=family=='T0';first=round(h/dt)+1;last=round((h+w)/dt)
 out={'target_start_offset_ms':h,'target_end_offset_ms':h+w,
      'first_bin_end_offset_ms':h+2. if wave else None,'bin_width_ms':2. if wave else None,
      'native_dt_ms':dt,'native_samples':round(w/dt),'native_samples_per_bin':32 if wave else None,
      'native_sample_support':{'cutoff_relative_first_index':first,'cutoff_relative_last_index':last,'index_convention':'one-based indices after cutoff; cutoff sample index0 excluded',
       'first_time_ms':cutoff_ms+h+dt,'last_time_ms':cutoff_ms+h+w},
      'physical_support_ms':{'left_open':cutoff_ms+h,'right_closed':cutoff_ms+h+w},
      'target_available_ms':cutoff_ms+h+w,'observation_cutoff_ms':cutoff_ms,
      'input_last_available_ms':cutoff_ms if input_availability_ms is None else float(input_availability_ms),
      'lead_mapping':'legacy lead_ms equals target_start_offset_ms, not first bin end; h=0 omits no future interval',
      'availability_note':'No sample backdating; Fourier/energy block available only after its complete one-second support.'}
 if out['input_last_available_ms']>cutoff_ms:raise ValueError('Ordinary input available after observation cutoff')
 return out

BANDS={'alpha':(8,12,False),'beta_low':(12,20,False),'beta_high':(20,30,True)}

def energy_from_coefficients(coefficients,frequency_hz,components,band,*,log=False,native_samples=16000,log_floor_v2=1e-40,log_reference_v2=1.):
 c=np.asarray(coefficients,dtype=np.float64);f=np.asarray(frequency_hz);q=np.asarray(components)
 if c.shape[-1]!=54 or native_samples!=16000 or not np.array_equal(f,np.repeat(np.arange(4,31),2)) or not np.array_equal(q,np.tile(['cos','sin'],27)):
  raise ValueError('Requires exact54 real orthonormal4..30Hz coordinates of16000native samples')
 if not np.isfinite(c).all():raise ValueError('Nonfinite Fourier coefficients')
 lo,hi,closed=BANDS[band];mask=(f>=lo)&((f<=hi) if closed else (f<hi))
 e=np.sum(c[...,mask]**2,axis=-1,keepdims=True,dtype=np.float64)/native_samples
 return np.log(np.maximum(e,log_floor_v2)/log_reference_v2) if log else e

def spec(band,log=False,*,lead_ms=10.,cutoff_ms=6000.):
 if lead_ms not in (0.,10.):raise ValueError('Energy target start offset must be0or10ms')
 lo,hi,closed=BANDS[band]
 out={'id':'T4_'+('log_energy_' if log else 'energy_')+band+f'_l{lead_ms:g}_d1000','family':'T4_log_energy' if log else 'T4_energy',
      'dimension':1,'lead_ms':lead_ms,'duration_ms':1000.,'physical_support_ms':{'left_open':cutoff_ms+lead_ms,'right_closed':cutoff_ms+lead_ms+1000.},
      'available_ms':cutoff_ms+lead_ms+1000.,'units':'natural log relative to1V^2' if log else 'V^2','risk_units':'squared natural-log units' if log else 'V^4',
      'band':band,'band_hz':[lo,hi],'include_upper':closed,'right_endpoint_included':closed,'native_samples':16000,'source_coordinates':'orthonormal real sin/cos,4..30Hz',
      'log_floor_v2':1e-40 if log else None,'log_reference_v2':1. if log else None,
      'operator_order':'Apply nonlinear energy to each future before conditional means/variances; not energy of predicted mean.'}
 out['operator_hash']=canonical_hash(out);return out

def gaussian_expected_energy(mu,variance,frequency_hz,components,band):
 mu=np.asarray(mu,dtype=np.float64);v=np.asarray(variance,dtype=np.float64)
 if np.any(v<0) or mu.shape!=v.shape:raise ValueError('Finite nonnegative matching marginal variances required')
 lo,hi,closed=BANDS[band];f=np.asarray(frequency_hz);mask=(f>=lo)&((f<=hi) if closed else (f<hi))
 energy_from_coefficients(mu,frequency_hz,components,band)
 return np.sum((mu**2+v)[...,mask],axis=-1,keepdims=True)/16000

def extract_physical_target(eeg_v,time_ms,target_id,cutoff_ms=6000.):
 """Apply the frozen physical transform to one complete native future block."""
 family,h,w=parse_target_id(target_id)
 base=TargetSpec(family='T3' if family.startswith('T4_') else family,lead_ms=h,duration_ms=w,cutoff_ms=cutoff_ms)
 value=extract_target(eeg_v,time_ms,base)
 if family.startswith('T4_'):
  ax=axes(base)
  return energy_from_coefficients(value,ax['target_frequency_hz'],ax['target_component'],family.split('energy_',1)[1],log=family.startswith('T4_log_'))
 return value

def target_metadata(target_id,cutoff_ms=6000.):
 """Uniform physical/operator and explicit timing metadata for reports/plans."""
 ts=target_spec(target_id,cutoff_ms)
 out=ts.to_dict() if isinstance(ts,TargetSpec) else dict(ts)
 physical_hash=ts.fingerprint if isinstance(ts,TargetSpec) else ts['operator_hash']
 return {**out,'id':target_id,**timing_metadata(target_id,cutoff_ms=cutoff_ms),'operator_hash':physical_hash,
         'timing_metadata_policy':'Explicit display metadata supplements unchanged historical operator hashes.'}
