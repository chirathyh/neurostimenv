"""Future-only band energies from physical orthonormal real Fourier coordinates."""
import numpy as np
from ..targets import canonical_hash

BANDS={'alpha':(8,12,False),'beta_low':(12,20,False),'beta_high':(20,30,True)}

def energy_from_coefficients(coefficients,frequency_hz,components,band,*,log=False,native_samples=16000,log_floor_v2=1e-40,log_reference_v2=1.):
 c=np.asarray(coefficients,dtype=np.float64);f=np.asarray(frequency_hz);q=np.asarray(components)
 if c.shape[-1]!=54 or native_samples!=16000 or not np.array_equal(f,np.repeat(np.arange(4,31),2)) or not np.array_equal(q,np.tile(['cos','sin'],27)):
  raise ValueError('Requires exact54 real orthonormal4..30Hz coordinates of16000native samples')
 if not np.isfinite(c).all():raise ValueError('Nonfinite Fourier coefficients')
 lo,hi,closed=BANDS[band];mask=(f>=lo)&((f<=hi) if closed else (f<hi))
 e=np.sum(c[...,mask]**2,axis=-1,keepdims=True,dtype=np.float64)/native_samples
 return np.log(np.maximum(e,log_floor_v2)/log_reference_v2) if log else e

def spec(band,log=False):
 lo,hi,closed=BANDS[band]
 out={'id':'T4_'+('log_energy_' if log else 'energy_')+band+'_l10_d1000','family':'T4_log_energy' if log else 'T4_energy',
      'dimension':1,'lead_ms':10.,'duration_ms':1000.,'physical_support_ms':{'left_open':6010.,'right_closed':7010.},
      'available_ms':7010.,'units':'natural log relative to1V^2' if log else 'V^2','risk_units':'squared natural-log units' if log else 'V^4',
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
