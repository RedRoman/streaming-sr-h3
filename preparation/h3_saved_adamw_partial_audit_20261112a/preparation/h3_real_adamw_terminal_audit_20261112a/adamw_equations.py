"""Independent first-step FP32 equation checks; not NPU kernel reenactment."""
import numpy as np
from audit_arrays import require, array_sha, verify_reduction

LR=1e-4*min(1.,1/100)
TOLERANCES={'exp_avg':{'rtol':3e-6,'atol':1e-20},
 'exp_avg_sq':{'rtol':3e-6,'atol':1e-30},'update':{'rtol':3e-6,'atol':2e-12}}

def close(actual, expected, tolerance, label):
    error=np.abs(actual.astype(np.float64)-expected)
    allowed=tolerance['atol']+tolerance['rtol']*np.abs(expected)
    normalized=error/allowed
    facts={'max_abs_error':float(error.max(initial=0)),
      'worst_normalized_residual':float(normalized.max(initial=0)),
      'failed_coordinates':int(np.count_nonzero(normalized>1)),
      'absolute_error_quantiles':dict(zip(('p0','p50','p95','p99','p100'),map(float,np.quantile(error,[0,.5,.95,.99,1])))),
      'normalized_residual_quantiles':dict(zip(('p0','p50','p95','p99','p100'),map(float,np.quantile(normalized,[0,.5,.95,.99,1]))))}
    require(facts['failed_coordinates']==0,label+' equation differs: '+str(facts))
    return facts

def first_step(before, gradient, after, moment, variance, row):
    require(before.dtype==gradient.dtype==after.dtype==moment.dtype==variance.dtype==np.float32,
            'AdamW tensor dtype differs')
    require(before.shape==gradient.shape==after.shape==moment.shape==variance.shape,'AdamW shapes differ')
    require(all(np.isfinite(v).all() for v in (before,gradient,after,moment,variance)),'nonfinite AdamW array')
    require(np.all(variance>=0),'negative AdamW variance')
    zero=gradient==0
    require(array_sha(after[zero])==array_sha(before[zero]),'zero-gradient parameter bytes changed')
    require(np.count_nonzero(moment[zero])==np.count_nonzero(variance[zero])==0,'zero-gradient state not zero')
    g=gradient.astype(np.float64)
    e1=close(moment,.1*g,TOLERANCES['exp_avg'],'first moment')
    e2=close(variance,.001*g*g,TOLERANCES['exp_avg_sq'],'second moment')
    expected=-LR*(moment.astype(np.float64)/(1-.9))/(np.sqrt(variance.astype(np.float64)/(1-.999))+1e-8)
    delta=after.astype(np.float64)-before.astype(np.float64)
    e3=close(delta,expected,TOLERANCES['update'],'first AdamW update')
    require(row['before_sha256']==array_sha(before) and row['after_sha256']==array_sha(after)
        and row['all_finite'] is True and row['zero_gradient_coordinates_unchanged'] is True,'producer update binding differs')
    require(row['changed_coordinates']==int(np.count_nonzero(delta)),'update changed count differs')
    verify_reduction(row['update_l2'],float(np.linalg.norm(delta)),'AdamW update L2')
    verify_reduction(row['update_rms'],float(np.sqrt(np.mean(delta*delta))),'AdamW update RMS')
    return {'zero_gradient_byte_exact_coordinates':int(zero.sum()),'changed_coordinates':int(np.count_nonzero(delta)),
      'first_moment':e1,'second_moment':e2,'update':e3,
      'NPU_kernel_bitexact_reexecution_claim':False,'tolerances':TOLERANCES}
