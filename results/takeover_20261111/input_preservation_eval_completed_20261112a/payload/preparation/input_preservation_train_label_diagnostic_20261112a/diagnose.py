"""One TRAIN-only cache diagnostic; no model, threshold selection, or updates."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
import pathlib, json, hashlib, signal, resource, time
import numpy as np

ROOT = pathlib.Path('/cache/zihaohe/omni_openh3_20261111')
MANIFEST = ROOT/'runs/input_preservation_train_cache_20261112b/output/manifest.json'
MANIFEST_SHA = '4fc49311e3c03b1e089e9eafbd33207965b806ddc5b250fb4fbe936297dc70b9'
OUT = ROOT/'runs/input_preservation_train_label_diagnostic_20261112a'

def sha(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()

def stat(y):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    bounds=[]
    for lower in (True,False):
        lo, hi = -4., 4.
        for _ in range(60):
            mid = (lo+hi)/2; g=np.clip(mid-y,-1.,1.).mean()
            if g>=0 if lower else g>0: hi=mid
            else: lo=mid
        bounds.append((lo+hi)/2)
    c=float(np.clip(0.,*bounds)); e=np.abs(c-y)
    return {'tiles':len(y),'positive_fraction':float((y>0).mean()),
            'negative_fraction':float((y<0).mean()),'zero_fraction':float((y==0).mean()),
            'saturated_positive_fraction':float((y==4).mean()),
            'saturated_negative_fraction':float((y==-4).mean()),
            'mean':float(y.mean()),'quantiles':np.quantile(y,[0,.1,.25,.5,.75,.9,1]).tolist(),
            'smoothL1_gradient_at_zero':float(np.clip(-y,-1,1).mean()),
            'TRAIN_empirical_constant_SmoothL1_minimizer':c,
            'constant_minimizer_interval':bounds,
            'constant_SmoothL1_loss':float(np.where(e<1,.5*e*e,e-.5).mean())}

def main():
    OUT.mkdir(exist_ok=False); start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120 second diagnostic budget')))
    signal.alarm(120); os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
    report={'schema':'rphi.input_preservation_train_label_diagnostic.v1','complete':False,
      'TRAIN_only':True,'dev_arrays_read':0,'model_calls':0,'training_steps':0,
      'source_sha256':sha(__file__),'manifest_sha256':MANIFEST_SHA,
      'scope':'Descriptive TRAIN distribution and global constant Huber optimum only; no learned predictor, dev threshold selection, or causal explanation.'}
    try:
        assert sha(MANIFEST)==MANIFEST_SHA
        m=json.loads(MANIFEST.read_text()); assert m['complete'] and m['role']=='train'
        docs={}
        for k in ('records','batches'):
            a=m[k]; assert sha(a['path'])==a['sha256']; docs[k]=[json.loads(l) for l in pathlib.Path(a['path']).read_text().splitlines()]
        assert len(docs['records'])==16000 and len(docs['batches'])==1000
        labels=np.empty((16000,36),np.float32)
        for i,b in enumerate(docs['batches']):
            assert b['batch_index']==i and sha(b['path'])==b['sha256']
            with np.load(b['path'],allow_pickle=False) as z:
                y=z['normalized_benefit'];assert y.shape==(16,1,6,6) and y.dtype==np.float32 and np.isfinite(y).all()
                assert hashlib.sha256(y.tobytes()).hexdigest()==b['arrays']['normalized_benefit']['sha256']
                labels[i*16:(i+1)*16]=y.reshape(16,36)
        profiles={}
        for i,row in enumerate(docs['records']):
            assert row['row_index']==i and row['split']=='train'
            profiles.setdefault(row['profile_id'],[]).append(i)
        report['all']=stat(labels)
        report['profiles']={k:{'rows':len(v),'row_fraction':len(v)/16000,**stat(labels[v])} for k,v in sorted(profiles.items())}
        report.update(complete=True,actual_label_arrays_read=1000,actual_tiles=576000,quality_accepted=False)
    except BaseException as e: report['error']={'type':type(e).__name__,'message':str(e)}
    report['elapsed_seconds']=time.monotonic()-start
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report));return 0 if report['complete'] else 1

if __name__=='__main__':raise SystemExit(main())
