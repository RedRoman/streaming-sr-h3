"""File/NumPy-only P1 terminal audit. Never import a model or replay an optimizer."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='2'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

SCHEMA='rphi.history_utility_one_step_array_audit.v1'
PRODUCER_SHA='8c2611302c3d3d9445ea6f839f6705bac7bc5e6dc16916c90b3c0940887478a6'
COUNTS={'teacher_training':1,'teacher_labels':3,'student_normal':1,'student_own_interventions':2,
        'restorer_total':7,'gate':1,'backward':2,'teacher_optimizer_steps':1,'student_optimizer_steps':0,'NPU_calls':0}
# Frozen before reading any actual-data result. Float64 reference vs Torch FP32 pooling.
RAW_ATOL=2e-7
NORMALIZED_ATOL=3e-4
MAX_ARRAY_BYTES=128<<20
MAX_TOTAL_ARRAY_BYTES=256<<20

def require(ok,message):
    if not ok:raise ValueError(message)

class NumericAuditError(ValueError):
    def __init__(self,message,diagnostics):
        super().__init__(message);self.diagnostics=diagnostics

def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def array_sha(a):return hashlib.sha256(memoryview(a).cast('B')).hexdigest()

def stat_id(p):
    s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

class Arrays:
    def __init__(self,root,records):self.root=Path(root).resolve();self.records=records;self.verified={}
    def path(self,row):
        rel=Path(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and rel.suffix=='.npy','unsafe array path')
        p=self.root/rel
        require(p.is_file() and not p.is_symlink() and self.root in p.resolve().parents,'array missing or escapes run')
        return p
    def load(self,name):
        row=self.records[name];p=self.path(row)
        a=np.load(p,allow_pickle=False,mmap_mode='r')
        require(str(a.dtype)==row['dtype'] and list(a.shape)==row['shape'],'array dtype/shape differs: '+name)
        require(a.dtype in (np.dtype('float32'),np.dtype('bool')) and a.flags.c_contiguous,'unexpected dtype/layout')
        require(a.nbytes<=MAX_ARRAY_BYTES,'array exceeds bounded audit size')
        return a
    def verify_all(self):
        total=0
        for name,row in self.records.items():
            p=self.path(row);before=stat_id(p)
            require(sha_file(p)==row['file_sha256'],'file SHA differs: '+name)
            a=self.load(name);total+=a.nbytes
            require(total<=MAX_TOTAL_ARRAY_BYTES,'total arrays exceed bounded audit size')
            require(np.isfinite(a).all(),'nonfinite actual array: '+name)
            require(array_sha(a)==row['content_sha256'],'content SHA differs: '+name)
            require(stat_id(p)==before,'array changed during verification')
            self.verified[name]={'path':row['path'],'file_sha256':row['file_sha256'],
                'content_sha256':row['content_sha256'],'shape':row['shape'],'dtype':row['dtype'],'bytes':a.nbytes}
            del a
        return {'arrays':len(self.verified),'payload_bytes':total,'all_finite':True,'all_file_and_content_SHA_exact':True}

def tile_mae(pred,target):
    require(pred.shape==target.shape and pred.ndim==4 and pred.shape[1]==3,'RGB shape mismatch')
    b,_,h,w=pred.shape;require(h%16==0 and w%16==0,'tile geometry')
    error=np.abs(pred.astype(np.float64)-target.astype(np.float64)).mean(axis=1)
    return error.reshape(b,h//16,16,w//16,16).mean(axis=(2,4))

def check_utility(all_rgb,removed0,removed1,target,availability,saved_raw,saved_normalized,saved_valid):
    b=all_rgb.shape[0]
    require(availability.shape==(b,2) and np.isin(availability,(0,1)).all(),'availability shape/range')
    valid=availability.astype(bool)
    require(saved_valid.dtype==np.bool_ and np.array_equal(saved_valid,valid),'utility valid mask differs')
    base=tile_mae(all_rgb,target)
    raw=np.stack([tile_mae(removed0,target)-base,tile_mae(removed1,target)-base],axis=1)
    raw=np.where(valid[:,:,None,None],raw,0.)
    normalized=np.clip(raw/(.25/255.),-4.,4.)
    require(saved_raw.shape==raw.shape and saved_normalized.shape==raw.shape,'utility map shape')
    require(np.isfinite(saved_raw).all() and np.isfinite(saved_normalized).all(),'nonfinite utility')
    missing=~np.broadcast_to(valid[:,:,None,None],raw.shape)
    raw_error=float(np.max(np.abs(saved_raw.astype(np.float64)-raw)))
    normalized_error=float(np.max(np.abs(saved_normalized.astype(np.float64)-normalized)))
    diagnostics={'raw_max_abs_error':raw_error,'raw_atol':RAW_ATOL,'normalized_max_abs_error':normalized_error,
        'normalized_atol':NORMALIZED_ATOL,'rtol':0,'valid_exact':True,'missing_history_exact_zero':True,
        'valid_tiles':int(np.broadcast_to(valid[:,:,None,None],raw.shape).sum()),
        'reference':'NumPy float64 mean(abs(RGB-GT)) over color and 16x16 tile; without minus all'}
    if not (np.all(saved_raw[missing]==0) and np.all(saved_normalized[missing]==0)):
        diagnostics['missing_history_exact_zero']=False
        raise NumericAuditError('missing-history utility must be exactly zero',diagnostics)
    if raw_error>RAW_ATOL:raise NumericAuditError('raw utility cross-implementation tolerance exceeded',diagnostics)
    if normalized_error>NORMALIZED_ATOL:raise NumericAuditError('normalized utility cross-implementation tolerance exceeded',diagnostics)
    return diagnostics

def state_hash(arrays,mapping,prefix):
    h=hashlib.sha256()
    for name,row in sorted(mapping.items()):
        require(row==arrays.records[prefix+'/'+name],'state array record differs')
        a=arrays.load(prefix+'/'+name)
        require(a.dtype==np.float32,'non-FP32 model state')
        h.update(name.encode());h.update(f'(torch.float32, {tuple(a.shape)})'.encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def check_gradients(arrays,report,prefix,expected_names):
    inv=report[prefix];require(inv['missing']==[] and set(inv['parameters'])==set(expected_names),'gradient names/missing differ')
    require(inv['present']==len(expected_names),'gradient count differs')
    sum_sq=0.;nonzero={}
    for name,row in inv['parameters'].items():
        record=arrays.records[prefix+'/'+name]
        require(all(row[k]==v for k,v in record.items()),'gradient array record mismatch')
        g=arrays.load(prefix+'/'+name);norm=float(np.linalg.norm(g.astype(np.float64).ravel()))
        count=int(np.count_nonzero(g));nonzero[name]=count;sum_sq+=norm*norm
        require(count==row['nonzero'] and np.isclose(norm,row['norm'],rtol=1e-6,atol=1e-10),'gradient statistics mismatch')
    norm=sum_sq**.5
    require(norm>0 and np.isfinite(norm) and np.isclose(norm,inv['aggregate_norm'],rtol=1e-6,atol=1e-10),'aggregate gradient mismatch/zero')
    return {'parameters':len(expected_names),'aggregate_norm':norm,'nonzero_elements':sum(nonzero.values()),'nonzero_by_name':nonzero}

def audit_success(report,arrays):
    require(report['schema']=='rphi.history_utility_one_step_report.v1' and report['status']=='completed_one_step_engineering','wrong producer schema/status')
    require(report['actual']==COUNTS and report['completed_restorer_calls']==7 and report['completed_backward_calls']==2
        and report['completed_teacher_optimizer_steps']==1,'producer-reported operation budget mismatch')
    input_names={'teacher_x':'input/teacher/observed','teacher_w':'input/teacher/weights',
        'teacher_target':'supervision/teacher/target','student_x':'input/student/observed',
        'student_a':'input/student/availability','student_target':'supervision/student/target'}
    require({k:arrays.records[v]['content_sha256'] for k,v in input_names.items()}==report['input_tensor_sha256_before'],
        'saved input bytes do not match producer initial input fingerprints')
    for role in ('teacher','student'):
        x=arrays.load('input/'+role+'/observed');a=arrays.load('input/'+role+'/availability');y=arrays.load('supervision/'+role+'/target')
        require(x.shape==(16,3,3,96,96) and y.shape==(16,3,96,96) and a.shape==(16,2),'input geometry differs')
        require(np.isin(a,(0,1)).all() and np.all((x>=0)&(x<=1)) and np.all((y>=0)&(y<=1)),'input ranges differ')
    tw=arrays.load('input/teacher/weights');ta=arrays.load('input/teacher/availability')
    require(tw.shape==(16,2,96,96) and np.all((tw>=0)&(tw<=1)) and np.all(tw*(1-ta[:,:,None,None])==0),'teacher mask revives missing history')
    before=report['teacher_before'];after=report['teacher_after'];student=report['student_initial']
    require(set(before)==set(after),'teacher state key mismatch')
    fingerprints={'teacher_before':state_hash(arrays,before,'teacher_before'),
        'teacher_after':state_hash(arrays,after,'teacher_after'),'student_initial':state_hash(arrays,student,'student_initial')}
    require(fingerprints['teacher_before']==report['teacher_before_sha256'] and fingerprints['teacher_after']==report['teacher_after_sha256'],'teacher state SHA mismatch')
    require(set(report['student_initial_sha256'])=={'generic_gate','utility_gate','shuffled_utility_gate'}
        and set(report['student_initial_sha256'].values())=={fingerprints['student_initial']},'student initial reported SHA mismatch')
    changed=[];delta=0.
    for name in before:
        x=arrays.load('teacher_before/'+name);y=arrays.load('teacher_after/'+name)
        if not np.array_equal(x,y):changed.append(name)
        delta=max(delta,float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64)))))
    require(bool(changed),'no actual teacher parameter change')
    restorer={k.removeprefix('restorer.'):v for k,v in student.items() if k.startswith('restorer.')}
    require(set(restorer)==set(after),'student inherited state keys differ')
    for name in after:
        require(np.array_equal(arrays.load('student_initial/restorer.'+name),arrays.load('teacher_after/'+name)),
            'student initial restorer differs from post-step teacher: '+name)
    gradients={name:check_gradients(arrays,report,name,before if name.startswith('teacher') else student)
        for name in ('teacher_grad_raw','teacher_grad_clipped','student_grad')}
    for prefix in ('gate.net.2.','restorer.backbone.tail.'):
        require(sum(v for k,v in gradients['student_grad']['nonzero_by_name'].items() if k.startswith(prefix))>0,'required student tail gradient is zero')
    a=arrays.load('input/student/availability');target=arrays.load('supervision/student/target')
    require(target.shape==(16,3,96,96) and arrays.load('input/student/observed').shape==(16,3,3,96,96),'actual probe geometry differs')
    utilities={}
    for group,normal in [('teacher_labels','teacher_labels/all'),('student_own','student_normal/rgb')]:
        utilities[group]=check_utility(arrays.load(normal),arrays.load(group+'/without0'),arrays.load(group+'/without1'),target,a,
            arrays.load(group+'/raw_utility'),arrays.load(group+'/normalized'),arrays.load(group+'/valid'))
    z=arrays.load('student_normal/z');g=arrays.load('student_normal/g');w=arrays.load('student_normal/weights')
    require(z.shape==(16,2,6,6) and g.shape==z.shape and w.shape==(16,2,96,96),'gate shapes differ')
    require(np.all(z==0) and np.all(g==.5),'initial gate logits/action differs')
    require(np.array_equal(w,np.broadcast_to(a[:,:,None,None]*np.float32(.5),w.shape)),'initial effective weights differ')
    reported_freeze={'baseline_unchanged':report['baseline_state_before']==report['baseline_state_after'],
        'teacher_final_equals_saved_post_step':report['teacher_final_sha256']==fingerprints['teacher_after'],
        'students_final_equals_saved_initial':report['student_final_sha256']==report['student_initial_sha256'],
        'teacher_gradients_cleared':report['teacher_gradients_cleared_after_update'] is True,
        'teacher_labels_unchanged':report['teacher_labels_state_unchanged'] is True,
        'hooks_removed':report['hooks_removed'] is True,'no_NPU_import':report['torch_npu_imported_after'] is False,
        'inputs_unchanged':report['input_tensor_sha256_before']==report['input_tensor_sha256_after'],
        'clone_hooks_absent':report['cloned_restorer_hooks']=={k:{'pre':0,'post':0} for k in report['student_initial_sha256']}}
    require(all(reported_freeze.values()),'producer-reported freeze/restoration failure')
    return {'state_fingerprints_recomputed':fingerprints,'teacher_changed_names':changed,'teacher_max_abs_parameter_delta':delta,
        'student_initial_restorer_exactly_equals_saved_teacher_after':True,'gradients':gradients,'utilities':utilities,
        'initial_gate':{'logits_exact_zero':True,'g_exact_half':True,'weights_exact_half_availability':True},
        'producer_reported_counts':report['actual'],'producer_reported_freeze':reported_freeze,
        'scope_limit':'Final live baseline/teacher/three-student states and operation counts are producer attestations; no saved final live tensors were independently loaded. No model/optimizer replay; no quality acceptance.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--expected-report-sha256',required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result={'schema':SCHEMA,'audit_complete':False,'producer_complete':None,'quality_verdict':'requires_review',
        'source_sha256':sha_file(__file__),'input_report':str(args.report.resolve()),'expected_report_sha256':args.expected_report_sha256,
        'fixed_tolerances':{'raw_utility_atol':RAW_ATOL,'normalized_utility_atol':NORMALIZED_ATOL,'rtol':0},'model_calls':0,'optimizer_replays':0,'NPU_calls':0}
    try:
        require(sha_file(args.report)==args.expected_report_sha256,'report SHA differs')
        report=json.loads(args.report.read_text());result['producer_complete']=report.get('complete')
        require(report['schema']=='rphi.history_utility_one_step_report.v1','wrong producer schema')
        pins=report['execution_contract']['implementation_sha256']
        require([v for k,v in pins.items() if Path(k).name=='check_history_utility_one_step.py']==[PRODUCER_SHA],'producer source pin differs')
        arrays=Arrays(args.report.parent,report.get('arrays',{}));result['array_validation']=arrays.verify_all()
        result['arrays_verified']=arrays.verified
        if report['complete'] is True:
            result['independent_checks']=audit_success(report,arrays);result['status']='passed_bounded_engineering_array_audit'
        else:
            require(report.get('status')=='failed','nonterminal report');result['status']='producer_failure_preserved';result['producer_failure']=report.get('failure');result['producer_reported_actual']=report.get('actual')
        require(sha_file(args.report)==args.expected_report_sha256,'report changed during audit')
        result['audit_complete']=True
    except BaseException as error:
        result.update(status='audit_failed',error=repr(error))
        if isinstance(error,NumericAuditError):result['numeric_failure_diagnostics']=error.diagnostics
    result['elapsed_seconds']=time.monotonic()-start
    with (args.out/'audit.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result.get(k) for k in ('audit_complete','producer_complete','status','error','elapsed_seconds')}))
    return 0 if result['audit_complete'] else 2
if __name__=='__main__':raise SystemExit(main())
