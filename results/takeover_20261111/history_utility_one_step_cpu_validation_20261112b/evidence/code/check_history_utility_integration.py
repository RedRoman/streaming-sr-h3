"""Bounded CPU-only inheritance probe for an authenticated real Rphi checkpoint.

Eight restoration forwards, one gate; zero backward/optimizer/data training.
Synthetic input or a separately pinned strict mixed-data receipt is explicit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

CHECKPOINT_SHA = '61ff3a2f696033791f2168410a15f02fc2f8b25274d8fb0c62a0e8cbf7a3f617'
EXPECTED_CONFIG = {'variant':'blind','k':3,'ch':64,'blocks':8,'condition_dim':6,'observation_encoder':'legacy'}
SHAPE = [1,3,3,96,96]
OUTPUT_POLICY = {'inheritance_atol':1e-6,'inheritance_rtol':1e-5,'direction':'candidate_vs_baseline',
    'bitexact_diagnostic':True,'same_operator_absence_bitexact_required':True}


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def save_json(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def verify_contract(path):
    c=json.loads(Path(path).read_text())
    require(c['schema']=='rphi.history_utility_integration_execution.v1','wrong integration schema')
    require(c['checkpoint']['sha256']==CHECKPOINT_SHA and c['model_config']==EXPECTED_CONFIG,'wrong fixed checkpoint/config')
    require(c['budget']=={'baseline_forwards':2,'mask_aware_forwards':6,'student_forwards':1,'gate_forwards':1,
        'backward':0,'optimizer_steps':0,'NPU_calls':0},'wrong finite forward-only budget')
    require(c['cpu_threads'] in (1,2) and c['hard_timeout_seconds']==180 and c['rss_limit_bytes']==2*(1<<30),'wrong CPU resource bounds')
    require(c['shape']==SHAPE and c['output_policy']==OUTPUT_POLICY,'geometry or predeclared output policy changed')
    require(c['input']['kind'] in ('synthetic','mixed_temporal'),'unknown input kind')
    require(c['implementation_sha256'] and all(sha_file(p)==s for p,s in c['implementation_sha256'].items()),'implementation/source pin mismatch')
    require(c['implementation_sha256'].get(str(Path(__file__).resolve()))==sha_file(__file__),'entry itself is not pinned')
    return c


def warmstart_evidence(baseline,expanded,torch):
    old,new=baseline.state_dict(),expanded.state_dict()
    require(old.keys()==new.keys(),'warmstart state keys changed')
    rows={}
    for name,value in old.items():
        inherited=new[name][:,:9] if name=='backbone.head.weight' else new[name]
        equal=torch.equal(value,inherited)
        require(equal and value.dtype==inherited.dtype==torch.float32,'warmstart tensor changed: '+name)
        require(value.data_ptr()!=inherited.data_ptr(),'warmstart aliased old state')
        rows[name]={'shape':list(value.shape),'sha256':hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest(),'exact':equal}
    added=new['backbone.head.weight'][:,9:]
    require(tuple(added.shape)==(baseline.model_config['ch'],2,3,3) and torch.count_nonzero(added)==0,'added head channels not zero')
    return {'inherited_tensors':rows,'added_kernel_shape':list(added.shape),'added_parameters':added.numel(),
        'added_sha256':hashlib.sha256(added.contiguous().numpy().tobytes()).hexdigest(),'added_all_zero':True}


def state_sha(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous();h.update(name.encode());h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()


def comparison(a,b,torch):
    delta=(a-b).double()
    return {'torch_equal':bool(torch.equal(a,b)),'byte_equal':hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest()==hashlib.sha256(b.contiguous().numpy().tobytes()).hexdigest(),
        'max_abs':float(delta.abs().max()),'engineering_close':bool(torch.allclose(b,a,atol=OUTPUT_POLICY['inheritance_atol'],rtol=OUTPUT_POLICY['inheritance_rtol']))}



def validate_comparisons(comparisons):
    for name in ('baseline_all_one','repeated_baseline_all_one'):
        require(comparisons[name]['engineering_close'],'predeclared inheritance engineering tolerance failed: '+name)
    for name in ('missing_content_invariance','repeated_mask_identity'):
        require(comparisons[name]['torch_equal'] and comparisons[name]['byte_equal'],'same-operator absence bitexact failed: '+name)

def execute(contract,out,report):
    require(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','CPU probe requires backend autoload0')
    import numpy as np
    import torch
    from rphi_model import RphiModel
    from rphi_history_utility import MaskAwareRestorer,HistoryUtilityStudent
    torch.set_num_threads(contract['cpu_threads'])
    require('torch_npu' not in sys.modules,'unexpected NPU extension import')
    report['environment']={'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'threads':torch.get_num_threads(),'torch_npu_imported':False}
    checkpoint=Path(contract['checkpoint']['path']);require(sha_file(checkpoint)==CHECKPOINT_SHA,'checkpoint full bytes SHA mismatch')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(sha_file(checkpoint)==CHECKPOINT_SHA,'checkpoint changed during load')
    require(payload['schema']=='rphi.checkpoint.v1' and payload['step']==3000 and payload['model_config']==EXPECTED_CONFIG,'actual checkpoint identity/config differs')
    for original,digest in payload['implementation_sha256'].items():
        bound=Path(__file__).parent/Path(original).name
        require(sha_file(bound)==digest,'actual inherited checkpoint implementation differs: '+original)
    baseline=RphiModel(**payload['model_config']).eval().requires_grad_(False)
    baseline.load_state_dict(payload['state_dict'],strict=True)
    mask=MaskAwareRestorer.from_rphi(payload['model_config'],payload['state_dict']).eval().requires_grad_(False)
    student=HistoryUtilityStudent(mask,20261111).eval().requires_grad_(False)
    report['checkpoint']={'path':str(checkpoint),'sha256':CHECKPOINT_SHA,'bytes':checkpoint.stat().st_size,'step':payload['step'],
        'model_config':payload['model_config'],'implementation_sha256':payload['implementation_sha256'],'training_contract':payload['training_contract']}
    report['warmstart']=warmstart_evidence(baseline,mask,torch)
    models={'baseline':baseline,'mask_aware':mask,'student':student}
    report['state_before']={k:state_sha(v) for k,v in models.items()}
    require(all(p.device.type=='cpu' and p.dtype==torch.float32 for m in models.values() for p in m.parameters()),'CPU FP32 model contract')
    report['parameter_counts']={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()}
    if contract['input']['kind']=='synthetic':
        generator=torch.Generator(device='cpu').manual_seed(contract['input']['seed'])
        observed=torch.rand(SHAPE,generator=generator,dtype=torch.float32)
        report['actual_data']={'status':'pending','reason':'this contract selects synthetic input; no actual-data integration claimed'}
    else:
        from history_utility_data import HistoryUtilityDataset
        spec=contract['input'];data=HistoryUtilityDataset(spec['receipts'],plan=spec['plan'],train_manifest=spec['train_manifest'],
            dev_manifest=spec['dev_manifest'],split='train',scope='engineering',verify_pixels=True,cache_bytes=512<<20)
        row=data._row(spec['source_id'],spec['profile_id'],spec['frame_index'],spec['crop_tlhw'],[1,1],None,0)
        batch=data.batch([row]);require(set(batch['model_inputs'])=={'observed','availability'},'oracle input keys')
        require(np.array_equal(batch['model_inputs']['availability'],np.ones((1,2),np.float32)),'integration requires two real predecessors')
        observed=torch.from_numpy(batch['model_inputs']['observed'])
        report['actual_data']={'status':'strict_pixels_verified','dataset_binding':data.binding,'dataset_sha256':data.dataset_sha256,
            'row':batch['metadata'][0],'target_sha256':hashlib.sha256(batch['target'].tobytes()).hexdigest(),
            'target_used_as_model_input':False,'target_usage':'hash only; no target loss in this forward-only probe'}
        del data,batch
    require(list(observed.shape)==SHAPE and observed.dtype==torch.float32,'actual crop geometry/dtype')
    weights=torch.ones(1,2,96,96);absent=torch.zeros_like(weights);availability=torch.ones(1,2)
    repeated=observed[:,-1:].expand(-1,3,-1,-1,-1).clone()
    changed=observed.clone();changed[:,:2]=float('nan')
    (out/'arrays').mkdir();report['arrays']={}
    def save(name,value):
        a=value.detach().cpu().contiguous().numpy();path=out/'arrays'/(name+'.npy');np.save(path,a,allow_pickle=False)
        report['arrays'][name]={'path':str(path.relative_to(out)),'shape':list(a.shape),'dtype':str(a.dtype),
            'file_sha256':sha_file(path),'pixel_sha256':hashlib.sha256(a.tobytes()).hexdigest()}
    save('observed',observed);save('repeated_current',repeated)
    actual={'baseline_forwards':0,'mask_aware_forwards':0,'student_forwards':0,'gate_forwards':0,'backward':0,'optimizer_steps':0,'NPU_calls':0}
    report['actual']=actual;hooks=[]
    def counter(key):
        def hook(*args):
            actual[key]+=1;require(actual[key]<=contract['budget'][key],'forward budget exceeded')
        return hook
    for m,key in ((baseline,'baseline_forwards'),(mask,'mask_aware_forwards'),(student.restorer,'mask_aware_forwards'),(student,'student_forwards'),(student.gate,'gate_forwards')):
        hooks.append(m.register_forward_pre_hook(counter(key)))
    try:
        with torch.no_grad():
            outputs={}
            def run(name,model,*args):
                result=model(*args);rgb=result['rgb'];require(torch.isfinite(rgb).all() and not rgb.requires_grad,'invalid CPU output')
                save(name,rgb);outputs[name]=rgb;save_json(out/'partial.json',report);return rgb
            a=run('baseline',baseline,observed);b=run('expanded_all_one',mask,observed,weights)
            absent_a=run('current_only',mask,observed,absent);absent_b=run('missing_content_changed',mask,changed,absent)
            ident_a=run('repeated_baseline',baseline,repeated);ident_b=run('repeated_expanded_one',mask,repeated,weights)
            ident_zero=run('repeated_expanded_zero',mask,repeated,absent)
            run('student_initial_half_gate',student,observed,availability)
            report['comparisons']={'baseline_all_one':comparison(a,b,torch),'missing_content_invariance':comparison(absent_a,absent_b,torch),
                'repeated_baseline_all_one':comparison(ident_a,ident_b,torch),'repeated_mask_identity':comparison(ident_b,ident_zero,torch)}
            report['input_control_semantics']='Repeated-current is a history-duplication control, not clean restoration quality or a ground-truth model input.'
    finally:
        for h in hooks:h.remove()
        report['state_after']={k:state_sha(v) for k,v in models.items()}
        report['checkpoint_sha256_after']=sha_file(checkpoint)
        report['all_frozen_unchanged']=report['state_before']==report['state_after']
        report['torch_npu_imported_after']='torch_npu' in sys.modules
        save_json(out/'partial.json',report)
    require(actual==contract['budget'],'operation count incomplete')
    require(report['all_frozen_unchanged'] and report['checkpoint_sha256_after']==CHECKPOINT_SHA and not report['torch_npu_imported_after'],'frozen CPU scope failed')
    validate_comparisons(report['comparisons'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();contract=verify_contract(args.execution_contract);out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'schema':'rphi.history_utility_integration.v1','complete':False,'execution_contract':contract,
        'execution_contract_sha256':sha_file(args.execution_contract),'scope':'bounded CPU forward inheritance only; no teacher/student training'}
    started=time.monotonic();save_json(out/'partial.json',report)
    try:
        execute(contract,out,report);report['complete']=True
    except BaseException as error:
        report['failure']=repr(error)
    report['elapsed_seconds']=time.monotonic()-started;save_json(out/('report.json' if report['complete'] else 'failure.json'),report)
    print(json.dumps({'complete':report['complete'],'failure':report.get('failure'),'actual':report.get('actual'),
        'comparisons':report.get('comparisons'),'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':raise SystemExit(main())
