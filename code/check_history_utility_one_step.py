"""One bounded actual-data CPU engineering step; never a research trainer.

Default inspection loads verified pixels but no model. Execution requires an
independently frozen contract and root admission. Existing P1 code is unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SCHEMA='rphi.history_utility_one_step_execution.v1'
CHECKPOINT_SHA='61ff3a2f696033791f2168410a15f02fc2f8b25274d8fb0c62a0e8cbf7a3f617'
CONFIG={'variant':'blind','k':3,'ch':64,'blocks':8,'condition_dim':6,'observation_encoder':'legacy'}
COUNTS={'teacher_training':1,'teacher_labels':3,'student_normal':1,'student_own_interventions':2,
        'restorer_total':7,'gate':1,'backward':2,'teacher_optimizer_steps':1,'student_optimizer_steps':0,'NPU_calls':0}
RESOURCE={'CPU_threads':2,'wall_seconds':300,'termination_grace_seconds':10,'owned_process_group_RSS_limit_bytes':4*(1<<30)}
FILES={'materialization_complete','train_receipt_index','dev_receipt_index','teacher_schedule','student_schedule','plan','train_manifest','dev_manifest'}


def require(value,message):
    if not value:raise ValueError(message)


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    p=Path(path);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def read(path):return json.loads(Path(path).read_text())

def execution_payload_sha256(c):
    payload={k:v for k,v in c.items() if k!='root_admission'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_contract(path):
    c=read(path)
    require(c['schema']==SCHEMA and c['scope']=='engineering_one_step_not_research','wrong schema/scope')
    require(c['counts']==COUNTS and c['resource']==RESOURCE,'budget changed')
    require(c['checkpoint']['sha256']==CHECKPOINT_SHA and c['model_config']==CONFIG,'fixed checkpoint/config changed')
    require(sha_file(c['checkpoint']['path'])==CHECKPOINT_SHA,'checkpoint bytes changed')
    for name in ('source_contract','root_admission'):
        require(sha_file(c[name]['path'])==c[name]['sha256'],'binding differs: '+name)
    admission=read(c['root_admission']['path'])
    require(admission['approved'] is True and admission['scope']=='engineering_one_step_not_research'
            and admission['source_contract_sha256']==c['source_contract']['sha256']
            and admission['entry_sha256']==sha_file(__file__)
            and admission['execution_payload_sha256']==execution_payload_sha256(c),'explicit root admission missing or changed')
    proposal=read(c['source_contract']['path'])
    require(proposal['schema']=='rphi.history_utility_one_step_engineering_contract.v1'
            and proposal['budget']['restorer_batch_forwards']==7 and proposal['budget']['backward_calls']==2,'source engineering contract changed')
    require(set(c['files'])==FILES,'missing data bindings')
    for row in c['files'].values():require(sha_file(row['path'])==row['sha256'],'input file changed: '+row['path'])
    require(c['implementation_sha256'] and all(sha_file(p)==s for p,s in c['implementation_sha256'].items()),'source pin changed')
    for name in ('check_history_utility_one_step.py','check_history_utility_integration.py','rphi_history_utility.py','history_utility_data.py','rphi_model.py','train_rphi.py','rphi_data.py'):
        p=str(Path(__file__).resolve().parent/name)
        require(c['implementation_sha256'].get(p)==sha_file(p),'required local source not pinned: '+name)
    # Link the frozen numerical sources to the reviewed engineering contract.
    for relative,digest in proposal['source_sha256'].items():
        if relative.startswith('code/'):
            require(c['implementation_sha256'].get(str(Path(__file__).resolve().parent/Path(relative).name))==digest,'reviewed numerical source changed')
    return c



def validate_materialization(complete):
    require(complete['complete'] is True and complete['status']=='completed_engineering_only','actual 2+2 materialization incomplete')
    schema=complete['schema']
    require(schema in ('openh3.history_utility_materialization.v1','openh3.history_utility_materialization_recovery.v1'),'unrecognized materializer schema')
    require(complete['model_calls']==0 and complete['NPU_calls']==0,'unexpected materializer model budget')
    if schema=='openh3.history_utility_materialization_recovery.v1':
        require(complete['original_data_contract_file_sha256']=='13b6e03f8b5387a101997e1545214a03b43d281454e1d1848a589a57fd9c47ad','recovery original data contract changed')
        expected={'reused_sources':1,'newly_materialized_sources':3,'new_target_decodes':3,'new_observation_codec_passes':18,
            'inherited_completed_target_decodes':1,'inherited_completed_codec_passes':6,'total_verified_target_NPZ':4,'total_verified_observation_NPZ':12}
        require(all(complete[k]==v for k,v in expected.items()),'explicit recovery accounting differs')
        require(len(complete['phases_completed'])==5 and all(p['complete'] is True for p in complete['phases_completed'])
                and len(set(complete['independent_worker_PIDs']))==5,'recovery phase completion differs')
        require(complete['scientific_continuity_approved'] is False and complete['natural_cut_status']=='unknown_engineering_only','recovery continuity overstated')
    else:
        require(complete['target_decodes']==4 and complete['observation_codec_passes']==24
                and complete['target_NPZ']==4 and complete['observation_NPZ']==12,'original complete accounting differs')
    return schema

def load_inputs(c):
    import numpy as np
    from history_utility_data import HistoryUtilityDataset
    files={k:row['path'] for k,row in c['files'].items()}
    complete=read(files['materialization_complete'])
    materialization_schema=validate_materialization(complete)
    require(complete['receipt_indices'].keys()=={'train','dev'},'source roles missing')
    indices={s:read(files[s+'_receipt_index']) for s in ('train','dev')}
    for split,index in indices.items():
        require(index['schema']=='openh3.history_utility_receipt_index.v1' and index['receipts']==complete['receipt_indices'][split],'materialized receipt index mismatch')
        require(len(index['receipts'])==2,'expected two sources per split')
    data=HistoryUtilityDataset(indices['train']['receipts'],plan=files['plan'],train_manifest=files['train_manifest'],
        dev_manifest=files['dev_manifest'],split='train',scope='engineering',verify_pixels=True,cache_bytes=512<<20)
    require(data.source_ids==['2BdYaGuVAKo','4EKWTiWjPTQ'] and not data.research_ready,'wrong engineering train sources')
    expected_dev={'6UZVovkR3IU','6rJrrfRO3cA'};actual_dev=set()
    for entry in indices['dev']['receipts']:
        require(sha_file(entry['path'])==entry['sha256'],'dev receipt changed')
        receipt=read(entry['path']);sid=receipt['source']['source_id'];actual_dev.add(sid)
        require(receipt['source']==data.registry[sid] and receipt['source']['split']=='dev','dev role binding changed')
    require(actual_dev==expected_dev,'wrong dev source set')
    batches={};schedules={}
    for role in ('teacher','student'):
        schedule=read(files[role+'_schedule'])
        require(schedule['role']==role and schedule['steps']==1 and schedule['batch_size']==16
                and schedule['scope']=='engineering' and not schedule['research_ready'] and len(schedule['rows'])==16,'one-step schedule required')
        data.validate_schedule(schedule);batch=data.batch(schedule['rows']);validate_batch(batch,role,np)
        batches[role]=batch;schedules[role]=schedule
    evidence={'materialization_schema':materialization_schema,'materialization_file_sha256':sha_file(files['materialization_complete']),
        'materialization_accounting_scope':'accepted complete artifact inventory; prior failed attempts are not rewritten as completed work',
        'dataset_binding':data.binding,'dataset_sha256':data.dataset_sha256,'schedules':schedules,
        'dev_validation_scope':'source/receipt/registry identity only here; no dev pixels/model calls; full materialization acceptance is a required external gate',
        'metadata':{role:batch['metadata'] for role,batch in batches.items()},'model_inputs':['observed','availability/teacher_weights'],
        'target_usage':'loss/labels and declared identity/nearclean control construction only; never an extra forward argument'}
    return batches,evidence


def validate_batch(batch,role,np):
    require(set(batch['model_inputs'])=={'observed','availability'},'oracle model input keys')
    x,a=batch['model_inputs']['observed'],batch['model_inputs']['availability'];y=batch['target']
    for name,v,shape in [('observed',x,(16,3,3,96,96)),('availability',a,(16,2)),('target',y,(16,3,96,96))]:
        require(v.dtype==np.float32 and v.shape==shape and np.isfinite(v).all(),'invalid '+name)
    require(((x>=0)&(x<=1)).all() and ((y>=0)&(y<=1)).all() and ((a==0)|(a==1)).all(),'input range/availability')
    require(len(batch['metadata'])==16 and all(r['split']=='train' for r in batch['metadata']),'non-train row')
    require(sorted(batch['control'].tolist())==[0]*12+[1]*2+[2]*2,'wrong 12+2+2 controls')
    require(all(len({r['source_id'] for r in batch['metadata'] if r['profile_id']==family})==2 for family in set(r['profile_id'] for r in batch['metadata'])),'source cycling lost')
    if role=='teacher':
        w=batch['teacher_weights'];require(w.dtype==np.float32 and w.shape==(16,2,96,96)
            and np.isfinite(w).all() and ((w>=0)&(w<=1)).all(),'teacher mask invalid')
        require(np.all(w*(1-a[:,:,None,None])==0),'teacher weights revive unavailable history')
    else:require('teacher_weights' not in batch,'student cannot receive teacher masks')


class Evidence:
    def __init__(self,out,report):self.out=Path(out);self.report=report;self.report.setdefault('arrays',{})
    def flush(self):write(self.out/'partial.json',self.report)
    def array(self,name,tensor):
        import numpy as np
        a=tensor.detach().cpu().contiguous().numpy() if hasattr(tensor,'detach') else np.ascontiguousarray(tensor)
        require(np.isfinite(a).all(),'nonfinite evidence: '+name)
        p=self.out/'arrays'/(name+'.npy');p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:np.save(f,a,allow_pickle=False)
        row={'path':str(p.relative_to(self.out)),'file_sha256':sha_file(p),'content_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'shape':list(a.shape),'dtype':str(a.dtype)}
        self.report['arrays'][name]=row;return row
    def state(self,name,model):
        return {key:self.array(name+'/'+key,value) for key,value in model.state_dict().items()}


def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def grad_inventory(model,evidence,prefix):
    import torch
    rows={};missing=[];sum_sq=0.
    for name,p in model.named_parameters():
        if p.grad is None:missing.append(name);continue
        g=p.grad.detach();require(torch.isfinite(g).all(),'nonfinite gradient '+name)
        row=evidence.array(prefix+'/'+name,g);row.update(norm=float(g.double().norm()),nonzero=int(torch.count_nonzero(g)))
        rows[name]=row;sum_sq+=row['norm']**2
    result={'parameters':rows,'missing':missing,'aggregate_norm':sum_sq**.5,'present':len(rows)}
    evidence.report[prefix.replace('/','_')]=result;evidence.flush()
    require(not missing and result['aggregate_norm']>0,'missing/zero aggregate gradients: '+prefix)
    return result


def run_core(baseline,batches,evidence,torch):
    """The exact bounded math path; small fixtures may supply smaller B/H/ch."""
    from rphi_history_utility import MaskAwareRestorer,HistoryUtilityStudent,remove_history,utility_targets,matched_losses
    from train_rphi import fidelity_losses,learning_rate
    from check_history_utility_integration import warmstart_evidence,state_sha
    report=evidence.report;actual={key:0 for key in COUNTS};report['actual']=actual
    report['completed_restorer_calls']=0;report['phase']='model_initialization'
    teacher=MaskAwareRestorer.from_rphi(baseline.model_config,baseline.state_dict()).train().requires_grad_(True)
    report['warmstart']=warmstart_evidence(baseline,teacher,torch)
    report['baseline_state_before']=state_sha(baseline)
    report['teacher_before']=evidence.state('teacher_before',teacher);report['teacher_before_sha256']=state_sha(teacher)
    teachers={k:torch.from_numpy(v) for k,v in batches['teacher']['model_inputs'].items()}
    students_input={k:torch.from_numpy(v) for k,v in batches['student']['model_inputs'].items()}
    y_teacher=torch.from_numpy(batches['teacher']['target']);y_student=torch.from_numpy(batches['student']['target'])
    weights=torch.from_numpy(batches['teacher']['teacher_weights'])
    for role,batch in batches.items():
        for name,value in batch['model_inputs'].items():evidence.array('input/'+role+'/'+name,value)
        evidence.array('supervision/'+role+'/target',batch['target'])
    evidence.array('input/teacher/weights',weights)
    input_before={name:tensor_sha(v) for name,v in {'teacher_x':teachers['observed'],'teacher_w':weights,'teacher_target':y_teacher,
        'student_x':students_input['observed'],'student_a':students_input['availability'],'student_target':y_student}.items()}
    report['input_tensor_sha256_before']=input_before;hooks=[];student_models={};optimizer=None
    def forward_pre(kind):
        def hook(module,args):
            phase=report['phase'];require(phase in ('teacher_training','teacher_labels','student_normal','student_own_interventions'),'unexpected forward phase')
            expected_x=teachers['observed'] if phase=='teacher_training' else students_input['observed']
            require(len(args)==2 and args[0] is expected_x,'unexpected/oracle forward arguments')
            require(args[1].ndim==(2 if kind=='gate' else 4),'forward mask rank differs')
            if kind=='gate':
                require(phase=='student_normal' and args[1] is students_input['availability'],'unexpected gate inputs')
                actual['gate']+=1;require(actual['gate']<=1,'gate budget exceeded')
            else:
                actual[phase]+=1;actual['restorer_total']+=1
                require(actual[phase]<=COUNTS[phase] and actual['restorer_total']<=7,'restorer budget exceeded')
        return hook
    def forward_done(module,args,result):
        require(torch.isfinite(result['rgb']).all(),'nonfinite RGB');report['completed_restorer_calls']+=1
    hooks+=[teacher.register_forward_pre_hook(forward_pre('restorer')),teacher.register_forward_hook(forward_done)]
    def backward(loss):
        require(torch.isfinite(loss) and loss.requires_grad,'invalid backward scalar')
        actual['backward']+=1;require(actual['backward']<=2,'backward budget exceeded');loss.backward()
        report.setdefault('completed_backward_calls',0);report['completed_backward_calls']+=1;evidence.flush()
    def set_phase(name):report['phase']=name;evidence.flush()
    try:
        set_phase('teacher_training')
        output=teacher(teachers['observed'],weights);evidence.array('teacher_training_rgb',output['rgb'])
        loss,parts=fidelity_losses(output['rgb'],y_teacher,teachers['observed'][:,-1],edge_weight=.1,guard_weight=.5)
        report['teacher_loss']={'total':float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()}}
        lr=learning_rate(1,1000,1e-4,50);require(lr==1e-4*(1/50),'step1 warmup differs')
        optimizer=torch.optim.AdamW(teacher.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,fused=False)
        optimizer.zero_grad(set_to_none=True);backward(loss)
        report['teacher_grad_raw']=grad_inventory(teacher,evidence,'teacher_grad_raw')
        norm=torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.,error_if_nonfinite=True,foreach=False)
        report['teacher_clip_norm_before']=float(norm);report['teacher_grad_clipped']=grad_inventory(teacher,evidence,'teacher_grad_clipped')
        actual['teacher_optimizer_steps']+=1;require(actual['teacher_optimizer_steps']==1,'teacher update budget exceeded')
        optimizer.step();report['completed_teacher_optimizer_steps']=1
        report['optimizer']={'class':'AdamW','actual_lr':lr,'actual_lr_float_hex':lr.hex(),'betas':[.9,.999],'eps':1e-8,'weight_decay':0,'amsgrad':False,'foreach':False,'fused':False,
            'state_steps':{name:float(optimizer.state[p]['step']) for name,p in teacher.named_parameters()}}
        require(all(v==1 for v in report['optimizer']['state_steps'].values()),'optimizer actual state step differs')
        for state in optimizer.state.values():
            for v in state.values():require(torch.isfinite(v).all(),'nonfinite optimizer state')
        report['teacher_after']=evidence.state('teacher_after',teacher);report['teacher_after_sha256']=state_sha(teacher)
        require(report['teacher_after_sha256']!=report['teacher_before_sha256'],'teacher one-step produced no parameter change')
        optimizer.zero_grad(set_to_none=True);teacher.eval().requires_grad_(False)
        del output,loss,parts;optimizer=None
        # deepcopy copies Module hooks. Remove our observer closures before any
        # student clone, then install one observer per actual callable below.
        for hook in hooks:hook.remove()
        hooks.clear()
        require(not teacher._forward_pre_hooks and not teacher._forward_hooks,'teacher hook leaked into clone boundary')
        student_models={name:HistoryUtilityStudent(teacher,20261111) for name in ('generic_gate','utility_gate','shuffled_utility_gate')}
        report['cloned_restorer_hooks']={name:{'pre':len(model.restorer._forward_pre_hooks),'post':len(model.restorer._forward_hooks)} for name,model in student_models.items()}
        require(all(row=={'pre':0,'post':0} for row in report['cloned_restorer_hooks'].values()),'cloned student inherited observer hooks')
        hooks+=[teacher.register_forward_pre_hook(forward_pre('restorer')),teacher.register_forward_hook(forward_done)]
        report['student_initial_sha256']={name:state_sha(model) for name,model in student_models.items()}
        report['student_parameter_counts']={name:sum(p.numel() for p in model.parameters()) for name,model in student_models.items()}
        require(len(set(report['student_initial_sha256'].values()))==1,'student initial bytes differ')
        for name,p in student_models['utility_gate'].named_parameters():
            peers=[dict(model.named_parameters())[name] for model in student_models.values()]
            require(len({q.data_ptr() for q in peers})==3 and all(torch.equal(p,q) and q.requires_grad for q in peers),'student shape/bytes/storage mismatch')
        selected=student_models['utility_gate'];report['student_initial']=evidence.state('student_initial',selected)
        hooks+=[selected.restorer.register_forward_pre_hook(forward_pre('restorer')),selected.restorer.register_forward_hook(forward_done),selected.gate.register_forward_pre_hook(forward_pre('gate'))]
        set_phase('teacher_labels')
        a=students_input['availability'];h,w=students_input['observed'].shape[-2:];normal=a[:,:,None,None].expand(-1,-1,h,w).clone()
        with torch.no_grad():
            base=teacher(students_input['observed'],normal)['rgb'];removed=[];evidence.array('teacher_labels/all',base)
            for slot in (0,1):
                mask=remove_history(normal,slot);require(torch.equal(mask[:,1-slot],normal[:,1-slot]),'other teacher history changed')
                value=teacher(students_input['observed'],mask)['rgb'];evidence.array('teacher_labels/without'+str(slot),value);removed.append(value)
            labels=utility_targets(base,torch.stack(removed,1),y_student,a)
        for name,value in labels.items():require(not value.requires_grad,'teacher labels attached to graph');evidence.array('teacher_labels/'+name,value)
        report['teacher_labels_state_unchanged']=state_sha(teacher)==report['teacher_after_sha256'];require(report['teacher_labels_state_unchanged'],'teacher changed during labels')
        set_phase('student_normal');selected.zero_grad(set_to_none=True)
        output=selected(students_input['observed'],a)
        for name in ('rgb','z','g','weights'):evidence.array('student_normal/'+name,output[name])
        require(torch.equal(output['g'],torch.full_like(output['g'],.5)),'initial gate not one-half')
        loss,parts=matched_losses(output,y_student,students_input['observed'][:,-1],labels['normalized'],labels['valid'],'utility_gate')
        report['student_loss']={'total':float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()}}
        backward(loss);grads=grad_inventory(selected,evidence,'student_grad')
        for prefix in ('gate.net.2.','restorer.backbone.tail.'):
            require(sum(v['nonzero'] for k,v in grads['parameters'].items() if k.startswith(prefix))>0,'expected gate/output gradient is zero: '+prefix)
        report['first_gate_layer_gradient_policy']='zero allowed at zero-initialized final convolution; present/finite required'
        normal_rgb=output['rgb'].detach();original_weights=output['weights'].detach().clone()
        del loss,parts,output
        set_phase('student_own_interventions')
        with torch.no_grad():
            removed=[]
            for slot in (0,1):
                value=selected.without_history(students_input['observed'],original_weights,slot)
                require(torch.equal(value['weights'][:,1-slot],original_weights[:,1-slot]) and torch.count_nonzero(value['weights'][:,slot])==0,'own intervention changed other gate')
                evidence.array('student_own/without'+str(slot),value['rgb']);removed.append(value['rgb'])
            own=utility_targets(normal_rgb,torch.stack(removed,1),y_student,a)
        for name,value in own.items():evidence.array('student_own/'+name,value)
        report['own_utility_scope']='training-batch wiring diagnostic only, not held-out utility transfer or teacher validation'
        require(actual==COUNTS and report['completed_restorer_calls']==7 and report['completed_backward_calls']==2,'actual operation count differs')
        require(all(p.grad is None for arm,model in student_models.items() if arm!='utility_gate' for p in model.parameters()),'inactive arm received gradients')
    finally:
        for hook in hooks:hook.remove()
        report['hooks_removed']=True;report['baseline_state_after']=state_sha(baseline);report['teacher_final_sha256']=state_sha(teacher)
        report['student_final_sha256']={name:state_sha(model) for name,model in student_models.items()}
        report['teacher_gradients_cleared_after_update']=all(p.grad is None for p in teacher.parameters())
        report['input_tensor_sha256_after']={name:tensor_sha(v) for name,v in {'teacher_x':teachers['observed'],'teacher_w':weights,'teacher_target':y_teacher,
            'student_x':students_input['observed'],'student_a':students_input['availability'],'student_target':y_student}.items()}
        report['torch_npu_imported_after']='torch_npu' in sys.modules;evidence.flush()
    require(report['baseline_state_before']==report['baseline_state_after'],'original checkpoint model changed')
    require(report['teacher_final_sha256']==report['teacher_after_sha256'] and report['teacher_gradients_cleared_after_update'],'frozen teacher changed after update')
    require(report['student_final_sha256']==report['student_initial_sha256'],'student parameters were updated')
    require(report['input_tensor_sha256_after']==input_before and not report['torch_npu_imported_after'],'input/backend scope changed')


def execute(c,out,report):
    require(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','backend autoload must be disabled')
    batches,report['data']=load_inputs(c)
    import torch
    from rphi_model import RphiModel
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(20261110)
    require('torch_npu' not in sys.modules,'unexpected NPU backend import')
    report['environment']={'torch':torch.__version__,'python':sys.version,'threads':torch.get_num_threads(),'device':'cpu','dtype':'float32','autocast':False,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    checkpoint=c['checkpoint']['path'];require(sha_file(checkpoint)==CHECKPOINT_SHA,'checkpoint changed before load')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(payload['schema']=='rphi.checkpoint.v1' and payload['step']==3000 and payload['model_config']==CONFIG,'actual checkpoint differs')
    for path,digest in payload['implementation_sha256'].items():require(sha_file(Path(__file__).parent/Path(path).name)==digest,'inherited code changed')
    baseline=RphiModel(**CONFIG).eval().requires_grad_(False);baseline.load_state_dict(payload['state_dict'],strict=True)
    report['checkpoint']={'path':checkpoint,'sha256':CHECKPOINT_SHA,'model_config':CONFIG,'step':3000}
    evidence=Evidence(out,report);evidence.flush()
    try:run_core(baseline,batches,evidence,torch)
    finally:
        report['checkpoint_sha256_after']=sha_file(checkpoint)
        report['implementation_sha256_after']={path:sha_file(path) for path in c['implementation_sha256']};evidence.flush()
    require(report['checkpoint_sha256_after']==CHECKPOINT_SHA and report['implementation_sha256_after']==c['implementation_sha256'],'frozen input changed')
    require(set(report['student_parameter_counts'].values())=={622900},'actual student parameter budget differs')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();c=verify_contract(args.execution_contract);out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'schema':'rphi.history_utility_one_step_report.v1','complete':False,'execution_contract_sha256':sha_file(args.execution_contract),'execution_contract':c,
        'scope':'CPU engineering; not research training or quality acceptance'}
    start=time.monotonic();write(out/'partial.json',report)
    try:
        if args.execute:execute(c,out,report);report['status']='completed_one_step_engineering'
        else:
            batches,report['data']=load_inputs(c);report['status']='inspected_actual_pixels_no_model';report['model_calls']=0
        report['complete']=True
    except BaseException as error:report.update(complete=False,status='failed',failure=repr(error))
    report['elapsed_seconds']=time.monotonic()-start;write(out/('report.json' if report['complete'] else 'failure.json'),report)
    print(json.dumps({'complete':report['complete'],'status':report['status'],'actual':report.get('actual'),'failure':report.get('failure'),'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    return 0 if report['complete'] else 2
if __name__=='__main__':raise SystemExit(main())
