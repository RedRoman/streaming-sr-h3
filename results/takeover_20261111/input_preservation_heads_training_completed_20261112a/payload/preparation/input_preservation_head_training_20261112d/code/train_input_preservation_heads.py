"""Admitted CPU-only matched/shuffled benefit-head training on audited caches.

No restorer, source pixels, development labels, eval pass, resume or automatic
follow-up. Research consumes exactly the fixed 16000 train rows once per arm.
The shuffled arm changes only which authenticated whole target map is selected.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import train_history_utility as T

SCHEMA = 'rphi.input_preservation_head_training_execution.v1'
REPORT_SCHEMA = 'rphi.input_preservation_head_training.v1'
CHECKPOINT_SCHEMA = 'rphi.input_preservation_head_checkpoint.v1'
PROPOSAL_SHA = 'd1dd2afd1b9daddea8340fe1edff7c71ed6215fbf6f3ef038adaa4b01aa262de'
SCREEN_SHA = '144120baa552ae8371a73a03dcd3c48da9204e86a9f7e87186f3697ee0178577'
STUDENT_SHA = 'b5e6d4c6031d79d5b9457c98fe215d49e393e335994559597498176f4e666f08'
DONOR_SHA = 'fdb872a4e1b20c07b67af2c7b40a322f6b968ea7efcac9b2a96239c8898a2d6a'
HEAD_SHA = '7a52656a3119dcaad515dee0ec591bccbb21ee7756ba150b7f59827ad3b8100d'
FILES = {'proposal', 'screen', 'train_cache_manifest', 'train_cache_audit', 'dev_cache_manifest',
    'dev_cache_audit', 'student_schedule', 'donor_manifest', 'donor_map', 'donor_audit', 'metric_readiness'}
STEPS, BATCH, ROWS, SEED = 1000, 16, 16000, 20261111
OPTIMIZER = copy.deepcopy(T.OPTIMIZER)
LOSS = {'type': 'SmoothL1Loss', 'beta': 1., 'reduction': 'mean_all_rows_and_tiles',
        'label': 'detached_normalized_whole_benefit_map', 'auxiliary_losses': []}
RESOURCES = {'wall_seconds': 900, 'CPU_cores': 2, 'RSS_limit_bytes': 4 << 30,
             'maximum_output_bytes': 32 << 20}
CONFIG = {'schema': 'rphi.input_preservation.v1', 'seed': SEED, 'parameters': 1745,
          'slot_reduction': 'arithmetic_mean', 'tile': 16, 'threshold': 0., 'tie': 'preserve_input'}


def need(ok, message):
    if not ok: raise ValueError(message)


def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def binary_tree_equal(a,b,torch):
    """Canonical CPU containers plus exact tensor dtype/shape/bytes, incl -0."""
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and T.tensor_sha(a)==T.tensor_sha(b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return set(a)==set(b) and all(binary_tree_equal(a[k],b[k],torch) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(binary_tree_equal(x,y,torch) for x,y in zip(a,b))
    return a==b


def comparison_identity(c):
    return T.value_sha({k: v for k, v in c.items() if k not in ('arm', 'admission', 'mode', 'backend_smokes')})


def binding(c):
    return {'files': copy.deepcopy(c['files']), 'backend_smokes': copy.deepcopy(c['backend_smokes']),
            'comparison_identity_sha256': comparison_identity(c)}


def check_donor_audit(da,files):
    need(da['schema']=='rphi.input_preservation_donor_independent_actual_audit.v1'
         and da['complete'] is True and da['donor_ready'] is True and da['input_bytes_unchanged'] is True,
         'actual donor audit incomplete')
    for name,key in [('manifest','donor_manifest'),('mapping','donor_map'),('schedule','student_schedule')]:
        need({k:da['inputs'][name][k] for k in ('path','sha256')}==files[key], 'donor audit input binding differs')
    v=da['validation']
    need(v['actual_rows']==v['row_semantic_SHA_verified']==16000 and v['actual_source_count']==19
         and v['actual_strata']==36 and v['actual_both_absent_rows']==1815
         and all(v[k] is True for k in ('nonself_cross_source_AND_group','all_recipients_and_donors_exactly_once',
            'all_strata_frequency_preserved','declared_seeded_rotation_verified','profile_phase_effective_availability_exact')),
         'donor audit coverage/bijection/00 differs')


def check_metric_readiness(m, c):
    need(m['schema'] == 'rphi.input_preservation_metric_readiness.v1' and m['complete'] is True
         and m['frozen'] is True and m['proposal_sha256'] == PROPOSAL_SHA
         and m['screen_sha256'] == SCREEN_SHA, 'metrics must be frozen before training')
    ep = T.bound_path(m['evaluator'], c['root'])
    need(m['implementation_sha256'].get(str(ep)) == m['evaluator']['sha256']
         and all(T.file_sha(p) == h for p, h in m['implementation_sha256'].items()), 'metric source pins differ')
    lp = m['LPIPS']
    need(lp['available'] is True and lp['network'] == 'alex' and lp['version']=='0.1'
         and lp['input']=='clipped_RGB_float32_mapped_minus1_plus1' and lp['downloads'] is False
         and len(lp['weights'])==2 and {w['role'] for w in lp['weights']}=={'alexnet','linear'}, 'complete LPIPS weights required')
    for item in lp['weights']: T.bound_path({k:item[k] for k in ('path','sha256')}, c['root'])
    need(m['runtime']['device'] == 'cpu' and m['runtime']['threads'] == 2
         and m['runtime']['batch_size'] == 16
         and {k:m['runtime']['expected_versions'][k] for k in c['runtime']['expected_versions']} == c['runtime']['expected_versions'],
         'frozen metric/head CPU backend differs')


def inspect_contract(path, *, execute=False, out=None):
    c = T.read(path); root = Path(c['root'])
    need(c['schema'] == SCHEMA and root.is_absolute() and root == root.resolve()
         and c['arm'] in ('matched', 'shuffled') and c['mode'] in ('backend_smoke', 'research'), 'training scope differs')
    need(c['resources'] == RESOURCES and c['optimizer'] == OPTIMIZER and c['loss'] == LOSS
         and c['model_config'] == CONFIG and c['batch_size'] == BATCH and c['steps'] == STEPS
         and c['seed'] == SEED and set(c['files']) == FILES, 'frozen head experiment differs')
    rt = c['runtime']
    need(rt['device'] == 'cpu' and rt['threads'] == 2 and rt['dtype'] == 'float32'
         and rt['autocast'] is False and rt['grad_scaler'] is False
         and set(rt['expected_versions']) == {'python', 'numpy', 'torch', 'torch_npu'}
         and rt['expected_versions']['torch_npu'] is None, 'CPU-only exact FP32 runtime required')
    pins = c['implementation_sha256']
    need(pins.get(str(Path(__file__).resolve())) == T.file_sha(__file__)
         and all(T.file_sha(p) == h for p, h in pins.items()), 'training source pin differs')
    local = Path(__file__).resolve().parent
    for name in (*T.DEPENDENCIES, 'rphi_input_preservation.py', 'prepare_input_preservation_donors.py'):
        need(pins.get(str(local/name)) == T.file_sha(local/name), 'required training helper not pinned: '+name)
    need(T.file_sha(local/'rphi_input_preservation.py') == HEAD_SHA, 'head numerical module changed')
    paths = {k:T.bound_path(v,root) for k,v in c['files'].items()}
    for name,h in [('proposal',PROPOSAL_SHA),('screen',SCREEN_SHA),('student_schedule',STUDENT_SHA),('donor_map',DONOR_SHA)]:
        need(c['files'][name]['sha256'] == h, 'fixed input changed: '+name)
    # Do not parse the large schedule/map or open any NPZ in default inspection.
    docs = {k:T.read(p) for k,p in paths.items() if k not in ('student_schedule','donor_map')}
    for role,n in [('train',16000),('dev',2880)]:
        producer,audit = docs[role+'_cache_manifest'],docs[role+'_cache_audit']
        need(producer['complete'] is True and producer['role'] == role
             and producer['status'] == 'cache_complete_pending_independent_audit_no_training_admission'
             and producer['completed_rows'] == n and producer['head_training_admitted'] is False, 'cache not ready: '+role)
        need(audit['schema'] == 'rphi.input_preservation_cache_actual_audit.v1' and audit['complete'] is True
             and audit['status'] == 'actual_cache_data_features_and_labels_verified_no_training_admission'
             and audit['role'] == role and audit['actual_rows'] == audit['raw_feature_rows_recomputed'] == n
             and audit['producer_manifest'] == c['files'][role+'_cache_manifest']
             and audit['head_training_admitted'] is False and audit['NPU_calls'] == 0,
             'independent actual cache audit required: '+role)
    dm,da = docs['donor_manifest'],docs['donor_audit']
    need(dm['complete'] is True and dm['donor_ready'] is True and dm['both_head_training_arms_blocked'] is False
         and dm['donor_map']['file_sha256'] == DONOR_SHA, 'donor not ready')
    # The metadata audit is source-bound; the actual map is independently checked again below.
    check_donor_audit(da,c['files'])
    check_metric_readiness(docs['metric_readiness'],c)
    if c['mode'] == 'research':
        need(set(c['backend_smokes']) == {'matched','shuffled'}, 'both bounded CPU backend smoke receipts required')
        for arm,b in c['backend_smokes'].items():
            smoke=T.read(T.bound_path(b,root))
            need(smoke['schema'] == REPORT_SCHEMA and smoke['complete'] is True and smoke['mode']=='backend_smoke'
                 and smoke['arm']==arm and smoke['completed_step']==2 and smoke['strict_checkpoint_reload'] is True
                 and smoke['comparison_identity_sha256']==comparison_identity(c), 'backend smoke not compatible')
    else: need(c['backend_smokes']=={}, 'backend smoke cannot inherit a research checkpoint')
    if execute:
        a=T.read(T.bound_path(c['admission'],root))
        need(a['schema']=='rphi.input_preservation_head_training_admission.v1' and a['approved'] is True
             and a['arm']==c['arm'] and a['mode']==c['mode'] and a['execution_payload_sha256']==T.execution_sha(c)
             and a['entry_sha256']==T.file_sha(__file__) and a['out']==str(Path(out).resolve()), 'exact separate training admission required')
    return c,paths,docs


def read_jsonl(artifact):
    path=Path(artifact['path'])
    need(path.stat().st_size==artifact['bytes'] and T.file_sha(path)==artifact['sha256'], 'cache JSONL bytes changed')
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_training_arrays(manifest, schedule, mapping, proposal, *, expected_rows=ROWS):
    """Read only raw features/normalized labels; never decompress K1/GT RGB."""
    import prepare_input_preservation_donors as D
    rows = D.validate_schedule(schedule,proposal) if expected_rows==ROWS else schedule
    validation=D.validate_donor_map(rows,mapping,expected_rows)
    need(validation.get('bijection_verified') is True,'full cross-source/group donor bijection required')
    records,batches=read_jsonl(manifest['records']),read_jsonl(manifest['batches'])
    need(len(records)==expected_rows and len(batches)*BATCH==expected_rows,'actual cache index coverage differs')
    features=np.empty((expected_rows,2,11,6,6),dtype=np.float32)
    labels=np.empty((expected_rows,1,6,6),dtype=np.float32)
    cache_root=Path(manifest['batches']['path']).resolve().parent
    for bi,b in enumerate(batches):
        start=bi*BATCH;selected=rows[start:start+BATCH];path=Path(b['path'])
        need(b['batch_index']==bi and b['row_indices']==list(range(start,start+BATCH))
             and b['row_sha256']==[r['row_sha256'] for r in selected], 'cache/schedule row identity differs')
        need(not path.is_symlink() and path.resolve().parent==cache_root/'arrays'
             and path.stat().st_size==b['bytes'] and T.file_sha(path)==b['sha256'], 'actual train cache NPZ differs')
        with zipfile.ZipFile(path) as z:
            members=z.infolist()
            need(len(members)==7 and len({m.filename for m in members})==7
                 and all(m.compress_type==zipfile.ZIP_STORED and m.file_size<2<<20 for m in members), 'cache NPZ format differs')
        with np.load(path,allow_pickle=False) as z:
            for name,destination,shape in [('raw_features',features,(BATCH,2,11,6,6)),
                                           ('normalized_benefit',labels,(BATCH,1,6,6))]:
                value=z[name]
                need(value.dtype==np.float32 and value.shape==shape and np.isfinite(value).all()
                     and b['arrays'][name]=={'dtype':'float32','shape':list(shape),'sha256':array_sha(value)}, 'head cache array differs: '+name)
                if name=='normalized_benefit':need(np.abs(value).max()<=4.,'normalized target outside frozen range')
                destination[start:start+BATCH]=value
                for j,r in enumerate(records[start:start+BATCH]):
                    need(r['row_index']==start+j and r['row_sha256']==selected[j]['row_sha256']
                         and r['cache_role']=='train' and r['usable_for_training'] is True
                         and r['batch_index']==bi and r['batch_row']==j and r['cache_file_sha256']==b['sha256']
                         and r['array_row_sha256'][name]==array_sha(value[j])
                         and r['cache_record_sha256']==T.value_sha({k:v for k,v in r.items() if k!='cache_record_sha256'}),
                         'train cache row provenance differs')
    donors=np.asarray([e['donor']['row_index'] for e in mapping['entries']],dtype=np.int64)
    return features,labels,donors,{'rows':expected_rows,'train_NPZ_files_verified':len(batches),
        'feature_and_label_arrays_loaded':len(batches)*2,'K1_or_GT_arrays_loaded':0,'dev_NPZ_files_opened':0,
        'feature_array_sha256':array_sha(features),'label_array_sha256':array_sha(labels),
        'donor_index_array_sha256':array_sha(donors),'student_row_SHA':T.value_sha([r['row_sha256'] for r in rows]),
        'donor_invariants':validation}


def selected_targets(labels, donors, start, arm):
    need(arm in ('matched','shuffled') and 0<=start<=len(labels)-BATCH,'invalid training arm/row')
    indices=np.arange(start,start+BATCH) if arm=='matched' else donors[start:start+BATCH]
    return np.ascontiguousarray(labels[indices]), indices.tolist()


def make_optimizer(model, torch):
    return torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,
        weight_decay=0.,amsgrad=False,foreach=False,fused=False)


def validate_optimizer(model,opt,step,torch):
    from train_rphi import learning_rate
    params=list(model.parameters());need(len(params)==4 and sum(p.numel() for p in params)==1745,'head parameter inventory changed')
    need(len(opt.param_groups)==1,'optimizer groups changed');g=opt.param_groups[0]
    need(all(a is b for a,b in zip(g['params'],params)) and len(g['params'])==4,'optimizer parameter identity changed')
    expected={'lr':learning_rate(step,STEPS,1e-4,50),'betas':(.9,.999),'eps':1e-8,
              'weight_decay':0.,'amsgrad':False,'foreach':False,'fused':False}
    need(all(g[k]==v for k,v in expected.items()) and not g.get('maximize',False)
         and not g.get('capturable',False) and not g.get('differentiable',False),'optimizer recipe changed')
    need(set(opt.state)==set(params),'optimizer state coverage differs')
    for p in params:
        need(p.device.type=='cpu' and p.dtype==torch.float32 and bool(torch.isfinite(p).all()),'invalid head parameter')
        state=opt.state[p];need(set(state)=={'step','exp_avg','exp_avg_sq'},'optimizer fields changed')
        for key,v in state.items():
            need(v.device.type=='cpu' and v.dtype==torch.float32 and bool(torch.isfinite(v).all())
                 and tuple(v.shape)==(() if key=='step' else tuple(p.shape)),'optimizer state invalid')
        need(float(state['step'])==step and bool((state['exp_avg_sq']>=0).all()),'optimizer step/moment invalid')


def training_step(model,opt,x,y,step,counts,torch):
    from train_rphi import learning_rate
    need(x.dtype==torch.float32 and y.dtype==torch.float32 and x.shape==(BATCH,2,11,6,6)
         and y.shape==(BATCH,1,6,6) and not x.requires_grad and not y.requires_grad,'raw cache/label training interface differs')
    opt.zero_grad(set_to_none=True);lr=learning_rate(step,STEPS,1e-4,50);opt.param_groups[0]['lr']=lr
    counts['forward_attempts']+=1;result=model.forward_features(x);counts['forwards_completed']+=1
    loss=torch.nn.functional.smooth_l1_loss(result['benefit_score'],y.detach(),beta=1.,reduction='mean')
    need(bool(torch.isfinite(loss)) and loss.requires_grad,'invalid head loss')
    counts['backward_attempts']+=1;loss.backward();counts['backwards_completed']+=1
    need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'missing/nonfinite head gradient')
    tail={n:float(p.grad.norm()) for n,p in model.named_parameters() if n.startswith('gate.net.2.')}
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
    counts['optimizer_attempts']+=1;opt.step();counts['optimizer_updates']+=1
    validate_optimizer(model,opt,step,torch)
    row={'step':step,'row_end_exclusive':step*BATCH,'lr':lr,'lr_hex':lr.hex(),'loss':float(loss.detach()),
        'gradient_norm_before_clip':float(norm),'tail_gradient_norms':tail,
        'correction_tile_fraction':float(result['correction_mask'].float().mean())}
    need(np.isfinite([row['loss'],row['gradient_norm_before_clip'],*tail.values()]).all(),'nonfinite diagnostics')
    opt.zero_grad(set_to_none=True);return row


def save_checkpoint(path,model,opt,step,c,initial,runtime,torch):
    validate_optimizer(model,opt,step,torch)
    role='research_final_head' if c['mode']=='research' and step==STEPS else 'backend_smoke_not_research_resume'
    need((c['mode']=='research' and step==1000) or (c['mode']=='backend_smoke' and step==2),'only fixed terminal checkpoints allowed')
    payload={'schema':CHECKPOINT_SCHEMA,'artifact_role':role,'arm':c['arm'],'completed_step':step,'row_cursor':step*BATCH,
        'model_config':copy.deepcopy(model.model_config),'state_dict':T.cpu_tree(model.state_dict()),
        'optimizer_state_dict':T.cpu_tree(opt.state_dict()),'rng_state':T.capture_rng(torch.device('cpu')),
        'training_contract':copy.deepcopy(c),'execution_payload_sha256':T.execution_sha(c),'binding':binding(c),
        'initial_state_sha256':initial,'final_state_sha256':T.state_sha(model.state_dict()),
        'optimizer_specification':copy.deepcopy(OPTIMIZER),'runtime':runtime,'quality_accepted':False}
    path=Path(path);need(not path.exists() and not path.is_symlink(),'checkpoint overwrite forbidden')
    temporary=path.with_suffix('.pt.partial')
    with temporary.open('xb') as f:torch.save(payload,f)
    os.link(temporary,path);temporary.unlink()
    saved=torch.load(path,map_location='cpu',weights_only=True);need(binary_tree_equal(payload,saved,torch),'serialized checkpoint differs')
    return {'path':str(path.resolve()),'sha256':T.file_sha(path),'bytes':path.stat().st_size,
            'model_sha256':payload['final_state_sha256'],'artifact_role':role},saved


def verify_fresh_reload(saved,torch):
    from rphi_input_preservation import InputPreservationHead
    model=InputPreservationHead(seed=SEED);opt=make_optimizer(model,torch)
    model.load_state_dict(saved['state_dict'],strict=True);opt.load_state_dict(saved['optimizer_state_dict'])
    # Saved cpu_tree deliberately canonicalizes OrderedDict into dict. Apply
    # the same container conversion to live state; tensor equality stays exact.
    need(binary_tree_equal(T.cpu_tree(model.state_dict()),saved['state_dict'],torch)
         and T.state_sha(model.state_dict())==saved['final_state_sha256']
         and binary_tree_equal(T.cpu_tree(opt.state_dict()),saved['optimizer_state_dict'],torch),
         'fresh reload model/optimizer differs')
    T.restore_rng(saved['rng_state'],torch.device('cpu'))
    need(binary_tree_equal(T.capture_rng(torch.device('cpu')),saved['rng_state'],torch),'fresh RNG reload differs')
    validate_optimizer(model,opt,saved['completed_step'],torch)
    return True


def execute(c,paths,docs,out):
    need(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0','disable Torch backend autoload before execution')
    import torch
    from rphi_input_preservation import InputPreservationHead
    need('torch_npu' not in sys.modules,'NPU backend must not be imported');torch.set_num_threads(2)
    versions={'python':'.'.join(map(str,sys.version_info[:3])),'numpy':np.__version__,'torch':str(torch.__version__),'torch_npu':None}
    need(versions==c['runtime']['expected_versions'],'exact CPU runtime differs')
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    model=InputPreservationHead(seed=SEED).train();need(model.model_config==CONFIG,'head config differs')
    initial=T.state_sha(model.state_dict());opt=make_optimizer(model,torch)
    report={'schema':REPORT_SCHEMA,'complete':False,'status':'loading_audited_train_features','mode':c['mode'],'arm':c['arm'],
        'comparison_identity_sha256':comparison_identity(c),'execution_payload_sha256':T.execution_sha(c),
        'binding':binding(c),'initial_state_sha256':initial,'completed_step':0,'row_cursor':0,
        'runtime':{'device':'cpu','threads':2,'versions':versions,'dtype':'float32','autocast':False,'grad_scaler':False},
        'counts':{k:0 for k in ('forward_attempts','forwards_completed','backward_attempts','backwards_completed','optimizer_attempts','optimizer_updates')},
        'new_restorer_forwards':0,'dev_label_arrays_read':0,'source_NPZ_reads':0,'NPU_calls':0,
        'original_history_headroom_remains_failed':True,'quality_accepted':False,'resumed':False}
    out=Path(out);need(not out.exists() and not out.is_symlink(),'fresh output required');out.mkdir(parents=True)
    started=time.monotonic();metrics=out/'metrics.jsonl';T.write(out/'partial.json',report)
    try:
        initial_payload={'schema':'rphi.input_preservation_head_initial_state.v1','artifact_role':'initial_state_not_resume',
            'model_config':copy.deepcopy(model.model_config),'state_dict':T.cpu_tree(model.state_dict()),
            'model_sha256':initial,'rng_state':T.capture_rng(torch.device('cpu'))}
        initial_path=out/'initial_state.pt'
        with initial_path.open('xb') as f:torch.save(initial_payload,f)
        need(binary_tree_equal(initial_payload,torch.load(initial_path,map_location='cpu',weights_only=True),torch), 'initial state serialization differs')
        report['initial_state']={'path':str(initial_path),'sha256':T.file_sha(initial_path),'model_sha256':initial}
        features,labels,donors,proof=load_training_arrays(docs['train_cache_manifest'],T.read(paths['student_schedule']),T.read(paths['donor_map']),docs['proposal'])
        report['cache_load']=proof;report['status']='training_head_only'
        steps=2 if c['mode']=='backend_smoke' else STEPS
        for step in range(1,steps+1):
            start=(step-1)*BATCH;y,indices=selected_targets(labels,donors,start,c['arm'])
            row=training_step(model,opt,torch.from_numpy(features[start:start+BATCH]),torch.from_numpy(y),step,report['counts'],torch)
            row.update(feature_rows=list(range(start,start+BATCH)),label_rows=indices,
                       feature_batch_sha256=array_sha(features[start:start+BATCH]),label_batch_sha256=array_sha(y))
            with metrics.open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            report.update(completed_step=step,row_cursor=step*BATCH,last_metric=row)
            if step==1 or step%25==0 or step==steps:
                need(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<RESOURCES['maximum_output_bytes']-(1<<20),'head output byte budget exceeded')
                T.write(out/'partial.json',report)
        need(all(v==steps for v in report['counts'].values()),'head operation count differs')
        need('torch_npu' not in sys.modules and torch.get_num_threads()==2,'CPU backend changed')
        checkpoint,saved=save_checkpoint(out/('head_step1000.pt' if steps==1000 else 'backend_step2.pt'),model,opt,steps,c,initial,report['runtime'],torch)
        del model,opt
        report['strict_checkpoint_reload']=verify_fresh_reload(saved,torch)
        report.update(checkpoint=checkpoint,final_state_sha256=saved['final_state_sha256'],
            final_parameters_changed=initial!=saved['final_state_sha256'],metrics={'path':str(metrics),'sha256':T.file_sha(metrics),'rows':steps},
            complete=True,status='final_head_ready_for_bound_evaluation' if steps==1000 else 'CPU_backend_smoke_complete_not_research_resume')
        need(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<RESOURCES['maximum_output_bytes']-(1<<20),'head final output byte budget exceeded')
    except BaseException as e:report.update(complete=False,status='failed_no_retry',error={'type':type(e).__name__,'message':str(e)})
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        T.write(out/'partial.json',report);T.write(out/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution-contract',required=True)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    c,paths,docs=inspect_contract(a.execution_contract,execute=a.execute,out=a.out)
    need(not a.out.exists() and not a.out.is_symlink(),'fresh output only')
    if not a.execute:
        a.out.mkdir(parents=True);r={'complete':True,'status':'metadata_inspected_no_features_or_model','mode':c['mode'],'arm':c['arm'],
            'execution_payload_sha256':T.execution_sha(c),'model_calls':0,'train_NPZ_reads':0,'dev_NPZ_reads':0}
        T.write(a.out/'inspection.json',r);print(json.dumps(r));return 0
    r=execute(c,paths,docs,a.out)
    try:inspect_contract(a.execution_contract,execute=True,out=a.out)
    except BaseException as e:
        r.update(complete=False,status='failed_final_binding',error={'type':type(e).__name__,'message':str(e)})
        T.write(a.out/'report.json',r);T.write(a.out/'partial.json',r)
    print(json.dumps({k:r.get(k) for k in ('complete','status','arm','mode','completed_step','error')}));return 0 if r['complete'] else 1


if __name__=='__main__':raise SystemExit(main())
