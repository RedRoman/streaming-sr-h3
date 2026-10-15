"""Fixed-budget, source-disjoint native H3 SR adaptation baseline.

All three controls use exactly the same training rows/noise/sigma and final400
checkpoint. Validation never enters optimizer updates or checkpoint selection.
The frozen BF16 foundation/source files are never written.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from training_math import inject_attention_lora,native_training_math
from train_native_lora import GuideProjection

MODES=['lora_only','guide_only','joint']
MODEL_SHA='a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(32*1024**2):h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def append_json(path,obj):
    with path.open('a') as f:f.write(json.dumps(obj)+'\n');f.flush()

def main():
    p=argparse.ArgumentParser()
    for field in ['comfy','model','train','val','context','out']:p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--steps',type=int,default=400);p.add_argument('--seed',type=int,default=703)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    p.add_argument('--readouts',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert a.steps==400,'This preregistered pilot is fixed at400updates'
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','');assert gpu.isdigit()
    row=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    free,util=map(int,row.split(','));assert free>75000 and util==0,(free,util)
    print(json.dumps({'event':'admission','gpu':gpu,'free_MiB':free,'util':util}),flush=True)
    tr=json.loads(a.train.read_text());va=json.loads(a.val.read_text());assert tr['complete'] and va['complete']
    train=sorted(tr['cases'],key=lambda x:x['id']);val=sorted(va['cases'],key=lambda x:x['id'])
    assert len(train)==8 and len(val)==4 and all(c['split']=='train' for c in train) and all(c['split']=='val' for c in val)
    assert {c['sequence'] for c in train}.isdisjoint({c['sequence'] for c in val})
    rng=random.Random(a.seed);schedule=[]
    for epoch in range(50):
        order=list(range(8));rng.shuffle(order)
        for index in order:schedule.append({'update':len(schedule),'case':train[index]['id'],'sigma':.05+.9*rng.random(),'noise_seed':a.seed*100000+len(schedule)})
    assert len(schedule)==a.steps
    protocol={'status':'preregistered','scope':'NativeH3SRadaptation baseline pilot, not a new memory method; final400weights only, no test fitting/hyperparameter or checkpoint selection',
              'seed':a.seed,'updates_each':a.steps,'controls':a.modes,'lr':a.lr,'optimizer':'AdamW betas(.9,.999),eps1e-8,weight_decay.01,gradient_clip1',
              'training_sigma':'Uniform actual video sigma [.05,.95]; no importanceweight; audio sigma exactnative shift12to3 conversion',
              'loss':'unweighted video velocityMSE to epsilon-zHR; HR only target/noisedtarget; LRguideonly condition',
              'train_cases':[c['id'] for c in train],'validation_cases':[c['id'] for c in val],'validation_sigmas':[.2,.5,.8],
              'validation_schedule':'0/100/200/300/400updates fixed noise; no selection, final400readouts only',
              'readout':'normal50sigmapoints/49Euler,CFG1,seed101,neutraltext only; all controls sameLRguide/prefix0..21',
              'lora':'rank2 qkv/out every50nativeblock','guide_projection':'24to32to24 spatial3x3 residual identityinit; parameters differ acrosscontrols',
              'context_sha256':sha(a.context),'model_sha256':MODEL_SHA,'train_manifest_sha256':sha(a.train),'val_manifest_sha256':sha(a.val),
              'schedule':schedule,'native_math':'scoped officialeagerRoPE+functionalresidual, unchangedexternalcode; zeroadapterparity audited'}
    proto_path=a.out/'protocol.json'
    if proto_path.exists():assert json.loads(proto_path.read_text())==protocol,'Refuse protocol drift during resume'
    else:atomic_json(proto_path,protocol)
    protocol_sha=sha(proto_path)
    torch.set_num_threads(4);torch.manual_seed(1703);torch.cuda.set_per_process_memory_fraction(.9);torch.cuda.reset_peak_memory_stats()
    began=time.perf_counter();status={'status':'running','pid':os.getpid(),'protocol_sha256':protocol_sha,'completed':[],
                                    'python':sys.version,'torch':torch.__version__,'script_sha256':sha(__file__),'training_math_sha256':sha(Path(__file__).with_name('training_math.py'))}
    def live(**kwargs):
        status.update(kwargs,epoch=time.time());atomic_json(a.out/'LIVE_STATE.json',status)
    live(phase='verify_full_model')
    assert a.model.stat().st_size==40225724176 and sha(a.model)==MODEL_SHA
    sys.path.insert(0,str(a.comfy))
    from comfy.cli_args import args as comfy_args
    comfy_args.reserve_vram=8.0
    import comfy.sd,comfy.sample,comfy.nested_tensor
    from comfy.ldm.minimax.model import PackedLayout,time_shift_sigma
    from torch.utils.checkpoint import checkpoint
    patcher=comfy.sd.load_diffusion_model(str(a.model),model_options={'dtype':torch.bfloat16})
    model=patcher.model.diffusion_model.eval().requires_grad_(False).to('cuda');assert len(model.blocks)==50
    status.update(native_block_count=50,native_source_sha256=sha(a.comfy/'comfy/ldm/minimax/model.py'))
    frozen=[(name,param,param._version) for name,param in model.named_parameters()]
    cache=torch.load(a.context,map_location='cpu',weights_only=True)
    assert bool((cache['text_token_tags']==1).all()),'This pilot preregisters neutraltext-only; no visual-contextswitch'
    tags=cache['text_token_tags'].cuda()
    with torch.no_grad():context=model.preprocess_text_embeds(cache['prompt_embeds'][None].cuda().to(torch.bfloat16)).detach()
    def load_cases(cases):
        data={}
        for case in cases:
            g,t=Path(case['guide']),Path(case['target'])
            assert sha(g)==case['guide_sha256'] and sha(t)==case['target_sha256']
            guide=load_file(str(g))['guide'].float();target=load_file(str(t))['guide'].float()
            assert guide.shape==target.shape and guide.shape[2]==7
            data[case['id']]={'case':case,'guide':guide,'target':target}
        return data
    train_data=load_cases(train);val_data=load_cases(val)
    def raw(xt,audio,guide,sigma):
        temporal,height,width=guide.shape[2:];keyframes=[{'resolved_frame_index':0,'latent':guide}]
        payload={'text_token_tags':tags,'keyframes':keyframes,'cond_video_latents':[guide],'seed':a.seed,'audio_scale':1.0,
                 'layout':PackedLayout(context.shape[1],temporal,height,width,audio.shape[-1],keyframes=keyframes)}
        return model([xt,audio],torch.tensor([sigma*1000],device='cuda'),context,minimax_payload=payload,transformer_options={})
    def probe(data,sigma,seed,projection):
        guide=data['guide'].cuda().detach().requires_grad_();target=data['target'].cuda()
        gen=torch.Generator(device='cuda').manual_seed(seed)
        epsilon=torch.randn(target.shape,device='cuda',generator=gen)
        xt=(1-sigma)*target+sigma*epsilon
        sa=float(time_shift_sigma(torch.tensor(sigma),12,3))
        audio=torch.randn((1,32,2,37),device='cuda',generator=gen)*sa
        output=raw(xt,audio,projection(guide),sigma)
        return F.mse_loss(output[0].float(),epsilon-target),output,guide,(xt,audio,target,epsilon)
    parity_data=train_data[train[0]['id']];identity=nn.Identity().cuda()
    with torch.no_grad():
        _,native,_,_=probe(parity_data,.5,999,identity)
        with native_training_math():_,scoped,_,_=probe(parity_data,.5,999,identity)
    parity={'native_vs_scoped_video_maxabs':float((native[0]-scoped[0]).abs().max()),'native_vs_scoped_audio_maxabs':float((native[1]-scoped[1]).abs().max())}
    inject_attention_lora(model,2);projection=GuideProjection().cuda()
    with torch.no_grad(),native_training_math():
        _,zero,_,_=probe(parity_data,.5,999,projection)
    parity.update(zero_adapter_video_maxabs=float((scoped[0]-zero[0]).abs().max()),zero_adapter_audio_maxabs=float((scoped[1]-zero[1]).abs().max()))
    assert parity['zero_adapter_video_maxabs']==parity['zero_adapter_audio_maxabs']==0
    del native,scoped,zero
    atomic_json(a.out/'zero_adapter_parity.json',parity);print(json.dumps({'event':'zero_adapter_parity',**parity}),flush=True)
    for block in model.blocks:
        original=block.forward
        def checkpointed(*inputs,_original=original,**kwargs):
            if not torch.is_grad_enabled():return _original(*inputs,**kwargs)
            if 'transformer_options' in kwargs:kwargs={**kwargs,'transformer_options':dict(kwargs['transformer_options'])}
            return checkpoint(_original,*inputs,use_reentrant=False,**kwargs)
        block.forward=checkpointed
    lora=[(n,v) for n,v in model.named_parameters() if n.endswith('.a') or n.endswith('.b')]
    initial={n:v.detach().clone() for n,v in lora};initial_proj={n:v.detach().clone() for n,v in projection.state_dict().items()}
    def frozen_audit():
        altered=[name for name,param,version in frozen if param._version!=version or param.grad is not None or param.requires_grad]
        assert not altered,altered[:8]
        return {'frozen_tensor_count':len(frozen),'frozen_parameter_count':sum(p.numel() for _,p,_ in frozen),'base_parameter_versions_unchanged':True,'base_weight_grad_count':0}
    def validate(mode,update):
        rows=[]
        with torch.no_grad(),native_training_math():
            for index,case in enumerate(val):
                for j,sigma in enumerate([.2,.5,.8]):
                    seed=90300+index*3+j;loss,output,_,_=probe(val_data[case['id']],sigma,seed,projection)
                    rows.append({'case':case['id'],'sigma':sigma,'noise_seed':seed,'velocity_mse':float(loss),'audio_finite':bool(torch.isfinite(output[1]).all())})
        record={'mode':mode,'update':update,'rows':rows,'mean_velocity_mse':sum(r['velocity_mse'] for r in rows)/len(rows)}
        append_json(a.out/'validation.jsonl',record);print(json.dumps({'event':'fixed_validation','mode':mode,'update':update,'mean_velocity_mse':record['mean_velocity_mse']}),flush=True)
    def save_adapter(path):
        state={n:p.detach().cpu().contiguous() for n,p in lora};state.update({'guide_projection.'+n:p.detach().cpu().contiguous() for n,p in projection.state_dict().items()})
        save_file(state,str(path));return state
    def readouts(label):
        for case in val:
            folder=a.out/'readouts'/label/case['id'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'sample_result.json').exists() and json.loads((folder/'sample_result.json').read_text()).get('status')=='complete':continue
            live(phase='native49_readout',mode=label,case=case['id'])
            guide=projection(val_data[case['id']]['guide'].cuda()).detach().cpu()
            positive=[[cache['prompt_embeds'][None],{'minimax_token_tags':cache['text_token_tags'],'minimax_keyframes':[{'resolved_frame_index':0,'latent':guide}]}]]
            height,width=case['shape_hw'];latent=comfy.nested_tensor.NestedTensor((torch.zeros(1,24,7,height//16,width//16),torch.zeros(1,32,2,37)))
            base=torch.linspace(1,0,50);sigmas=12*base/(1+11*base);start=time.perf_counter()
            with torch.no_grad(),native_training_math():
                samples=comfy.sample.sample(patcher,comfy.sample.prepare_noise(latent,101),49,1.,'euler','simple',positive,[],latent,seed=101,disable_pbar=True,sigmas=sigmas)
            torch.cuda.synchronize();video,audio=samples.unbind()
            assert torch.isfinite(video).all() and torch.isfinite(audio).all()
            save_file({'video':video.cpu().contiguous(),'audio':audio.cpu().contiguous()},str(folder/'sampled_av_latents.safetensors'))
            save_file({'guide':val_data[case['id']]['guide'].contiguous()},str(folder/'guide.safetensors'))
            save_file({'guide':guide.contiguous()},str(folder/'actual_condition_guide.safetensors'))
            result={'status':'complete','mode':label,'case':case['id'],'split':'val','shape_hw':case['shape_hw'],'input_prefix':[0,21],'bounded_wait_frames':21,
                    'observed_lr_sha256':case['observed_lr_sha256'],'guide_sha256':case['guide_sha256'],'context_sha256':protocol['context_sha256'],
                    'actual_projected_condition_sha256':sha(folder/'actual_condition_guide.safetensors'),
                    'protocol_sha256':protocol_sha,'steps_model_evaluations':49,'sigma_points':50,'cfg':1,'seed':101,'seconds':time.perf_counter()-start,
                    'decoder_pending':True,'HR_condition':False,'gt_target_not_read_by_sampler':True,'native_math':'scopedautogradcompatible sourceequivalent math, frozenexternalfiles'}
            atomic_json(folder/'sample_result.json',result);print(json.dumps({'event':'native49_readout_complete',**result}),flush=True)
    try:
        with torch.no_grad():
            if a.readouts:readouts('unadapted')
        for mode in a.modes:
            folder=a.out/mode;folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text()).get('status')=='complete':
                status['completed'].append(mode);continue
            with torch.no_grad():
                for n,v in lora:v.copy_(initial[n])
                projection.load_state_dict(initial_proj)
            for _,v in lora:v.requires_grad_(mode in ['lora_only','joint'])
            projection.requires_grad_(mode in ['guide_only','joint'])
            params=[v for _,v in lora if v.requires_grad]+[v for v in projection.parameters() if v.requires_grad]
            opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=.01)
            checkpoint_file=folder/'resume.pt';start_update=0
            if checkpoint_file.exists():
                restored=torch.load(checkpoint_file,map_location='cuda',weights_only=False);assert restored['protocol_sha256']==protocol_sha
                with torch.no_grad():
                    for n,v in lora:v.copy_(restored['adapter'][n])
                    projection.load_state_dict({k.removeprefix('guide_projection.'):v for k,v in restored['adapter'].items() if k.startswith('guide_projection.')})
                opt.load_state_dict(restored['optimizer']);start_update=restored['next_update']
            counts={'mode':mode,'trainable_parameters':sum(v.numel() for v in params),'lora_parameters':sum(v.numel() for _,v in lora if v.requires_grad),'guide_parameters':sum(v.numel() for v in projection.parameters() if v.requires_grad)}
            print(json.dumps({'event':'training_control_start',**counts,'resume_update':start_update}),flush=True)
            if start_update==0:validate(mode,0)
            control_start=time.perf_counter()
            for update in range(start_update,a.steps):
                row=schedule[update];live(phase='train',mode=mode,update=update,case=row['case']);opt.zero_grad(set_to_none=True)
                start=time.perf_counter()
                with native_training_math():
                    loss,output,input_guide,_=probe(train_data[row['case']],row['sigma'],row['noise_seed'],projection)
                    assert torch.isfinite(loss);loss.backward()
                assert input_guide.grad is not None and torch.isfinite(input_guide.grad).all()
                assert all(v.grad is None or torch.isfinite(v.grad).all() for v in params)
                audit=frozen_audit();norm=float(torch.nn.utils.clip_grad_norm_(params,1))
                assert norm>0 and torch.isfinite(torch.tensor(norm));opt.step()
                record={**row,**counts,'velocity_mse':float(loss.detach()),'guide_input_grad_norm':float(input_guide.grad.float().norm()),'adapter_grad_norm_before_clip':norm,
                        'base_weight_grad_count':audit['base_weight_grad_count'],'seconds':time.perf_counter()-start}
                append_json(folder/'train.jsonl',record)
                if (update+1)%10==0:print(json.dumps({'event':'train_progress',**record}),flush=True)
                if (update+1)%50==0:
                    state=save_adapter(folder/f'adapter_step{update+1:04d}.safetensors')
                    tmp=folder/'resume.pt.tmp';torch.save({'protocol_sha256':protocol_sha,'next_update':update+1,'adapter':state,'optimizer':opt.state_dict()},tmp);tmp.replace(checkpoint_file)
                if (update+1)%100==0:validate(mode,update+1)
            final_audit=frozen_audit();save_adapter(folder/'adapter_final400.safetensors')
            if a.readouts:
                with torch.no_grad():readouts(mode)
            result={'status':'complete',**counts,**final_audit,'updates_completed':a.steps,'protocol_sha256':protocol_sha,'seconds_training_and_fixed_eval':time.perf_counter()-control_start,
                    'checkpoint_selection':'none, final400fixed','peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            atomic_json(folder/'result.json',result);status['completed'].append(mode)
        live(status='complete',phase='complete',seconds=time.perf_counter()-began,**frozen_audit())
    except Exception as exc:
        live(status='failed',exception=repr(exc),seconds=time.perf_counter()-began);raise
    finally:print(json.dumps(status),flush=True)

if __name__=='__main__':main()
