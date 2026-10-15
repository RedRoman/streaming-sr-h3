"""Native frozen MiniMax-H3 SR LoRA/guide entry, with a two-step gradient audit.

Supervised HR latents enter only the training target/noised target stream. LR
guide latents and a fixed neutral text are the conditioning. A tiny default run
is feasibility evidence only, not a restoration-quality experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from training_math import inject_attention_lora,native_training_math


class GuideProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv3d(24,32,(1,3,3),padding=(0,1,1)),nn.SiLU(),nn.Conv3d(32,24,(1,3,3),padding=(0,1,1)))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return x+self.net(x.float())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--comfy',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--guide',type=Path,required=True);p.add_argument('--target',type=Path,required=True)
    p.add_argument('--context',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--rank',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--sigma',type=float,default=.5)
    p.add_argument('--checkpoint-blocks',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--train-guide-projection',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--lora-blocks',default='all');p.add_argument('--verify-model-sha',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert a.model.is_file() and a.model.stat().st_size==40225724176,'Full canonical BF16 model must be complete; partial download is forbidden'
    assert a.guide.resolve()!=a.target.resolve(),'HR supervision must not be passed as the LR condition'
    assert 0<a.sigma<1 and a.steps>0
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    assert gpu.isdigit(),'Set exactly one numeric CUDA_VISIBLE_DEVICES'
    row=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    free,util=map(int,row.split(','));assert free>75000 and util==0,(free,util)
    sys.path.insert(0,str(a.comfy));import comfy.sd
    from comfy.ldm.minimax.model import PackedLayout,time_shift_sigma
    from torch.utils.checkpoint import checkpoint
    torch.set_num_threads(4);torch.manual_seed(703);torch.cuda.set_per_process_memory_fraction(.9)
    torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    result={'status':'running','scope':'Whole native50-block frozen H3 BF16 LoRA/guide SR gradient feasibility by default; not method quality evidence',
            'model_path':str(a.model),'model_bytes':a.model.stat().st_size,'core_storage':'BF16 attn/MLP, official prunedFP16AdaLN curve coefficients/F32 time table, no Turbo/quantizedDiT',
            'text_precision':'Cached actual Qwen3VL32B NVFP4-AWQ encoder/BF16 hidden50; fixed neutral prompt',
            'conditioning':'LR guide only, no HR guide; HR is supervised velocity/noised-target construction only',
            'sign_protocol':'Native MiniMaxH3Model returns negative data-ward velocity; target=epsilon-zHR, x0hat=xt-sigma*prediction',
            'rank':a.rank,'lora_blocks':a.lora_blocks,'checkpoint_blocks':a.checkpoint_blocks,'guide_projection':a.train_guide_projection,
            'steps':a.steps,'seed':703,'sigma_video':a.sigma,'lr':a.lr,'records':[]}
    result['input_file_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.guide,a.target,a.context]}
    result['code_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),Path(__file__).with_name('training_math.py'),a.comfy/'comfy/ldm/minimax/model.py']}
    try:
        if a.verify_model_sha:
            digest=hashlib.sha256()
            with a.model.open('rb') as handle:
                while block:=handle.read(32*1024**2):digest.update(block)
            assert digest.hexdigest()=='a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4'
            result['verified_model_sha256']=digest.hexdigest()
        print(json.dumps({'event':'load_whole_native_H3','model':str(a.model)}),flush=True)
        patcher=comfy.sd.load_diffusion_model(str(a.model),model_options={'dtype':torch.bfloat16})
        model=patcher.model.diffusion_model
        model.eval().requires_grad_(False).to('cuda')
        assert len(model.blocks)==50
        cache=torch.load(a.context,map_location='cpu',weights_only=True)
        with torch.no_grad():context=model.preprocess_text_embeds(cache['prompt_embeds'][None].to('cuda',dtype=torch.bfloat16)).detach()
        tags=cache['text_token_tags'].to('cuda')
        selected=None if a.lora_blocks=='all' else set(map(int,a.lora_blocks.split(',')))
        result['lora_parameter_names']=inject_attention_lora(model,a.rank,selected)
        if a.checkpoint_blocks:
            for block in model.blocks:
                original=block.forward
                def checkpointed(*inputs,_original=original,**kwargs):
                    if 'transformer_options' in kwargs:
                        kwargs={**kwargs,'transformer_options':dict(kwargs['transformer_options'])}
                    return checkpoint(_original,*inputs,use_reentrant=False,**kwargs)
                block.forward=checkpointed
        guide=load_file(str(a.guide),device='cpu')['guide'].float().cuda().detach().requires_grad_()
        target=load_file(str(a.target),device='cpu')['guide'].float().cuda().detach()
        assert guide.shape==target.shape and guide.shape[:2]==(1,24)
        projection=GuideProjection().cuda() if a.train_guide_projection else nn.Identity().cuda()
        params=[v for v in model.parameters() if v.requires_grad]+list(projection.parameters())
        optimizer=torch.optim.AdamW(params,lr=a.lr)
        temporal=guide.shape[2];frames=1 if temporal==1 else ((temporal-2)//5)*17+5
        assert frames>=5
        audio_t=round(frames/24*40)
        sigma_a=float(time_shift_sigma(torch.tensor(a.sigma),12,3))
        result.update(frames=frames,latent_shape=list(guide.shape),audio_latents=audio_t,projected_context_shape=list(context.shape),sigma_audio=sigma_a,trainable_parameters=sum(v.numel() for v in params))
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True);guide.grad=None
            epsilon=torch.randn_like(target);xt=(1-a.sigma)*target+a.sigma*epsilon
            audio=torch.randn(1,32,2,audio_t,device='cuda')*sigma_a
            current=projection(guide)
            keyframes=[{'resolved_frame_index':0,'latent':current}]
            payload={'text_token_tags':tags,'keyframes':keyframes,'cond_video_latents':[current],'seed':703,'audio_scale':1.0,
                     'layout':PackedLayout(context.shape[1],temporal,guide.shape[-2],guide.shape[-1],audio_t,keyframes=keyframes)}
            started=time.perf_counter()
            with native_training_math():
                prediction=model([xt,audio],torch.tensor([a.sigma*1000],device='cuda'),context,minimax_payload=payload,transformer_options={})
                loss=F.mse_loss(prediction[0].float(),epsilon-target)
                loss.backward()
            assert guide.grad is not None and torch.isfinite(guide.grad).all() and guide.grad.float().norm()>0
            assert all(v.grad is None or torch.isfinite(v.grad).all() for v in params)
            frozen_grad_count=sum(v.grad is not None for v in model.parameters() if not v.requires_grad)
            assert frozen_grad_count==0
            lora_grad_norm=float(torch.stack([v.grad.float().square().sum() for v in model.parameters() if v.requires_grad and v.grad is not None]).sum().sqrt())
            assert lora_grad_norm>0
            projection_grad_norm=float(torch.stack([v.grad.float().square().sum() for v in projection.parameters() if v.grad is not None]).sum().sqrt()) if a.train_guide_projection else 0.0
            if a.train_guide_projection:assert projection_grad_norm>0
            grad_norm=float(torch.nn.utils.clip_grad_norm_(params,1))
            record={'step':step,'velocity_mse':float(loss.detach()),'guide_input_grad_norm':float(guide.grad.float().norm()),'adapter_grad_norm_before_clip':grad_norm,
                    'lora_grad_norm':lora_grad_norm,'guide_projection_grad_norm':projection_grad_norm,
                    'base_weight_grad_count':frozen_grad_count,'audio_forward_finite':bool(torch.isfinite(prediction[1]).all()),
                    'x0_latent_rmse':float((xt-a.sigma*prediction[0]-target).float().square().mean().sqrt().detach()),'seconds':time.perf_counter()-started}
            result['records'].append(record);print(json.dumps({'event':'native_full_DiT_gradient',**record}),flush=True)
            optimizer.step()
        state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items() if k.endswith('.a') or k.endswith('.b')}
        state.update({'guide_projection.'+k:v.detach().cpu().contiguous() for k,v in projection.state_dict().items()})
        save_file(state,str(a.out/'adapter_gradient_audit.safetensors'))
        result['status']='complete'
    except Exception as exc:
        result.update(status='failed',exception=repr(exc));raise
    finally:
        result.update(seconds=time.perf_counter()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
