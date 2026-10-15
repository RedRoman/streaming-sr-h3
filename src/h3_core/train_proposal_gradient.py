"""Five native22 full50-block updates of the proposal-memory condition adapter.

This deliberately small train-only experiment verifies gradient paths, exact
zero-state behavior, and frozen foundation weights. It establishes no SR gain,
no sequential-flow novelty, and no held-out effectiveness.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from proposal_condition import NativeProposalCondition,CausalProposalState


def main():
    p=argparse.ArgumentParser();p.add_argument('--comfy',required=True);p.add_argument('--model',required=True);p.add_argument('--context',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=5);args=p.parse_args()
    device_index=os.environ.get('CUDA_VISIBLE_DEVICES','');assert device_index.isdigit()
    free,util=map(int,subprocess.check_output(['nvidia-smi','--id='+device_index,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));assert free>75000 and util==0,(free,util)
    torch.set_num_threads(4);torch.manual_seed(703);torch.cuda.set_per_process_memory_fraction(.9);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0,str(Path(args.comfy).resolve()));sys.path.insert(0,str(Path('experiments/h3_backbone').resolve()))
    import comfy.sd
    from comfy.ldm.minimax.model import PackedLayout,time_shift_sigma
    from torch.utils.checkpoint import checkpoint
    from training_math import native_training_math
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    result={'status':'running','scope':'5-step actualnative22 proposal-memory adapter gradient integration; not quality/novelty evidence','config':vars(args),'seed':703,'training_target':'Separate train-only physicalnew-support hybrid, then commonfar/boxDC beforeVAE; box-inducedidentitychange explicitlyrecorded inpreparationaudit','sampling_control_note':'Basicsequentialflow/nativeadapter sameX0 control required for effectiveness; thisrun only verifies interface.','records':[]}
    try:
        model=comfy.sd.load_diffusion_model(args.model,model_options={'dtype':torch.bfloat16}).model.diffusion_model.eval().requires_grad_(False).to('cuda');assert len(model.blocks)==50
        for block in model.blocks:
            original=block.forward
            def checkpointed(*inputs,_original=original,**kwargs):
                if 'transformer_options' in kwargs:kwargs={**kwargs,'transformer_options':dict(kwargs['transformer_options'])}
                return checkpoint(_original,*inputs,use_reentrant=False,**kwargs)
            block.forward=checkpointed
        text=torch.load(args.context,map_location='cpu',weights_only=True)
        with torch.no_grad():context=model.preprocess_text_embeds(text['prompt_embeds'][None].to('cuda',dtype=torch.bfloat16)).detach()
        tags=text['text_token_tags'].cuda();data={k:v.cuda().float().detach() for k,v in load_file(str(Path(args.data)/'conditioning.safetensors')).items()};target=load_file(str(Path(args.data)/'supervision_train_only.safetensors'))['target'].cuda().float().detach()
        adapter=NativeProposalCondition().cuda();optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4);sigma=.5;audio_t=37;sigma_audio=float(time_shift_sigma(torch.tensor(sigma),12,3))
        with torch.no_grad():
            zero=torch.zeros(1,32,device='cuda');zero_difference=float((adapter.condition(data['reference'],data['current'],zero)-data['reference']).abs().max());assert zero_difference==0
            ledger=CausalProposalState(data['reference'],zero.clone());ledger.observe(adapter,'far0',0,data['innovation'],data['geometry'],False);far_difference=float(ledger.code.abs().max());assert far_difference==0
            ledger.observe(adapter,'near17',17,data['innovation'],data['geometry'],True);old=ledger.code.clone();assert not ledger.observe(adapter,'near17',17,data['innovation']+1,data['geometry'],True);assert torch.equal(ledger.code,old)
        result.update(trainable_parameters=sum(p.numel() for p in adapter.parameters()),frozen_native_blocks=len(model.blocks),native_guide_shape=list(data['reference'].shape),condition_tokens=448,zero_state_maxabs=zero_difference,far_only_state_change=far_difference,duplicate_state_maxabs=float((old-ledger.code).abs().max()))
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True);state=adapter.transition(torch.zeros(1,32,device='cuda'),data['reference'],data['innovation'],data['geometry'],True);state.retain_grad();guide=adapter.condition(data['reference'],data['current'],state);guide.retain_grad()
            noise=torch.randn_like(target);xt=(1-sigma)*target+sigma*noise;audio=torch.randn(1,32,2,audio_t,device='cuda')*sigma_audio
            keyframes=[{'resolved_frame_index':0,'latent':guide}];payload={'text_token_tags':tags,'keyframes':keyframes,'cond_video_latents':[guide],'seed':703,'audio_scale':1.,'layout':PackedLayout(context.shape[1],7,16,16,audio_t,keyframes=keyframes)}
            with native_training_math():
                prediction=model([xt,audio],torch.tensor([sigma*1000],device='cuda'),context,minimax_payload=payload,transformer_options={});loss=F.mse_loss(prediction[0].float(),noise-target);loss.backward()
            assert torch.isfinite(loss) and guide.grad is not None and torch.isfinite(guide.grad).all()
            frozen_count=sum(v.grad is not None for v in model.parameters());assert frozen_count==0
            gradients=[v.grad for v in adapter.parameters() if v.grad is not None];assert gradients and all(torch.isfinite(v).all() for v in gradients)
            update_norm=float(torch.stack([v.grad.float().square().sum() for v in adapter.update.parameters() if v.grad is not None]).sum().sqrt());read_norm=float(torch.stack([v.grad.float().square().sum() for v in adapter.read.parameters() if v.grad is not None]).sum().sqrt());assert update_norm>0
            record={'step':step+1,'velocity_mse':float(loss.detach()),'guide_input_grad_norm':float(guide.grad.norm()),'state_grad_norm':float(state.grad.norm()),'update_parameter_grad_norm':update_norm,'readout_parameter_grad_norm':read_norm,'state_norm':float(state.detach().norm()),'base_weight_grad_count':frozen_count,'native_video_finite':bool(torch.isfinite(prediction[0]).all()),'native_audio_finite':bool(torch.isfinite(prediction[1]).all())};result['records'].append(record)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);optimizer.step();print(json.dumps({'event':'native_proposal_memory_gradient',**record}),flush=True)
        assert result['records'][-1]['readout_parameter_grad_norm']>0
        save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},str(out/'adapter.safetensors'));result['status']='complete'
    except Exception as exc:result.update(status='failed',exception=repr(exc));raise
    finally:
        result.update(seconds=time.monotonic()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
