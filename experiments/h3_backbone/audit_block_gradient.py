"""Real pretrained native block0 backward/LoRA test; not whole-DiT training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from safetensors.torch import load_file,save_file
from training_math import inject_attention_lora,native_training_math


def main():
    p=argparse.ArgumentParser();p.add_argument('--comfy',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.comfy));import comfy.ops
    from comfy.ldm.minimax.model import DiTBlock,rope_rotation_table
    a.out.mkdir(parents=True,exist_ok=True);torch.manual_seed(701);torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.25);torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    h=hashlib.sha256()
    with a.weights.open('rb') as f:
        while b:=f.read(16*1024**2):h.update(b)
    assert h.hexdigest()=='bc712fc8152b8d0fb657374d556219e674dfc37dfcf75667624b6f10f0e1811e'
    state=load_file(str(a.weights),device='cpu');table=state.pop('adaln_t_table').cuda()
    model=DiTBlock(5376,56,128,14336,8,1e-5,1e-5,apply_silu=False,adaln_dtype=torch.float32,dtype=torch.bfloat16,device='cuda',operations=comfy.ops.manual_cast)
    model.load_state_dict(state,strict=True);del state
    model.eval().requires_grad_(False)
    x=torch.randn(64,5376,device='cuda',dtype=torch.bfloat16)*.1
    t=table[torch.tensor([1023,512,819],device='cuda')]
    segments=[(0,8,2),(8,24,0),(24,32,7),(32,64,3)]
    angles=torch.linspace(0,.7,64,device='cuda')[:,None]*torch.linspace(.1,1,96,device='cuda')[None]
    rope=rope_rotation_table(angles,torch.bfloat16)
    result={'scope':'Only one exact pretrained H3 DiT block, synthetic token input; tests autograd/LoRA plumbing, never restoration or whole-DiT training evidence','weights_sha256':h.hexdigest(),'base_storage':'BF16 attn/MLP, FP16 pruned AdaLN coefficients copied exactly to native FP32 compute','tokens_shape':list(x.shape)}
    with torch.no_grad():
        native=model(x.clone(),t,segments,rope)
        with native_training_math():eager=model(x.clone(),t,segments,rope)
    result.update(native_vs_scoped_forward_maxabs=float((native.float()-eager.float()).abs().max()),native_vs_scoped_forward_rmse=float((native.float()-eager.float()).square().mean().sqrt()))
    inject_attention_lora(model,rank=2)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-5)
    try:
        with native_training_math(functional_residuals=False):
            probe=(x.detach().clone().requires_grad_()*1)
            model(probe,t,segments,rope).float().square().mean().backward()
        result['unmodified_residual_backward']='succeeded'
    except RuntimeError as exc:result['unmodified_residual_backward_error']=str(exc)
    opt.zero_grad(set_to_none=True)
    target=native.detach().float()*.9;steps=[]
    for i in range(2):
        probe=x.detach().clone().requires_grad_()
        with native_training_math():
            output=model(probe,t,segments,rope)
            loss=(output.float()-target).square().mean();loss.backward()
        row={'step':i,'loss':float(loss),'input_grad_norm':float(probe.grad.float().norm()),'lora_grad_norm':float(torch.stack([p.grad.float().square().sum() for p in params if p.grad is not None]).sum().sqrt()),'base_weight_grad_count':sum(p.grad is not None for p in model.parameters() if not p.requires_grad)}
        assert row['input_grad_norm']>0 and row['lora_grad_norm']>0 and row['base_weight_grad_count']==0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
        torch.nn.utils.clip_grad_norm_(params,1);opt.step();opt.zero_grad(set_to_none=True);steps.append(row)
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items() if k.endswith('.a') or k.endswith('.b')},str(a.out/'lora_gradient_smoke.safetensors'))
    result.update(status='complete',steps=steps,trainable_parameters=sum(p.numel() for p in params),seconds=time.perf_counter()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
