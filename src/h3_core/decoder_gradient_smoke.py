"""Verify actual frozen H3 decoder gradients with respect to its latent input."""
import argparse
import json
import sys
import time
from pathlib import Path
import torch
from h3_interface import load_frozen_vae,differentiable_decoder_math


def main():
    p=argparse.ArgumentParser();p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--dtype',choices=['float16','float32'],default='float16');p.add_argument('--frame',type=int,default=21);p.add_argument('--out',default='results/h3_core/decoder_gradient.json')
    p.add_argument('--eager-autograd',action='store_true');args=p.parse_args()
    sys.path.insert(0,str(Path(args.comfy).resolve()));torch.set_num_threads(4)
    dtype=torch.float16 if args.dtype=='float16' else torch.float32
    vae,info=load_frozen_vae(args.vae,dtype=dtype)
    item=torch.load(args.cache,map_location='cpu',weights_only=False)
    latent=item['inputs']['anchor'].cuda().to(dtype).detach().requires_grad_(True)
    target=item['supervision']['canonical_hr'].cuda().permute(2,0,1)[None].float()/255
    torch.cuda.reset_peak_memory_stats();start=time.time()
    with torch.no_grad():reference=vae.decode(latent.detach())[:,:,args.frame].to(device=latent.device,dtype=torch.float32)
    if args.eager_autograd:
        with differentiable_decoder_math():
            rgb=vae.decode(latent)[:,:,args.frame].to(device=latent.device,dtype=torch.float32)
            loss=(rgb-target).square().mean();loss.backward()
    else:
        rgb=vae.decode(latent)[:,:,args.frame].to(device=latent.device,dtype=torch.float32)
        loss=(rgb-target).square().mean();loss.backward()
    torch.cuda.synchronize()
    result={'passed':bool(latent.grad is not None and torch.isfinite(latent.grad).all() and latent.grad.abs().max()>0),
            'scope':'Actual native H3 VAE, frozen weights, RGB supervision gradient to latent input; not DiT training',
            'vae':info,'dtype':args.dtype,'latent_shape':list(latent.shape),'loss_mse':float(loss.detach()),
            'eager_autograd':args.eager_autograd,'forward_parity_rmse':float((rgb.detach()-reference).square().mean().sqrt()),
            'latent_gradient_rms':float(latent.grad.float().square().mean().sqrt()),
            'latent_gradient_max':float(latent.grad.abs().max()),'frozen_parameter_grads':sum(p.grad is not None for p in vae.parameters()),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'seconds':time.time()-start}
    if not result['passed']:raise RuntimeError(result)
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
