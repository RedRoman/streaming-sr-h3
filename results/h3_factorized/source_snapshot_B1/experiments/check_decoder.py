"""Audit exact B native D path before any RGB supervision."""
import argparse,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from native_vae import load_vae,decode_last,differentiable_decoder_math


def main():
    p=argparse.ArgumentParser();p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);args=p.parse_args();torch.set_num_threads(4);t=time.time()
    vae=load_vae(args.vae,args.comfy);item=torch.load(Path(args.cache)/'train_100075.pt',weights_only=False);z=item['inputs']['observation_latents'][0:1].cuda().float().requires_grad_(True);target=item['supervision']['rgb'][0:1].cuda().float()
    with torch.no_grad():native=decode_last(vae,z).detach()
    with differentiable_decoder_math():
        eager=decode_last(vae,z);loss=(eager-target).abs().mean();loss.backward()
    result=dict(passed=True,seconds=time.time()-t,input_shape=list(z.shape),native_eager_rgb_maxabs=float((native-eager.detach()).abs().max()),native_eager_rgb_rmse=float((native-eager.detach()).square().mean().sqrt()),input_gradient_rms=float(z.grad.square().mean().sqrt()),input_gradient_maxabs=float(z.grad.abs().max()),weight_gradients=sum(p.grad is not None for p in vae.parameters()),actual_decoder_dtype=str(next(vae.decoder.parameters()).dtype),peak_cuda_bytes=torch.cuda.max_memory_allocated())
    assert result['native_eager_rgb_maxabs']<1e-5
    assert torch.isfinite(z.grad).all() and result['input_gradient_rms']>0 and result['weight_gradients']==0
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
