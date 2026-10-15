"""Align cached RGB baseline with the exact saved-query latent runtime route.

No model/label tuning or new observations: original encoder-output D differed
slightly from D(saved stacked half latent), while repeated runtime D is exact.
"""
import argparse,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from native_vae import load_vae,decode_last


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;root=Path(args.cache);manifest=json.loads((root/'manifest.json').read_text());start=time.time();vae=load_vae(args.vae,args.comfy);rows=[]
    with torch.no_grad():
        for prov in manifest['items']:
            path=root/(prov['id']+'.pt');data=torch.load(path,map_location='cpu',weights_only=False);anchor=data['inputs']['query_latents'][None].cuda().float();old=data['inputs']['query_base_rgb'];new=torch.stack([decode_last(vae,anchor[:,i])[0].cpu() for i in range(anchor.shape[1])]);row=dict(source=prov['id'],old_to_runtime_maxabs=float((old-new).abs().max()),old_to_runtime_RMSE=float((old-new).square().mean().sqrt()));data['inputs']['query_base_rgb']=new.float();data['provenance']['baseline_decode_route']='D(saved stacked half query latent) exactly as runtime; original E-output route had small numerical layout/rounding drift';torch.save(data,path);rows.append(row);print(json.dumps(row),flush=True)
    meta=dict(seconds=time.time()-start,rows=rows,reason='Actual protocol probe directD/cache RMSE5.87e-5,max.000447; repeat runtimeD exact0. Baseline redecoded from exact saved runtime query tensor. No change to observations/latent labels/model architecture/budget.',no_training_performed=True);(root.parent/'baseline_route_alignment.json').write_text(json.dumps(meta,indent=2));manifest['baseline_route_alignment']=meta;(root/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
