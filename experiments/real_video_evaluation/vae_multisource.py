"""Fixed multi-source H3 codec baseline, including native 480p spatial context.

Preparation uses HR only to define a reproducible synthetic observation dataset.
The inference phase opens only LR paths. No training or test-based selection.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/real_video'
OUT = ROOT/'results/real_video_vae'
sys.path.insert(0, str(ROOT/'src'))
from pixel_consistency import BicubicConsistency, audit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2, 0, 1).float()/255


def save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(value.permute(1, 2, 0).cpu().numpy().clip(0, 1)*255+.5)).save(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    windows = json.loads((DATA/'h3_windows_39.json').read_text())['windows']
    videos = json.loads((DATA/'manifest.json').read_text())['videos']
    cases = []
    for window in windows:
        cases.append(dict(sequence=window['sequence'], split=window['split'], resolution='256',
                          hr_paths=window['hr_paths'][:22], lr_paths=window['lr_paths'][:22],
                          crop_xywh=window['hr_crop_xywh'], shape_hw=[256,256]))
        video = next(v for v in videos if v['sequence']==window['sequence'])
        hr_paths, lr_paths = [], []
        for frame in video['frames_metadata'][:22]:
            original = Image.open(DATA/frame['hr']).convert('RGB')
            w, h = original.size; cw, ch = w//32*32, h//32*32
            x, y = ((w-cw)//2)//4*4, ((h-ch)//2)//4*4
            high = original.crop((x,y,x+cw,y+ch))
            low = high.resize((cw//4,ch//4), Image.Resampling.BICUBIC)
            folder = DATA/'h3_native480'/window['split']/window['sequence']
            hp, lp = folder/'hr'/f"{frame['frame_index']:05d}.png", folder/'lr_x4'/f"{frame['frame_index']:05d}.png"
            hp.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
            high.save(hp);low.save(lp)
            hr_paths.append(str(hp.relative_to(DATA)));lr_paths.append(str(lp.relative_to(DATA)))
        cases.append(dict(sequence=window['sequence'], split=window['split'], resolution='native480',
                          hr_paths=hr_paths, lr_paths=lr_paths, crop_xywh=[x,y,cw,ch],shape_hw=[ch,cw]))
    for case in cases:
        case['id']=case['sequence']+'_'+case['resolution']
        case['observed_lr_sha256']=[digest(DATA/p) for p in case['lr_paths']]
    result={'protocol':'8 source-disjoint clips, first22 frames, two spatial contexts; no model fitting',
            'degradation':'Pillow uint8 bicubic x4; fixed center mod32 crop at native480, no HR upsampling',
            'frames':22,'fps':24,'cases':cases}
    (OUT/'input_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'prepared_cases':len(cases),'native_shapes':[c['shape_hw'] for c in cases if c['resolution']=='native480']}),flush=True)


def inference(args):
    # Prevent Comfy's global CLI from consuming this experiment's arguments.
    sys.argv=[sys.argv[0]];sys.path.insert(0,str(args.comfy))
    import comfy.sd
    import comfy.utils
    torch.cuda.set_per_process_memory_fraction(.4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state,metadata=comfy.utils.load_torch_file(str(args.weights),return_metadata=True)
    vae=comfy.sd.VAE(sd=state,device=torch.device('cuda'),dtype=torch.float16,metadata=metadata)
    del state
    cases=json.loads((OUT/'input_manifest.json').read_text())['cases']
    with torch.inference_mode():
        for case in cases:
            folder=OUT/case['id'];folder.mkdir(exist_ok=True)
            done=folder/'inference.json'
            if done.exists():
                prior=json.loads(done.read_text())
                assert prior['observed_lr_sha256']==case['observed_lr_sha256']
                assert all((folder/f'{i:04d}.png').exists() for i in range(22))
                continue
            observed=torch.stack([load(DATA/p) for p in case['lr_paths']])
            assert [digest(DATA/p) for p in case['lr_paths']]==case['observed_lr_sha256']
            up=F.interpolate(observed,size=case['shape_hw'],mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            pixels=up.permute(0,2,3,1).contiguous()
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            latent=vae.encode(pixels);torch.cuda.synchronize();encoded=time.perf_counter()
            decoded=vae.decode(latent).float().cpu();torch.cuda.synchronize();end=time.perf_counter()
            if decoded.ndim==5:decoded=decoded[0]
            assert list(decoded.shape)==list(pixels.shape) and torch.isfinite(decoded).all()
            for i,value in enumerate(decoded):save(value.permute(2,0,1),folder/f'{i:04d}.png')
            result={**case,'status':'complete','method':'Frozen official H3 FP16 VAE E(upLR) then D; no DiT',
                    'input_hr_read':False,'latest_observed_frame':21,'latent_shape':list(latent.shape),
                    'encode_seconds':encoded-start,'decode_seconds':end-encoded,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'weights_sha256':'7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522',
                    'output_sha256':[digest(folder/f'{i:04d}.png') for i in range(22)]}
            done.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['id','status','latent_shape','encode_seconds','decode_seconds','peak_allocated_bytes']}),flush=True)
            del observed,up,pixels,latent,decoded


def evaluate():
    import lpips
    from evaluate import ssim_rgb, comparison_panel
    model=lpips.LPIPS(net='alex',version='0.1',verbose=False).eval().requires_grad_(False)
    cases=json.loads((OUT/'input_manifest.json').read_text())['cases'];rows=[];summaries=[]
    (OUT/'projection_audit.json').write_text(json.dumps(audit(),indent=2))
    with torch.inference_mode():
        for case in cases:
            folder=OUT/case['id']; cached=folder/'evaluation.json'
            if cached.exists():
                summaries.append(json.loads(cached.read_text()));continue
            assert json.loads((folder/'inference.json').read_text())['observed_lr_sha256']==case['observed_lr_sha256']
            lr=torch.stack([load(DATA/p) for p in case['lr_paths']]);hr=torch.stack([load(DATA/p) for p in case['hr_paths']])
            bicubic=F.interpolate(lr,size=case['shape_hw'],mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            # Both pre-DC readouts have the same 8-bit storage precision.
            bicubic=(bicubic*255).round()/255
            decoded=torch.stack([load(folder/f'{i:04d}.png') for i in range(22)])
            op=BicubicConsistency(tuple(hr.shape[-2:]),tuple(lr.shape[-2:]))
            methods={'bicubic':bicubic,'h3_vae':decoded};projections={}
            for name,pred in list(methods.items()):
                start=time.perf_counter();out,residual=op.project(pred,lr,iterations=25)
                projections[name]={'seconds':time.perf_counter()-start,'max_float_lr_rmse':float(residual.max())}
                methods[name+'_dc']=out
            localrows=[];summary={}
            for name,pred in methods.items():
                distances=[]
                for start in range(0,22,2):distances.extend(model(2*pred[start:start+2]-1,2*hr[start:start+2]-1).flatten().tolist())
                for i,value in enumerate(pred):
                    low=Image.fromarray(np.uint8(value.permute(1,2,0).numpy().clip(0,1)*255+.5)).resize((lr.shape[-1],lr.shape[-2]),Image.Resampling.BICUBIC)
                    low=torch.from_numpy(np.asarray(low).copy()).permute(2,0,1).float()/255
                    localrows.append({'method':name,'frame':i,'psnr':float(-10*torch.log10((value-hr[i]).square().mean().clamp_min(1e-12))),
                                      'ssim':ssim_rgb(value.numpy(),hr[i].numpy()),'lpips':distances[i],
                                      'lr_pillow_rmse':float((low-lr[i]).square().mean().sqrt())})
                summary[name]={k:float(np.mean([r[k] for r in localrows if r['method']==name])) for k in ['psnr','ssim','lpips','lr_pillow_rmse']}
            result={'case':case['id'],'sequence':case['sequence'],'split':case['split'],'resolution':case['resolution'],'shape_hw':case['shape_hw'],
                    'methods':summary,'projection':projections,'projection_iterations':25,'selection':'Fixed protocol; no test-dependent selection'}
            cached.write_text(json.dumps(result,indent=2));(folder/'per_frame.json').write_text(json.dumps(localrows,indent=2))
            comparison_panel(hr,methods,['bicubic_dc','h3_vae_dc'],[0,11,21],folder/'comparison_projected.png')
            summaries.append(result);print(json.dumps(result),flush=True)
    (OUT/'summary.json').write_text(json.dumps({'cases':summaries,'limits':['Only8 source videos','No training or persistent state','22-frame blocks, not zero-lookahead','Native480 and256 are different fixed fields of view, not an isolated resolution ablation']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','infer','evaluate'])
    parser.add_argument('--comfy',type=Path);parser.add_argument('--weights',type=Path);args=parser.parse_args()
    torch.set_num_threads(4)
    if args.phase=='prepare':prepare()
    elif args.phase=='infer':inference(args)
    else:evaluate()
