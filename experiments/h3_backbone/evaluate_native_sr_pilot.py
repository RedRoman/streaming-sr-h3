"""Fixed validation readout characterization, no training or selection."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
import lpips

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from pixel_consistency import BicubicConsistency
spec=importlib.util.spec_from_file_location('sr_fixed_eval',ROOT/'experiments/real_video_evaluation/evaluate.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2,0,1).float()/255

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--val',type=Path,required=True);p.add_argument('--data',type=Path,default=ROOT/'data/real_video');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    torch.set_num_threads(4);out=a.run/'evaluation';out.mkdir(exist_ok=True)
    cases=json.loads(a.val.read_text())['cases'];assert len(cases)==4 and all(c['split']=='val' for c in cases)
    net=Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth';cal=Path(lpips.__file__).parent/'weights/v0.1/alex.pth';assert net.is_file() and cal.is_file(),'Use existing validatedweights; do not silently download'
    model=lpips.LPIPS(net='alex',version='0.1',verbose=False).eval().requires_grad_(False).to(a.device)
    weights={str(path):sha(path) for path in [net,cal]};records=[];start=time.perf_counter()
    for case in cases:
        target=out/case['id'];target.mkdir(exist_ok=True)
        hashes=[sha(a.data/p) for p in case['lr_paths']];assert hashes==case['observed_lr_sha256']
        low=torch.stack([load(a.data/p) for p in case['lr_paths']]);high=torch.stack([load(a.data/p) for p in case['hr_paths']])
        methods={'bicubic':F.interpolate(low,size=case['shape_hw'],mode='bicubic',align_corners=False,antialias=True).clamp(0,1)}
        for mode in ['vae_only','unadapted','lora_only','guide_only','joint']:
            folder=a.run/'readouts'/mode/case['id'];decoded=json.loads((folder/'decode_result.json').read_text());assert decoded['status']=='complete'
            if mode!='vae_only':
                sampled=json.loads((folder/'sample_result.json').read_text());assert sampled['observed_lr_sha256']==hashes and not sampled['HR_condition']
            methods[mode]=torch.stack([load(folder/f'output_{i:02d}.png') for i in range(22)])
        op=BicubicConsistency(tuple(high.shape[-2:]),tuple(low.shape[-2:]))
        rows=[]
        for label,image in methods.items():
            for dc in [False,True]:
                name=label+('_dc25' if dc else '');began=time.perf_counter()
                if dc:value,residual=op.project(image,low,iterations=25)
                else:value=image;residual=(op.down(value.to(op.w))-low.to(op.w)).square().mean((-1,-2,-3)).sqrt()
                mse=(value-high).square().mean((1,2,3));psnr=(-10*mse.clamp_min(1e-12).log10()).tolist()
                similarity=[shared.ssim_rgb(x.numpy(),y.numpy()) for x,y in zip(value,high)]
                with torch.no_grad():perceptual=[float(model(x[None].to(a.device)*2-1,y[None].to(a.device)*2-1)) for x,y in zip(value,high)]
                row={'case':case['id'],'sequence':case['sequence'],'resolution':case['resolution'],'split':'val','method':name,
                     'mean_frame_psnr_db':float(np.mean(psnr)),'pooled_psnr_db':float(-10*mse.mean().clamp_min(1e-12).log10()),
                     'mean_ssim_rgb':float(np.mean(similarity)),'mean_lpips_alex_v01':float(np.mean(perceptual)),
                     'max_lr_float_operator_rmse':float(residual.max()),'mean_lr_float_operator_rmse':float(residual.mean()),
                     'per_frame_psnr_db':psnr,'per_frame_ssim':similarity,'per_frame_lpips':perceptual,'seconds_evaluation':time.perf_counter()-began}
                rows.append(row);print(json.dumps({'event':'pilot_metric_complete',**{k:v for k,v in row.items() if not k.startswith('per_frame')}}),flush=True)
        (target/'metrics.json').write_text(json.dumps({'scope':'Fixedtwo-sourcevalidationonly; final400weights, no checkpoint/hyperparameterselection; DC25 outputpostprocessing never newevidence','rows':rows,'observed_lr_sha256':hashes,'lpips_weights':weights},indent=2)+'\n')
        records.extend(rows)
        indices=[0,10,21];labels=['bicubic','vae_only','unadapted','lora_only','guide_only','joint'];h,w=case['shape_hw'];panel=Image.new('RGB',(w*7,(h+26)*3),'white');draw=ImageDraw.Draw(panel)
        for ri,i in enumerate(indices):
            for ci,(label,tensor) in enumerate([('HR evaluation',high)]+[(l,methods[l]) for l in labels]):
                draw.text((ci*w+5,ri*(h+26)+5),f'{label}, t={i}',fill='black');panel.paste(Image.fromarray(np.uint8(tensor[i].permute(1,2,0).numpy().clip(0,1)*255+.5)),(ci*w,ri*(h+26)+26))
        panel.save(target/'fixed_frame_panel_raw.png')
    aggregate=[]
    for resolution in ['256','native480']:
        for method in sorted({r['method'] for r in records}):
            rows=[r for r in records if r['resolution']==resolution and r['method']==method]
            aggregate.append({'resolution':resolution,'method':method,'source_count':len(rows),**{key:float(np.mean([r[key] for r in rows])) for key in ['mean_frame_psnr_db','mean_ssim_rgb','mean_lpips_alex_v01','mean_lr_float_operator_rmse']}})
    result={'status':'complete','scope':'Two held-outvalidation source videos x2spatialcontexts; fixedfinal400, smallpilot nottestbenchmark/memorymethod; rawandmatchedDC25 sharedcontrols',
            'aggregate':aggregate,'rows':records,'lpips_weights':weights,'protocol_sha256':sha(a.run/'protocol.json'),'seconds':time.perf_counter()-start}
    (out/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'complete','aggregate':aggregate,'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
