"""Actual finite-byte classical controls for the pre-cue natural image task.

Separate preparation, encoding, code-only reading and GT evaluation processes.
No training, H3 execution, future-conditioned writer, or per-image GT tuning.
"""
import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, features
from scipy.fft import dctn, idctn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'results/h3_core/decoder_side_natural_v2'
OUT = ROOT/'results/decoder_side_classical_v1'
BUDGET = 896


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
def load(path): return torch.load(path, map_location='cpu', weights_only=False)


def indices(high=False):
    # Common fixed frequency order, no coefficient positions sent per image.
    order = [(u+v, max(u,v), u, v, c) for u in range(64) for v in range(64)
             for c in range(3) if not high or max(u,v)>=32]
    order.sort()
    return [(c,u,v) for _,_,u,v,c in order[:BUDGET//2]]


def prepare():
    if OUT.exists(): raise FileExistsError('Use a new version instead of overwriting')
    OUT.mkdir(parents=True)
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'payload_budget': BUDGET,
        'sources': 'All fixed32TRAIN/8VAL; natural primary only; no auxiliary worlds',
        'primary': 'VAL8 source mean of four future phase PSNR; TRAIN descriptive',
        'writer': 'Only observed past uint8 RGB64x64; future phase and pixels unavailable',
        'reader': 'Only serialized payload, one arrived future uint8 RGB32x32, known physical camera and shared format',
        'codecs': {
            'dct_low': '448 fixed zigzag-like (u+v,max,u,v,RGB) orthonormal DCT-II coefficients, little-endian float16 =896bytes',
            'dct_high': '448 coefficients ordered as above restricted max(u,v)>=32; rest supplied by the actual future-only base; 896bytes',
            'webp': 'Pillow WebP method6, integer quality0..100 exhaustively encoded; highest quality whose entire file<=896bytes; no GT',
            'uncompressed': 'Observed uint8 near RGB bytes=12288, capacity upper reference, not rate matched'},
        'future_only': 'Upsample arrived far32 to128, inverse its known HR0/1 shift with border padding, reflect-pad4, then optional physical projection',
        'physical_reader': 'Exactly25 Dykstra iterations on full136, using actual far Ay/Ax and right inverses and [0,1] box, then crop4:132. No GT/phase tuning.',
        'dct_high_reader': 'Project future-only base into near camera; replace its selected DCT coefficients with transmitted ones; bicubic-to128 and reflect-pad4, then future P25',
        'metrics': 'PSNR and float physical future LR residual; raw/futureP25 independently named. No learned-arm postprocessor is changed.',
        'scope': 'Classical rate-utility controls, no newH3 E/D/DiT or GPU or optimizer updates',
        'source_sha256': sha(__file__), 'manifest_sha256': sha(DATA/'manifest.json'),
        'operators_sha256': sha(DATA/'operators.pt')}
    write(OUT/'LOCKED_PROTOCOL.json', protocol)
    manifest=json.loads((DATA/'manifest.json').read_text()); records=[]
    for record in manifest['records']:
        path=ROOT/record['file']; data=load(path)['primary_natural']
        directory=OUT/'inputs'/record['id']; directory.mkdir(parents=True)
        torch.save(data['writer_input']['past_lr_uint8'], directory/'past_only.pt')
        torch.save(data['reader_input'], directory/'future_only.pt')
        torch.save(data['supervision']['target_RGB'], directory/'GT_evaluation_only.pt')
        records.append({'id':record['id'], 'split':record['split'], 'original_data_sha256':sha(path),
                        'past_sha256':sha(directory/'past_only.pt'), 'future_sha256':sha(directory/'future_only.pt')})
    torch.save(load(DATA/'operators.pt'), OUT/'operators.pt')
    write(OUT/'manifest.json', {'records':records})
    print(json.dumps({'prepared':len(records),'protocol_sha256':sha(OUT/'LOCKED_PROTOCOL.json')}),flush=True)


def check():
    p=json.loads((OUT/'LOCKED_PROTOCOL.json').read_text())
    assert p['source_sha256']==sha(__file__)
    return json.loads((OUT/'manifest.json').read_text())['records']


def dct_pack(past, high):
    coefficients=dctn(past.numpy().astype(np.float64)/255, axes=(-2,-1), type=2, norm='ortho')
    chosen=np.asarray([coefficients[c,u,v] for c,u,v in indices(high)],dtype='<f2')
    assert np.isfinite(chosen).all()
    return chosen.tobytes()


def dct_unpack(payload, high, base=None):
    assert len(payload)==BUDGET
    coefficients=np.zeros((3,64,64),np.float64) if base is None else dctn(base.numpy().astype(np.float64),axes=(-2,-1),type=2,norm='ortho')
    values=np.frombuffer(payload,dtype='<f2')
    for value,(c,u,v) in zip(values,indices(high)): coefficients[c,u,v]=float(value)
    return torch.from_numpy(idctn(coefficients,axes=(-2,-1),type=2,norm='ortho')).float()


def encode():
    records=check(); assert features.check('webp')
    if (OUT/'encode.json').exists(): raise FileExistsError('Existing encoding')
    started=time.perf_counter(); rows=[]
    for r in records:
        source=r['id']; inp=OUT/'inputs'/source/'past_only.pt'
        assert sha(inp)==r['past_sha256']; past=load(inp)
        assert past.shape==(3,64,64) and past.dtype==torch.uint8
        directory=OUT/'payloads'/source; directory.mkdir(parents=True)
        t=time.perf_counter()
        for name,high in [('dct_low',False),('dct_high',True)]:
            (directory/f'{name}.bin').write_bytes(dct_pack(past,high))
        (directory/'uncompressed.bin').write_bytes(past.contiguous().numpy().tobytes())
        dct_seconds=time.perf_counter()-t
        image=Image.fromarray(past.permute(1,2,0).numpy()); candidates=[]; t=time.perf_counter()
        for q in range(101):
            buff=io.BytesIO();image.save(buff,format='WEBP',quality=q,method=6)
            payload=buff.getvalue()
            if len(payload)<=BUDGET:candidates.append((q,payload))
        if not candidates:raise RuntimeError(f'No eligible WebP file for {source}')
        quality,payload=candidates[-1];(directory/'webp.webp').write_bytes(payload)
        rows.append({'source':source,'split':r['split'],'webp_quality':quality,
                     'webp_encode_trials':101,'webp_seconds':time.perf_counter()-t,'other_encode_seconds':dct_seconds,
                     'bytes':{p.name:p.stat().st_size for p in directory.iterdir()},
                     'sha256':{p.name:sha(p) for p in directory.iterdir()}})
        print(json.dumps({'phase':'encode','source':source,'webp_quality':quality,'bytes':len(payload)}),flush=True)
    write(OUT/'encode.json',{'rows':rows,'seconds':time.perf_counter()-started,'future_or_GT_loaded':False})


def expand(near):
    return F.pad(F.interpolate(near[None],size=(128,128),mode='bicubic',align_corners=False,antialias=True),
                 (4,4,4,4),mode='reflect')[0].double()


def project(x,y,op):
    x=x.double(); y=y.double(); p=torch.zeros_like(x); q=torch.zeros_like(x)
    for _ in range(25):
        shifted=x+p
        affine=shifted+op['right_y']@(y-op['Ay']@shifted@op['Ax'].T)@op['right_x'].T
        p=shifted-affine; shifted=affine+q; x=shifted.clamp(0,1); q=shifted-x
    return x


def future_base(y,phase):
    x=F.interpolate(y[None],(128,128),mode='bicubic',align_corners=False,antialias=True)
    yy,xx=torch.meshgrid(torch.arange(128),torch.arange(128),indexing='ij')
    dx,dy=2*float(phase[0]),2*float(phase[1])
    coords=torch.stack(((xx-dx+.5)*2/128-1,(yy-dy+.5)*2/128-1),-1).float()[None]
    x=F.grid_sample(x,coords,mode='bilinear',padding_mode='border',align_corners=False)
    return F.pad(x,(4,4,4,4),mode='reflect')[0].double()


@torch.no_grad()
def decode():
    records=check();ops=load(OUT/'operators.pt')
    if (OUT/'decode.json').exists():raise FileExistsError('Existing decode')
    encoded=json.loads((OUT/'encode.json').read_text())['rows'];started=time.perf_counter();rows=[]
    for r,e in zip(records,encoded):
        source=r['id'];assert source==e['source']
        context_path=OUT/'inputs'/source/'future_only.pt';assert sha(context_path)==r['future_sha256']
        context=load(context_path); directory=OUT/'payloads'/source
        for name,digest in e['sha256'].items():assert sha(directory/name)==digest
        low=dct_unpack((directory/'dct_low.bin').read_bytes(),False)
        raw=torch.from_numpy(np.frombuffer((directory/'uncompressed.bin').read_bytes(),np.uint8).copy().reshape(3,64,64)).float()/255
        with Image.open(directory/'webp.webp') as im:webp=torch.from_numpy(np.asarray(im.convert('RGB')).copy()).permute(2,0,1).float()/255
        stored={'DCT_low_raw':[],'DCT_low_futureP25':[],'DCT_high_futureP25':[],
                'WebP_raw':[],'WebP_futureP25':[],'raw_past_raw':[],'raw_past_futureP25':[],
                'future_only_raw':[],'future_only_P25':[]}
        lr_residuals={k:[] for k in stored}; t=time.perf_counter()
        for phase_id in range(4):
            y=context['future_lr_uint8'][phase_id].float()/255
            phase=context['future_phase_nearLR'][phase_id]
            op=ops['future'][phase_id]; initial=future_base(y,phase)
            near_est=ops['near']['Ay']@initial@ops['near']['Ax'].T
            high=dct_unpack((directory/'dct_high.bin').read_bytes(),True,near_est)
            outputs={}
            for label,near in [('DCT_low',low),('WebP',webp),('raw_past',raw)]:
                full=expand(near).clamp(0,1)
                outputs[label+'_raw']=full
                outputs[label+'_futureP25']=project(full,y,op)
            outputs['DCT_high_futureP25']=project(expand(high),y,op)
            outputs['future_only_raw']=initial.clamp(0,1)
            outputs['future_only_P25']=project(initial,y,op)
            for name,value in outputs.items():
                stored[name].append(value[:,4:132,4:132].float())
                lr_residuals[name].append(float((op['Ay']@value@op['Ax'].T-y.double()).square().mean().sqrt()))
        outdir=OUT/'decoded';outdir.mkdir(exist_ok=True)
        target=outdir/f'{source}.pt';torch.save({k:torch.stack(v) for k,v in stored.items()},target)
        rows.append({'source':source,'split':r['split'],'seconds':time.perf_counter()-t,
                     'output_sha256':sha(target),'future_lr_rmse':lr_residuals})
        print(json.dumps({'phase':'code-only decode','source':source}),flush=True)
    write(OUT/'decode.json',{'rows':rows,'seconds':time.perf_counter()-started,
                           'past_or_GT_loaded':False,'own_current_future_count_per_query':1})


@torch.no_grad()
def evaluate():
    records=check();decoded=json.loads((OUT/'decode.json').read_text());rows=[]
    if (OUT/'evaluation.json').exists():raise FileExistsError('Existing evaluation')
    for r,d in zip(records,decoded['rows']):
        source=r['id']; path=OUT/'decoded'/f'{source}.pt';assert sha(path)==d['output_sha256']
        outputs=load(path);gt=load(OUT/'inputs'/source/'GT_evaluation_only.pt').float()
        for name,value in outputs.items():
            psnr=(-10*(value-gt).square().mean((1,2,3)).clamp_min(1e-12).log10()).tolist()
            for q,number in enumerate(psnr):rows.append({'source':source,'split':r['split'],'method':name,'phase':q,
                                                         'psnr':number,'future_lr_rmse':d['future_lr_rmse'][name][q]})
    methods=sorted({r['method'] for r in rows});means={}
    for split in ['train','val']:
        means[split]={m:float(np.mean([r['psnr'] for r in rows if r['split']==split and r['method']==m])) for m in methods}
    write(OUT/'evaluation.json',{'rows':rows,'source_equal_four_phase_means':means,
                                'E':0,'D':0,'DiT':0,'GPU':0,'optimizer_updates':0})
    print(json.dumps(means),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','encode','decode','evaluate'])
    args=p.parse_args();globals()[args.phase]()
