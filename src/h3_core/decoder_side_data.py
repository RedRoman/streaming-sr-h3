"""CPU-only natural near/far data and counterfactual observability audit.

Primary natural photographs are unchanged. Auxiliary nullspace twins use an
explicit full136x136 physical field, a mechanical feasible line step, no HR
clipping, and are only saved as training examples on TRAIN. VAL twins are
discarded after a separately labelled mechanical audit. No H3 call/training.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

PHASES=[(0.,0.),(.5,0.),(0.,.5),(.5,.5)]  # Near-LR pixels; HR offsets0/1.


class Sensor:
    def __init__(self,factor,dx=0,dy=0):
        self.factor=factor;self.dx=dx;self.dy=dy
        basis=torch.eye(128,dtype=torch.float64)[:,None]
        coord=torch.arange(-2,3,dtype=torch.float64);kernel=(-coord.square()/(2*.6**2)).exp();kernel/=kernel.sum()
        blurred=F.conv1d(F.pad(basis,(2,2),mode='reflect'),kernel[None,None])
        core=F.interpolate(blurred[:,None],size=(1,128//factor),mode='bicubic',align_corners=False,antialias=True)[:,0,0].T.contiguous()
        def axis(shift):
            crop=torch.zeros((128,136),dtype=torch.float64);crop[torch.arange(128),torch.arange(128)+4+shift]=1
            a=core@crop;right=a.T@torch.linalg.solve(a@a.T,torch.eye(len(a),dtype=a.dtype));return a,right
        self.ax,self.rx=axis(dx);self.ay,self.ry=axis(dy)
    def observe(self,x):return self.ay@x@self.ax.T
    def null(self,x):return x-self.ry@self.observe(x)@self.rx.T
    def direct(self,x):
        x=x[...,4+self.dy:132+self.dy,4+self.dx:132+self.dx]
        coord=torch.arange(-2,3,dtype=x.dtype);k=(-coord.square()/(2*.6**2)).exp();k/=k.sum();kernel=(k[:,None]*k[None,:])[None,None].repeat(3,1,1,1)
        blurred=F.conv2d(F.pad(x,(2,2,2,2),mode='reflect'),kernel,groups=3)
        return F.interpolate(blurred,size=(128//self.factor,128//self.factor),mode='bicubic',align_corners=False,antialias=True)
    def payload(self):return dict(Ay=self.ay,Ax=self.ax,right_y=self.ry,right_x=self.rx,factor=self.factor,HR_crop_origin=[4+self.dx,4+self.dy])


def natural(path):
    image=Image.open(path).convert('RGB');w,h=image.size;assert min(w,h)>=136
    array=np.asarray(image.crop(((w-136)//2,(h-136)//2,(w-136)//2+136,(h-136)//2+136))).copy()
    return torch.from_numpy(array).permute(2,0,1)[None].double()/255


def feasible_twin(x,direction,fraction=.5):
    # Pick the larger of the two feasible signed natural directions, by pixel
    # range only. No restoration metric or held-source threshold is optimized.
    maximum=float(direction.abs().max())
    if maximum<1e-15:return x.clone(),0.,0,0.
    d=direction/maximum
    def bound(v):
        pos=v>1e-15;neg=v< -1e-15;choices=[]
        if bool(pos.any()):choices.append(((1-x[pos])/v[pos]).min())
        if bool(neg.any()):choices.append((x[neg]/(-v[neg])).min())
        return float(torch.stack(choices).min()) if choices else 0.
    bp,bn=bound(d),bound(-d);sign=1 if bp>=bn else -1;limit=max(bp,bn);alpha=math.nextafter(fraction*limit,0.) if limit else 0.;y=x+alpha*sign*d
    assert float(y.min())>=-1e-12 and float(y.max())<=1+1e-12
    return y,alpha,sign,limit


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--noise',type=float,default=.003);p.add_argument('--train-aux-interior',action='store_true');p.add_argument('--alpha-fraction',type=float,choices=[.5,1.],default=.5);p.add_argument('--primary-reference');a=p.parse_args()
    started=time.monotonic();torch.set_num_threads(4);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads(Path(a.manifest).read_text());sources=[s for s in source_manifest['items'] if s['split'] in ['train','val']];assert sum(s['split']=='train' for s in sources)==32 and sum(s['split']=='val' for s in sources)==8
    near=Sensor(2);future=[Sensor(4,int(2*x),int(2*y)) for x,y in PHASES]
    probe=torch.rand((1,3,136,136),generator=torch.Generator().manual_seed(817),dtype=torch.float64)
    errors=[float((s.observe(probe)-s.direct(probe)).abs().max()) for s in [near,*future]];assert max(errors)<1e-12
    torch.save(dict(near=near.payload(),future=[s.payload() for s in future],phases_nearLR=PHASES),out/'operators.pt')
    audits=[];records=[]
    for si,source in enumerate(sources):
        path=Path(source['source']);donor=Path(source['donor_source']);sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==source['source_sha256'];assert hashlib.sha256(donor.read_bytes()).hexdigest()==source['donor_sha256']
        x=natural(path);v=natural(donor);smooth=F.avg_pool2d(F.pad(v,(1,1,1,1),mode='reflect'),3,stride=1);direction=v-smooth;window=torch.zeros_like(direction);window[...,4:132,4:132]=1;direction*=window
        rng=torch.Generator().manual_seed(81300+si);past_noise=torch.randn((1,3,64,64),generator=rng,dtype=torch.float64)*a.noise;future_noise=[torch.randn((1,3,32,32),generator=rng,dtype=torch.float64)*a.noise for _ in future]
        def payload(field):
            def measurement(y,noise):return ((y+noise).clamp(0,1)*255).round().byte()[0]
            return dict(writer_input=dict(past_lr_uint8=measurement(near.observe(field),past_noise)),
                        reader_input=dict(future_lr_uint8=torch.stack([measurement(s.observe(field),n) for s,n in zip(future,future_noise)]),future_phase_nearLR=torch.tensor(PHASES)),
                        supervision=dict(target_RGB=field[0,:,4:132,4:132].float(),full_physical_field=field[0].float()))
        primary=payload(x);auxiliary=[];pair_records=[]
        if a.primary_reference:
            original=torch.load(Path(a.primary_reference)/source['split']/(source['id']+'.pt'),map_location='cpu',weights_only=False)['primary_natural']
            for branch in ['writer_input','reader_input','supervision']:
                assert all(torch.equal(v,original[branch][key]) for key,v in primary[branch].items()),'Primarynaturalchanged'
        interior=a.train_aux_interior and source['split']=='train';auxbase=.05+.90*x if interior else x
        baseline=payload(auxbase)
        if interior:auxiliary.append(dict(kind='fixed_range_interior_base',**baseline))
        for q,s in enumerate(future):
            twin,alpha,sign,limit=feasible_twin(auxbase,s.null(direction),a.alpha_fraction);other=payload(twin);before=near.observe(twin)-near.observe(auxbase);after=s.observe(twin)-s.observe(auxbase)
            audit=dict(source=source['id'],split=source['split'],kind='future_null_collision',auxiliary_base='0.05+0.90X TRAINonly' if interior else 'originalnatural',phase=q,phase_nearLR=PHASES[q],alpha=alpha,sign=sign,range_limit=limit,future_noiseless_maxabs=float(after.abs().max()),past_noiseless_RMS=float(before.square().mean().sqrt()),past_gt_ROI_difference_RMS=float((twin-auxbase)[...,4:132,4:132].square().mean().sqrt()),future_quantized_different_bytes=int((other['reader_input']['future_lr_uint8'][q]!=baseline['reader_input']['future_lr_uint8'][q]).sum()),past_quantized_different_bytes=int((other['writer_input']['past_lr_uint8']!=baseline['writer_input']['past_lr_uint8']).sum()),above_noise=float(before.square().mean().sqrt())>a.noise,above_one_LSB=float(before.square().mean().sqrt())>1/255,degenerate_zero_alpha=alpha==0)
            assert audit['future_noiseless_maxabs']<1e-12 and audit['future_quantized_different_bytes']==0
            audits.append(audit);pair_records.append(audit)
            if source['split']=='train':auxiliary.append(dict(kind='future_null_collision',construction_phase=q,**other))
        twin,alpha,sign,limit=feasible_twin(auxbase,near.null(direction),a.alpha_fraction);other=payload(twin);pastdiff=near.observe(twin)-near.observe(auxbase)
        for q,s in enumerate(future):
            fd=s.observe(twin)-s.observe(auxbase);audit=dict(source=source['id'],split=source['split'],kind='past_null_reveal',auxiliary_base='0.05+0.90X TRAINonly' if interior else 'originalnatural',phase=q,phase_nearLR=PHASES[q],alpha=alpha,sign=sign,range_limit=limit,past_noiseless_maxabs=float(pastdiff.abs().max()),future_noiseless_RMS=float(fd.square().mean().sqrt()),past_gt_ROI_difference_RMS=float((twin-auxbase)[...,4:132,4:132].square().mean().sqrt()),past_quantized_different_bytes=int((other['writer_input']['past_lr_uint8']!=baseline['writer_input']['past_lr_uint8']).sum()),future_quantized_different_bytes=int((other['reader_input']['future_lr_uint8'][q]!=baseline['reader_input']['future_lr_uint8'][q]).sum()),above_noise=float(fd.square().mean().sqrt())>a.noise,above_one_LSB=float(fd.square().mean().sqrt())>1/255,degenerate_zero_alpha=alpha==0)
            assert audit['past_noiseless_maxabs']<1e-12 and audit['past_quantized_different_bytes']==0;audits.append(audit);pair_records.append(audit)
        if source['split']=='train':auxiliary.append(dict(kind='past_null_reveal',**other))
        # Every auxiliary physical world is paired with all four future queries;
        # construction phase is a training label, never a writer feature.
        saved=dict(source=source,primary_natural=primary,train_auxiliary=auxiliary,scope='Pixel/observed-information diagnostic, not H3 core training; encoder may read only writer_input; no sourceID/template lookup.')
        dest=out/source['split'];dest.mkdir(exist_ok=True);torch.save(saved,dest/(source['id']+'.pt'));records.append(dict(id=source['id'],split=source['split'],source=str(path),source_sha256=sha,file=str(dest/(source['id']+'.pt')),primary_worlds=1,TRAIN_auxiliary_worlds=len(auxiliary),future_queries_per_world=4))
        print(json.dumps(dict(event='decoder_side_CPU_data',source=source['id'],train_aux_worlds=len(auxiliary),nondegenerate=sum(not r['degenerate_zero_alpha'] for r in pair_records))),flush=True)
    report=dict(status='complete',audit_rows=audits,source_count=40,operator_direct_maxabs=errors,actual_E_calls=0,actual_D_calls=0,new_DiT_calls=0,GPU_used=False,optimizer_steps=0,seconds=time.monotonic()-started,alpha_rule=f'{a.alpha_fraction} times larger feasible signed naturalHF nullspace direction line bound, nextafter toward0. No clipping or manual gain search; zero bound retained.',TRAIN_auxiliary_base='Fixed0.05+0.90X only' if a.train_aux_interior else 'Original natural',primary_reference_bitwise_checked=a.primary_reference,sensor='FP64 full136field→128 crop atHRorigin4+(phase*2), reflectGaussian.6/5tap+TorchAA bicubic; quantizeduint8 withsigma.003. Primary natural photos unchanged. Counterfactual worlds use common noise realisations to isolate physical ambiguity, not extra independent measurements.',VAL_scope='Only natural world is stored as eligibleVAL data. MechanicalVAL counterfactual metrics use original natural fields; no VAL auxiliary images or pair labels are stored for training.',future_distribution='All4future phases per physical world, including each TRAIN auxiliary world. Writer sees only past_lr_uint8 and no futurephase/operator, labels, sourceID or camera answer.',scope='Controlled natural-photo measurement/codec data, not real video, naturalH3 identity diversity, or learned H3 method efficacy.')
    (out/'ambiguity_audit.json').write_text(json.dumps(report,indent=2));(out/'manifest.json').write_text(json.dumps(dict(records=records,phases_nearLR=PHASES,near_factor=2,future_factor=4,physical_field=136,target_ROI=128,noise=a.noise,source_manifest=a.manifest,past_writer_pixels=64,future_reader_pixels=32,training_updates=0,H3_call_count=0,scope=report['scope']),indent=2));print(json.dumps(dict(status='complete',seconds=report['seconds'],sources=len(records))),flush=True)


if __name__=='__main__':main()
