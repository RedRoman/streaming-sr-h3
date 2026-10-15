"""Fixed natural-source two-cue data and native-H3 guide preparation.

The restoration payload is physically consistent LR/geometry only. Ground
truth is saved separately. Native22 copies are canonical-memory contexts,
never22 distinct observed frames or future physical measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint
from h3_interface import load_frozen_vae
from observation_quotient import ObservationQuotient


VIEWS={'far':(0.,0.,256.,256.),'A':(32.,32.,160.,160.),'B':(96.,96.,224.,224.)}
SUBSETS={'empty':[],'A':['A'],'B':['B'],'AB':['A','B']}


class SeparableCamera:
    def __init__(self,view,q):
        def axis(vlo,vhi):
            u=(torch.arange(256,device=q.matrix.device,dtype=torch.float64)+.5)/256*(vhi-vlo)+vlo;coordinate=u-.5;left=coordinate.floor().long();fraction=coordinate-left
            warp=torch.zeros(256,256,device=u.device,dtype=u.dtype);rows=torch.arange(256,device=u.device)
            for index,value in [(left,1-fraction),(left+1,fraction)]:
                valid=(index>=0)&(index<256);warp[rows[valid],index[valid]]=value[valid]
            return q.matrix@warp
        self.wy=axis(view[1],view[3]);self.wx=axis(view[0],view[2])
    def observe(self,rgb):return (self.wy@rgb.double()@self.wx.T).to(rgb.dtype)
    def adjoint(self,lr):return (self.wy.T@lr.double()@self.wx).to(lr.dtype)


def classical_map(reference,observations,cameras,subset,q):
    if not subset:return reference.clone(),{'iterations':0,'normal_equation_residual':0.}
    # Hard first-LR quotient, ridge RGB prior std.05 / likelihood sigma.005.
    ridge=(.005/.05)**2
    def forward(value):return ridge*value+sum(q.null(cameras[k].adjoint(cameras[k].observe(q.null(value)))) for k in subset)
    rhs=sum(q.null(cameras[k].adjoint(observations[k]-cameras[k].observe(reference))) for k in subset)
    value=torch.zeros_like(reference);residual=rhs.clone();direction=residual.clone();norm=residual.square().sum()
    for _ in range(40):
        response=forward(direction);alpha=norm/(direction*response).sum().clamp_min(1e-30);value=value+alpha*direction;next_residual=residual-alpha*response;next_norm=next_residual.square().sum();direction=next_residual+(next_norm/norm.clamp_min(1e-30))*direction;residual=next_residual;norm=next_norm
    image=q.bounded(reference+q.null(value),observations['far'])
    return image,{'iterations':40,'normal_equation_residual':float(norm.sqrt()),'prior_std':.05,'effective_noise':.005,'ridge':ridge}


def prepare(args):
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True);q=ObservationQuotient();cameras={k:SeparableCamera(v,q) for k,v in VIEWS.items()};vae,info=load_frozen_vae(args.vae);jobs=[];sources=[]
    def encode(rgb):return vae.encode((2*rgb-1)[:,:,None].repeat(1,1,22,1,1).half().clone()).float().cpu().contiguous()
    for split,count in [('train',4),('val',2)]:
        for number,path in enumerate(sorted((Path(args.data)/split).glob('*.jpg'))[:count]):
            sid=f'{split}_{path.stem}';folder=root/sid;folder.mkdir(exist_ok=True)
            image=Image.open(path).convert('RGB');w,h=image.size;target=np.array(image.resize((256,256),Image.Resampling.BICUBIC,box=((w-256)/2,(h-256)/2,(w+256)/2,(h+256)/2)))
            rgb=torch.from_numpy(target).cuda().permute(2,0,1)[None].float()/255;generator=torch.Generator(device='cuda').manual_seed(91700+number+(0 if split=='train' else 100));observations={};errors={}
            for key,cam in cameras.items():
                expected=cam.observe(rgb);independent=camera_footprint(rgb,(0,0,256,256),VIEWS[key]);errors[key]=float((expected-independent).abs().max());assert errors[key]<2e-6
                observations[key]=((expected+torch.randn(expected.shape,device='cuda',generator=generator)*.003).clamp(0,1)*255).round()/255
            source={'id':sid,'split':split,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'canonical_rectangle':[0,0,256,256]}
            torch.save({'observations':{k:v.cpu() for k,v in observations.items()},'views':VIEWS,'source':source,'operator_probe_maxabs':errors,'packet_ids':{k:f"{source['sha256']}:{k}" for k in VIEWS}},folder/'inputs.pt')
            torch.save({'target':torch.from_numpy(target)},folder/'supervision.pt')
            up=F.interpolate(observations['far'],(256,256),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            with torch.no_grad():guide=encode(up)
            for seed in ([101,202] if split=='val' else [101]):
                destination=folder/f'initial_seed{seed}';destination.mkdir(exist_ok=True);save_file({'guide':guide},str(destination/'guide.safetensors'));jobs.append({'id':f'{sid}/initial{seed}','source':sid,'directory':str(destination),'seed':seed,'role':'far-only initial proposal'})
            sources.append(source)
    manifest={'scope':'Fixed4train2val natural-source developmentpilot, not heldouttest; no fabricated H3 diversity','sources':sources,'views':VIEWS,'candidate_seeds':[101,202],'native_frames':22,'query':'Static canonical memory; repeatednativecontext, not realmotionvideo','operator':'Exact declared Torch reflectGaussian.6+bilinearwarp+AA bicubic; sigma.003/8bitround; effectiveSigma.005','conditional_subsets':SUBSETS,'total_actual_H3_samples':56,'full50block_forwards':2744,'private_queries':'Four32x32corners, operational query definition; not identified semanticindependence','vae':info}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));(root/'initial_jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps({'event':'connection_inputs_ready','sources':len(sources),'initial_jobs':len(jobs)}),flush=True)


def guides(args):
    root=Path(args.out);manifest=json.loads((root/'manifest.json').read_text());vae,_=load_frozen_vae(args.vae);q=ObservationQuotient();cameras={k:SeparableCamera(v,q) for k,v in VIEWS.items()};jobs=[]
    def decode(z):return vae.decode(z.half())[:,:,21].cuda().float()
    def encode(x):return vae.encode((2*x-1)[:,:,None].repeat(1,1,22,1,1).half().clone()).float().cpu().contiguous()
    with torch.no_grad():
        for source in manifest['sources']:
            folder=root/source['id'];inputs=torch.load(folder/'inputs.pt',weights_only=False,map_location='cpu');observations={k:v.cuda() for k,v in inputs['observations'].items()};z0=load_file(str(folder/'initial_seed101/sampled_av_latents.safetensors'))['video'].cuda().float();raw=decode(z0);reference=q.bounded(raw,observations['far'])
            features=z0[0].flatten(1).double();features=features-features.mean(1,keepdim=True);values,vectors=torch.linalg.eigh(features@features.T/features.shape[1]);basis=vectors[:,-8:].flip(1).float();sign=basis.gather(0,basis.abs().argmax(0)[None]).sign();basis=basis*sign
            matrices={};maps={};audits={}
            for subset,keys in SUBSETS.items():
                maps[subset],audits[subset]=classical_map(reference,observations,cameras,keys,q);guide=encode(maps[subset]);matrices[subset]=guide
                for seed in [101,202]:
                    destination=folder/f'response_{subset}_seed{seed}';destination.mkdir(exist_ok=True);save_file({'guide':guide},str(destination/'guide.safetensors'));jobs.append({'id':f"{source['id']}/{subset}/{seed}",'source':source['id'],'directory':str(destination),'seed':seed,'role':'ActualH3 conditionalresponse','evidence_subset':keys})
            initial={'z0':z0.cpu(),'rgb_reference':reference.cpu(),'rgb_raw':raw.cpu(),'basis':basis.cpu(),'basis_protocol':'Top8channelPCAofX0, immutable perobject; computationalcoordinates, not semanticfactoridentification','classical_rgb':{k:v.cpu() for k,v in maps.items()},'classical_audits':audits}
            if source['split']=='val':
                z2=load_file(str(folder/'initial_seed202/sampled_av_latents.safetensors'))['video'].cuda().float();x2=q.bounded(decode(z2),observations['far']);initial['second_natural_initial']={'z0':z2.cpu(),'rgb_reference':x2.cpu(),'rgb_pair_rms':float((x2-reference).square().mean().sqrt())}
            torch.save(initial,folder/'initial.pt');print(json.dumps({'event':'connection_guides_ready','source':source['id'],'jobs':8}),flush=True)
    (root/'conditional_jobs.json').write_text(json.dumps(jobs,indent=2));print(json.dumps({'event':'conditional_jobs_ready','jobs':len(jobs)}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','guides']);p.add_argument('--out',required=True);p.add_argument('--data',default='data/BSR/BSDS500/data/images');p.add_argument('--comfy',required=True);p.add_argument('--vae',required=True);args=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;sys.path.insert(0,str(Path(args.comfy).resolve()))
    if args.phase=='prepare':prepare(args)
    else:guides(args)


if __name__=='__main__':main()
