"""Strong LR-only revision controls; all operations are classical.

The inverse uses the actual camera and an unclipped affine photometric model.
Clipping/model mismatch is handled only by the shared causal LR confidence.
No ground-truth image, nuisance value, or content/change mask is accepted here.
"""
import torch
from torch.nn import functional as F
from cache_revision import modes
from revision_utils import FP


def photometric_field(q):
    shade=(1+(q[6:,None,None]*modes(256,256,device=q.device)).sum(0)).clamp(.3,2)
    return (1+q[:3,None,None]).clamp(.5,1.5)*shade,q[3:6,None,None]


def conjugate_gradient(normal,b,iterations=48):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rr=r.square().sum();initial=float(rr)
    for _ in range(iterations):
        ap=normal(p);alpha=rr/(p.mul(ap).sum().clamp_min(1e-30));x=x+alpha*p;r=r-alpha*ap;new=r.square().sum()
        if float(new)<max(initial*1e-12,1e-25):break
        p=r+(new/rr)*p;rr=new
    return x,float(r.square().sum().sqrt()/b.square().sum().sqrt().clamp_min(1e-15))


class ClassicalRevision:
    def __init__(self,readout):self.r=readout;self.k=readout.near.matrix;self.u=readout.near.right
    def crop(self,x,fp):top,left=FP[fp];return x[...,top:top+128,left:left+128]
    def embed(self,x,fp):
        top,left=FP[fp];z=torch.zeros((*x.shape[:-2],256,256),device=x.device,dtype=x.dtype);z[...,top:top+128,left:left+128]=x;return z
    def observe(self,x,fp):return self.k@self.crop(x,fp)@self.k.T
    def relative(self,qsource,qquery,fp):
        si,bi=photometric_field(qsource);sq,bq=photometric_field(qquery);ratio=si/sq;offset=bi-ratio*bq
        return self.crop(ratio,fp)[None],self.crop(offset,fp)[None]
    def packets(self,inputs,near_LR,qi,retrospective=True,appearance=None):
        step=int(inputs['query_steps'][0,qi]);qg=inputs['group_features'][0,step,:,:12] if appearance is None else appearance
        qq=inputs['query_features'][0,qi,:12] if int(inputs['query_groups'][0,qi])>=0 else torch.zeros(12,device=qg.device)
        records=[]
        for j,arrival in enumerate(inputs['packet_steps'][0].tolist()):
            if arrival>step:continue
            fp=int(inputs['packet_fp'][0,j]);group=int(inputs['packet_groups'][0,j]);s=step if retrospective else arrival;q=qg[group] if retrospective else inputs['group_features'][0,s,group,:12]
            weight=inputs['weights'][0,s,j:j+1];wf=F.interpolate(weight,size=(256,256),mode='bilinear',align_corners=False);w=F.interpolate(self.crop(wf,fp),size=(64,64),mode='area');ratio,offset=self.relative(q,qq,fp)
            records.append(dict(index=j,fp=fp,ratio=ratio,offset=offset,weight=w,y=near_LR[j:j+1],q=q,query_q=qq))
        return records
    def atlas(self,base,records):
        total=torch.zeros_like(base);count=torch.zeros_like(base[:,:1])
        for p in records:
            fp=p['fp'];pred=p['ratio']*self.crop(base,fp)+p['offset'];res=p['y']-self.k@pred@self.k.T;correction=(self.u@res@self.u.T)/p['ratio'].clamp_min(.1);w=F.interpolate(p['weight'],size=(128,128),mode='bilinear',align_corners=False);total+=self.embed(w*correction,fp);count+=self.embed(w,fp)
        return base+self.r.null(total/count.clamp_min(.001))
    def inverse(self,base,records,ridge=.01,iterations=48):
        def adjoint(y,p):return self.embed(p['ratio']*(self.k.T@y@self.k),p['fp'])
        b=torch.zeros_like(base)
        for p in records:
            pred=p['ratio']*self.crop(base,p['fp'])+p['offset'];res=p['y']-self.k@pred@self.k.T;b+=adjoint(p['weight']*res,p)
        b=self.r.null(b)
        def normal(x):
            nx=self.r.null(x);v=torch.zeros_like(x)
            for p in records:
                y=self.k@(p['ratio']*self.crop(nx,p['fp']))@self.k.T;v+=adjoint(p['weight']*y,p)
            return self.r.null(v)+ridge*x
        h,res=conjugate_gradient(normal,b,iterations);return base+self.r.null(h),res
    def decoded_atlas(self,base,decoded,records):
        total=torch.zeros_like(base);count=torch.zeros_like(base[:,:1])
        for p in records:
            fp=p['fp'];# Same per-packet physical preconditioned native codes.
            canonical=(self.crop(decoded[p['index']],fp)-p['offset'])/p['ratio'].clamp_min(.1)
            delta=self.r.pn@(canonical-self.crop(base,fp))@self.r.pn.T;w=F.interpolate(p['weight'],size=(128,128),mode='bilinear',align_corners=False);total+=self.embed(w*delta,fp);count+=self.embed(w,fp)
        return base+self.r.null(total/count.clamp_min(.001))


def frozen_controller(inputs,freeze_photometry=True,freeze_reliability=True):
    result={k:v.clone() for k,v in inputs.items()};steps=inputs['packet_steps'][0].tolist();groups=inputs['packet_groups'][0].tolist()
    if freeze_photometry:
        for g in [0,1]:
            first=min(t for t,gg in zip(steps,groups) if gg==g)
            result['group_features'][:,first:,g]=inputs['group_features'][:,first:first+1,g]
    if freeze_reliability:
        for j,first in enumerate(steps):result['weights'][:,first:,j]=inputs['weights'][:,first:first+1,j]
    return result
