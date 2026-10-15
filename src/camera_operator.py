"""Differentiable canonical texture -> LR camera footprint for H3 memory.

World rectangles use continuous source-image pixel coordinates. The caller
keeps instantaneous appearance and object visibility separate from identity.
The Gaussian/bicubic implementation approximates the PIL synthetic degradation;
that implementation mismatch must be measured, not treated as exact physics.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def camera_footprint(canonical, canonical_rectangle, view_rectangle, view_size=(256,256), lr_size=(64,64), blur=.6):
    """Apply a linear, differentiable observation operator to BCHW values.

    Also valid for signed decoder tangent directions: no clipping, colour
    conversion or quantisation takes place. Outside the canonical surface,
    zero directions make no claim about the unknown background image.
    """
    assert canonical.ndim == 4
    device, dtype = canonical.device, canonical.dtype
    x0,y0,x1,y1 = canonical_rectangle
    vx0,vy0,vx1,vy1 = view_rectangle
    vh,vw=view_size
    ys=(torch.arange(vh,device=device,dtype=dtype)+.5)/vh*(vy1-vy0)+vy0
    xs=(torch.arange(vw,device=device,dtype=dtype)+.5)/vw*(vx1-vx0)+vx0
    gy,gx=torch.meshgrid(2*(ys-y0)/(y1-y0)-1,2*(xs-x0)/(x1-x0)-1,indexing='ij')
    grid=torch.stack((gx,gy),-1)[None].expand(canonical.shape[0],-1,-1,-1)
    image=F.grid_sample(canonical,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    if blur>0:
        radius=max(1,math.ceil(3*blur))
        u=torch.arange(-radius,radius+1,device=device,dtype=dtype)
        k=torch.exp(-u.square()/(2*blur*blur));k=k/k.sum()
        kernel=(k[:,None]*k[None,:])[None,None].expand(image.shape[1],1,-1,-1)
        image=F.conv2d(F.pad(image,(radius,)*4,mode='reflect'),kernel,groups=image.shape[1])
    return F.interpolate(image,size=lr_size,mode='bicubic',align_corners=False,antialias=True)


def nuisance_project(design, residual, nuisance, damping=1e-5):
    """Schur elimination of linear appearance nuisance (classical control).

    design: [N,K], residual: [N], nuisance: [N,P].
    Returns projected design/residual so brightness/gain directions do not
    automatically become permanent content evidence. This algebra is not novel.
    """
    gram=nuisance.T@nuisance+damping*torch.eye(nuisance.shape[1],device=nuisance.device,dtype=nuisance.dtype)
    fit_design=torch.linalg.solve(gram,nuisance.T@design)
    fit_residual=torch.linalg.solve(gram,nuisance.T@residual)
    return design-nuisance@fit_design,residual-nuisance@fit_residual


def audit():
    torch.manual_seed(21)
    x=torch.randn(1,3,32,32,dtype=torch.float64,requires_grad=True)
    y=torch.randn_like(x)
    kwargs=dict(canonical_rectangle=(0.,0.,32.,32.),view_rectangle=(4.,6.,28.,30.),view_size=(48,48),lr_size=(12,12))
    ax=camera_footprint(x,**kwargs);ay=camera_footprint(y,**kwargs)
    linear=float((camera_footprint(1.3*x-.7*y,**kwargs)-(1.3*ax-.7*ay)).abs().max().detach())
    v=torch.randn_like(ax);adj=torch.autograd.grad((ax*v).sum(),x)[0]
    lhs=(camera_footprint(y,**kwargs)*v).sum();rhs=(y*adj).sum()
    adjoint=float((lhs-rhs).abs())
    b=torch.randn(100,8,dtype=torch.float64);r=torch.randn(100,dtype=torch.float64);n=torch.ones(100,1,dtype=torch.float64)
    bp,rp=nuisance_project(b,r,n,damping=1e-12)
    leak=max(float((n.T@bp).abs().max()),float((n.T@rp).abs().max()))
    assert linear<1e-10 and adjoint<1e-10 and leak<1e-10
    return {'linearity_max_abs':linear,'adjoint_abs':adjoint,'brightness_nuisance_leak':leak,'passed':True,'degradation_is_pil_approximation':True}


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args();result=audit();print(json.dumps(result,indent=2))
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
