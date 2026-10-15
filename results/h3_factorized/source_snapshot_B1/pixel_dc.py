"""Classical separable minimum-change affine/box Dykstra control, not novel.

The1D matrix is built by executing the same blur+Torch bicubic on basis inputs.
Its separable2D operator is checked against the actual simulator at runtime.
"""
import torch
from torch.nn import functional as F
from cache_pairs import degrade


class Projection:
    def __init__(self,size,factor,device='cuda',steps=64):
        self.size=size;self.factor=factor;self.steps=steps
        coords=torch.arange(-2,3,device=device,dtype=torch.float32);k=torch.exp(-coords.square()/(2*.6**2));k=k/k.sum()
        basis=torch.eye(size,device=device)[:,None]
        blurred=F.conv1d(F.pad(basis,(2,2),mode='reflect'),k[None,None])[:,0]
        matrix=F.interpolate(blurred[:,None,None,:],size=(1,size//factor),mode='bicubic',align_corners=False,antialias=True)[:,0,0].T.contiguous()
        self.matrix=matrix;gram=matrix@matrix.T
        self.right=matrix.T@torch.linalg.solve(gram,torch.eye(len(gram),device=device))
        generator=torch.Generator(device=device).manual_seed(19);probe=torch.rand((1,3,size,size),device=device,generator=generator)
        self.operator_maxabs=float((self.observe(probe)-degrade(probe,factor)).abs().max())
        assert self.operator_maxabs<2e-5,('Wrong projection operator',self.operator_maxabs)

    def observe(self,x):return self.matrix@x@self.matrix.T

    def affine(self,x,y):return x+self.right@(y-self.observe(x))@self.right.T

    def apply(self,x,y):
        x=x.float();y=y.float();p=torch.zeros_like(x);q=torch.zeros_like(x)
        for _ in range(self.steps):
            candidate=x+p;affine=self.affine(candidate,y);p=candidate-affine
            candidate=affine+q;x=candidate.clamp(0,1);q=candidate-x
        return x


def project_visible(rgb,measurements,stage,far,near):
    assert bool(measurements['visible_stage'][stage]),'No invented observation for query'
    if bool(measurements['near_stage'][stage]):
        lo,hi=measurements['roi'];result=rgb.clone();target=measurements['near_y'][stage:stage+1].to(rgb.device)
        result[:,:,lo:hi,lo:hi]=near.apply(rgb[:,:,lo:hi,lo:hi],target)
        residual=degrade(result[:,:,lo:hi,lo:hi],2)-target
    else:
        target=measurements['far_y'][stage:stage+1].to(rgb.device);result=far.apply(rgb,target);residual=degrade(result,4)-target
    return result,float(residual.square().mean().sqrt())
