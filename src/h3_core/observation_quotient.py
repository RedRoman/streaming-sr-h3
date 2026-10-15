"""Classical observed/coarse quotient for a nonlinear H3 decoder readout.

This projection is a control/constraint, not a novel restoration algorithm.
The 1D matrix executes the declared reflect-Gaussian + Torch antialiased
bicubic operator. It approximates the separate uint8 Pillow simulator.
"""
import math
import numpy as np
import torch
from torch.nn import functional as F


class ObservationQuotient:
    def __init__(self,size=256,factor=4,blur=.6,device='cuda',dtype=torch.float64):
        self.size=size;self.factor=factor;self.blur=blur
        basis=torch.eye(size,device=device,dtype=dtype)[:,None]
        if blur>0:
            radius=max(1,math.ceil(3*blur));u=torch.arange(-radius,radius+1,device=device,dtype=dtype)
            kernel=torch.exp(-u.square()/(2*blur*blur));kernel=kernel/kernel.sum()
            basis=F.conv1d(F.pad(basis,(radius,radius),mode='reflect'),kernel[None,None])
        self.matrix=F.interpolate(basis[:,None],size=(1,size//factor),mode='bicubic',align_corners=False,antialias=True)[:,0,0].T.contiguous()
        gram=self.matrix@self.matrix.T
        self.right=self.matrix.T@torch.linalg.solve(gram,torch.eye(len(gram),device=device,dtype=dtype))

    def observe(self,image):
        value=image.to(self.matrix);return (self.matrix@value@self.matrix.T).to(image.dtype)

    def range_component(self,observed):
        value=observed.to(self.right);return (self.right@value@self.right.T).to(observed.dtype)

    def null(self,image):
        value=image.to(self.matrix)
        residual=value-self.right@(self.matrix@value@self.matrix.T)@self.right.T
        return residual.to(image.dtype)

    def readout(self,decoded,observed):
        value=decoded.to(self.matrix);target=observed.to(self.matrix)
        result=value+self.right@(target-self.matrix@value@self.matrix.T)@self.right.T
        return result.to(decoded.dtype)

    def bounded(self,decoded,observed,steps=64):
        value=decoded.to(self.matrix);target=observed.to(self.matrix)
        p=torch.zeros_like(value);q=torch.zeros_like(value)
        for _ in range(steps):
            candidate=value+p
            affine=candidate+self.right@(target-self.matrix@candidate@self.matrix.T)@self.right.T
            p=candidate-affine;candidate=affine+q;value=candidate.clamp(0,1);q=candidate-value
        return value.to(decoded.dtype)


class SameFootprintCoarseMean:
    """Independent repeated packets reduce coarse noise; copied IDs do nothing.

    Different footprints are left to the detail likelihood. Geometry/visibility
    is oracle in this controlled protocol, stated separately in its manifest.
    """
    def __init__(self,rectangle,shape,device='cuda'):
        self.rectangle=np.asarray(rectangle);self.total=torch.zeros(shape,device=device,dtype=torch.float64)
        self.count=torch.zeros_like(self.total[:,:1]);self.packet_ids=set()

    def write(self,packet):
        if packet.observation_id in self.packet_ids:return False
        if not np.allclose(self.rectangle,packet.world_rectangle,atol=1e-7,rtol=0):return False
        rgb=torch.as_tensor(packet.rgb.copy(),device=self.total.device).permute(2,0,1)[None].double()/255
        valid=(torch.as_tensor(packet.visible_mask.copy(),device=self.total.device)>0)[None,None]
        self.total+=rgb*valid;self.count+=valid;self.packet_ids.add(packet.observation_id)
        return True

    def mean(self):
        if not bool((self.count>0).all()):raise ValueError('The initial canonical coarse observation must cover the whole frame')
        return (self.total/self.count).float()
