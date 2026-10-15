"""Shared classical decoder-support quotient around the fixed current LR.

The learnable contribution is ResidualMemory; these projectors are deliberately
shared with all four controls and never claimed novel.
"""
import torch
from native_vae import decode_last
from pixel_dc import Projection


class ResidualReadout:
    def __init__(self,vae):
        self.vae=vae;self.far=Projection(256,4);self.near=Projection(128,2)
        self.pf=self.far.right@self.far.matrix;self.pn=self.near.right@self.near.matrix

    def historical_support(self,x):
        out=torch.zeros_like(x);out[:,:,64:192,64:192]=self.pn@x[:,:,64:192,64:192]@self.pn.T;return out

    def current_null(self,x,is_near):
        return x-self.historical_support(x) if is_near else x-self.pf@x@self.pf.T

    def supported_null(self,x,is_near):
        return self.current_null(self.historical_support(self.current_null(x,is_near)),is_near)

    def rgb(self,model,inputs,detail,a,stage):
        z=inputs['observation_latents'][:,stage];base=inputs['base_decoded_rgb'][:,stage];is_near=stage in [2,3,5]
        # Initial proposal is explicitly zero in all recorded B2 experiments.
        # Optional future fixed H3-DiT proposal is never counted as evidence.
        proposal=inputs.get('proposal_canonical_residual')
        if proposal is None:
            prior=torch.zeros_like(z);prior_rgb=base;prior_effect=torch.zeros_like(base)
        else:
            prior=model.read_detail(proposal,a);prior_rgb=decode_last(self.vae,z+prior)
            prior_effect=self.current_null(prior_rgb-base,is_near)
        decoded=decode_last(self.vae,z+prior+detail)
        return base+prior_effect+self.supported_null(decoded-prior_rgb,is_near)
