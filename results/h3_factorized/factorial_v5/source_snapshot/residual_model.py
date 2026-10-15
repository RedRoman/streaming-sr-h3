"""B2: current-LR anchored memory of observation-supported detail only."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from model import FactorizedMemory,flatten_time,restore_time


@dataclass
class DetailState:
    detail:torch.Tensor
    reference:torch.Tensor
    seen:frozenset


class ResidualMemory(FactorizedMemory):
    """Same four charts/parameter budget; new state and observation contract."""
    def read_detail(self,r,a):
        # Translation of a bijection remains bijective, while every control
        # has exactly zero residual at r=0, including its conditional biases.
        return self.chart.read(r,a)-self.chart.read(torch.zeros_like(r),a)

    def canonical_detail(self,r,a):
        offset=self.chart.read(torch.zeros_like(r),a)
        return self.chart.canonicalize(r+offset,a)-self.chart.canonicalize(offset,a)

    def begin(self,z_observed,reference,packet_id):
        # z_observed supplies shape only. No observed or HR absolute code is
        # ever mistaken for an actual initial detail/proposal identity.
        return DetailState(torch.zeros_like(z_observed),reference,frozenset([packet_id]))

    def observe(self,state,z_observed,rgb_observed,packet_id,visible,reveal,footprint):
        a=self.appearance(state.reference,rgb_observed)
        if packet_id in state.seen or not bool(visible.any()):return state,a
        detail=state.detail
        if bool(reveal.any()):
            detail=self.global_detail_update(detail,self.canonical_detail(z_observed,a),a,visible,reveal,footprint)
        return DetailState(detail,state.reference,state.seen|{packet_id}),a

    def query(self,state,appearance):return self.read_detail(state.detail,appearance)

    def global_detail_update(self,old,observed,a,visible,reveal,footprint):
        # The actual pixel sensor footprint is a CONDITION, not a hard latent
        # spatial mask: H3 encoder/decoder and chart have nonlocal footprints.
        o,shape=flatten_time(old);z,_=flatten_time(observed);b,c,t,h,w=shape
        visible=F.interpolate(visible,size=(h,w),mode='area');reveal=F.interpolate(reveal,size=(h,w),mode='area');channels=torch.cat((visible,reveal,torch.full_like(visible,footprint)),1)
        channels=channels[:,None].expand(-1,t,-1,-1,-1).reshape(b*t,3,h,w);aa=a[:,None,:,None,None].expand(-1,t,-1,h,w).reshape(b*t,12,h,w)
        delta,gate=self.update_head.net(torch.cat((o,z,channels,aa),1)).chunk(2,1)
        return restore_time(o+gate.sigmoid()*(z-o+2*delta.tanh()),shape)

    def rollout(self,inputs):
        obs=inputs['detail_observation_latents'];rgb=inputs['observation_rgb'];zero=torch.zeros((obs.shape[0],12),device=obs.device)
        state=self.begin(obs[:,0],rgb[:,0],'far_A0')
        out=[self.query(state,zero)];appearances=[zero];states=[state.detail]
        ids=['far_A0','far_A1','near_A1','query','return_A0','near_B0','far_B2','far_B2']
        for step in range(1,8):
            # reveal denotes a finer packet, not novelty/count==0: a fresh
            # revisited near can overwrite truly changed detail.
            state,a=self.observe(state,obs[:,step],rgb[:,step],ids[step],inputs['visible'][:,step],inputs['reveal'][:,step],float(inputs['footprint'][step]))
            out.append(self.query(state,a));appearances.append(a);states.append(state.detail)
        return torch.stack(out,1),torch.stack(appearances,1),states


def residual_losses(model,inputs,targets):
    from train import TARGET
    output,a,states=model.rollout(inputs);stages=[2,4,5,6]
    target=targets['detail_latents'];q=targets['appearance']
    reconstruction=F.l1_loss(output[:,stages],target[:,[TARGET[s] for s in stages]])
    appearance=F.mse_loss(a[:,[1,2,4,5,6]],q[:,[TARGET[s] for s in [1,2,4,5,6]]])
    swap=target.new_zeros(())
    if model.kind!='factorized_no_swap':
        ca=model.canonical_detail(target[:,1],q[:,1]);cb=model.canonical_detail(target[:,5],q[:,5])
        swap=F.l1_loss(model.read_detail(ca,q[:,2]),target[:,2])+F.l1_loss(model.read_detail(cb,q[:,4]),target[:,4])+.5*(F.l1_loss(ca,target[:,0])+F.l1_loss(cb,target[:,3]))
    total=reconstruction+10*appearance+.5*swap
    return total,dict(total=total,reconstruction=reconstruction,appearance=appearance,swap=swap),output,a,states
