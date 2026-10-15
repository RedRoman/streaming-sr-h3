"""B3R candidate: closed affine actions on learned signed evidence moments.

The algebra is established; independent value is a testable H3 quality/cost
hypothesis. MLP factor-cache, nonlinear group and replay-GRU remain controls.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model import flatten_time,restore_time

KINDS=['closed_action','closed_action_no_equiv','factor_cache_mlp','nonlinear_group','gru_online','gru_replay']


def coords(batch,h,w,device):
    y,x=torch.meshgrid(torch.linspace(-1,1,h,device=device),torch.linspace(-1,1,w,device=device),indexing='ij');return torch.stack((x,y))[None].expand(batch,-1,-1,-1)


class FieldNet(nn.Module):
    def __init__(self,inc,outc,width):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(inc,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,outc,1));nn.init.normal_(self.net[-1].weight,std=.005);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)


class AffineAction(nn.Module):
    def __init__(self,width=32):super().__init__();self.net=FieldNet(14,8,width)
    def field(self,a,h=16,w=16):
        xy=coords(len(a),h,w,a.device);q=a[:,:,None,None].expand(-1,-1,h,w);z=torch.zeros_like(q);out=self.net(torch.cat((q,xy),1))-self.net(torch.cat((z,xy),1));g,b=out.chunk(2,1);return (.5*g.tanh()).exp().unsqueeze(2),(.5*b.tanh()).unsqueeze(2)
    def canonical(self,h,a):g,b=self.field(a,*h.shape[-2:]);return (h-b)/g
    def style(self,h,a):g,b=self.field(a,*h.shape[-2:]);return g*h+b
    def rebase(self,m,w,old,new):
        go,bo=self.field(old,*m.shape[-2:]);gn,bn=self.field(new,*m.shape[-2:]);return (go*m+(bo-bn)*w)/gn


class NonlinearAction(nn.Module):
    def __init__(self,width):super().__init__();self.inverse=FieldNet(18,4,width);self.forward_net=FieldNet(18,4,width)
    def apply(self,h,a,net):
        x,shape=flatten_time(h);b,c,t,hh,ww=shape;q=a[:,None,:,None,None].expand(-1,t,-1,hh,ww).reshape(b*t,12,hh,ww);xy=coords(b*t,hh,ww,h.device);out=net(torch.cat((x,q,xy),1));zero=net(torch.cat((x,torch.zeros_like(q),xy),1));return restore_time(x+.5*(out-zero),shape)
    def canonical(self,h,a):return self.apply(h,a,self.inverse)
    def style(self,h,a):return self.apply(h,a,self.forward_net)


class GRUAction(nn.Module):
    def __init__(self,width):
        super().__init__();self.first=nn.Conv2d(32,width,3,padding=1);self.gates=nn.Conv2d(width+4,8,3,padding=1);self.candidate=nn.Conv2d(width+4,4,3,padding=1);self.query=FieldNet(18,4,16)
    def event(self,state,h,oldq,newq,oldw,neww):
        x,shape=flatten_time(h);s,_=flatten_time(state);b,c,t,hh,ww=shape
        def rep(q):return q[:,None,:,None,None].expand(-1,t,-1,hh,ww).reshape(b*t,-1,hh,ww)
        ow=oldw.expand(-1,1,t,-1,-1).permute(0,2,1,3,4).reshape(b*t,1,hh,ww);nw=neww.expand(-1,1,t,-1,-1).permute(0,2,1,3,4).reshape(b*t,1,hh,ww);f=F.silu(self.first(torch.cat((x,rep(oldq),rep(newq),ow,nw,coords(b*t,hh,ww,h.device)),1)));reset,update=self.gates(torch.cat((f,s),1)).sigmoid().chunk(2,1);proposal=self.candidate(torch.cat((f,reset*s),1)).tanh();return restore_time((1-update)*s+update*proposal,shape)
    def style(self,h,a):
        x,shape=flatten_time(h);b,c,t,hh,ww=shape;q=a[:,None,:,None,None].expand(-1,t,-1,hh,ww).reshape(b*t,12,hh,ww);xy=coords(b*t,hh,ww,h.device);out=self.query(torch.cat((x,q,xy),1))-self.query(torch.cat((x,torch.zeros_like(q),xy),1));return restore_time(x+.5*out,shape)


class RevisionMemory(nn.Module):
    def __init__(self,kind,width=64,core_width=None):
        super().__init__();assert kind in KINDS;self.kind=kind;self.encoder=FieldNet(51,4,width);self.decoder=FieldNet(41,24,width);self.appearance=nn.Sequential(nn.Linear(18,48),nn.SiLU(),nn.Linear(48,48),nn.SiLU(),nn.Linear(48,12));nn.init.zeros_(self.appearance[-1].weight);nn.init.zeros_(self.appearance[-1].bias)
        if kind.startswith('closed'):self.core=AffineAction(32)
        elif kind in ['factor_cache_mlp','nonlinear_group']:self.core=NonlinearAction(core_width or 20)
        else:self.core=GRUAction(core_width or 26)
    def appearance_value(self,features):return features[...,:12]+.05*self.appearance(features).tanh()
    def lift(self,delta,proxy,footprint):
        x,shape=flatten_time(delta);z,_=flatten_time(proxy);b,c,t,h,w=shape;mask=F.interpolate(footprint,size=(h,w),mode='area')[:,None].expand(-1,t,-1,-1,-1).reshape(b*t,1,h,w);out=self.encoder(torch.cat((x,z,mask,coords(b*t,h,w,x.device)),1));return restore_time(.5*out.tanh(),shape)
    def read(self,h,anchor,a,confidence):
        x,shape=flatten_time(h);z,_=flatten_time(anchor);b,c,t,hh,ww=shape;q=a[:,None,:,None,None].expand(-1,t,-1,hh,ww).reshape(b*t,12,hh,ww);w=confidence.expand(-1,1,t,-1,-1).permute(0,2,1,3,4).reshape(b*t,1,hh,ww);out=self.decoder(torch.cat((x,z,w,q),1))-self.decoder(torch.cat((torch.zeros_like(x),z,w,q),1));return restore_time(2*out.tanh(),shape)
    def lift_packets(self,inputs,neutral=False,targets=None):
        delta=targets['neutral_delta'] if neutral else inputs['packet_delta'];proxy=targets['neutral_proxy'] if neutral else inputs['packet_proxy'];b,n=delta.shape[:2];masks=inputs['packet_masks'];h=self.lift(delta.flatten(0,1),proxy.flatten(0,1),masks.flatten(0,1));return h.reshape(b,n,*h.shape[1:])
    def rollout(self,inputs,return_details=False):
        h=self.lift_packets(inputs);b,n,c,t,hh,ww=h.shape;device=h.device;zero=h[:,0]*0;groups=[zero.clone(),zero.clone()];ws=[torch.zeros((b,1,1,hh,ww),device=device) for _ in range(2)];oldq=torch.zeros(b,2,12,device=device);oldw=torch.zeros(b,n,1,1,hh,ww,device=device);hidden=zero.clone();states=[];outputs=[];appearances=[];query_features=[];qsteps=inputs['query_steps'][0].tolist();pg=inputs['packet_groups'][0].tolist();info=[];q_all=[];gains=[torch.ones(b,4,1,hh,ww,device=device) for _ in range(2)];biases=[torch.zeros(b,4,1,hh,ww,device=device) for _ in range(2)]
        for step in range(18):
            features=inputs['group_features'][:,step];newq=self.appearance_value(features);available=(features.abs().sum(-1)>0);newq=newq*available[:,:,None];neww=inputs['weights'][:,step].unsqueeze(3);changed_g=[not torch.equal(newq[:,g].detach(),oldq[:,g].detach()) for g in range(2)];changed_j=[not torch.equal(neww[:,j],oldw[:,j]) or (changed_g[pg[j]] and bool(oldw[:,j].any())) for j in range(n)]
            if self.kind.startswith('closed'):
                for g in range(2):
                    if changed_g[g]:
                        gain,bias=self.core.field(newq[:,g]);groups[g]=(gains[g]*groups[g]+(biases[g]-bias)*ws[g])/gain;gains[g],biases[g]=gain,bias
                for j in range(n):
                    if not torch.equal(neww[:,j],oldw[:,j]):
                        g=pg[j];difference=neww[:,j]-oldw[:,j];groups[g]=groups[g]+difference*(h[:,j]-biases[g])/gains[g];ws[g]=ws[g]+difference
                weight=ws[0]+ws[1];canonical=(groups[0]+groups[1])/weight.clamp_min(.001)
            elif self.kind=='factor_cache_mlp':
                if any(changed_j):
                    ca=self.core.canonical(h.flatten(0,1),newq[:,pg].flatten(0,1)).reshape_as(h)
                    for g in range(2):
                        ii=[j for j in range(n) if pg[j]==g];groups[g]=(neww[:,ii]*ca[:,ii]).sum(1);ws[g]=neww[:,ii].sum(1)
                weight=ws[0]+ws[1];canonical=(groups[0]+groups[1])/weight.clamp_min(.001)
            elif self.kind=='nonlinear_group':
                for j in range(n):
                    if not torch.equal(neww[:,j],oldw[:,j]):
                        g=pg[j];difference=neww[:,j]-oldw[:,j];groups[g]=groups[g]+difference*h[:,j];ws[g]=ws[g]+difference
                weight=ws[0]+ws[1];canonical=sum((ws[g]*self.core.canonical(groups[g]/ws[g].clamp_min(.001),newq[:,g]) for g in range(2)),zero.clone())/weight.clamp_min(.001)
            else:
                if any(changed_j):
                    if self.kind=='gru_replay':
                        hidden=zero.clone()
                        for j in range(n):
                            if bool(neww[:,j].any()):hidden=self.core.event(hidden,h[:,j],oldq[:,pg[j]]*0,newq[:,pg[j]],oldw[:,j]*0,neww[:,j])
                    else:
                        for j in range(n):
                            if changed_j[j]:hidden=self.core.event(hidden,h[:,j],oldq[:,pg[j]],newq[:,pg[j]],oldw[:,j],neww[:,j])
                weight=neww.sum(1);canonical=hidden
            oldq=newq;oldw=neww;canonical=canonical*(weight>0);q_all.append(newq);states.append(canonical)
            if step in qsteps:
                qi=qsteps.index(step);qgroup=int(inputs['query_groups'][0,qi]);a=torch.zeros(b,12,device=device) if qgroup<0 else self.appearance_value(inputs['query_features'][:,qi]);styled=self.core.style(canonical,a)*(weight>0);output=self.read(styled,inputs['query_latents'][:,qi],a,(weight/8).clamp(0,1));outputs.append(output);appearances.append(a);query_features.append(canonical);info.append(dict(step=step,feature_RMS=canonical.square().mean().sqrt(),evidence_mass=weight.mean(),active_factor_count=sum(bool(neww[:,j].any()) for j in range(n))))
        result=(torch.stack(outputs,1),torch.stack(appearances,1),states)
        if return_details:return (*result,dict(packet_features=h,group_appearance=torch.stack(q_all,1),query_features=torch.stack(query_features,1),information=info))
        return result


def losses(model,inputs,targets):
    pred,a,states,details=model.rollout(inputs,True);reconstruction=F.l1_loss(pred,targets['detail_latents']);trueq=targets['query_q'];appearance=F.mse_loss(a,trueq);gtrue=targets['q_groups'][:,None].expand(-1,18,-1,-1);valid=(inputs['group_features'].abs().sum(-1)>0);appearance=appearance+((details['group_appearance']-gtrue).square()*valid[:,:,:,None]).sum()/(valid.sum()*12).clamp_min(1)
    equivariance=pred.new_zeros(())
    if model.kind!='closed_action_no_equiv':
        neutral=model.lift_packets(inputs,True,targets);obs=details['packet_features'];pg=inputs['packet_groups'][0];q=targets['q_groups'][:,pg];styled=model.core.style(neutral.flatten(0,1),q.flatten(0,1));equivariance=F.l1_loss(styled,obs.flatten(0,1))
    total=reconstruction+10*appearance+.5*equivariance;return total,dict(total=total,reconstruction=reconstruction,appearance=appearance,equivariance=equivariance),pred,a,states,details


def build(kind):
    # Parameter matching is a mechanical width choice, not a validation sweep.
    if kind.startswith('closed'):return RevisionMemory(kind)
    with torch.random.fork_rng(devices=[]):
        target=sum(p.numel() for p in RevisionMemory('closed_action').parameters());counts=[]
        for width in range(8,49):
            model=RevisionMemory(kind,core_width=width);count=sum(p.numel() for p in model.parameters());counts.append((abs(count-target),width,count))
    _,width,count=min(counts);assert abs(count-target)/target<.015,(kind,count,target);return RevisionMemory(kind,core_width=width)
