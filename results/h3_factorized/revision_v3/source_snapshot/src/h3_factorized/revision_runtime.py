"""Incremental inference state, retaining compressed factors rather than codes.

Controller descriptors/reference LR are explicitly retained by the caller and
included in the cost audit. Group tags are the known simulator grouping.
"""
import torch


class RevisionRuntime:
    def __init__(self,model):
        self.model=model;self.h={};self.group={};self.weights={};self.groups=None;self.ws=None;self.q=None;self.gains=None;self.biases=None;self.hidden=None;self.canonical=None;self.mass=None
    def ingest(self,packet_id,delta,proxy,mask,group):
        if packet_id in self.h:return False
        h=self.model.lift(delta,proxy,mask).detach();self.ingest_feature(packet_id,h,group);return True
    def ingest_feature(self,packet_id,h,group):
        if packet_id in self.h:return False
        self.h[packet_id]=h.detach();self.group[packet_id]=int(group);b,c,t,hh,ww=h.shape
        self.weights[packet_id]=torch.zeros((b,1,1,hh,ww),device=h.device)
        if self.groups is None:
            zero=h*0;self.groups=[zero.clone(),zero.clone()];self.ws=[torch.zeros((b,1,1,hh,ww),device=h.device) for _ in range(2)];self.q=torch.zeros(b,2,12,device=h.device);self.gains=[torch.ones(b,4,1,hh,ww,device=h.device) for _ in range(2)];self.biases=[torch.zeros(b,4,1,hh,ww,device=h.device) for _ in range(2)];self.hidden=zero.clone();self.canonical=zero.clone();self.mass=self.ws[0]+self.ws[1]
        return True
    def update(self,group_features,weights):
        if not self.h:return
        model=self.model;kind=model.kind;keys=list(self.h);available=group_features.abs().sum(-1)>0;newq=model.appearance_value(group_features)*available[:,:,None];cg=[not torch.equal(newq[:,g],self.q[:,g]) for g in range(2)];nw={key:weights.get(key,torch.zeros_like(self.weights[key])).detach().clone() for key in keys};cj={key:(not torch.equal(nw[key],self.weights[key]) or (cg[self.group[key]] and bool(self.weights[key].any()))) for key in keys}
        if kind.startswith('closed'):
            for g in range(2):
                if cg[g]:
                    gain,bias=model.core.field(newq[:,g]);self.groups[g]=(self.gains[g]*self.groups[g]+(self.biases[g]-bias)*self.ws[g])/gain;self.gains[g],self.biases[g]=gain,bias
            for key in keys:
                if not torch.equal(nw[key],self.weights[key]):
                    g=self.group[key];difference=nw[key]-self.weights[key];self.groups[g]+=difference*(self.h[key]-self.biases[g])/self.gains[g];self.ws[g]+=difference
            self.mass=self.ws[0]+self.ws[1];canonical=(self.groups[0]+self.groups[1])/self.mass.clamp_min(.001)
        elif kind=='factor_cache_mlp':
            if any(cj.values()):
                hh=torch.cat([self.h[key] for key in keys]);qq=torch.cat([newq[:,self.group[key]] for key in keys]);ca=model.core.canonical(hh,qq);ca=dict(zip(keys,ca.split(group_features.shape[0])))
                for g in range(2):
                    ii=[key for key in keys if self.group[key]==g];self.groups[g]=sum((nw[key]*ca[key] for key in ii),self.groups[g]*0);self.ws[g]=sum((nw[key] for key in ii),self.ws[g]*0)
            self.mass=self.ws[0]+self.ws[1];canonical=(self.groups[0]+self.groups[1])/self.mass.clamp_min(.001)
        elif kind=='nonlinear_group':
            for key in keys:
                if not torch.equal(nw[key],self.weights[key]):
                    g=self.group[key];difference=nw[key]-self.weights[key];self.groups[g]+=difference*self.h[key];self.ws[g]+=difference
            self.mass=self.ws[0]+self.ws[1];canonical=sum((self.ws[g]*model.core.canonical(self.groups[g]/self.ws[g].clamp_min(.001),newq[:,g]) for g in range(2)),self.hidden*0)/self.mass.clamp_min(.001)
        else:
            if any(cj.values()):
                if kind=='gru_replay':
                    self.hidden=self.hidden*0
                    for key in keys:
                        if bool(nw[key].any()):self.hidden=model.core.event(self.hidden,self.h[key],self.q[:,self.group[key]]*0,newq[:,self.group[key]],nw[key]*0,nw[key])
                else:
                    for key in keys:
                        if cj[key]:self.hidden=model.core.event(self.hidden,self.h[key],self.q[:,self.group[key]],newq[:,self.group[key]],self.weights[key],nw[key])
            self.mass=sum(nw.values(),self.mass*0);canonical=self.hidden
        self.canonical=canonical*(self.mass>0);self.q=newq;self.weights=nw
    def query(self,anchor,features,neutral=False):
        a=torch.zeros(len(anchor),12,device=anchor.device) if neutral else self.model.appearance_value(features)
        if self.canonical is None:return anchor*0
        styled=self.model.core.style(self.canonical,a)*(self.mass>0);return self.model.read(styled,anchor,a,(self.mass/8).clamp(0,1))
    def payload(self):
        # Tensor values, not Python allocator overhead; cached derivable arrays
        # are included rather than optimistically omitted.
        fields=dict(h=self.h,weights=self.weights,groups=self.groups,ws=self.ws,q=self.q,gains=self.gains,biases=self.biases,hidden=self.hidden,canonical=self.canonical,mass=self.mass);seen=set();counts={}
        def size(x):
            if isinstance(x,torch.Tensor):
                key=(x.untyped_storage().data_ptr(),x.untyped_storage().nbytes())
                if key in seen:return 0
                seen.add(key);return x.untyped_storage().nbytes()
            if isinstance(x,dict):return sum(size(v) for v in x.values())
            if isinstance(x,(list,tuple)):return sum(size(v) for v in x)
            return 0
        for name,value in fields.items():counts[name]=size(value)
        counts['packet_ID_group_geometry_metadata_assumed_int64x3']=24*len(self.h);return counts
