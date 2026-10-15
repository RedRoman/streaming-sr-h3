"""Same-storage full24 H3 conditional-function heads; no novelty assertion."""
import torch
from torch import nn
from torch.nn import functional as F

KINDS=['mixed','additive','joint_set_cnp']


class ReadNet(nn.Module):
    def __init__(self,inc,width):
        super().__init__();self.a=nn.Conv3d(inc,width,(1,3,3),padding=(0,1,1));self.b=nn.Conv3d(width,24,(1,3,3),padding=(0,1,1));nn.init.zeros_(self.b.weight);nn.init.zeros_(self.b.bias)
    def forward(self,x):return self.b(F.silu(self.a(x)))


class OperatorHead(nn.Module):
    def __init__(self,kind,width=32):
        super().__init__();self.kind=kind;self.encoder=nn.Sequential(nn.Conv3d(51,32,1),nn.SiLU(),nn.Conv3d(32,8,1))
        # Three immutable support coordinates; every head has the same24-channel
        # function sketch and same full24-channel persistent source carrier.
        design=torch.tensor([[1.,-1.,-1.],[1.,-1.,1.],[1.,1.,-1.]])
        self.register_buffer('inverse_design',torch.linalg.inv(design))
        if kind=='mixed':self.read=ReadNet(37,width)
        elif kind=='joint_set_cnp':self.read=ReadNet(53,width)
        else:self.read=ReadNet(50,width);self.geometry=ReadNet(5,width)
    def memory(self,response,carrier):
        # response [B,3,24,T,H,W], same samples for every method.
        b,n,c,t,h,w=response.shape;coords=response.new_tensor([[-1.,-1.,1.],[-1.,1.,-1.],[1.,-1.,-1.]])
        # Each row is(gx,gy,gx*gy), not an identity/semantic label.
        g=coords[None,:,:,None,None,None].expand(b,-1,-1,t,h,w);source=carrier[:,None].expand(-1,n,-1,-1,-1,-1)
        features=torch.cat((response,source,g),2).flatten(0,1);encoded=self.encoder(features).reshape(b,n,8,t,h,w)
        return torch.einsum('ij,bjcthw->bicthw',self.inverse_design,encoded).flatten(1,2)
    def query(self,memory,carrier,g):
        b,c,t,h,w=carrier.shape
        if g.ndim==1:g=g[None].expand(b,-1)
        gx,gy=g[:,0],g[:,1];g3=torch.stack((gx,gy,gx*gy),1)[:,:,None,None,None].expand(-1,-1,t,h,w)
        y,x=torch.meshgrid(torch.linspace(-1,1,h,device=carrier.device),torch.linspace(-1,1,w,device=carrier.device),indexing='ij');xy=torch.stack((x,y))[None,:,None].expand(b,-1,t,-1,-1)
        if self.kind=='mixed':
            m0,mx,my=memory.chunk(3,1);value=m0+gx[:,None,None,None,None]*mx+gy[:,None,None,None,None]*my;output=self.read(torch.cat((value,carrier,g3,xy),1))
        elif self.kind=='joint_set_cnp':output=self.read(torch.cat((memory,carrier,g3,xy),1))
        else:output=self.read(torch.cat((memory,carrier,xy),1))+self.geometry(torch.cat((g3,xy),1))
        # A common linear output preserves the additive control's defining
        # property; tanh(a(m)+b(g)) would silently reintroduce mixed effects.
        return output


def build(kind):
    if kind=='mixed':return OperatorHead(kind,32)
    with torch.random.fork_rng(devices=[]):
        target=sum(p.numel() for p in OperatorHead('mixed',32).parameters());choices=[]
        for width in range(8,65):
            model=OperatorHead(kind,width);count=sum(p.numel() for p in model.parameters());choices.append((abs(count-target),width,count))
    _,width,count=min(choices);assert abs(count-target)/target<.025,(kind,count,target)
    return OperatorHead(kind,width)


def opposite_correlation_probe():
    c=torch.tensor([-1.,-1.,1.,1.]);g=torch.tensor([-1.,1.,-1.,1.]);same=(c==g).float()/2;opposite=(c!=g).float()/2
    def conditional(p):return float((p*(g==1)*c).sum()/(p*(g==1)).sum())
    assert float((same*c).sum())==float((opposite*c).sum())==0
    assert float((same*g).sum())==float((opposite*g).sum())==0
    assert conditional(same)==1 and conditional(opposite)==-1
    return dict(equal_content_mean=0.,equal_geometry_mean=0.,same_joint_Ecg=float((same*c*g).sum()),opposite_joint_Ecg=float((opposite*c*g).sum()),after_geometry_cue_content=[conditional(same),conditional(opposite)],exact_joint_MHT_passes=True,independent_marginal_reconstruction_content=[0.,0.],scope='Mathematical interface check only; not H3 natural diversity or learned-method performance')
