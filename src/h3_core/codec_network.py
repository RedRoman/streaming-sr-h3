"""Shared past-only writer and single-future reader for the fixed pixel pilot.

This is an information-retention codec, not an H3 E/D or DiT model. The only
arm-dependent module is the separately owned frozen-book quantizer.
"""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.h3_backbone.decoder_side_codec import DecoderSideCodec


def patch_groups(past):
    b=past.shape[0];blocks=F.unfold(past,kernel_size=4,stride=4).T if False else F.unfold(past,kernel_size=4,stride=4)
    blocks=blocks.transpose(1,2).reshape(b,16,16,3,4,4)
    return torch.stack([blocks[...,r:r+2,c:c+2].reshape(b,16,16,12)
                        for r,c in [(0,0),(0,2),(2,0),(2,2)]],-2)


class PastWriter(nn.Module):
    def __init__(self):
        super().__init__();self.analysis=nn.Conv2d(3,32,4,stride=4)
        self.residual=nn.Sequential(nn.Conv2d(32,48,3,padding=1),nn.SiLU(),nn.Conv2d(48,32,3,padding=1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,past):
        x=self.analysis(past);x=x+self.residual(x)
        return x.permute(0,2,3,1).reshape(len(x),16,16,4,8)
    def initialize_pca(self,past):
        data=patch_groups(past).double().reshape(-1,4,12);audit=[]
        with torch.no_grad():
            self.analysis.weight.zero_();self.analysis.bias.zero_()
            for g,(r,c) in enumerate([(0,0),(0,2),(2,0),(2,2)]):
                x=data[:,g];mean=x.mean(0);center=x-mean;cov=center.T@center/len(x);values,basis=torch.linalg.eigh(cov);basis=basis[:,-8:].flip(1).float()
                # Deterministic signs keep PCA initialization reproducible.
                basis*=basis[basis.abs().argmax(0),torch.arange(8)].sign()
                self.analysis.weight[g*8:(g+1)*8,:,r:r+2,c:c+2]=basis.T.reshape(8,3,2,2)
                self.analysis.bias[g*8:(g+1)*8]=-(mean.float()@basis)
                audit.append(dict(group=g,retained_energy=float(values[-8:].sum()/values.sum()),mean=mean.tolist()))
        exact=(self(past)-torch.stack([(data[:,g].float()-data[:,g].mean(0).float())@self.analysis.weight[g*8:(g+1)*8,:,r:r+2,c:c+2].reshape(8,12).T for g,(r,c) in enumerate([(0,0),(0,2),(2,0),(2,2)])],1).reshape(len(past),16,16,4,8)).abs().max()
        assert float(exact)<2e-6
        return dict(groups=audit,initial_PCA_forward_maxabs=float(exact),source='TRAIN past pixel patches only; common to all arms; all writer weights remain trainable')


class FutureSelector(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(5,48,3,padding=1),nn.SiLU(),nn.Conv2d(48,64,4,stride=2,padding=1),nn.SiLU(),nn.Conv2d(64,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,1024,1))
    def forward(self,future,phase):
        p=phase[:,:,None,None].expand(-1,-1,32,32);x=self.net(torch.cat((future,p),1))
        return x.permute(0,2,3,1).reshape(len(x),16,16,4,256)


class PixelReader(nn.Module):
    def __init__(self):
        super().__init__();self.first=nn.Sequential(nn.Conv2d(69,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,64,3,padding=1),nn.SiLU());self.up=nn.ModuleList([nn.Conv2d(64,48,3,padding=1),nn.Conv2d(48,32,3,padding=1),nn.Conv2d(32,24,3,padding=1)]);self.last=nn.Conv2d(24,3,3,padding=1);nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)
    def forward(self,feature,side_feature,future,phase):
        b=len(feature);a=torch.cat((feature,side_feature),-1).reshape(b,16,16,64).permute(0,3,1,2)
        f=F.interpolate(future,(16,16),mode='bilinear',align_corners=False);p=phase[:,:,None,None].expand(-1,-1,16,16);x=self.first(torch.cat((a,f,p),1))
        for layer in self.up:x=F.silu(layer(F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)))
        return F.interpolate(future,(128,128),mode='bicubic',align_corners=False,antialias=True)+self.last(x)


class PixelCodec(nn.Module):
    def __init__(self,fine,coarse,mode):
        super().__init__();self.writer=PastWriter();self.selector=FutureSelector();self.reader=PixelReader();self.quantizer=DecoderSideCodec(fine,coarse,mode)
    def forward(self,past,future,phase):
        feature=self.writer(past);logits=self.selector(future,phase);q=self.quantizer(feature,logits);rgb=self.reader(q['feature'],q['side_feature'],future,phase)
        return dict(rgb=rgb,writer_feature=feature,logits=logits,**q)
    @torch.no_grad()
    def write(self,past):return self.quantizer.write(self.writer(past))
    def read(self,blob,future,phase):
        logits=self.selector(future,phase);q=self.quantizer.read(blob,logits);return dict(rgb=self.reader(q['feature'],q['side_feature'],future,phase),logits=logits,**q)


def fit_books(features,seed=17,iterations=20):
    """One common, TRAIN-only20-step deterministic Lloyd fit per book/group."""
    pools=features.reshape(-1,4,8).float();permutation=torch.randperm(len(pools),generator=torch.Generator().manual_seed(seed));pools=pools[permutation];books={};audit=[]
    for k in [256,128]:
        groups=[]
        for g in range(4):
            x=pools[:,g];centers=x[torch.linspace(0,len(x)-1,k).long()].clone();empty=[]
            for _ in range(iterations):
                distance=x.square().sum(1,keepdim=True)+centers.square().sum(1)[None]-2*x@centers.T;index=distance.argmin(1);count=torch.bincount(index,minlength=k);sums=torch.zeros_like(centers);sums.index_add_(0,index,x);valid=count>0;centers[valid]=sums[valid]/count[valid,None];empty.append(int((~valid).sum()))
            groups.append(centers);audit.append(dict(K=k,group=g,empty_centers_per_iteration=empty))
        books[k]=torch.stack(groups)
    return books[256],books[128],dict(iterations=iterations,seed=seed,tokens_per_group=len(pools),groups=audit,scope='Common fixed preparation, no learned optimizer updates and no future/VAL/GT used.')
