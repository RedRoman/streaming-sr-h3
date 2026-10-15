"""Learned canonical-coordinate memory for frozen MiniMax-H3 guides.

The observation teacher is a *local decoder linearization*, not an exact latent
inverse operator. Its RGB reprojection error must be audited with the real VAE.
All evidence enters via LR packets; rendered/generated guides are never data.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


def bures_covariance_map(old_cov,new_cov):
    """Stable Gaussian Bures control; tiny covariance products need no ABS floor."""
    old=.5*(old_cov.double()+old_cov.double().transpose(-1,-2))
    new=.5*(new_cov.double()+new_cov.double().transpose(-1,-2))
    def sqrt_psd(cov,inverse=False):
        cov=.5*(cov+cov.transpose(-1,-2))
        values,vectors=torch.linalg.eigh(cov)
        floor=values.abs().amax(-1,keepdim=True).clamp_min(torch.finfo(cov.dtype).tiny)*torch.finfo(cov.dtype).eps
        powers=values.clamp_min(floor).pow(-.5 if inverse else .5)
        return (vectors*powers[...,None,:])@vectors.transpose(-1,-2)
    old_sqrt=sqrt_psd(old);old_inv=sqrt_psd(old,inverse=True)
    transport=old_inv@sqrt_psd(old_sqrt@new@old_sqrt)@old_inv
    return .5*(transport+transport.transpose(-1,-2))


def dct_atoms(channels=24, tile=4, atoms=32):
    """Fixed orthonormal channel × spatial DCT span; no HR-dependent atoms."""
    candidates=[]
    for fy in range(tile):
        for fx in range(tile):
            for channel in range(channels):
                yy=torch.arange(tile).float()+.5
                spatial=torch.cos(math.pi*fy*yy[:,None]/tile)*torch.cos(math.pi*fx*yy[None,:]/tile)
                spatial=spatial/spatial.norm()
                atom=torch.zeros(channels,tile,tile);atom[channel]=spatial
                candidates.append(atom.flatten())
    # Include low and high spatial modes while retaining channel diversity.
    generator=torch.Generator().manual_seed(3208)
    indices=torch.randperm(len(candidates),generator=generator)[:atoms]
    basis=torch.stack([candidates[int(i)] for i in indices],1)
    assert torch.allclose(basis.T@basis,torch.eye(atoms),atol=1e-6)
    return basis


class CanonicalChart(nn.Module):
    """A learned rank-eight basis inside a fixed rank-32 decoder teacher span.

    Compute C(anchor) once per object/stream. Changing network weights while
    training changes C, while cached K0 remains valid because B=K0@C.
    """
    def __init__(self, channels=24, tile=4, atoms=32, rank=8, width=64,temporal=1,basis_mode='learned'):
        super().__init__();self.tile=tile;self.atoms=atoms;self.rank=rank;self.temporal=temporal;self.basis_mode=basis_mode
        dimension=channels*temporal*tile*tile
        self.register_buffer('p0',dct_atoms(channels*temporal,tile,atoms))
        self.context=nn.Sequential(nn.Linear(dimension,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.basis=nn.Linear(width,atoms*rank)
        self.prior=nn.Linear(width,rank)
        nn.init.normal_(self.basis.weight,std=.01)
        fixed=torch.eye(atoms,rank)
        if basis_mode=='random':
            generator=torch.Generator().manual_seed(80129)
            fixed=torch.linalg.qr(torch.randn(atoms,rank,generator=generator)).Q
        self.register_buffer('fixed_coordinates',fixed)

    def forward(self,anchor_roi):
        context=self.context(anchor_roi.flatten(1))
        raw=self.fixed_coordinates[None]+self.basis(context).reshape(-1,self.atoms,self.rank)
        # QR is over fixed canonical coordinates, never over future frames.
        matrix=torch.linalg.qr(raw,mode='reduced').Q if self.basis_mode=='learned' else self.fixed_coordinates[None].expand(anchor_roi.shape[0],-1,-1)
        prior=self.prior(context)
        return matrix,prior,context

    def embed(self,anchor_latent,matrix,coefficients,roi):
        """Return native H3 normalized guide [B,24,1,Hlat,Wlat]."""
        y,x=roi
        patch=torch.einsum('dk,bkr,br->bd',self.p0,matrix,coefficients)
        patch=patch.reshape(anchor_latent.shape[0],24,self.temporal,self.tile,self.tile)
        out=anchor_latent.clone()
        out[:,:,:self.temporal,y:y+self.tile,x:x+self.tile]+=patch
        return out


@dataclass
class EvidenceSnapshot:
    gram: torch.Tensor
    rhs: torch.Tensor
    packets: int
    last_observed_index: int


class PacketLedger:
    """Fixed-coordinate sufficient statistics; independent packets add precision."""
    def __init__(self,atoms=32,device='cpu',dtype=torch.float32):
        self.gram=torch.zeros(atoms,atoms,device=device,dtype=dtype)
        self.rhs=torch.zeros(atoms,device=device,dtype=dtype)
        self.packet_ids=set();self.last_observed_index=-1

    def write(self,design,residual,observation_id,index,variance):
        if observation_id in self.packet_ids:return False
        if index<self.last_observed_index:raise ValueError('Packets must arrive in source order')
        self.gram=self.gram+design.T@design/variance
        self.rhs=self.rhs+design.T@residual/variance
        self.packet_ids.add(observation_id);self.last_observed_index=index
        return True

    def snapshot(self):
        return EvidenceSnapshot(self.gram.clone(),self.rhs.clone(),len(self.packet_ids),self.last_observed_index)


class SpectralDetailTransport(nn.Module):
    """Nonlinear transport in a posterior-whitened canonical detail chart.

    Cholesky posterior coupling is the analytic control. The trainable component
    learns non-Gaussian detail transitions from new packet information. The
    information trigger uses precision/logdet gain, including same-footprint
    independent noisy measurements. Exact packet copies cause no transition.
    """
    def __init__(self,rank=8,context=64,width=64,prior_precision=1.):
        super().__init__();self.rank=rank;self.prior_precision=prior_precision
        features=rank*4+context+2
        self.net=nn.Sequential(nn.Linear(features,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,rank*2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,matrix,prior,context,grams,rhs,packet_counts,mode='learned',identity_noise=None,fresh_identity_noises=None):
        batch,_,rank=matrix.shape
        eye=torch.eye(rank,device=matrix.device,dtype=matrix.dtype)[None].expand(batch,-1,-1)
        old_mean=prior;old_cov=eye/self.prior_precision
        if mode in ('analytic_transport','shared_seed','fresh_noise','bures_transport') and identity_noise is None:
            raise ValueError('Stochastic coupling controls require explicit nonzero identity noise')
        coefficient=prior if identity_noise is None else prior+identity_noise/math.sqrt(self.prior_precision)
        old_logdet=torch.full((batch,),rank*math.log(self.prior_precision),device=matrix.device,dtype=matrix.dtype)
        old_packets=torch.zeros(batch,device=matrix.device,dtype=packet_counts.dtype)
        outputs=[];diagnostics=[]
        for t in range(grams.shape[1]):
            gram=matrix.transpose(1,2)@grams[:,t]@matrix
            natural=(matrix.transpose(1,2)@rhs[:,t,:,None])[...,0]
            precision=gram+self.prior_precision*eye
            covariance=torch.linalg.inv(precision)
            mean=torch.linalg.solve(precision,(natural+self.prior_precision*prior)[...,None])[...,0]
            logdet=torch.linalg.slogdet(precision).logabsdet
            info=(logdet-old_logdet).clamp_min(0)
            fresh=(packet_counts[:,t]>old_packets)
            if mode=='map':new_coefficient=mean
            elif mode in ('shared_seed','fresh_noise','bures_transport'):
                new_chol=torch.linalg.cholesky(covariance+eye*1e-6)
                if mode=='bures_transport':
                    transport=bures_covariance_map(old_cov,covariance)
                    residual=(transport@(coefficient-old_mean).double()[...,None])[...,0].to(mean)
                    new_coefficient=mean+residual
                else:
                    if mode=='fresh_noise':
                        if fresh_identity_noises is None:raise ValueError('Pass explicit noise per step for fresh-noise control')
                        noise=fresh_identity_noises[:,t]
                    else:noise=identity_noise
                    new_coefficient=mean+(new_chol@noise[...,None])[...,0]
            else:
                old_chol=torch.linalg.cholesky(old_cov+eye*1e-6)
                new_chol=torch.linalg.cholesky(covariance+eye*1e-6)
                white=torch.linalg.solve_triangular(old_chol,(coefficient-old_mean)[...,None],upper=False)[...,0]
                eigenvalues=torch.linalg.eigvalsh(gram.detach()).clamp_min(0).log1p()
                features=torch.cat([mean,coefficient,mean-old_mean,eigenvalues,context,info[:,None],torch.log1p(packet_counts[:,t].float())[:,None]],1)
                delta,gates=self.net(features).chunk(2,1)
                trigger=-torch.expm1(-info/rank)
                if mode=='analytic_transport':delta=delta*0
                white=white+trigger[:,None]*gates.sigmoid()*delta
                new_coefficient=mean+(new_chol@white[...,None])[...,0]
            # Override numerical Cholesky noise on exact duplicate packets.
            coefficient=torch.where(fresh[:,None],new_coefficient,coefficient)
            outputs.append(coefficient)
            diagnostics.append({'logdet_information_gain':info,'fresh_packet':fresh})
            old_mean=torch.where(fresh[:,None],mean,old_mean)
            old_cov=torch.where(fresh[:,None,None],covariance,old_cov)
            old_logdet=torch.where(fresh,logdet,old_logdet)
            old_packets=packet_counts[:,t]
        return torch.stack(outputs,1),diagnostics


class PosteriorDetailHead(nn.Module):
    """Same-budget stateless and GRU controls with the SAME analytic ledger.

    Giving GRU the complete posterior mean avoids attributing the bookkeeping
    advantage of an analytic solver to the learned transport architecture.
    """
    def __init__(self,kind='stateless',rank=8,context=64,prior_precision=1.):
        super().__init__();self.kind=kind;self.rank=rank;self.prior_precision=prior_precision
        features=rank*3+context+2
        if kind=='gru':
            self.cell=nn.GRUCell(features,31);self.output=nn.Linear(31,rank)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        else:
            self.net=nn.Sequential(nn.Linear(features,68),nn.SiLU(),nn.Linear(68,68),nn.SiLU(),nn.Linear(68,rank))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,matrix,prior,context,grams,rhs,packet_counts,identity_noise):
        batch,_,rank=matrix.shape;eye=torch.eye(rank,device=matrix.device)[None].expand(batch,-1,-1)
        old_gram=torch.zeros_like(grams[:,0]);old_rhs=torch.zeros_like(rhs[:,0]);old_logdet=torch.full((batch,),rank*math.log(self.prior_precision),device=matrix.device)
        old_count=torch.zeros_like(packet_counts[:,0]);coefficient=prior+identity_noise/math.sqrt(self.prior_precision)
        h=torch.zeros(batch,31,device=matrix.device);out=[]
        for t in range(grams.shape[1]):
            g0=grams[:,t]-old_gram if self.kind=='single_packet' else grams[:,t]
            r0=rhs[:,t]-old_rhs if self.kind=='single_packet' else rhs[:,t]
            gram=matrix.transpose(1,2)@g0@matrix;natural=(matrix.transpose(1,2)@r0[...,None])[...,0]
            precision=gram+self.prior_precision*eye;covariance=torch.linalg.inv(precision)
            mean=torch.linalg.solve(precision,(natural+self.prior_precision*prior)[...,None])[...,0]
            logdet=torch.linalg.slogdet(precision).logabsdet;info=(logdet-old_logdet).clamp_min(0)
            eigenvalues=torch.linalg.eigvalsh(gram.detach()).clamp_min(0).log1p()
            features=torch.cat([mean,eigenvalues,identity_noise,context,info[:,None],torch.log1p(packet_counts[:,t].float())[:,None]],1)
            fresh=packet_counts[:,t]>old_count
            if self.kind=='gru':
                new_h=self.cell(features,h);h=torch.where(fresh[:,None],new_h,h);delta=self.output(h)
            else:delta=self.net(features)
            chol=torch.linalg.cholesky(covariance+eye*1e-6)
            new_coefficient=mean+(chol@(identity_noise+delta)[...,None])[...,0]
            coefficient=torch.where(fresh[:,None],new_coefficient,coefficient);out.append(coefficient)
            old_count=packet_counts[:,t];old_gram=grams[:,t];old_rhs=rhs[:,t];old_logdet=logdet
        return torch.stack(out,1)
