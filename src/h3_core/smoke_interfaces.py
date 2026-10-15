"""Synthetic tensor/API checks. This is NOT an actual H3 weight smoke."""
import json
from pathlib import Path
import torch
from memory_adapter import CanonicalChart,PacketLedger,SpectralDetailTransport,PosteriorDetailHead,bures_covariance_map
from h3_interface import stream_blocks,guide_payload

torch.manual_seed(29);torch.set_num_threads(2)
anchor=torch.randn(2,24,1,8,8)*.1
chart=CanonicalChart();matrix,prior,context=chart(anchor[:,:,0,2:6,2:6])
orth=float((matrix.transpose(1,2)@matrix-torch.eye(8)).abs().max().detach())
assert orth<1e-5
records=[]
for batch in range(2):
    ledger=PacketLedger();snapshots=[]
    design=torch.randn(20,32)*.02
    for t in range(5):
        packet_id='p1' if t==2 else f'p{t}'
        d=design if t in (1,2,3) else torch.randn(20,32)*.02
        ledger.write(d,torch.randn(20)*.02,packet_id,t,.01)
        snapshots.append(ledger.snapshot())
    records.append(snapshots)
grams=torch.stack([torch.stack([s.gram for s in row]) for row in records])
rhs=torch.stack([torch.stack([s.rhs for s in row]) for row in records])
counts=torch.tensor([[s.packets for s in row] for row in records])
noise=torch.randn(2,8);fresh=torch.randn(2,5,8)
model=SpectralDetailTransport()
results={}
for mode in ['learned','map','analytic_transport','shared_seed','fresh_noise','bures_transport']:
    full,diag=model(matrix,prior,context,grams,rhs,counts,mode,noise,fresh)
    prefix,_=model(matrix,prior,context,grams[:,:3],rhs[:,:3],counts[:,:3],mode,noise,fresh[:,:3])
    error=float((full[:,:3]-prefix).abs().max())
    duplicate=float((full[:,2]-full[:,1]).abs().max())
    assert error==0,(mode,error)
    assert duplicate==0,(mode,duplicate)
    assert float(diag[3]['logdet_information_gain'].min())>0
    results[mode]={'prefix_max_abs':error,'duplicate_max_abs':duplicate,
                   'independent_same_footprint_information_min':float(diag[3]['logdet_information_gain'].min())}
    if mode=='learned':
        loss=full.square().mean();loss.backward(retain_graph=True)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(chart.parameters())+list(model.parameters()))
        results[mode]['backward_finite']=True
    if mode=='map':deterministic=full.detach()
    if mode=='analytic_transport':analytic=full.detach()
assert float((analytic-deterministic).norm())>.01
head_parameters={'transport':sum(p.numel() for p in model.parameters())}
for kind in ['stateless','gru','single_packet']:
    head=PosteriorDetailHead(kind)
    out=head(matrix.detach(),prior.detach(),context.detach(),grams,rhs,counts,noise)
    prefix=head(matrix.detach(),prior.detach(),context.detach(),grams[:,:3],rhs[:,:3],counts[:,:3],noise)
    assert torch.equal(out[:,:3],prefix)
    assert torch.equal(out[:,2],out[:,1])
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    head_parameters[kind]=sum(p.numel() for p in head.parameters())
blocks=stream_blocks();assert [(b.start,b.stop,b.emit_start) for b in blocks]==[(0,22,0),(17,39,22)]
payload=guide_payload([(0,anchor[:1],21)],blocks[0])
assert 'denoise_mask' not in payload and payload['cond_video_latents'][0] is payload['keyframes'][0]['latent']
future_rejected=False
try:guide_payload([(0,anchor[:1],22)],blocks[0])
except ValueError:future_rejected=True
assert future_rejected
result={'passed':True,'scope':'H3-shaped synthetic tensors; actual H3 VAE weights not used by this check',
        'basis_orthogonality_max_abs':orth,'controls':results,'nonzero_analytic_vs_map_norm':float((analytic-deterministic).norm()),
        'blocks':[(b.start,b.stop,b.emit_start) for b in blocks],'unavailable_future_guide_rejected':future_rejected}
result['equal_budget_neural_head_parameters']=head_parameters
q=torch.linalg.qr(torch.randn(8,8,dtype=torch.float64)).Q
old=q@torch.diag(torch.logspace(-8,-3,8,dtype=torch.float64))@q.T
q2=torch.linalg.qr(torch.randn(8,8,dtype=torch.float64)).Q
new=q2@torch.diag(torch.logspace(-9,-4,8,dtype=torch.float64))@q2.T
transport=bures_covariance_map(old[None],new[None])[0]
covariance_error=float((transport@old@transport.T-new).norm()/new.norm())
assert covariance_error<1e-7,covariance_error
result['bures_ill_scaled_covariance_relative_error']=covariance_error
output=Path(__file__).resolve().parents[2]/'results/h3_core/interface_smoke.json'
output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
