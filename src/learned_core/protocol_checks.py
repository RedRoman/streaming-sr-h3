"""Small, independent assertions about the causal inference protocol."""
import json
import torch
from experiment import CoreModel, observations, gaussian_map

torch.manual_seed(9)
torch.set_num_threads(2)
target = torch.rand(2,16,4,4)
cov = torch.eye(144)*.1
stats = (torch.full((144,),.5),cov,torch.arange(16)*9+4)
ys,masks = observations(target,199)
bases = gaussian_map(ys,masks,stats)
checks = {}
with torch.no_grad():
    prefix_base=gaussian_map(ys[:3],masks[:3],stats)
    error=max(float((a-b).abs().max()) for a,b in zip(bases[:3],prefix_base))
    assert error==0
    checks['gaussian_map/prefix_causality_max_abs']=error
    for kind in ['current','gru','ledger','transport','map_residual','map_transport']:
        model = CoreModel(kind).eval()
        full = model(ys,masks,bases)
        prefix = model(ys[:3],masks[:3],bases[:3])
        error = max(float((a-b).abs().max()) for a,b in zip(full[:3],prefix))
        assert error == 0, (kind,error)
        checks[kind+'/prefix_causality_max_abs'] = error
    repeated,repeated_masks = observations(target,199,mode='repeat')
    repeated_bases = gaussian_map(repeated,repeated_masks,stats)
    for kind in ['ledger','transport','map_residual','map_transport']:
        model=CoreModel(kind).eval()
        out=model(repeated,repeated_masks,repeated_bases)
        error=max(float((a-out[0]).abs().max()) for a in out)
        assert error == 0,(kind,error)
        checks[kind+'/exact_duplicate_stability_max_abs'] = error
    for kind in ['map_residual','map_transport']:
        model=CoreModel(kind).eval()
        out=model(ys,masks,bases)
        seen=torch.zeros_like(target)
        errors=[]
        for prediction,base,mask in zip(out,bases,masks):
            seen=(seen+mask).clamp_max(1)
            errors.append(float(((prediction-base)*seen).abs().max()))
        assert max(errors)==0,(kind,errors)
        checks[kind+'/observed_direction_projection_max_abs']=max(errors)
    # Same footprint, genuinely independent noisy packets: information adds.
    mask=torch.zeros_like(target);mask[:,0]=1
    independent=[target*mask+.02*mask,target*mask-.02*mask]
    independent_bases=gaussian_map(independent,[mask,mask],stats,.0001,[mask,mask])
    expected=(.1/(.1+.0001/2))*target[:,0]+(.0001/2/(.1+.0001/2))*.5
    error=float((independent_bases[-1][:,0]-expected).abs().max())
    assert error<1e-6,error
    checks['gaussian_map/independent_packet_precision_max_abs']=error
print(json.dumps({'passed':True,'checks':checks},indent=2))
