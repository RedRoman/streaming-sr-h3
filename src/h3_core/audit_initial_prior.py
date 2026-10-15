"""Reveal audit of three fixed true-H3 proposals, not a method benchmark.

All proposals exactly share first-LR DC. A newly observed real LR packet tests
their compatibility. Exact three-atom Bayes weights and a fitted rank<=2
Gaussian update are strong diagnostics with explicit three-sample prior cost.
No HR is used to choose weights, coefficients, or samples.
"""
import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint,nuisance_project
from observation_quotient import ObservationQuotient


def main():
    p=argparse.ArgumentParser();p.add_argument('--proposals',required=True);p.add_argument('--item',required=True);p.add_argument('--out',required=True);args=p.parse_args();torch.set_num_threads(4)
    root=Path(args.proposals);manifest=json.loads((root/'manifest.json').read_text());item=torch.load(args.item,map_location='cpu',weights_only=False);packet=item['inputs']['packets'][17];observed=packet['rgb'].permute(2,0,1)[None].double()/255
    samples=torch.cat([torch.load(root/f'seed{seed}'/'immutable_identity.pt',map_location='cpu',weights_only=False)['rgb'] for seed in manifest['seeds']]).double();q=ObservationQuotient(device='cpu')
    geometry=dict(canonical_rectangle=item['source']['canonical_rectangle'],view_rectangle=packet['world_rectangle']);predict=lambda rgb:camera_footprint(rgb,**geometry)
    valid=packet['visible_mask']>0;observed_vector=observed[0,:,valid].flatten();predicted=predict(samples)[:,:,valid].flatten(1)
    # Shared instantaneous observed-LR bias/gain nuisance basis, no GT.
    bias=torch.eye(3,dtype=torch.float64)[:,:,None,None].expand(-1,-1,64,64);mask=valid[None].expand(3,-1,-1);nuisance=torch.cat([bias,bias*observed],0)[:,mask].T
    design=predicted.T;design,_=nuisance_project(design,torch.zeros_like(observed_vector),nuisance,damping=1e-6)
    _,target=nuisance_project(torch.zeros(len(observed_vector),1,dtype=torch.float64),observed_vector,nuisance,damping=1e-6)
    residual=design.T-target;sigmasq=.005**2;loglike=-.5*residual.square().sum(1)/sigmasq;weights=torch.softmax(loglike-loglike.max(),0)
    mean=samples.mean(0,keepdim=True);directions=(samples-mean)/(len(samples)-1)**.5;near_directions=predict(directions)[:,:,valid].flatten(1).T
    near_directions,mean_residual=nuisance_project(near_directions,(observed-predict(mean))[0,:,valid].flatten(),nuisance,damping=1e-6)
    gram=near_directions.T@near_directions/sigmasq;natural=near_directions.T@mean_residual/sigmasq;coefficient=torch.linalg.solve(torch.eye(len(samples),dtype=torch.float64)+gram,natural)
    gaussian_mean=mean+(directions*coefficient[:,None,None,None]).sum(0,keepdim=True);particle_mean=(samples*weights[:,None,None,None]).sum(0,keepdim=True)
    gt=item['supervision']['canonical_hr'].permute(2,0,1)[None].double()/255;first=item['inputs']['coarse'][0:1].double()
    def metric(rgb):
        error=(observed-predict(rgb))[0,:,valid].flatten();_,clean=nuisance_project(torch.zeros(len(error),1,dtype=torch.float64),error,nuisance,damping=1e-6)
        return {'psnr_db':float(-10*torch.log10((rgb-gt).square().mean())),'near_real_LR_rmse':float(error.square().mean().sqrt()),'near_LR_schur_rmse':float(clean.square().mean().sqrt()),'first_real_LR_rmse':float((q.observe(rgb)-first).square().mean().sqrt())}
    near_variance=near_directions.square().sum(1)+sigmasq;coverage=float((mean_residual.abs()<3*near_variance.sqrt()).double().mean())
    result={'scope':'TRAIN one-source prior-support/reveal diagnostic, not model selection or population calibration. All controls here cost3actual49callH3initializations; no extraDiTupdates.','proposal_seeds':manifest['seeds'],'actual_initial_full50block_forwards':49*len(samples),'near_packet_index':17,'likelihood':'Gaussian effectiveSigma.005 plus common observedLR perchannel bias/gain Schur; no HR weighting','exact_three_atom_posterior_weights':weights.tolist(),'near_schur_rmse_each_seed':residual.square().mean(1).sqrt().tolist(),'effective_atoms':float(1/weights.square().sum()),'fitted_gaussian_coefficient_norm':float(coefficient.norm()),'fitted_gaussian_gram_eigenvalues':torch.linalg.eigvalsh(gram).tolist(),'observed_near_fraction_within_empirical3sigma':coverage,'metrics':{'prior_mean':metric(mean),'exact_three_atom_mean':metric(particle_mean),'same_three_samples_gaussian_map':metric(gaussian_mean)},'caution':'Three draws cannot establish full H3 support/calibration. Reweighting cannot create a new detail absent from all draws; any next learned transport must test newly generated evidence-compatible detail and unrelated-identity retention beyond these diagnostics.'}
    out=Path(args.out);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
