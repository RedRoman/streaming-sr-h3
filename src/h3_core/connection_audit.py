"""ActualD audit and causal learned-head inputs from fixed H3 responses."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from connection_data import SeparableCamera,VIEWS,SUBSETS
from h3_interface import load_frozen_vae
from observation_quotient import ObservationQuotient


def query_masks(device='cuda'):
    y,x=torch.meshgrid(torch.arange(256,device=device),torch.arange(256,device=device),indexing='ij')
    private=((x<32)|(x>=224))&((y<32)|(y>=224));near=((x>=32)&(x<160)&(y>=32)&(y<160))|((x>=96)&(x<224)&(y>=96)&(y<224))
    return private,~(private|near)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--comfy',required=True);p.add_argument('--vae',required=True);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(Path(args.comfy).resolve()))
    root=Path(args.root);manifest=json.loads((root/'manifest.json').read_text());vae,info=load_frozen_vae(args.vae);q=ObservationQuotient();cams={k:SeparableCamera(v,q) for k,v in VIEWS.items()};private,globalquery=query_masks();result={'scope':'Fixed4train2val actualH3 nonlinearresponse audit; HR usedonlyinmetrics, notconditionalgeneration/learnerfeatures','vae':info,'sources':{},'all56_sample_records_complete':True}
    def decode(z):return vae.decode(z.half())[:,:,21].cuda().float()
    with torch.no_grad():
        for source in manifest['sources']:
            folder=root/source['id'];inputs=torch.load(folder/'inputs.pt',map_location='cpu',weights_only=False);initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);target=torch.load(folder/'supervision.pt',map_location='cpu',weights_only=False)['target'].cuda().permute(2,0,1)[None].float()/255
            z0=initial['z0'].cuda();basis=initial['basis'].cuda();reference=initial['rgb_reference'].cuda();observations={k:v.cuda() for k,v in inputs['observations'].items()};latents={};guides={};responses={};rgb_candidates={};cuefeatures={}
            def metric(rgb):
                error=(rgb-target).square();delta=q.null(rgb-reference)
                return {'psnr_full_db':float(-10*torch.log10(error.mean().clamp_min(1e-12))),'psnr_global_query_db':float(-10*torch.log10(error[:,:,globalquery].mean().clamp_min(1e-12))),'private_query_revision_rms':float(delta[:,:,private].square().mean().sqrt()),'global_query_revision_rms':float(delta[:,:,globalquery].square().mean().sqrt()),'far_lr_rmse':float((q.observe(rgb)-observations['far']).square().mean().sqrt()),'A_lr_rmse':float((cams['A'].observe(rgb)-observations['A']).square().mean().sqrt()),'B_lr_rmse':float((cams['B'].observe(rgb)-observations['B']).square().mean().sqrt()),'highfrequency_rms':float(q.null(rgb).square().mean().sqrt())}
            source_result={'source':source,'U_protocol':initial['basis_protocol'],'initial':metric(reference),'classical':{},'conditional':{},'interaction':{}}
            for subset in SUBSETS:
                samples=[];guide_samples=[]
                for seed in [101,202]:
                    where=folder/f'response_{subset}_seed{seed}';record=json.loads((where/'sample_result.json').read_text());assert record['status']=='complete' and len(record['forward_events'])==49 and all(v['video_finite'] and v['audio_finite'] for v in record['forward_events']);samples.append(load_file(str(where/'sampled_av_latents.safetensors'))['video'].float());guide_samples.append(load_file(str(where/'guide.safetensors'))['guide'].float())
                latents[subset]=torch.cat(samples).cuda();guides[subset]=guide_samples[0].cuda()
                source_result['classical'][subset]=metric(initial['classical_rgb'][subset].cuda())
            for subset in SUBSETS:
                responses[subset]=latents[subset]-latents['empty'];rgb_candidates[subset]=[];rows=[];coverage=[]
                for candidate in range(2):
                    r=responses[subset][candidate:candidate+1];shared=torch.einsum('ck,bcthw->bkthw',basis,r);projected=torch.einsum('ck,bkthw->bcthw',basis,shared);rgb=q.bounded(decode(z0+projected),observations['far']);rgb_candidates[subset].append(rgb.cpu());rows.append(metric(rgb));coverage.append(float(projected.square().sum()/r.square().sum().clamp_min(1e-20)))
                source_result['conditional'][subset]={'shared_carrier_candidates':rows,'U_response_energy_fraction':coverage,'raw_candidate_pair_rms':float((latents[subset][0]-latents[subset][1]).square().mean().sqrt())}
            interaction=latents['AB']-latents['A']-latents['B']+latents['empty'];guide_interaction=guides['AB']-guides['A']-guides['B']+guides['empty'];source_result['interaction']={'latent_relative':float(interaction.norm()/responses['AB'].norm().clamp_min(1e-20)),'guide_relative':float(guide_interaction.norm()/(guides['AB']-guides['empty']).norm().clamp_min(1e-20)),'caveat':'Nonlinear conditionalpipeline includes classicaljoint-guide/Einteraction; this statistic doesnot isolate pureDiT curvature.'}
            if 'second_natural_initial' in initial:source_result['natural_initial_pair_rms']=initial['second_natural_initial']['rgb_pair_rms']
            for key in ['A','B']:
                residual=(observations[key]-cams[key].observe(reference))/.005;lifted=cams[key].adjoint(residual);coverage=cams[key].adjoint(torch.ones_like(observations[key][:,:1]));coverage=coverage/coverage.max().clamp_min(1e-12)
                cuefeatures[key]=torch.cat([F.adaptive_avg_pool2d(lifted,(16,16)),F.adaptive_avg_pool2d(coverage,(16,16)),torch.full((1,1,16,16),.5,device='cuda')],1)[:,:,None].repeat(1,1,7,1,1).cpu()
            torch.save({'z0':initial['z0'],'basis':initial['basis'],'rgb_reference':initial['rgb_reference'],'observations':inputs['observations'],'packet_ids':inputs['packet_ids'],'responses':{k:v.cpu() for k,v in responses.items()},'cue_features':cuefeatures,'source':source,'private_mask':private.cpu(),'global_query_mask':globalquery.cpu()},folder/'learning_inputs.pt')
            torch.save({'rgb_candidates':rgb_candidates,'latents':{k:v.cpu() for k,v in latents.items()}},folder/'ensemble_control.pt')
            result['sources'][source['id']]=source_result;(root/'response_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({'event':'actualDresponseaudit','source':source['id'],'interaction':source_result['interaction'],'initial_global_db':source_result['initial']['psnr_global_query_db'],'ABshared_global_db':[v['psnr_global_query_db'] for v in source_result['conditional']['AB']['shared_carrier_candidates']]}),flush=True)
    result['training_decision']='Complete fixed4headsx300 whenever finite/nonzero, includingweakrawglobalheadroom; no selection/sweep.';(root/'response_audit.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
