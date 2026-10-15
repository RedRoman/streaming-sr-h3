"""VAL-ONLY structural audit of a classical observation-quotient readout.

No learned-model tuning. HR-code transfer and per-image RGB optimization are
explicit capacity diagnostics, never valid inference controls or test results.
Existing fixed training continues unchanged before this separate audit runs.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint
from h3_data import NaturalCameraSequence
from h3_interface import load_frozen_vae,differentiable_decoder_math
from observation_quotient import ObservationQuotient,SameFootprintCoarseMean
from train_nonlinear import masks,rgb_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True)
    p.add_argument('--wait-suite');p.add_argument('--sources',type=int,default=8);p.add_argument('--oracle-rgb-steps',type=int,default=80)
    args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    if args.wait_suite:
        while True:
            try:state=json.loads((Path(args.wait_suite)/'suite_state.json').read_text())
            except (OSError,json.JSONDecodeError):state={'jobs':[]}
            if len(state['jobs'])==4:
                if any(v['exit_code'] for v in state['jobs']):raise ValueError('Fixed suite failed; do not silently continue')
                break
            time.sleep(10)
    sys.path.insert(0,str(Path(args.comfy).resolve()));vae,vae_info=load_frozen_vae(args.vae)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);q=ObservationQuotient()
    result={'config':vars(args),'scope':'VAL-only observation-quotient structural audit; no trained-method claim; per-image HR optimizers are explicitly diagnostic','vae':vae_info,'projection':'Classical A0^+ybar+(I-A0^+A0)D(z), exact declared Torch Gaussian+bicubic, not exact Pillow simulator','items':{}}
    for path in sorted(Path(args.cache).glob('val_*.pt'))[:args.sources]:
        item=torch.load(path,map_location='cpu',weights_only=False);z=item['inputs']['anchor'].cuda();y,x=item['inputs']['roi'];p0=item['teacher']['p0'].cuda()
        responses=item['teacher']['decoder_responses'].cuda();null_responses=q.null(responses);base=item['teacher']['base_rgb'].cuda()
        sequence=NaturalCameraSequence(item['source']['path'],size=256,frames=39,protocol=item['source']['protocol'],seed=item['source']['sequence_seed'])
        packets=list(sequence.packets());coarse=SameFootprintCoarseMean(item['source']['canonical_rectangle'],(1,3,64,64));coarse.write(packets[0])
        target=item['supervision']['canonical_hr'].cuda().permute(2,0,1)[None].float()/255;roi,support=masks(item,'cuda')
        gram=torch.zeros(32,32,device='cuda',dtype=torch.float64);prefix={};projected_maps={};variance=.005**2
        def operator(rgb,packet):
            return camera_footprint(rgb,item['source']['canonical_rectangle'],packet.world_rectangle,view_size=(256,256),lr_size=(64,64))
        # Geometry-only projected evidence audits all39prefixes, with no HR.
        for t,packet in enumerate(packets):
            coarse.write(packet)
            images=operator(null_responses,packet)
            cover=operator(torch.ones_like(base[:,:1]),packet)[0,0]
            valid=(cover>.999)&(cover<1.001)&(torch.as_tensor(packet.visible_mask.copy(),device='cuda')>0)
            mask=valid[None].expand(3,-1,-1);design=images[:,mask].T.double()
            gram+=design.T@design/variance
            if t in [0,17,38]:
                values=torch.linalg.eigvalsh(gram)
                prefix[t]={'direction_rms':float(images.square().mean().sqrt()),'gram_eigenvalues':values.tolist(),'effective_directions_lambda_gt1':int((values>1).sum()),'coarse_packets_written':len(coarse.packet_ids)}
                # Rebase ALL past residuals when independent same-footprint
                # coarse measurements update ybar. Old h must not stay stale.
                projected_base=q.readout(base,coarse.mean());natural=torch.zeros(32,device='cuda',dtype=torch.float64)
                for past in packets[:t+1]:
                    b=operator(null_responses,past);coverage=operator(torch.ones_like(base[:,:1]),past)[0,0]
                    v=(coverage>.999)&(coverage<1.001)&(torch.as_tensor(past.visible_mask.copy(),device='cuda')>0);m=v[None].expand(3,-1,-1)
                    observed=torch.as_tensor(past.rgb.copy(),device='cuda').permute(2,0,1).float()/255
                    residual=(observed-operator(projected_base,past)[0])[m].double()
                    natural+=b[:,m].double()@residual/variance
                projected_maps[t]=torch.linalg.solve(gram+torch.eye(32,device='cuda',dtype=torch.float64),natural).float()
        ybar=coarse.mean()
        record={'protocol':item['source']['protocol'],'source_hash':item['source']['source_hash'],'high_frequency_response_energy_fraction':float(null_responses.square().sum()/responses.square().sum()),'same_footprint_null_rms':float(q.observe(null_responses).square().mean().sqrt()),'prefix':prefix,'rgb':{}}
        def embed(coefficient):
            anchor=z.clone();anchor[:,:,:,y:y+4,x:x+4]+=(p0@coefficient).reshape(1,24,7,4,4);return anchor
        def decode(coefficient):return vae.decode(embed(coefficient).half())[:,:,21].cuda()
        coefficients={'anchor':torch.zeros(32,device='cuda'),'projected_map':projected_maps[38],'local_hr_code_transfer_diagnostic':item['supervision']['target_p0'][0].cuda()}
        panels=[]
        with torch.no_grad():
            for name,coefficient in coefficients.items():
                rgb=q.readout(decode(coefficient),ybar)
                record['rgb'][name]={**rgb_metrics(rgb,target,roi,support),'outside_rgb_box_fraction':float(((rgb<0)|(rgb>1)).float().mean()),'exact_torch_lr_rmse':float((q.observe(rgb)-ybar).square().mean().sqrt()),'code_norm':float(coefficient.norm())}
                panels.append((name,rgb.cpu()))
        if args.oracle_rgb_steps:
            # The diagnostic optimizes32coordinates against this VAL image's
            # HR, demonstrating capacity only. It is not a trained predictor.
            coefficient=torch.zeros(32,device='cuda',requires_grad=True);optimizer=torch.optim.Adam([coefficient],lr=.1)
            for step in range(args.oracle_rgb_steps):
                with differentiable_decoder_math():rgb=q.readout(decode(coefficient),ybar)
                loss=(rgb-target).square()[:,:,roi].mean();optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            with torch.no_grad():
                rgb=q.readout(decode(coefficient),ybar);bounded=q.bounded(rgb,ybar)
                record['rgb']['per_image_hr_optimized_capacity_diagnostic']={**rgb_metrics(rgb,target,roi,support),'code_norm':float(coefficient.norm()),'exact_torch_lr_rmse':float((q.observe(rgb)-ybar).square().mean().sqrt()),'steps':args.oracle_rgb_steps,'not_inference':True}
                record['rgb']['same_oracle_with_box_control']={**rgb_metrics(bounded,target,roi,support),'exact_torch_lr_rmse':float((q.observe(bounded)-ybar).square().mean().sqrt()),'not_inference':True}
                panels.append(('hr_optimized_capacity',rgb.cpu()))
        result['items'][path.stem]=record
        arrays=[item['supervision']['canonical_hr'].numpy()]+[(v[0].permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8) for _,v in panels]
        Image.fromarray(np.concatenate(arrays,1)).save(out/f'{path.stem}.png')
        (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps({'event':'quotient_audit','source':path.stem,'hf_response_energy_fraction':record['high_frequency_response_energy_fraction'],'prefix_effective_directions':{k:v['effective_directions_lambda_gt1'] for k,v in prefix.items()},'rgb_roi_psnr':{k:v['psnr_nominal_roi_db'] for k,v in record['rgb'].items()}}),flush=True)
    print(json.dumps({'event':'complete','val_sources':len(result['items'])}),flush=True)


if __name__=='__main__':main()
