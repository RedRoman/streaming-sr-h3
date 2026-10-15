"""Actual H3 B2 readout with fixed B1 zero-extra-training strong controls."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from model import FactorizedMemory
from residual_model import ResidualMemory
from residual_readout import ResidualReadout
from native_vae import load_vae,decode_last,CANONICAL_SHA
from train import batch,TARGET,write_live
from evaluate import metrics,save_csv,STAGES,NAMES
from pixel_dc import project_visible


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--runs',required=True);p.add_argument('--b1-runs',required=True);p.add_argument('--observations',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);p.add_argument('--split',choices=['val','test'],default='val');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;start=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'projected').mkdir(exist_ok=True)
    vae=load_vae(args.vae,args.comfy);reader=ResidualReadout(vae);manifest=json.loads((Path(args.cache)/'manifest.json').read_text());assert manifest['initial_proposal'].startswith('explicit zero');items=[x for x in manifest['items'] if x['split']==args.split];assert items
    models=[]
    for path in sorted(Path(args.runs).glob('*/best.pt')):
        ckpt=torch.load(path,map_location='cpu',weights_only=False);assert ckpt['config']['residual_v2'];model=ResidualMemory(ckpt['kind']).cuda().eval();model.load_state_dict(ckpt['model']);models.append((model,ckpt,path))
    assert models
    seeds=sorted({c['seed'] for _,c,_ in models});b1={}
    for seed in seeds:
        ckpt=torch.load(Path(args.b1_runs)/f'factorized_swap_seed{seed}'/'best.pt',map_location='cpu',weights_only=False);model=FactorizedMemory('factorized_swap').cuda().eval();model.load_state_dict(ckpt['model']);b1[seed]=model
    rows=[];dcrows=[];diagnostics=[];capacity=[]
    def record(source,method,seed,stage,rgb,target,measurement):
        # Raw image is boxed for display; invariant is audited BEFORE boxing.
        # DC starts from the same unboxed readout for every method.
        raw=rgb.clamp(0,1);dc,rmse=project_visible(rgb,measurement,stage,reader.far,reader.near)
        key=dict(source=source,split=args.split,method=method,seed=seed,stage=stage,scenario=NAMES[stage]);rows.append(dict(**key,**metrics(raw,target)));dcrows.append(dict(**key,**metrics(dc,target),actual_measurement_rmse=rmse));return dc
    with torch.no_grad():
        for number,prov in enumerate(items):
            data=torch.load(Path(args.cache)/(prov['id']+'.pt'),weights_only=False);inputs,targets=batch([data],'cuda');measurement=torch.load(Path(args.observations)/(prov['id']+'.pt'),weights_only=False)['inputs'];gallery={};gallery_dc={}
            for stage in STAGES:
                target=targets['rgb'][:,TARGET[stage]];baselines={'input_bicubic':inputs['observation_rgb'][:,stage],'H3_current_LR':inputs['base_decoded_rgb'][:,stage],'H3_HR_roundtrip':decode_last(vae,targets['latents'][:,TARGET[stage]])}
                for name,rgb in baselines.items():
                    dc=record(prov['id'],name,-1,stage,rgb,target,measurement)
                    if stage==4:gallery.update(GT=target);gallery[name]=rgb;gallery_dc.update(GT=target);gallery_dc[name]=dc
            for model,ckpt,path in models:
                residual,a,states=model.rollout(inputs);assert not bool(states[0].any());assert torch.equal(states[2],states[4]) and torch.equal(states[5],states[6]) and torch.equal(states[6],states[7]);anchor_errors=[];near_errors=[]
                for stage in STAGES:
                    rgb=reader.rgb(model,inputs,residual[:,stage],a[:,stage],stage);target=targets['rgb'][:,TARGET[stage]];base=inputs['base_decoded_rgb'][:,stage]
                    if stage in [2,5]:near_errors.append(float((rgb-base).abs().max()))
                    else:anchor_errors.append(float(reader.far.observe(rgb-base).square().mean().sqrt()))
                    dc=record(prov['id'],ckpt['kind'],ckpt['seed'],stage,rgb,target,measurement)
                    if stage==4 and ckpt['seed']==17:gallery[ckpt['kind']]=rgb;gallery_dc[ckpt['kind']]=dc
                q=targets['appearance'][:,1];z=targets['detail_latents'][:,1];inv=None
                if ckpt['kind'].startswith('factorized'):inv=float((model.read_detail(model.canonical_detail(z,q),q)-z).abs().max())
                diagnostics.append(dict(source=prov['id'],seed=ckpt['seed'],method=ckpt['kind'],selected_step=ckpt['step'],parameters=ckpt['parameters'],initial_proposal='zero',zero_initial_state=True,far_retains_near_state_exactly=True,query_read_only=True,duplicate_idempotent=True,near_readout_baseline_maxabs=max(near_errors),prebox_current_far_observation_difference_rms=max(anchor_errors),inverse_maxabs=inv,detail_state_norms=[float(s.norm()) for s in states],proposal_is_evidence=False))
                if ckpt['kind']=='factorized_swap':
                    for stage in STAGES:
                        historic=2 if stage in [2,4] else 5 if stage in [5,6] else None
                        direct=torch.zeros_like(residual[:,stage]) if historic is None else model.read_detail(model.canonical_detail(inputs['detail_observation_latents'][:,historic],a[:,historic]),a[:,stage])
                        rgb=reader.rgb(model,inputs,direct,a[:,stage],stage);target=targets['rgb'][:,TARGET[stage]];dc=record(prov['id'],'B2_direct_observed_detail',ckpt['seed'],stage,rgb,target,measurement)
                        if stage==4 and ckpt['seed']==17:gallery['B2_direct_observed_detail']=rgb;gallery_dc['B2_direct_observed_detail']=dc
                        if stage in [4,6]:
                            # Supervised 2x-minus4x detail teacher is an ORACLE
                            # capacity diagnostic, never a valid method/state.
                            teacher=reader.rgb(model,inputs,targets['detail_latents'][:,TARGET[stage]],a[:,stage],stage);teacher_dc,rmse=project_visible(teacher,measurement,stage,reader.far,reader.near);capacity.append(dict(source=prov['id'],seed=ckpt['seed'],stage=stage,oracle_supervised_teacher=True,valid_LR_only_method=False,**metrics(teacher_dc,target),actual_measurement_rmse=rmse))
            # These two structural controls use frozen B1 learned charts and
            # require ZERO additional training. They do not receive HR pixels
            # or HR residuals at inference. Both must remain in B2 comparisons.
            for seed,model in b1.items():
                _,appearance,states=model.rollout(inputs)
                for stage in STAGES:
                    a=appearance[:,stage];base=inputs['base_decoded_rgb'][:,stage];target=targets['rgb'][:,TARGET[stage]]
                    absolute=decode_last(vae,model.chart.read(states[stage],a));dc_abs=record(prov['id'],'B1_absolute_state',seed,stage,absolute,target,measurement)
                    historic=2 if stage in [2,4] else 5 if stage in [5,6] else None
                    if historic is None:retained=direct=base
                    else:
                        retained_rgb=decode_last(vae,model.chart.read(states[historic],a));nearcode=model.chart.canonicalize(inputs['observation_latents'][:,historic],appearance[:,historic]);direct_rgb=decode_last(vae,model.chart.read(nearcode,a));near=stage in [2,5]
                        retained=base+reader.supported_null(retained_rgb-base,near);direct=base+reader.supported_null(direct_rgb-base,near)
                    for name,rgb in [('fixed_B1_retained_detail',retained),('fixed_B1_direct_near',direct)]:
                        dc=record(prov['id'],name,seed,stage,rgb,target,measurement)
                        if stage==4 and seed==17:gallery[name]=rgb;gallery_dc[name]=dc
                    if stage==4 and seed==17:gallery['B1_absolute_state']=absolute;gallery_dc['B1_absolute_state']=dc_abs
            if number<4:
                for dest,collection in [(out,gallery),(out/'projected',gallery_dc)]:
                    w=256;canvas=Image.new('RGB',(w*4,(w+24)*3),'white');draw=ImageDraw.Draw(canvas)
                    for j,(name,rgb) in enumerate(collection.items()):
                        arr=(rgb[0].clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype('uint8');canvas.paste(Image.fromarray(arr),((j%4)*w,(j//4)*(w+24)));draw.text(((j%4)*w+2,(j//4)*(w+24)+w+2),name,fill='black')
                    canvas.save(dest/(prov['id']+'_return_A0.png'))
            save_csv(out/'rgb_metrics.csv',rows);save_csv(out/'projected/rgb_metrics.csv',dcrows);(out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));write_live(out,status='B2_fixed_protocol_actual_D_evaluation',phase=args.split+'_B2_and_strong_frozen_B1_controls',step=number+1,steps=len(items),source=prov['id'],seconds=time.time()-start);print(json.dumps(dict(source=prov['id'],rows=len(rows),seconds=time.time()-start)),flush=True)
            if capacity:save_csv(out/'oracle_teacher_capacity_rgb.csv',capacity)
    config=dict(**vars(args),initial_proposal='zero for all B2',HR_only_evaluation=True,actual_H3_D='native FP16',chart_projector='strict FP32/noTF32',DiT_used=False,b1_probes='Zero ADDITIONAL training; frozen B1 learned appearance/chart, current LR anchor, historical-near/current-null support. No HR inference inputs.',raw='boxed readout; anchor invariance audited prebox',DC='same64 iteration affine/box Dykstra all models and probes, actual physical current footprint only',seconds=time.time()-start,vae_sha256=CANONICAL_SHA,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'complete.json').write_text(json.dumps(config,indent=2));(out/'projected/complete.json').write_text(json.dumps(config,indent=2));write_live(out,status='B2_actual_D_evaluation_complete',phase=args.split+'_B2_and_strong_frozen_B1_controls',step=len(items),steps=len(items),seconds=time.time()-start)


if __name__=='__main__':main()
