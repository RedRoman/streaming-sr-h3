"""True frozen-H3 RGB evaluation; includes source-wise baselines and images."""
import argparse,csv,hashlib,json,math,os,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from model import FactorizedMemory,IdentityState
from train import batch,TARGET,write_live
from native_vae import load_vae,decode_last,CANONICAL_SHA
from pixel_dc import Projection,project_visible

STAGES=[1,2,4,5,6]
NAMES={1:'appearance_A1',2:'reveal_A1',4:'return_A0',5:'content_replace_B0',6:'appearance_B2'}


def psnr(x,y,mask=None):
    mse=(x-y).square()
    if mask is not None:mse=(mse*mask).sum()/(mask.sum()*3)
    else:mse=mse.mean()
    return float(-10*torch.log10(mse.clamp_min(1e-12)))


def metrics(x,y):
    h=x.shape[-1];mask=torch.zeros_like(x[:,:1]);mask[:,:,h//4:h*3//4,h//4:h*3//4]=1
    return dict(rgb_psnr=psnr(x,y),roi_psnr=psnr(x,y,mask),outside_psnr=psnr(x,y,1-mask))


def save_csv(path,rows):
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def error_source_oracle(model,inputs,targets,mode):
    """Evaluation-only oracle. Never a valid LR-only competing method."""
    obs=inputs['observation_latents'];rgb=inputs['observation_rgb'];zero=torch.zeros((1,12),device=obs.device)
    state=model.begin(obs[:,0],rgb[:,0],'far_A0')
    if mode=='oracle_initial_HR_identity':state=IdentityState(model.chart.canonicalize(targets['latents'][:,0],zero),state.reference,state.seen)
    outputs=[model.query(state,zero)];ids=['far_A0','far_A1','near_A1','query','return_A0','near_B0','far_B2','far_B2']
    for step in range(1,8):
        a=targets['appearance'][:,TARGET[step]] if mode=='oracle_appearance' else model.appearance(state.reference,rgb[:,step])
        if ids[step] not in state.seen and bool(inputs['visible'][:,step].any()):
            canonical=model.chart.canonicalize(obs[:,step],a);content=model.update_head(state.content,canonical,a,inputs['visible'][:,step],inputs['reveal'][:,step],float(inputs['footprint'][step]));state=IdentityState(content,state.reference,state.seen|{ids[step]})
        outputs.append(model.query(state,a))
    return torch.stack(outputs,1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--runs',required=True);p.add_argument('--out',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--split',default='test',choices=['val','test']);p.add_argument('--observations');args=p.parse_args()
    torch.set_num_threads(4)
    # Conditional chart invertibility is a float32 arithmetic claim. TF32
    # rounding in repeated conditional convs magnifies inverse roundoff;
    # keep native FP16 VAE unchanged and evaluate charts with strict FP32.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);start=time.time()
    vae=load_vae(args.vae,args.comfy);manifest=json.loads((Path(args.cache)/'manifest.json').read_text());items=[i for i in manifest['items'] if i['split']==args.split]
    write_live(out,status='loading_fixed_validation_selected_checkpoints',phase='heldout_RGB_raw_DC_evaluation',step=0,steps=len(items),sources=len(items),sources_evaluated=0)
    far=near=None
    if args.observations:
        assert json.loads((Path(args.observations)/'manifest.json').read_text())['passed']
        far=Projection(256,4);near=Projection(128,2);(out/'projected').mkdir(exist_ok=True)
    models=[]
    for path in sorted(Path(args.runs).glob('*/best.pt')):
        ckpt=torch.load(path,map_location='cpu',weights_only=False);model=FactorizedMemory(ckpt['kind']).cuda().eval();model.load_state_dict(ckpt['model'])
        models.append((model,ckpt,path))
    assert models,'No actual trained checkpoints'
    rows=[];diagnostics=[];swap_rows=[];neutral_rows=[];projected_rows=[];oracle_rows=[]
    with torch.no_grad():
        for source_index,prov in enumerate(items):
            item=torch.load(Path(args.cache)/(prov['id']+'.pt'),map_location='cpu',weights_only=False);inputs,targets=batch([item],'cuda');gallery={};gallery_dc={}
            measurement=torch.load(Path(args.observations)/(prov['id']+'.pt'),map_location='cpu',weights_only=False)['inputs'] if args.observations else None
            # Reuse actual decode once/source/stage, independently of methods.
            for stage in STAGES:
                target=targets['rgb'][:,TARGET[stage]]
                images={'input_bicubic':inputs['observation_rgb'][:,stage], 'H3_current_LR':decode_last(vae,inputs['observation_latents'][:,stage]),'H3_HR_roundtrip':decode_last(vae,targets['latents'][:,TARGET[stage]])}
                if stage==4:gallery.update(GT=target,**images)
                for name,rgb in images.items():
                    rows.append(dict(source=prov['id'],split=args.split,method=name,seed=-1,stage=stage,scenario=NAMES[stage],**metrics(rgb,target)))
                    if measurement is not None:
                        dc,rmse=project_visible(rgb,measurement,stage,far,near);projected_rows.append(dict(source=prov['id'],split=args.split,method=name,seed=-1,stage=stage,scenario=NAMES[stage],**metrics(dc,target),actual_measurement_rmse=rmse))
                        if stage==4:gallery_dc.update(GT=target);gallery_dc[name]=dc
            for model,ckpt,path in models:
                latent,appearance,states=model.rollout(inputs)
                assert torch.equal(states[2],states[3]) and torch.equal(states[6],states[7]),'Causal protocol failed'
                assert torch.equal(latent[:,6],latent[:,7]),'Duplicate changed readout'
                leak=float((model.chart.canonicalize(targets['latents'][:,1],targets['appearance'][:,1])-targets['latents'][:,0]).square().mean().sqrt())
                inverse=None;inverse_rms=None
                if ckpt['kind'].startswith('factorized'):
                    z=targets['latents'][:,1];a=targets['appearance'][:,1];err=model.chart.read(model.chart.canonicalize(z,a),a)-z;inverse=float(err.abs().max());inverse_rms=float(err.square().mean().sqrt())
                content_bytes=states[0].numel()*states[0].element_size();reference_bytes=inputs['observation_rgb'][:,0].numel()*inputs['observation_rgb'].element_size()
                diagnostics.append(dict(source=prov['id'],method=ckpt['kind'],seed=ckpt['seed'],selected_step=ckpt['step'],parameters=ckpt['parameters'],content_state_bytes=content_bytes,reference_state_bytes=reference_bytes,total_tensor_state_bytes=content_bytes+reference_bytes,packet_id_metadata_excluded=True,canonical_appearance_leakage_rms=leak,appearance_rmse=float((appearance[:,STAGES]-targets['appearance'][:,[TARGET[s] for s in STAGES]]).square().mean().sqrt()),inverse_maxabs=inverse,inverse_rms=inverse_rms,query_read_only=True,duplicate_idempotent=True))
                for stage in STAGES:
                    decoded=decode_last(vae,latent[:,stage]);target=targets['rgb'][:,TARGET[stage]]
                    rows.append(dict(source=prov['id'],split=args.split,method=ckpt['kind'],seed=ckpt['seed'],stage=stage,scenario=NAMES[stage],**metrics(decoded,target)))
                    if stage==4 and ckpt['seed']==17:gallery[ckpt['kind']]=decoded
                    if measurement is not None:
                        dc,rmse=project_visible(decoded,measurement,stage,far,near);projected_rows.append(dict(source=prov['id'],split=args.split,method=ckpt['kind'],seed=ckpt['seed'],stage=stage,scenario=NAMES[stage],**metrics(dc,target),actual_measurement_rmse=rmse))
                        if stage==4 and ckpt['seed']==17:gallery_dc[ckpt['kind']]=dc
                # Mechanism-only chart audit: HR latent inputs are explicitly
                # oracle and never mixed with the causal LR rollout metrics.
                for source_target,query_target in [(1,2),(5,4)]:
                    content=model.chart.canonicalize(targets['latents'][:,source_target],targets['appearance'][:,source_target])
                    swapped=decode_last(vae,model.chart.read(content,targets['appearance'][:,query_target]))
                    swap_rows.append(dict(source=prov['id'],method=ckpt['kind'],seed=ckpt['seed'],from_target=source_target,to_target=query_target,oracle_HR_latent_input=True,**metrics(swapped,targets['rgb'][:,query_target])))
                for state_index,target_index in [(2,0),(6,3)]:
                    neutral=decode_last(vae,model.chart.read(states[state_index],torch.zeros_like(appearance[:,0])))
                    neutral_rows.append(dict(source=prov['id'],method=ckpt['kind'],seed=ckpt['seed'],state_after_stage=state_index,query='neutral_a0',HR_input=False,**metrics(neutral,targets['rgb'][:,target_index])))
                if ckpt['kind']=='factorized_swap':
                    for mode in ['oracle_appearance','oracle_initial_HR_identity']:
                        oracle=error_source_oracle(model,inputs,targets,mode)
                        for stage in STAGES:
                            decoded_oracle=decode_last(vae,oracle[:,stage]);oracle_rows.append(dict(source=prov['id'],method=ckpt['kind'],seed=ckpt['seed'],mode=mode,stage=stage,scenario=NAMES[stage],HR_supervision_used=True,valid_LR_only_method=False,**metrics(decoded_oracle,targets['rgb'][:,TARGET[stage]])))
            if source_index<4:
                for dest,collection in [(out,gallery)]+([(out/'projected',gallery_dc)] if gallery_dc else []):
                    width=256;canvas=Image.new('RGB',(width*4,(width+24)*2),'white');draw=ImageDraw.Draw(canvas)
                    for j,(name,tensor) in enumerate(collection.items()):
                        a=(tensor[0].clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype('uint8');canvas.paste(Image.fromarray(a),((j%4)*width,(j//4)*(width+24)));draw.text(((j%4)*width+3,(j//4)*(width+24)+width+3),name,fill='black')
                    canvas.save(dest/(prov['id']+'_return_A0.png'))
            save_csv(out/'rgb_metrics.csv',rows);save_csv(out/'oracle_chart_swap_rgb.csv',swap_rows);save_csv(out/'LR_memory_neutral_query_rgb.csv',neutral_rows);(out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));write_live(out,status='actual_H3_RGB_evaluation',phase='heldout_RGB_raw_DC_evaluation',step=source_index+1,steps=len(items),source=prov['id'],sources_evaluated=source_index+1,sources=len(items),seconds=time.time()-start)
            if projected_rows:save_csv(out/'projected'/'rgb_metrics.csv',projected_rows)
            if oracle_rows:save_csv(out/'error_source_oracles_rgb.csv',oracle_rows)
            print(json.dumps(dict(event='evaluated',source=prov['id'],items=len(rows),seconds=time.time()-start)),flush=True)
    config=dict(**vars(args),vae_sha256=CANONICAL_SHA,encoder_storage_dtype='float16',actual_decoder_dtype='float16',chart_dtype='float32',chart_TF32=False,RGB_range='0..1',native_H3_VAE=True,DiT_used=False,selected_only_by_validation=True,checkpoint_paths=[str(p) for _,_,p in models],seconds=time.time()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'complete.json').write_text(json.dumps(config,indent=2));write_live(out,status='actual_H3_RGB_evaluation_complete',seconds=time.time()-start)
    if projected_rows:(out/'projected'/'complete.json').write_text(json.dumps(dict(**config,projection='Shared classical min-change affine/box Dykstra64 iterations with exact physical LR measurements; far4x full image, near2x central128 ROI only. No query or oracle-swap projection.',far_operator_maxabs=far.operator_maxabs,near_operator_maxabs=near.operator_maxabs),indent=2))


if __name__=='__main__':main()
