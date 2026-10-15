"""Fixed B3R val8 actual H3 D evaluation, including strong classical controls."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
from revision_model import build,KINDS
from revision_utils import batch,Readout,FP
from revision_classical import ClassicalRevision,frozen_controller
from native_vae import load_vae,decode_last,encode_repeated,CANONICAL_SHA
from cache_pairs import degrade


def write_csv(path,rows):
    if rows:
        with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--models',required=True);p.add_argument('--checkpoint',choices=['best','last'],default='best');p.add_argument('--expected-step',type=int,default=200);p.add_argument('--dense',action='store_true');p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=17);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);manifest=json.loads((Path(args.cache)/'manifest.json').read_text());sources=[x for x in manifest['items'] if x['split']=='val'];assert len(sources)==8
    vae=load_vae(args.vae,args.comfy);reader=Readout(vae);classic=ClassicalRevision(reader);models={};checkpoints={}
    if args.dense:
        from revision_dense import build_dense,DENSE_KINDS
        kinds=DENSE_KINDS;builder=build_dense
    else:kinds=KINDS;builder=build
    for kind in kinds:
        path=Path(args.models)/f'{kind}_seed{args.seed}'/(args.checkpoint+'.pt');checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if args.checkpoint=='last':assert checkpoint['step']==args.expected_step
        model=builder(kind).cuda().eval();model.load_state_dict(checkpoint['model']);models[kind]=model;checkpoints[kind]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),step=checkpoint['step'],parameters=checkpoint['parameters'],active_parameters=checkpoint.get('active_parameters'),initial_checkpoint=checkpoint.get('initial_checkpoint'))
    rows=[];diagnostics=[];timings=[];no_observation=[];calls={'encode':0,'decode':0};gallery={}
    def timed(source,method,operation,fn):
        torch.cuda.synchronize();t=time.perf_counter();v=fn();torch.cuda.synchronize();timings.append(dict(source=source,method=method,operation=operation,seconds=time.perf_counter()-t));return v
    def encdec(source,method,x):
        calls['encode']+=1;z=timed(source,method,'native_E22',lambda:encode_repeated(vae,x.clamp(0,1)));calls['decode']+=1;return timed(source,method,'native_D22',lambda:decode_last(vae,z))
    with torch.no_grad():
        for number,prov in enumerate(sources):
            item=torch.load(Path(args.cache)/(prov['id']+'.pt'),map_location='cpu',weights_only=False);inputs,targets=batch([item],'cuda');near=item['classical_inputs']['near_LR'].cuda().float();far=item['classical_inputs']['query_LR'].cuda().float();base=inputs['query_base_rgb'];predictions={};full_details=None
            for kind,model in models.items():
                result=timed(prov['id'],kind,'all18_transition_and_7latent_queries',lambda:model.rollout(inputs,True));z,a,states,details=result;predictions[kind]=[]
                if kind=='closed_action':full_details=details
                for qi in range(7):
                    calls['decode']+=1;predictions[kind].append(timed(prov['id'],kind,'native_D_supported_readout',lambda:reader.rgb(inputs,z[:,qi],qi)))
                    diagnostics.append(dict(source=prov['id'],method=kind,query_step=int(inputs['query_steps'][0,qi]),factor_feature_RMS=float(details['packet_features'].square().mean().sqrt()),factor_feature_channel_variance=float(details['packet_features'].var(dim=(0,1,3,4,5)).mean()),detail_code_RMS=float(z[:,qi].square().mean().sqrt()),state_feature_RMS=float(details['query_features'][:,qi].square().mean().sqrt()),query_appearance_MSE=float((a[:,qi]-targets['query_q'][:,qi]).square().mean())))
            # Fixed-checkpoint intervention, not additional optimization/selection.
            for variant,photo,reliability in [('closed_action_frozen_photo',True,False),('closed_action_frozen_weights',False,True),('closed_action_frozen_both',True,True)]:
                vi=frozen_controller(inputs,photo,reliability);vz,_,_=models['closed_action'].rollout(vi);predictions[variant]=[]
                for qi in range(7):
                    calls['decode']+=1;predictions[variant].append(reader.rgb(vi,vz[:,qi],qi))
            # Does a nonzero state merely switch on an anchor-conditioned SR
            # prior? These are interventions, not eligible reconstruction
            # methods and never affect training/checkpoint selection.
            other=torch.load(Path(args.cache)/(sources[(number+1)%len(sources)]['id']+'.pt'),map_location='cpu',weights_only=False)
            for variant in ['diagnostic_closed_no_measured_detail','diagnostic_closed_shuffled_history']:
                vi={k:v.clone() for k,v in inputs.items()}
                if variant.endswith('no_measured_detail'):vi['packet_delta'].zero_()
                else:
                    for key in ['packet_delta','packet_proxy']:vi[key]=other['inputs'][key][None].cuda().float()
                vz,_,_=models['closed_action'].rollout(vi);predictions[variant]=[]
                for qi in range(7):
                    calls['decode']+=1;predictions[variant].append(reader.rgb(vi,vz[:,qi],qi))
            if args.dense:
                # The neural current-image control has exactly the same output
                # under historical content poison. Its common physical filter
                # can still depend on history-derived reliability statistics.
                vi={k:v.clone() for k,v in inputs.items()};vi['packet_delta'].fill_(101);vi['packet_proxy'].fill_(-81);vi['group_features'].fill_(61);vi['coarse_descriptors'].fill_(19);plain=models['anchor_only'].rollout(inputs)[0];poison=models['anchor_only'].rollout(vi)[0];assert torch.equal(plain,poison)
                vi={k:v.clone() for k,v in inputs.items()}
                for t in range(18):
                    for j,arrival in enumerate(inputs['packet_steps'][0].tolist()):vi['weights'][:,t,j]=inputs['packet_masks'][:,j] if arrival<=t else 0
                predictions['anchor_only_geometry_filter']=[]
                for qi in range(7):calls['decode']+=1;predictions['anchor_only_geometry_filter'].append(reader.rgb(vi,plain[:,qi],qi))
            decoded=[]
            for z in item['classical_inputs']['packet_full_latents']:
                calls['decode']+=1;decoded.append(decode_last(vae,z[None].cuda().float()))
            for qi,step in enumerate(inputs['query_steps'][0].tolist()):
                current=base[:,qi];records=classic.packets(inputs,near,qi);frozen_records=classic.packets(inputs,near,qi,False);learned_records=classic.packets(inputs,near,qi,True,appearance=full_details['group_appearance'][0,step]);last=records[-1:]
                pixel_atlas=timed(prov['id'],'LR_photo_revision_atlas','query_inverse_free',lambda:classic.atlas(current,records));pixel_inverse,cgres=timed(prov['id'],'LR_photo_revision_inverse','query_CG48',lambda:classic.inverse(current,records));last_pixel=classic.atlas(current,last)
                methods={name:values[qi] for name,values in predictions.items()};methods.update(current_LR_VAE=current,LR_photo_last_near=last_pixel,LR_photo_nonretro_atlas=classic.atlas(current,frozen_records),LR_photo_revision_atlas=pixel_atlas,LR_photo_revision_inverse=pixel_inverse,H3_preconditioned_decoded_atlas=classic.decoded_atlas(current,decoded,records),H3_preconditioned_last_code=classic.decoded_atlas(current,decoded,last),LR_atlas_with_full_learned_photometry=classic.atlas(current,learned_records))
                for name,x in [('H3_ED_last_near',last_pixel),('H3_ED_revision_atlas',pixel_atlas),('H3_ED_revision_inverse',pixel_inverse)]:
                    d=encdec(prov['id'],name,x);methods[name]=current+reader.historical_band(d-current,inputs,qi)
                seen=torch.zeros_like(current[:,:1],dtype=torch.bool);coverage=torch.zeros_like(current[:,:1]);fresh=torch.zeros_like(seen)
                for rec in records:
                    top,left=FP[rec['fp']];seen[...,top:top+128,left:left+128]=True;coverage[...,top:top+128,left:left+128]+=1
                changed=torch.zeros_like(seen);changed[...,64:128,64:128]=step>=13
                if step in [3,4]:fresh=seen.clone()
                elif step in [7,8,11]:fresh[...,160:224,32:224]=True;fresh=fresh&seen
                else:fresh=changed.clone()
                occlusion=torch.zeros_like(seen)
                if step>=7:occlusion[...,112:144,64:96]=True
                if step>=17:occlusion[...,64:96,128:160]=True
                regions=dict(union=seen,old_unmodified=seen&~changed,new_detail=fresh,changed=changed,overlap=coverage>1,transient_foreground=occlusion)
                target=targets['rgb'][:,qi]
                for method,pred in methods.items():
                    anchor_rms=float((reader.far.observe(pred-current)).square().mean().sqrt());detail_rms=float((pred-current).square().mean().sqrt())
                    for family,rgb in [('raw',pred.clamp(0,1)),('DC',reader.far.apply(pred,far[qi:qi+1]))]:
                        error=(rgb-target).square();past_mse=[];past_weights=[]
                        for rec in records:
                            y=classic.k@(rec['ratio']*classic.crop(rgb,rec['fp'])+rec['offset'])@classic.k.T;past_mse.append(((y-rec['y']).square()*rec['weight']).sum());past_weights.append(rec['weight'].sum()*3)
                        entry=dict(source=prov['id'],seed=args.seed,method=method,family=family,query_step=step,condition='appearance_cue' if step in [3,7,11] else ('neutral_return' if step in [4,8] else 'true_content_change'),rgb_psnr=float(-10*torch.log10(error.mean())),current_far_RMSE=float((reader.far.observe(rgb)-far[qi:qi+1]).square().mean().sqrt()),prebox_anchor_difference_RMSE=anchor_rms,detail_RGB_RMS=detail_rms,past_confidence_weighted_LR_RMSE=float((sum(past_mse)/sum(past_weights).clamp_min(1)).sqrt()),CG_relative_residual=cgres if 'inverse' in method else None)
                        for name,mask in regions.items():entry[name+'_pixels']=int(mask.sum());entry[name+'_psnr']=float(-10*torch.log10(error[mask.expand_as(error)].mean())) if bool(mask.any()) else None
                        rows.append(entry)
                        if number<2 and family=='DC' and step in [8,17] and method in ['current_LR_VAE','closed_action','anchor_only','factor_cache_mlp','gru_replay','LR_photo_revision_atlas','LR_photo_revision_inverse','H3_ED_revision_inverse']:gallery[(prov['id'],step,method)]=rgb[0].cpu()
                if number<2 and step in [8,17]:gallery[(prov['id'],step,'GT')]=target[0].cpu()
            if args.dense:
                # Query15 has no new far observation. Reuse actual far13 as a
                # retained anchor; near14 is genuine arrived evidence. Do not
                # run any current-observation DC or invent a current LR target.
                ni={k:v.clone() for k,v in inputs.items()}
                for key in ['query_latents','query_base_rgb','query_features','query_groups']:
                    v=inputs[key];ni[key]=torch.cat((v[:,:6],v[:,5:6],v[:,6:]),1)
                ni['query_steps']=torch.tensor([[3,4,7,8,11,13,15,17]],device='cuda');target=targets['rgb'][:,5];current=ni['query_base_rgb'][:,6];nr=classic.packets(ni,near,6);inverse,_=classic.inverse(current,nr);nm=dict(current_LR_VAE=current,LR_photo_revision_atlas=classic.atlas(current,nr),LR_photo_revision_inverse=inverse)
                for kind,model in models.items():
                    z,_,states=model.rollout(ni);assert torch.equal(states[14],states[15]);repeat=model.rollout(ni)[0];assert torch.equal(z,repeat);calls['decode']+=1;nm[kind]=reader.rgb(ni,z[:,6],6)
                union=torch.zeros_like(current[:,:1],dtype=torch.bool)
                for rec in nr:top,left=FP[rec['fp']];union[...,top:top+128,left:left+128]=True
                for method,pred in nm.items():
                    rgb=pred.clamp(0,1);error=(rgb-target).square();no_observation.append(dict(source=prov['id'],method=method,query_step=15,last_real_far_step=13,new_current_observation=False,DC_applied=False,rgb_PSNR=float(-10*torch.log10(error.mean())),union_PSNR=float(-10*torch.log10(error[union.expand_as(error)].mean())),retained_far13_RMSE=float((reader.far.observe(rgb)-far[5:6]).square().mean().sqrt()),query_readonly_state14_equals15=True,repeat_query_exact=True))
                write_csv(out/'no_observation_query.csv',no_observation)
            write_csv(out/'rgb_metrics.csv',rows);write_csv(out/'runtime_operations.csv',timings);(out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2));live=dict(status='B3R_val8_actual_nativeD_running',command=' '.join(sys.argv),pid=os.getpid(),source=prov['id'],step=number+1,steps=8,seconds=time.time()-start,foreground=True,unix_time=time.time());(out/'LIVE_STATE.json').write_text(json.dumps(live,indent=2));print(json.dumps(live),flush=True)
    for source,step in sorted({(key[0],key[1]) for key in gallery}):
        entries=[(m,x) for (s,t,m),x in gallery.items() if s==source and t==step];entries.sort(key=lambda v:v[0]!='GT');canvas=Image.new('RGB',(4*256,((len(entries)+3)//4)*280),'white');draw=ImageDraw.Draw(canvas)
        for i,(method,rgb) in enumerate(entries):canvas.paste(Image.fromarray((rgb.permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8')),((i%4)*256,(i//4)*280));draw.text(((i%4)*256+3,(i//4)*280+258),method,fill='black')
        canvas.save(out/f'{source}_step{step}_DC.png')
    config=dict(**vars(args),seconds=time.time()-start,checkpoints=checkpoints,sources=[x['id'] for x in sources],capture_group_provenance=manifest['capture_group_provenance'],native_calls=calls,peak_GPU_bytes=torch.cuda.max_memory_allocated(),VAE_parameter_bytes=sum(p.numel()*p.element_size() for p in vae.parameters()),shared_camera_operators_bytes=524288,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),VAE_sha256=CANONICAL_SHA,precision='nativeFP16 E/D; cameraFP32; TF32off; RGBtargetscommonFP16 cache',classical_inverse='LR-only relative multiplicative spatial gain+bias, confidence weighted48CG ridge.01 fixed earliertrain-only calibration; current-far-null,sharedDC64',scope='32train/8val development natural photos with simulated camera/nuisance, no old16test, no real video or DiT claim',evaluation_source_independence='Average queries within each source before bootstrap/pairwise statistics; one trained seed, not8x7 independent images',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=len(rows));(out/'complete.json').write_text(json.dumps(config,indent=2));(out/'LIVE_STATE.json').write_text(json.dumps(dict(status='B3R_val8_complete',seconds=config['seconds'],rows=len(rows),unix_time=time.time()),indent=2));print(json.dumps(config),flush=True)


if __name__=='__main__':main()
