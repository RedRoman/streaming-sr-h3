"""Fixed B5 last672 VAL8, correct-base neural interventions and physical controls."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_bridge import build_bridge,BRIDGE_ARMS,attach_native,bridge_rgb,shared_bridge
from revision_utils import batch,Readout,FP
from revision_classical import ClassicalRevision,frozen_controller
from native_vae import load_vae,decode_last


def write(path,rows):
    if rows:
        with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def masks(inputs,step):
    seen=torch.zeros((1,1,256,256),device='cuda',dtype=torch.bool);coverage=torch.zeros_like(seen,dtype=torch.float32)
    for j,t in enumerate(inputs['packet_steps'][0].tolist()):
        if t<=step:top,left=FP[int(inputs['packet_fp'][0,j])];seen[...,top:top+128,left:left+128]=True;coverage[...,top:top+128,left:left+128]+=1
    change=torch.zeros_like(seen);change[...,64:128,64:128]=step>=13;fresh=seen.clone() if step in [3,4] else torch.zeros_like(seen)
    if step in [7,8,11]:fresh[...,160:224,32:224]=True;fresh&=seen
    if step>=13:fresh=change.clone()
    return dict(union=seen,old_unmodified=seen&~change,new_detail=fresh,changed=change,overlap=coverage>1)
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--phi',required=True);p.add_argument('--models',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());sources=[s for s in manifest['items'] if s['split']=='val'];assert len(sources)==8;vae=load_vae(a.vae,a.comfy);read=Readout(vae);classic=ClassicalRevision(read);models={};checkpoints={};start=time.time();rows=[];features=[];times=[];noobs=[];gallery={};calls=0;zero_max=0
    for arm in BRIDGE_ARMS:
        path=Path(a.models)/f'{arm}_seed17/last.pt';cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['step']==672;model=build_bridge(arm).cuda().eval();model.load_state_dict(cp['model']);models[arm]=model;checkpoints[arm]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),step=672,parameters=cp['parameters'],active_parameters=cp['active_parameters'])
    with torch.no_grad():
        for si,source in enumerate(sources):
            item=torch.load(cache/(source['id']+'.pt'),map_location='cpu',weights_only=False);other=torch.load(cache/(sources[(si+1)%8]['id']+'.pt'),map_location='cpu',weights_only=False);inputs,target=batch([item],'cuda');attach_native(inputs,item);phi=torch.load(Path(a.phi)/(source['id']+'.pt'),map_location='cpu',weights_only=False);otherphi=torch.load(Path(a.phi)/(sources[(si+1)%8]['id']+'.pt'),map_location='cpu',weights_only=False);base=phi['bridge'][None].cuda();decoded=[x[None].cuda() for x in phi['decoded']];otherdecoded=[x[None].cuda() for x in otherphi['decoded']];near=item['classical_inputs']['near_LR'].cuda().float();far=item['classical_inputs']['query_LR'].cuda().float();shuffled_base=shared_bridge(read,inputs,near,otherdecoded);predictions={};actual_z={}
            for arm,model in models.items():
                torch.cuda.synchronize();t=time.perf_counter();z,q,states,details=model.rollout(inputs,True);torch.cuda.synchronize();times.append(dict(source=source['id'],method=arm,operation='all18_NN_transition_and7latent_reads',seconds=time.perf_counter()-t));actual_z[arm]=z;predictions[arm]=[]
                for qi in range(7):
                    torch.cuda.synchronize();t=time.perf_counter();predictions[arm].append(bridge_rgb(read,inputs,z[:,qi],qi,base,model.bridge_skip));calls+=1;torch.cuda.synchronize();times.append(dict(source=source['id'],method=arm,operation='actualD_plus_H_pixel_read',seconds=time.perf_counter()-t));features.append(dict(source=source['id'],method=arm,query_step=int(inputs['query_steps'][0,qi]),packet_h_RMS=float(details['packet_features'].square().mean().sqrt()),canonical_h_RMS=float(details['query_features'][:,qi].square().mean().sqrt()),latent_correction_RMS=float(z[:,qi].square().mean().sqrt())))
                for suffix in ['NN_delta0','NN_shuffled_history','NN_frozen_photo','NN_frozen_weights']:
                    vi={k:v.clone() for k,v in inputs.items()}
                    if suffix=='NN_delta0':vi['packet_delta'].zero_()
                    if suffix=='NN_shuffled_history':vi['packet_proxy']=other['inputs']['packet_proxy'][None].cuda().float();attach_native(vi,other)
                    if suffix=='NN_frozen_photo':vi=frozen_controller(inputs,True,False)
                    if suffix=='NN_frozen_weights':vi=frozen_controller(inputs,False,True)
                    zz=model.rollout(vi)[0];predictions[arm+'__'+suffix]=[]
                    # Correct original physical base and H are deliberately held
                    # fixed: this isolates learned history/action dependence.
                    for qi in range(7):predictions[arm+'__'+suffix].append(bridge_rgb(read,inputs,zz[:,qi],qi,base,model.bridge_skip));calls+=1
                if model.bridge_skip:
                    predictions[arm+'__bridge_removed']=[predictions[arm][qi]-base[:,qi]+inputs['query_base_rgb'][:,qi] for qi in range(7)];predictions[arm+'__bridge_shuffled']=[predictions[arm][qi]-base[:,qi]+shuffled_base[:,qi] for qi in range(7)]
            for qi,step in enumerate(inputs['query_steps'][0].tolist()):
                current=inputs['query_base_rgb'][:,qi];zero=bridge_rgb(read,inputs,actual_z[BRIDGE_ARMS[0]][:,qi]*0,qi,base,0);calls+=1;zero_max=max(zero_max,float((zero-current).abs().max()));assert torch.equal(zero,current)
                records=classic.packets(inputs,near,qi);inverse,res=classic.inverse(current,records);methods={m:v[qi] for m,v in predictions.items()};methods.update(current_LR_VAE=current,H3_preconditioned_decoded_atlas=base[:,qi],LR_photo_revision_atlas=classic.atlas(current,records),LR_photo_revision_inverse=inverse);region=masks(inputs,step)
                for method,value in methods.items():
                    for family,rgb in [('raw',value.clamp(0,1)),('DC',read.far.apply(value,far[qi:qi+1]))]:
                        error=(rgb-target['rgb'][:,qi]).square();entry=dict(source=source['id'],seed=17,method=method,family=family,query_step=step,condition='appearance_cue' if step in [3,7,11] else ('neutral_return' if step in [4,8] else 'true_content_change'),rgb_psnr=float(-10*torch.log10(error.mean())),current_far_RMSE=float((read.far.observe(rgb)-far[qi:qi+1]).square().mean().sqrt()),detail_RGB_RMS=float((value-current).square().mean().sqrt()),CG_relative_residual=res if method=='LR_photo_revision_inverse' else None)
                        for name,mask in region.items():entry[name+'_pixels']=int(mask.sum());entry[name+'_psnr']=float(-10*torch.log10(error[mask.expand_as(error)].mean())) if bool(mask.any()) else None
                        rows.append(entry)
                        if si<2 and family=='DC' and step in [8,17] and (method in BRIDGE_ARMS or method in ['current_LR_VAE','H3_preconditioned_decoded_atlas','LR_photo_revision_atlas']):gallery[(source['id'],step,method)]=rgb[0].cpu()
                if si<2 and step in [8,17]:gallery[(source['id'],step,'GT')]=target['rgb'][0,qi].cpu()
            # No new far measurement at15. Same source content as13, near14 is
            # newly available; do not invent a DC/observation for query15.
            ni={k:v.clone() for k,v in inputs.items()}
            for key in ['query_latents','query_base_rgb','query_features','query_groups']:v=inputs[key];ni[key]=torch.cat((v[:,:6],v[:,5:6],v[:,6:]),1)
            ni['query_steps']=torch.tensor([[3,4,7,8,11,13,15,17]],device='cuda');nb=shared_bridge(read,ni,near,decoded);nr=classic.packets(ni,near,6);current=ni['query_base_rgb'][:,6];inverse,_=classic.inverse(current,nr);values=dict(current_LR_VAE=current,H3_preconditioned_decoded_atlas=nb[:,6],LR_photo_revision_atlas=classic.atlas(current,nr),LR_photo_revision_inverse=inverse)
            for arm,model in models.items():
                z,_,states=model.rollout(ni);assert torch.equal(states[14],states[15]);assert torch.equal(z,model.rollout(ni)[0]);values[arm]=bridge_rgb(read,ni,z[:,6],6,nb,model.bridge_skip);calls+=1
            mask=masks(ni,15)['union']
            for method,value in values.items():rgb=value.clamp(0,1);err=(rgb-target['rgb'][:,5]).square();noobs.append(dict(source=source['id'],method=method,query_step=15,new_current_observation=False,DC_applied=False,repeatquery_readonly=True,rgb_PSNR=float(-10*torch.log10(err.mean())),union_PSNR=float(-10*torch.log10(err[mask.expand_as(err)].mean()))))
            write(out/'rgb_metrics.csv',rows);write(out/'no_observation_query.csv',noobs);write(out/'runtime_operations.csv',times);write(out/'features.csv',features);live=dict(status='B5_last672_VAL8',pid=os.getpid(),step=si+1,steps=8,seconds=time.time()-start,foreground=True,unix_time=time.time());(out/'LIVE_STATE.json').write_text(json.dumps(live,indent=2));print(json.dumps(live),flush=True)
    for source,step in sorted({(s,t) for s,t,m in gallery}):
        values=[(m,v) for (s,t,m),v in gallery.items() if s==source and t==step];values.sort(key=lambda x:x[0]!='GT');canvas=Image.new('RGB',(4*256,((len(values)+3)//4)*280),'white');draw=ImageDraw.Draw(canvas)
        for i,(m,rgb) in enumerate(values):canvas.paste(Image.fromarray((rgb.permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8')),((i%4)*256,(i//4)*280));draw.text(((i%4)*256+3,(i//4)*280+258),m,fill='black')
        canvas.save(out/f'{source}_step{step}_DC.png')
    (out/'complete.json').write_text(json.dumps(dict(**vars(a),checkpoints=checkpoints,sources=[s['id'] for s in sources],source_average_before_statistics=True,actual_D_calls=calls,actual_E_calls=0,commonphi_D_calls_separate=64,zero_base_all_val_pixel_maxabs=zero_max,seconds=time.time()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),rows=len(rows),optimizer_updates=0,test_accessed=False,scope='One trainedseed17,VAL8development; all primarylast672. NN interventions hold physical base/H fixed. Bridge interventions hold trainedNN prediction fixed. NativeD frozenFP16,TF32off; no DiT/real-video claim.'),indent=2))


if __name__=='__main__':main()
