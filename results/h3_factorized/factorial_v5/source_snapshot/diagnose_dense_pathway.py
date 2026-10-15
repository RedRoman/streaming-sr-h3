"""Read-only staged interventions of fixed dense checkpoints; no optimization."""
import argparse,csv,json,sys,time,types
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_dense import build_dense
from revision_utils import batch,Readout,FP
from native_vae import load_vae,decode_last


def rms(x):return float(x.square().mean().sqrt())
def psnr(x,y,mask):return float(-10*torch.log10((x-y).square()[mask.expand_as(x)].mean()))
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--models',required=True);p.add_argument('--single',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());sources=[s for s in manifest['items'] if s['split']=='val'];assert len(sources)==8;model=build_dense('closed_action').cuda().eval();cp=torch.load(Path(a.models)/'closed_action_seed17/last.pt',map_location='cpu',weights_only=False);assert cp['step']==672;model.load_state_dict(cp['model']);vae=load_vae(a.vae,a.comfy);read=Readout(vae);rows=[];hfields=[];hzfields=[];start=time.time();fixed_h=None;calls=0
    with torch.no_grad():
        for si,source in enumerate(sources):
            item=torch.load(cache/(source['id']+'.pt'),map_location='cpu',weights_only=False);other=torch.load(cache/(sources[(si+1)%8]['id']+'.pt'),map_location='cpu',weights_only=False);inputs,targets=batch([item],'cuda');far=item['classical_inputs']['query_LR'].cuda().float();z0,_,_,d0=model.rollout(inputs,True);hfields.append(d0['packet_features'].cpu());fixed_h=d0['packet_features'].clone() if fixed_h is None else fixed_h
            reference=[]
            for qi in range(7):
                raw=decode_last(vae,inputs['query_latents'][:,qi]+z0[:,qi]);calls+=1;band=read.historical_band(raw-inputs['query_base_rgb'][:,qi],inputs,qi);rgb=inputs['query_base_rgb'][:,qi]+band;dc=read.far.apply(rgb,far[qi:qi+1]);reference.append((raw,band,rgb,dc))
            for variant in ['actual','delta_zero_proxy_fixed','shuffled_delta_proxy','neural_anchor_other_source','constant_packet_h_other_source']:
                vi={k:v.clone() for k,v in inputs.items()};original=model.lift_packets
                if variant=='delta_zero_proxy_fixed':vi['packet_delta'].zero_()
                if variant=='shuffled_delta_proxy':
                    for k in ['packet_delta','packet_proxy']:vi[k]=other['inputs'][k][None].cuda().float()
                if variant=='neural_anchor_other_source':vi['query_latents']=other['inputs']['query_latents'][None].cuda().float()
                if variant=='constant_packet_h_other_source':model.lift_packets=types.MethodType(lambda self,inp,neutral=False,targets=None:fixed_h,model)
                pred,_,_,details=model.rollout(vi,True);model.lift_packets=original
                if variant=='delta_zero_proxy_fixed':hzfields.append(details['packet_features'].cpu())
                for qi,step in enumerate(inputs['query_steps'][0].tolist()):
                    if variant=='actual':raw,band,rgb,dc=reference[qi]
                    else:
                        # Only neural conditioning may change; nativeD uses the
                        # original current source anchor, never the other source.
                        raw=decode_last(vae,inputs['query_latents'][:,qi]+pred[:,qi]);calls+=1;band=read.historical_band(raw-inputs['query_base_rgb'][:,qi],inputs,qi);rgb=inputs['query_base_rgb'][:,qi]+band;dc=read.far.apply(rgb,far[qi:qi+1])
                    union=torch.zeros_like(rgb[:,:1],dtype=torch.bool)
                    for j,t in enumerate(inputs['packet_steps'][0].tolist()):
                        if t<=step:top,left=FP[int(inputs['packet_fp'][0,j])];union[...,top:top+128,left:left+128]=True
                    rr,bb,xx,dd=reference[qi];rows.append(dict(source=source['id'],variant=variant,query_step=step,packet_h_RMS=rms(details['packet_features']),packet_h_change_RMS=rms(details['packet_features']-d0['packet_features']),canonical_h_RMS=rms(details['query_features'][:,qi]),canonical_h_change_RMS=rms(details['query_features'][:,qi]-d0['query_features'][:,qi]),readout_delta_RMS=rms(pred[:,qi]),readout_delta_change_RMS=rms(pred[:,qi]-z0[:,qi]),nativeD_change_RMS=rms(raw-rr),supported_change_RMS=rms(band-bb),postDC_change_RMS=rms(dc-dd),supported_detail_RMS=rms(band),raw_union_PSNR=psnr(rgb.clamp(0,1),targets['rgb'][:,qi],union),DC_union_PSNR=psnr(dc,targets['rgb'][:,qi],union),baseline_DC_union_PSNR=psnr(dd,targets['rgb'][:,qi],union),DC_source_anchor_RMSE=rms(read.far.observe(dc)-far[qi:qi+1])))
            print(json.dumps(dict(source=source['id'],step=si+1,steps=8,seconds=time.time()-start)),flush=True)
    def across(v):
        h=torch.cat(v);return dict(total_RMS=rms(h),across_source_standard_deviation_RMS=rms(h.std(0,unbiased=False)),source_mean_standard_deviation_RMS=rms(h.mean((1,3,4,5)).std(0,unbiased=False)))
    with (out/'stages.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Compare the already-completed single TRAIN fit with the multsource model,
    # under exactly its old one-packet/same-q diagnostic (not formal inference).
    train=next(s for s in manifest['items'] if s['split']=='train');item=torch.load(cache/(train['id']+'.pt'),map_location='cpu',weights_only=False);inputs,targets=batch([item],'cuda');inputs['weights'][:,:,1:]=0;weight=inputs['weights'][:,3,0].unsqueeze(2);q=targets['query_q'][:,0];anchor=inputs['query_latents'][:,0];single=[]
    with torch.no_grad():
        for name,path in [('multi_source_last672',Path(a.models)/'closed_action_seed17/last.pt'),('single_TRAIN_RGB300',Path(a.single)/'actualD_RGB_only/step300.pt')]:
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);zref=href=None;rawref=bandref=dcref=None
            for variant in ['actual','delta_zero_proxy_fixed']:
                delta=inputs['packet_delta'][:,0] if variant=='actual' else inputs['packet_delta'][:,0]*0;h=model.lift(delta,inputs['packet_proxy'][:,0],inputs['packet_masks'][:,0]);styled=model.core.style(model.core.canonical(h,q),q)*(weight>0);z=model.read(styled,anchor,q,(weight/8).clamp(0,1));raw=decode_last(vae,anchor+z);calls+=1;band=read.historical_band(raw-inputs['query_base_rgb'][:,0],inputs,0);rgb=inputs['query_base_rgb'][:,0]+band;dc=read.far.apply(rgb,item['classical_inputs']['query_LR'][0:1].cuda().float());mask=torch.zeros_like(rgb[:,:1],dtype=torch.bool);mask[...,32:160,32:160]=True
                if variant=='actual':zref,href,rawref,bandref,dcref=z,h,raw,band,dc
                single.append(dict(checkpoint=name,variant=variant,packet_h_change_RMS=rms(h-href),readout_delta_change_RMS=rms(z-zref),nativeD_change_RMS=rms(raw-rawref),supported_change_RMS=rms(band-bandref),postDC_change_RMS=rms(dc-dcref),DC_single_near_PSNR=psnr(dc,targets['rgb'][:,0],mask)))
    (out/'summary.json').write_text(json.dumps(dict(seconds=time.time()-start,actualD_calls=calls,optimizer_updates=0,across_source_packet_h=across(hfields),across_source_packet_h_after_delta_zero=across(hzfields),variant_source_query_means={v:{k:float(np.mean([r[k] for r in rows if r['variant']==v])) for k in rows[0] if k not in ['source','variant','query_step']} for v in sorted({r['variant'] for r in rows})},single_TRAIN_comparison=single,scope='Fixed last672 VAL8 intervention. Neural anchor swap affects only readout conditioning; physical frozen decoder/current source anchor and shared controller fixed. Constant-h counterfactual is a diagnostic, not an eligible reconstruction method. Single TRAIN trueq only reproduces prior diagnostic; no training/test.'),indent=2))


if __name__=='__main__':main()
