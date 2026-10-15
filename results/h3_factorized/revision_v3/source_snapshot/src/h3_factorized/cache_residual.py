"""Actual H3 B2 inputs/teachers; zero proposal, no HR initialization."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from cache_pairs import make_pair,degrade
from native_vae import load_vae,encode_repeated,decode_last,CANONICAL_SHA
from train import write_live


def up(x,size):return F.interpolate(x[None],size=(size,size),mode='bicubic',align_corners=False,antialias=True)[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--splits',nargs='+',default=['train','val']);args=p.parse_args();torch.set_num_threads(4)
    source=Path(args.cache);out=Path(args.out);out.mkdir(parents=True,exist_ok=True);original=json.loads((source/'manifest.json').read_text());items=[x for x in original['items'] if x['split'] in args.splits];start=time.time();vae=load_vae(args.vae,args.comfy)
    manifest=dict(config=dict(counts=[sum(x['split']==s for x in items) for s in ['train','val','test']],size=256,context=22),items=[],native_H3=True,vae_sha256=CANONICAL_SHA,origin_B1_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),initial_proposal='explicit zero for all methods; no HR state initialization',input_detail='E(actual current near LR)-E(LR-only far proxy of that same observed near); available LR only',teacher_detail='Training supervision only: E(noisefree near2x observed photo)-E(noisefree far4x observed photo), 6 photometric/content labels; never actual initial state',base_RGB='Exact native FP16 H3 D output cached float32, making zero-residual baseline bitwise reproducible',scope='Controlled natural-photo H3 representation experiment; no DiT proposals or real-motion claim')
    with torch.no_grad():
        for i,prov in enumerate(items):
            path=out/(prov['id']+'.pt')
            if path.exists():manifest['items'].append(torch.load(path,weights_only=False)['provenance']);continue
            old=torch.load(source/(prov['id']+'.pt'),weights_only=False);observed,targets,q,visible,reveal,order=make_pair(Path(prov['source']),Path(prov['donor_source']),256,prov['seed'],'cuda')
            assert torch.equal(observed[order].cpu().half(),old['inputs']['observation_rgb']),'Changed original LR construction'
            detail=torch.zeros_like(old['inputs']['observation_latents'])
            for stage,unique_index in [(2,2),(5,4)]:
                proxy=observed[unique_index].clone();proxy[:,64:192,64:192]=up(degrade(proxy[:,64:192,64:192][None],4)[0].clamp(0,1),128)
                zproxy=encode_repeated(vae,proxy[None],22)[0];detail[stage]=(old['inputs']['observation_latents'][stage].cuda().float()-zproxy).cpu().half()
            detail[3]=detail[2]
            decoded={stage:decode_last(vae,old['inputs']['observation_latents'][stage:stage+1].cuda().float())[0].cpu() for stage in [0,1,2,4,5,6]};decoded[3]=decoded[2];decoded[7]=decoded[6]
            teacher=[]
            for target in targets:
                far=up(degrade(target[None],4)[0].clamp(0,1),256).clamp(0,1);near=far.clone();near[:,64:192,64:192]=up(degrade(target[:,64:192,64:192][None],2)[0].clamp(0,1),128)
                zfar=encode_repeated(vae,far[None],22);znear=encode_repeated(vae,near[None],22);teacher.append((znear-zfar)[0].cpu().half())
            inputs=dict(old['inputs'],detail_observation_latents=detail,base_decoded_rgb=torch.stack([decoded[j] for j in range(8)]));supervision=dict(old['supervision'],detail_latents=torch.stack(teacher));newprov=dict(prov,path=str(path),B1_source_cache=str(source/(prov['id']+'.pt')),zero_proposal=True)
            torch.save(dict(inputs=inputs,supervision=supervision,provenance=newprov),path);manifest['items'].append(newprov);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));write_live(out,status='B2_actual_H3_detail_cache',phase='zero_proposal_residual_data',step=i+1,steps=len(items),seconds=time.time()-start);print(json.dumps(dict(source=prov['id'],step=i+1,total=len(items),seconds=time.time()-start)),flush=True)
    manifest['seconds']=time.time()-start;manifest['additional_H3_encoder_calls']=14*len(items);manifest['additional_H3_decoder_calls']=6*len(items);manifest['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(out/'manifest.json').write_text(json.dumps(manifest,indent=2));write_live(out,status='B2_detail_cache_complete',phase='zero_proposal_residual_data',step=len(items),steps=len(items),seconds=time.time()-start)


if __name__=='__main__':main()
