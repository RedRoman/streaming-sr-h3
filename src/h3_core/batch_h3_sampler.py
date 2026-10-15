"""Foreground native official-grid H3 samples with one reusable frozen model.

Same public Comfy sampler API and exact49-Euler grid as the independently
audited staged sampler. Every job reads only its LR-derived guide and neutral
text, records all actual full50-block forward shapes/finite flags, and saves
raw AV latents. No HR file is read in this process.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file,save_file


def completed(path):
    try:
        value=json.loads(path.read_text());return value['status']=='complete' and len(value['forward_events'])==49
    except (OSError,KeyError,json.JSONDecodeError):return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);p.add_argument('--comfy',required=True);p.add_argument('--model',required=True);p.add_argument('--context',required=True);p.add_argument('--state',required=True);args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES','').isdigit();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.9);torch.cuda.reset_peak_memory_stats();sys.path.insert(0,str(Path(args.comfy).resolve()))
    from comfy.cli_args import args as comfy_args
    comfy_args.reserve_vram=8.
    import comfy.sd
    import comfy.sample
    import comfy.nested_tensor
    jobs=json.loads(Path(args.jobs).read_text());began=time.monotonic();state={'status':'running','scope':'ActualBF16H3 native22canonical conditionalresponses, not motionvideo','config':vars(args),'pid':os.getpid(),'jobs':[]};record=Path(args.state);record.parent.mkdir(exist_ok=True,parents=True)
    def update():state.update(seconds=time.monotonic()-began);record.write_text(json.dumps(state,indent=2))
    cache=torch.load(args.context,map_location='cpu',weights_only=True);context,tags=cache['prompt_embeds'],cache['text_token_tags'];assert context.shape[-1]==5120 and tags.shape==context.shape[:1]
    pending=[job for job in jobs if not completed(Path(job['directory'])/'sample_result.json')]
    if not pending:state['status']='complete';update();return
    model=comfy.sd.load_diffusion_model(args.model,model_options={'dtype':torch.bfloat16});assert len(model.model.diffusion_model.blocks)==50
    base=torch.linspace(1,0,50);sigmas=12*base/(1+11*base)
    for job in jobs:
        folder=Path(job['directory']);destination=folder/'sample_result.json';entry={**job,'status':'already_complete' if completed(destination) else 'running'};state['jobs'].append(entry);update()
        if completed(destination):continue
        guide_path=folder/'guide.safetensors';guide=load_file(str(guide_path))['guide'];assert guide.shape==(1,24,7,16,16)
        result={'status':'running','scope':'Static canonical-memory query using only arrived LR evidence; syntheticnative22context, NOT futurephysical22frames','job':job,'observed_evidence_subset':job.get('evidence_subset',['far']),'resolved_frame_index':0,'keyframe_index_meaning':'Internalnativecanonicalquerylayout; realLRpacket IDs/geometry are in inputs.pt','guide_shape':list(guide.shape),'guide_tokens':448,'guide_sha256':hashlib.sha256(guide_path.read_bytes()).hexdigest(),'context_sha256':hashlib.sha256(Path(args.context).read_bytes()).hexdigest(),'native_block_count':50,'model_path':args.model,'model_bytes':Path(args.model).stat().st_size,'dit_precision':'FullcanonicalBF16; native official sampler streamIO','steps':49,'seed':job['seed'],'sampler':'euler','cfg':1.,'sigma_grid':sigmas.tolist(),'forward_events':[],'HR_read':False}
        positive=[[context[None],{'minimax_token_tags':tags,'minimax_keyframes':[{'resolved_frame_index':0,'latent':guide}]}]]
        latent=comfy.nested_tensor.NestedTensor((torch.zeros(1,24,7,16,16),torch.zeros(1,32,2,37)))
        def hook(module,inputs,kwargs,output):
            value=inputs[0] if inputs else kwargs['x'];timestep=inputs[1] if len(inputs)>1 else kwargs['timestep'];event={'call':len(result['forward_events']),'video_shape':list(value[0].shape),'audio_shape':list(value[1].shape),'timestep':timestep.detach().float().cpu().flatten().tolist(),'video_finite':bool(torch.isfinite(output[0]).all()),'audio_finite':bool(torch.isfinite(output[1]).all())};assert event['video_finite'] and event['audio_finite'];result['forward_events'].append(event)
        handle=model.model.diffusion_model.register_forward_hook(hook,with_kwargs=True);started=time.monotonic()
        try:
            with torch.no_grad():samples=comfy.sample.sample(model,comfy.sample.prepare_noise(latent,job['seed']),49,1.,'euler','simple',positive,[],latent,seed=job['seed'],disable_pbar=True,sigmas=sigmas)
            video,audio=samples.unbind();save_file({'video':video.cpu().contiguous(),'audio':audio.cpu().contiguous()},str(folder/'sampled_av_latents.safetensors'));assert len(result['forward_events'])==49;result['status']='complete';entry['status']='complete'
        except Exception as exc:result.update(status='failed',exception=repr(exc));entry['status']='failed';raise
        finally:
            handle.remove();result.update(sampling_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());destination.write_text(json.dumps(result,indent=2));update()
        print(json.dumps({'event':'actualH3jobcomplete','id':job['id'],'seconds':result['sampling_seconds'],'full50block_calls':49}),flush=True)
    state.update(status='complete',total_full50block_forwards=49*len(jobs));update()


if __name__=='__main__':main()
