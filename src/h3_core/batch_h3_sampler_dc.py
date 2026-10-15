"""Official native49 H3 with independently versioned actual-far sampling DC.

Post-CFG clean video += E(P25(D(video)))-E(D(video)); audio/guide unchanged.
Own Gaussian.6/AA far operator, one measured packet repeated22 contexts.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from observation_quotient import ObservationQuotient

POINTS=[28,45,48]


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);p.add_argument('--state',required=True);p.add_argument('--comfy',required=True);p.add_argument('--model',required=True);p.add_argument('--context',required=True);p.add_argument('--vae',required=True);a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.9);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;sys.path.insert(0,str(Path(a.comfy).resolve()));from comfy.cli_args import args as ca;ca.reserve_vram=8.
    import comfy.sd,comfy.sample,comfy.nested_tensor,comfy.utils
    jobs=json.loads(Path(a.jobs).read_text());began=time.monotonic();state=dict(status='running',pid=os.getpid(),GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),jobs=[],scope='SeparatefarinnerDCteacher version; allprefix inputs alreadyarrived, static22 onefarpacket');statepath=Path(a.state)
    def update():state['seconds']=time.monotonic()-began;statepath.write_text(json.dumps(state,indent=2))
    update();patcher=comfy.sd.load_diffusion_model(a.model,model_options={'dtype':torch.bfloat16});native=patcher.model.diffusion_model.eval().requires_grad_(False);assert len(native.blocks)==50;frozen=[(v,v._version) for v in native.parameters()];cache=torch.load(a.context,map_location='cpu',weights_only=True);grid=torch.linspace(1,0,50);sigmas=12*grid/(1+11*grid);q=ObservationQuotient(device='cpu');vae=None
    def get_vae():
        nonlocal vae
        if vae is None:
            weights,metadata=comfy.utils.load_torch_file(a.vae,return_metadata=True);vae=comfy.sd.VAE(sd=weights,device=torch.device('cuda'),dtype=torch.float16,metadata=metadata);del weights;vae.first_stage_model.eval().requires_grad_(False)
        return vae
    for job in jobs:
        folder=Path(job['directory']);dest=folder/'sample_result.json';entry={**job,'status':'running'};state['jobs'].append(entry);update()
        if dest.exists() and json.loads(dest.read_text()).get('status')=='complete':entry['status']='already_complete';update();continue
        guide_path=folder/'guide.safetensors';guide=load_file(str(guide_path))['guide'].float();data=torch.load(job['far_inputs'],map_location='cpu',weights_only=False);low=data['observations']['far'].float().expand(22,-1,-1,-1).clone();far_id=data['packet_ids']['far'];mode=job.get('mode','sampling_dc');assert guide.shape==(1,24,7,16,16) and low.shape==(22,3,64,64)
        positive=[[cache['prompt_embeds'][None],{'minimax_token_tags':cache['text_token_tags'],'minimax_keyframes':[{'resolved_frame_index':0,'latent':guide}]}]];latent=comfy.nested_tensor.NestedTensor((torch.zeros(1,24,7,16,16),torch.zeros(1,32,2,37)));_,shapes=comfy.utils.pack_latents(list(latent.unbind()));events=[];corrections=[];hook_calls=0
        def observer(module,args,kwargs,output):
            index=len(events);record=dict(call=index,video_finite=bool(torch.isfinite(output[0]).all()),audio_finite=bool(torch.isfinite(output[1]).all()));assert record['video_finite'] and record['audio_finite']
            if index in [0,*POINTS]:
                actual=kwargs['minimax_payload']['cond_video_latents'][0].detach().cpu();record['guide_maxabs']=float((actual-guide).abs().max());assert record['guide_maxabs']==0
            events.append(record)
        handle=native.register_forward_hook(observer,with_kwargs=True);current=patcher.clone()
        def hook(args):
            nonlocal hook_calls
            index=hook_calls;hook_calls+=1;assert len(events)==index+1 and abs(float(args['sigma'].flatten()[0])-float(sigmas[index]))<1e-6
            if index not in POINTS:return args['denoised']
            start=time.monotonic();denoised=args['denoised'];z0,audio=comfy.utils.unpack_latents(denoised,shapes);rgb=get_vae().decode(z0).float().cpu()
            if rgb.ndim==5:rgb=rgb[0]
            assert rgb.shape==(22,256,256,3) and torch.isfinite(rgb).all();zbase=get_vae().encode(rgb.clone())
            if mode=='identity':delta=torch.zeros_like(zbase);E_calls=1;projected=rgb.permute(0,3,1,2)
            else:
                projected=q.bounded(rgb.permute(0,3,1,2),low,steps=25);zc=get_vae().encode(projected.permute(0,2,3,1).contiguous().clone());delta=zc-zbase;E_calls=2
            delta=delta.to(z0);assert delta.shape==z0.shape and torch.isfinite(delta).all();packed,_=comfy.utils.pack_latents([z0+delta,audio]);_,aa=comfy.utils.unpack_latents(packed,shapes);assert torch.equal(audio,aa)
            if mode=='identity':assert torch.equal(packed,denoised) and float(delta.abs().max())==0
            corrections.append(dict(index=index,sigma=float(args['sigma'].flatten()[0]),D_calls=1,E_calls=E_calls,delta_RMS=float(delta.float().square().mean().sqrt()),delta_maxabs=float(delta.abs().max()),input_far_LR_RMSE=float((q.observe(rgb.permute(0,3,1,2))-low).square().mean().sqrt()),projected_far_LR_RMSE=float((q.observe(projected)-low).square().mean().sqrt()),audio_return_maxabs=0.,seconds=time.monotonic()-start,identity=mode=='identity'))
            return denoised if mode=='identity' else packed
        current.set_model_sampler_post_cfg_function(hook);start=time.monotonic()
        try:
            with torch.no_grad():samples=comfy.sample.sample(current,comfy.sample.prepare_noise(latent,job['seed']),49,1.,'euler','simple',positive,[],latent,seed=job['seed'],disable_pbar=True,sigmas=sigmas)
            video,audio=samples.unbind();assert len(events)==49 and hook_calls==49 and len(corrections)==3;assert all(v._version==version and v.grad is None for v,version in frozen);save_file({'video':video.cpu().contiguous(),'audio':audio.cpu().contiguous()},str(folder/'sampled_av_latents.safetensors'))
            result=dict(status='complete',job=job,mode=mode,forward_events=events,steps=49,native_block_count=50,dit_precision='FullBF16 official sampler streamIO',correction_events=corrections,E_calls=sum(v['E_calls'] for v in corrections),D_calls_inside_sampler=3,guide_sha256=hashlib.sha256(guide_path.read_bytes()).hexdigest(),context_sha256=hashlib.sha256(Path(a.context).read_bytes()).hexdigest(),sampling_seconds=time.monotonic()-start,far_packet_id=far_id,distinct_far_measurements=1,repeated_native_context=22,far_physical_frame_index=0,bounded_wait_for_repeated_far=0,HR_read=False,near_guide_unchanged=True,audio_inside_hook_unchanged=True,pixel_operator='Own actualGaussian.6+reflect+AA bicubic4x, Dykstra25',base_weight_grad_count=0,base_parameter_versions_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if mode=='identity':
                baseline=load_file(job['raw_reference']);parity={k:float((baseline[k]-v.cpu()).abs().max()) for k,v in [('video',video),('audio',audio)]};result['raw_reference_parity_maxabs']=parity;assert max(parity.values())==0,parity
            dest.write_text(json.dumps(result,indent=2));entry['status']='complete';print(json.dumps(dict(event='DC_operator_H3_job_complete',id=job['id'],mode=mode,seconds=result['sampling_seconds'],calls=49)),flush=True)
        except Exception as exc:
            entry.update(status='failed',exception=repr(exc));state.update(status='failed');update();raise
        finally:handle.remove();update()
    state.update(status='complete',full50_DiT_forward_calls=49*len(jobs));update()


if __name__=='__main__':main()
