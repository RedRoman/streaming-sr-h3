"""One fixed train case: native SR training velocity vs official sampler math.

The known HR latent constructs a supervised noisy bridge only in this diagnostic.
It is NOT an SR inference result or a causal method input.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from training_math import inject_attention_lora,native_training_math
from train_native_lora import GuideProjection
from native_io import official_stream_inputs
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diff(a,b):
    d=a.float()-b.float()
    return {'max_abs':float(d.abs().max()),'rmse':float(d.square().mean().sqrt())}

def main():
    p=argparse.ArgumentParser()
    for key in ['comfy','model','guide','target','context','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','');assert gpu.isdigit()
    free,util=map(int,subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));assert free>75000 and util==0,(free,util)
    torch.set_num_threads(4);torch.manual_seed(1703);torch.cuda.set_per_process_memory_fraction(.9);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    sys.path.insert(0,str(a.comfy));from comfy.cli_args import args as ca;ca.reserve_vram=8.
    import comfy.sd,comfy.sample,comfy.nested_tensor,comfy.utils
    from comfy.ldm.minimax.model import PackedLayout,time_shift_sigma
    result={'status':'running','scope':'Fixed bear_256 train case mathematical interface diagnostic only, not restoration quality. KnownHRlatent only constructs supervised x_sigma, never guide.',
            'source_case':'bear_256, train, frames0..21','model_path':str(a.model),'model_bytes':a.model.stat().st_size,'canonical_SHA_previously_verified':'a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4',
            'source_sha256':{str(a.comfy/f):sha(a.comfy/f) for f in ['comfy/model_base.py','comfy/model_sampling.py','comfy/ldm/minimax/model.py','comfy/latent_formats.py']},
            'input_sha256':{str(p):sha(p) for p in [a.guide,a.target,a.context]},'conditions':[]}
    try:
        patcher=comfy.sd.load_diffusion_model(str(a.model),model_options={'dtype':torch.bfloat16});base=patcher.model;model=base.diffusion_model.eval().requires_grad_(False).cuda()
        inject_attention_lora(model,2);model.requires_grad_(False);projection=GuideProjection().cuda().requires_grad_(False)
        frozen=[(n,v,v._version) for n,v in model.named_parameters() if not (n.endswith('.a') or n.endswith('.b'))]
        cache=torch.load(a.context,map_location='cpu',weights_only=True);tags=cache['text_token_tags'].cuda()
        context=model.preprocess_text_embeds(cache['prompt_embeds'][None].cuda().to(torch.bfloat16)).detach()
        guide=load_file(str(a.guide))['guide'].float();target=load_file(str(a.target))['guide'].float();assert guide.shape==target.shape==(1,24,7,16,16)
        latent=comfy.nested_tensor.NestedTensor((target,torch.zeros(1,32,2,37)))
        noise=comfy.sample.prepare_noise(latent,101);nv,na=noise.unbind();nv=nv.float().cuda();na=na.float().cuda();target_gpu=target.cuda()
        grid=torch.linspace(1,0,50);grid=12*grid/(1+11*grid);sigma=float(grid[46]);sigma_tensor=torch.tensor([sigma],device='cuda')
        sigma_a=float(time_shift_sigma(torch.tensor(sigma),12,3));xt=(1-sigma)*target_gpu+sigma*nv;audio=sigma_a*na
        result.update(sigma_grid_index=46,sigma_video=sigma,sigma_audio=sigma_a,expected_timestep=sigma*1000,video_noise_seed=101,condition_seed_aligned_for_audit=703,
                      actual_training_condition_seed=703,actual_pilot_readout_condition_seed=101,noise_scale=float(base.model_sampling.noise_scale),
                      video_head_weight_dtype=str(model.final_layer.video_out.weight.dtype),video_head_count=model.final_layer.video_out.weight.shape[0]//model.final_layer.video_out.out_features,
                      latent_format_scale_factor=base.latent_format.scale_factor)
        assert result['noise_scale']==1 and result['video_head_count']==1
        packed,shapes=comfy.utils.pack_latents([xt,sigma*na]);perfect,_=comfy.utils.pack_latents([nv-target_gpu,torch.zeros_like(na)])
        exact=base.model_sampling.calculate_denoised(sigma_tensor,perfect,packed);wrong=base.model_sampling.calculate_denoised(sigma_tensor,-perfect,packed)
        exact_video=comfy.utils.unpack_latents(exact,shapes)[0];wrong_video=comfy.utils.unpack_latents(wrong,shapes)[0]
        result['analytic_perfect_v_x0_vs_HR']=diff(exact_video,target_gpu);result['opposite_sign_control_x0_vs_HR']=diff(wrong_video,target_gpu)
        assert result['analytic_perfect_v_x0_vs_HR']['max_abs']<2e-6 and result['opposite_sign_control_x0_vs_HR']['rmse']>.1
        expected_t=base.model_sampling.timestep(sigma_tensor);result['timestep_runtime']=expected_t.tolist()
        assert float((expected_t-sigma_tensor*1000).abs().max())==0
        outputs={}
        with torch.no_grad(),native_training_math():
            for label,bias in [('zero_adapter',0.),('one_guide_bias_plus_0p01',.01)]:
                projection.net[-1].bias.zero_();projection.net[-1].bias[0]=bias
                current=projection(guide.cuda());current_cpu=current.cpu()
                keyframes=[{'resolved_frame_index':0,'latent':current}]
                raw_payload={'text_token_tags':tags,'keyframes':keyframes,'cond_video_latents':[current],'seed':703,'audio_scale':1.,
                             'layout':PackedLayout(context.shape[1],7,16,16,37,keyframes=keyframes)}
                train_video,train_audio,train_timestep=official_stream_inputs(xt,na,sigma,time_shift_sigma)
                train_pred=model([train_video,train_audio],train_timestep,context,minimax_payload=raw_payload,transformer_options={})
                captured={}
                def pre(module,inputs,kwargs):
                    captured['calls']=captured.get('calls',0)+1
                    captured['x']=[v.detach().clone() for v in inputs[0]];captured['t']=inputs[1].detach().clone()
                    captured['context']=kwargs['context'].detach().clone();captured['payload']=kwargs['minimax_payload'];captured['options']=kwargs['transformer_options'].copy()
                def post(module,inputs,kwargs,output):captured['output']=[v.detach().clone() for v in output]
                h1=model.register_forward_pre_hook(pre,with_kwargs=True);h2=model.register_forward_hook(post,with_kwargs=True)
                positive=[[cache['prompt_embeds'][None],{'minimax_token_tags':cache['text_token_tags'],'minimax_keyframes':[{'resolved_frame_index':0,'latent':current_cpu}]}]]
                try:
                    sample=comfy.sample.sample(patcher,noise,1,1.,'euler','simple',positive,[],latent,seed=703,disable_pbar=True,sigmas=torch.tensor([sigma,0.]))
                finally:h1.remove();h2.remove()
                assert captured['calls']==1
                sample_video=sample.unbind()[0].cuda().float();infer_v=captured['output'][0].float()
                carry=time_shift_sigma((captured['t'].flatten()[0]/1000).float().clamp(min=1e-6),12,3)/(captured['t'].flatten()[0]/1000).float().clamp(min=1e-6)
                physical_audio=captured['x'][1]*carry.to(captured['x'][1].dtype)
                matched_payload=dict(captured['payload']);matched_payload['audio_scale']=1.
                matched=model([captured['x'][0],physical_audio],captured['t'],captured['context'],minimax_payload=matched_payload,transformer_options=captured['options'])
                raw_rows=model._cond_video_rows(raw_payload,'cuda');infer_rows=model._cond_video_rows(captured['payload'],'cuda')
                p703=dict(captured['payload']);p101=dict(captured['payload']);p703['seed']=703;p101['seed']=101
                row={'label':label,'guide_last_bias_channel0':bias,'sampler_forward_calls':captured['calls'],'sampler_timestep':captured['t'].float().tolist(),
                     'input_video_dtype_train':str(train_video.dtype),'input_video_dtype_official_wrapper':str(captured['x'][0].dtype),
                     'output_video_dtype_train':str(train_pred[0].dtype),'output_video_dtype_official_wrapper':str(captured['output'][0].dtype),
                     'official_audio_scale_flag':captured['payload']['audio_scale'],
                     'guide_shapes_train':[list(z.shape) for z in raw_payload['cond_video_latents']],
                     'guide_shapes_official':[list(z.shape) for z in captured['payload']['cond_video_latents']],
                     'layout_segments_identical':raw_payload['layout'].segments==captured['payload']['layout'].segments,
                     'context_preprocess_train_vs_official':diff(context,captured['context']),
                     'guide_before_aug_train_vs_official':diff(current,current_cpu.cuda()),
                     'guide_after_aug_GPUtrain_vs_CPUinfer':diff(raw_rows,infer_rows),
                     'actual_condition_seed703_vs101_aug_rows':diff(model._cond_video_rows(p703,'cuda'),model._cond_video_rows(p101,'cuda')),
                     'video_input_v2_vs_wrapper':diff(train_video,captured['x'][0]),
                     'video_input_after_matching_wrapper_cast':diff(xt.to(captured['x'][0].dtype),captured['x'][0]),
                     'audio_physical_v2_vs_wrapper':diff(train_audio,physical_audio),
                     'v_native_io_v2_vs_official_wrapper':diff(train_pred[0],infer_v),
                     'v_matched_wrapper_precision_and_payload':diff(matched[0],infer_v),
                     'official_oneEuler_output_vs_xt_minus_sigma_v':diff(sample_video,xt-sigma*infer_v),
                     'training_x0_vs_official_oneEuler_output':diff(xt-sigma*train_pred[0].float(),sample_video),
                     'guide_condition_only_LR':True,'HR_only_target_stream_for_math':True}
                assert row['layout_segments_identical'] and row['context_preprocess_train_vs_official']['max_abs']==0 and row['guide_before_aug_train_vs_official']['max_abs']==0
                assert row['v_matched_wrapper_precision_and_payload']['max_abs']==0,row
                assert row['v_native_io_v2_vs_official_wrapper']['max_abs']==0,row
                assert row['video_input_v2_vs_wrapper']['max_abs']==row['audio_physical_v2_vs_wrapper']['max_abs']==0,row
                assert row['training_x0_vs_official_oneEuler_output']['max_abs']<2e-5,row
                assert row['official_oneEuler_output_vs_xt_minus_sigma_v']['max_abs']<2e-5,row
                result['conditions'].append(row);outputs[label]=sample_video.cpu();print(json.dumps({'event':'math_alignment_condition',**row}),flush=True)
            restored=base.process_latent_out(base.process_latent_in(latent));result['video_process_in_out_roundtrip']=diff(restored.unbind()[0],target)
        result['one_parameter_effect_on_official_x0']=diff(outputs['zero_adapter'],outputs['one_guide_bias_plus_0p01'])
        assert result['one_parameter_effect_on_official_x0']['rmse']>0 and result['video_process_in_out_roundtrip']['max_abs']==0
        # Real all-50-block backward, with the same official IO helper used by v2.
        for block in model.blocks:
            original=block.forward
            def checkpointed(*inputs,_original=original,**kwargs):
                if not torch.is_grad_enabled():return _original(*inputs,**kwargs)
                if 'transformer_options' in kwargs:kwargs={**kwargs,'transformer_options':dict(kwargs['transformer_options'])}
                return checkpoint(_original,*inputs,use_reentrant=False,**kwargs)
            block.forward=checkpointed
        lora=[v for n,v in model.named_parameters() if n.endswith('.a') or n.endswith('.b')]
        for v in lora:v.requires_grad_(True)
        projection.requires_grad_(True)
        with torch.no_grad():projection.net[-1].bias.zero_()
        grad_guide=guide.cuda().detach().requires_grad_();projected=projection(grad_guide)
        keyframes=[{'resolved_frame_index':0,'latent':projected}]
        payload={'text_token_tags':tags,'keyframes':keyframes,'cond_video_latents':[projected],'seed':703,'audio_scale':1.,
                 'layout':PackedLayout(context.shape[1],7,16,16,37,keyframes=keyframes)}
        video_io,audio_io,timestep=official_stream_inputs(xt,na,sigma,time_shift_sigma)
        with native_training_math():
            pred=model([video_io,audio_io],timestep,context,minimax_payload=payload,transformer_options={})
            loss=F.mse_loss(pred[0].float(),nv-target_gpu)
            assert loss.dtype==torch.float32 and pred[0].dtype==torch.bfloat16
            loss.backward()
        norm=lambda seq:float(torch.stack([v.grad.float().square().sum() for v in seq if v.grad is not None]).sum().sqrt())
        backward={'loss_float32':float(loss.detach()),'output_dtype':str(pred[0].dtype),'guide_input_grad_norm':float(grad_guide.grad.float().norm()),
                  'lora_grad_norm':norm(lora),'guide_projection_grad_norm':norm(list(projection.parameters())),
                  'base_weight_grad_count':sum(v.grad is not None for _,v,_ in frozen),'frozen_base_tensors':len(frozen)}
        assert backward['guide_input_grad_norm']>0 and backward['lora_grad_norm']>0 and backward['guide_projection_grad_norm']>0 and backward['base_weight_grad_count']==0
        result['native_io_v2_real_backward']=backward;result['native_io_v2_audit_passed']=True
        result['native_io_helper_sha256']=sha(Path(__file__).with_name('native_io.py'))
        assert all(v._version==version and v.grad is None and not v.requires_grad for _,v,version in frozen)
        save_file({k:v.contiguous() for k,v in outputs.items()},str(a.out/'diagnostic_x0_latents.safetensors'))
        result.update(status='complete',base_frozen_tensor_versions_unchanged=True,base_weight_grad_count=0,
                      interpretation='Native IO v2 exactly matches actual official wrapper BF16 streams, AV carry, video velocity and one-step x0 for zero adapter and +.01 guide bias. Full50 real backward nonzero adapters/guide, frozen base unchanged. No optimizer update in this audit.')
    except Exception as exc:result.update(status='failed',exception=repr(exc));raise
    finally:
        result.update(seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
