"""Fixed, prior-work-inspired sampling DC control; no training/parameter search."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file,save_file
from training_math import native_training_math

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from pixel_consistency import BicubicConsistency
SHA_MODEL='a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4'
SHA_VAE='7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522'
POINTS=[28,45,48]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/h3_setup/sampling_dc_pilot'));a=p.parse_args();os.chdir(ROOT);a.out.mkdir(parents=True,exist_ok=True)
 gpu=os.environ.get('CUDA_VISIBLE_DEVICES','');assert gpu.isdigit()
 free,util=map(int,subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));assert free>75000 and util==0,(free,util)
 torch.set_num_threads(4);torch.manual_seed(1703);torch.cuda.set_per_process_memory_fraction(.9);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
 comfy=ROOT/'external/ComfyUI_H3';sys.path.insert(0,str(comfy));from comfy.cli_args import args as ca;ca.reserve_vram=8.
 import comfy.sd,comfy.sample,comfy.nested_tensor,comfy.utils
 baseline=ROOT/'results/h3_setup/native_sr_pilot400_native_io_v2'
 tr=json.loads((ROOT/'results/h3_setup/native_sr_train_pairs/pairs_manifest.json').read_text());va=json.loads((ROOT/'results/h3_setup/native_sr_val_pairs/pairs_manifest.json').read_text())
 train=next(c for c in tr['cases'] if c['id']=='bear_256');validation=sorted(va['cases'],key=lambda x:x['id']);assert len(validation)==4
 grid=torch.linspace(1,0,50);sigmas=12*grid/(1+11*grid)
 protocol={'status':'preregistered','scope':'DDNM/FlowDPS-inspired engineering baseline, not paper-exact reproduction or memory method; no training/selection',
  'validation_cases':[c['id'] for c in validation],'train_guard_case':'bear_256','same_v2_protocol_sha256':sha(baseline/'protocol.json'),
  'seed':101,'cfg':1.,'sigma_grid':sigmas.tolist(),'steps':49,'correction_indices':POINTS,'correction_sigmas':[float(sigmas[i]) for i in POINTS],'strength':1.,
  'correction':'post-CFG clean z0 + E(P25(D(z0)))-E(D(z0)); originalEuler,guide,audio unchanged','pixel_operator_sha256':sha(ROOT/'src/pixel_consistency.py'),
  'context_sha256':sha(ROOT/'external/h3_assets/h3_qwen_context_neutral_restore.pt'),'model_sha256_previously_verified':SHA_MODEL,'vae_sha256_previously_verified':SHA_VAE,
  'vae_precision':'canonicalFP16storage/compute, standardwrapper; native22, floatRGB internal, no PNGquantization',
  'train_guard':'original49 vs identityprojection hooks; shared E(r) => delta=0 exactly; no val tuning','HR_input':False,
  'main_hooks_per_case':3,'main_E_calls':6,'main_mid_D_calls':3,'main_final_D_separate':1,
  'cost_note':'Guard identityhook uses3D+3E; main perhook1D+2E. No hidden perhook correctionD; finaltrueD/LRerror evaluated separately.'}
 proto=a.out/'protocol.json'
 if proto.exists():assert json.loads(proto.read_text())==protocol,'Refuse protocol drift'
 else:write(proto,protocol)
 protocol_sha=sha(proto)
 state={'status':'running','pid':os.getpid(),'gpu':gpu,'admission_free_MiB':free,'admission_util':util,'protocol_sha256':protocol_sha,'script_sha256':sha(__file__),'completed':[]}
 def live(**kw):state.update(kw,epoch=time.time());write(a.out/'LIVE_STATE.json',state)
 model_path=ROOT/'external/h3_assets/minimax_h3_fl2va_pruned_bf16.safetensors';vae_path=ROOT/'external/h3_assets/minimax_h3_video_vae_fp16.safetensors'
 assert model_path.stat().st_size==40225724176 and vae_path.stat().st_size==5207808496
 live(phase='load_frozen_foundation');patcher=comfy.sd.load_diffusion_model(str(model_path),model_options={'dtype':torch.bfloat16});model=patcher.model.diffusion_model.eval().requires_grad_(False).cuda();assert len(model.blocks)==50
 assert patcher.model.latent_format.scale_factor==1
 frozen=[(n,v,v._version) for n,v in model.named_parameters()]
 cache=torch.load(ROOT/'external/h3_assets/h3_qwen_context_neutral_restore.pt',map_location='cpu',weights_only=True)
 vae=None
 def get_vae():
  nonlocal vae
  if vae is None:
   begin=time.perf_counter();sd,meta=comfy.utils.load_torch_file(str(vae_path),return_metadata=True);vae=comfy.sd.VAE(sd=sd,device=torch.device('cuda'),dtype=torch.float16,metadata=meta);del sd
   vae.first_stage_model.eval().requires_grad_(False);state['vae_wrapper_init_seconds']=time.perf_counter()-begin
  return vae
 def freeze_audit():
  assert all(v._version==ver and v.grad is None and not v.requires_grad for _,v,ver in frozen)
  return {'base_parameter_versions_unchanged':True,'base_weight_grad_count':0,'frozen_base_tensors':len(frozen)}
 def sample(case,label):
  folder=a.out/('guard' if label in ['original','identity'] else 'readouts')/label/case['id'];folder.mkdir(parents=True,exist_ok=True)
  if (folder/'sample_result.json').exists() and json.loads((folder/'sample_result.json').read_text()).get('status')=='complete':return load_file(str(folder/'sampled_av_latents.safetensors'))
  live(phase='official49',mode=label,case=case['id'])
  guide=load_file(case['guide'])['guide'].float();assert sha(case['guide'])==case['guide_sha256']
  files=[ROOT/'data/real_video'/q for q in case['lr_paths']];assert [sha(q) for q in files]==case['observed_lr_sha256']
  low=torch.stack([torch.from_numpy(np.array(Image.open(q).convert('RGB')).copy()).permute(2,0,1) for q in files]).float()/255
  h,w=case['shape_hw'];assert low.shape==(22,3,h//4,w//4)
  op=BicubicConsistency((h,w),(h//4,w//4));positive=[[cache['prompt_embeds'][None],{'minimax_token_tags':cache['text_token_tags'],'minimax_keyframes':[{'resolved_frame_index':0,'latent':guide}]}]]
  latent=comfy.nested_tensor.NestedTensor((torch.zeros(1,24,7,h//16,w//16),torch.zeros(1,32,2,37)));_,shapes=comfy.utils.pack_latents(list(latent.unbind()))
  observations=[];calls=0;hook_calls=0;guide_checks=[]
  def observer(module,args,kwargs,output):
   nonlocal calls
   if calls in [0,*POINTS]:
    actual=kwargs['minimax_payload']['cond_video_latents'][0].detach().cpu();difference=float((actual-guide).abs().max());assert difference==0;guide_checks.append({'index':calls,'guide_maxabs':difference})
   calls+=1
  observer_handle=model.register_forward_hook(observer,with_kwargs=True)
  current=patcher.clone()
  def hook(args):
   nonlocal hook_calls
   index=hook_calls;hook_calls+=1;assert calls==index+1 and abs(float(args['sigma'].flatten()[0])-float(sigmas[index]))<1e-6
   if index not in POINTS:return args['denoised']
   begin=time.perf_counter();denoised=args['denoised'];z0,audio=comfy.utils.unpack_latents(denoised,shapes)
   torch.cuda.synchronize();t=time.perf_counter();rgb=get_vae().decode(z0).float().cpu();torch.cuda.synchronize();dtime=time.perf_counter()-t
   if rgb.ndim==5:rgb=rgb[0]
   assert rgb.shape==(22,h,w,3) and torch.isfinite(rgb).all()
   t=time.perf_counter();zbase=get_vae().encode(rgb.clone());torch.cuda.synchronize();etime0=time.perf_counter()-t
   projection_time=0.;projected_residual=None
   if label=='identity':delta=torch.zeros_like(zbase);etime1=0.;E_calls=1
   else:
    t=time.perf_counter();projected,residual=op.project(rgb.permute(0,3,1,2),low,iterations=25);projection_time=time.perf_counter()-t;projected_residual=float(residual.max())
    t=time.perf_counter();zc=get_vae().encode(projected.permute(0,2,3,1).contiguous().clone());torch.cuda.synchronize();etime1=time.perf_counter()-t;delta=zc-zbase;E_calls=2
   delta=delta.to(device=z0.device,dtype=z0.dtype);assert delta.shape==z0.shape and torch.isfinite(delta).all()
   corrected=z0+delta;packed,_=comfy.utils.pack_latents([corrected,audio]);_,audio_after=comfy.utils.unpack_latents(packed,shapes)
   assert torch.equal(audio,audio_after)
   if label=='identity':assert float(delta.abs().max())==0 and torch.equal(packed,denoised)
   observation={'index':index,'sigma':float(args['sigma'].flatten()[0]),'D_calls':1,'E_calls':E_calls,'D_seconds':dtime,'E_original_seconds':etime0,'E_projected_seconds':etime1,
    'projection_seconds':projection_time,'hook_total_seconds':time.perf_counter()-begin,'delta_maxabs':float(delta.abs().max()),'delta_rms':float(delta.float().square().mean().sqrt()),
    'input_D_LR_rmse':float((op.down(rgb.permute(0,3,1,2).to(op.w))-low.to(op.w)).square().mean().sqrt()),'projected_RGB_max_LR_rmse':projected_residual,
    'audio_return_maxabs':0.,'guide_unchanged':True,'identity_projection':label=='identity'}
   observations.append(observation);save_file({'clean_video':z0.detach().cpu().contiguous(),'delta':delta.detach().cpu().contiguous()},str(folder/f'hook_{index:03d}.safetensors'))
   print(json.dumps({'event':'sampling_DC_hook','case':case['id'],'mode':label,**observation}),flush=True)
   return denoised if label=='identity' else packed
  if label!='original':current.set_model_sampler_post_cfg_function(hook)
  t=time.perf_counter()
  try:
   with torch.no_grad(),native_training_math():samples=comfy.sample.sample(current,comfy.sample.prepare_noise(latent,101),49,1.,'euler','simple',positive,[],latent,seed=101,disable_pbar=True,sigmas=sigmas)
   torch.cuda.synchronize()
  finally:observer_handle.remove()
  video,audio=samples.unbind();assert calls==49 and torch.isfinite(video).all() and torch.isfinite(audio).all()
  if label!='original':assert hook_calls==49 and [v['index'] for v in observations]==POINTS
  save_file({'video':video.cpu().contiguous(),'audio':audio.cpu().contiguous()},str(folder/'sampled_av_latents.safetensors'));shutil.copy2(case['guide'],folder/'guide.safetensors')
  result={'status':'complete','mode':label,'case':case['id'],'split':case['split'],'shape_hw':case['shape_hw'],'input_prefix':[0,21],'bounded_wait_frames':21,'observed_lr_sha256':case['observed_lr_sha256'],
   'guide_sha256':case['guide_sha256'],'context_sha256':protocol['context_sha256'],'protocol_sha256':protocol_sha,'steps_model_evaluations':calls,'sigma_points':50,'cfg':1.,'seed':101,'seconds':time.perf_counter()-t,
   'decoder_pending':True,'HR_condition':False,'gt_target_not_read_by_sampler':True,'correction_events':observations,'E_calls':sum(x['E_calls'] for x in observations),'D_calls_inside_sampler':sum(x['D_calls'] for x in observations),
   'guide_checks':guide_checks,'audio_inside_hook_unchanged':True,**freeze_audit()}
  write(folder/'sample_result.json',result);state['completed'].append(label+'/'+case['id']);print(json.dumps({'event':'sampling_complete',**{k:v for k,v in result.items() if k!='observed_lr_sha256'}}),flush=True)
  return {'video':video.cpu(),'audio':audio.cpu()}
 try:
  # A feasible RGB fixed point for the exact floating operator, distinct from
  # the explicitly identity-projection no-innovation sampler test.
  op=BicubicConsistency(32,8);g=torch.Generator().manual_seed(722);x=.2+.6*torch.rand((1,3,32,32),generator=g,dtype=torch.float64);y=op.down(x);px,_=op.project(x,y,25);fixed_error=float((px-x).abs().max());assert fixed_error<1e-12
  original=sample(train,'original');identity=sample(train,'identity');parity={key:float((original[key]-identity[key]).abs().max()) for key in ['video','audio']};assert max(parity.values())==0,parity
  write(a.out/'zero_correction_parity.json',{'passed':True,'original_vs_identity_latent_maxabs':parity,'identity_delta_exactly_zero':True,'actual_P25_feasible_RGB_fixedpoint_maxabs':fixed_error,**freeze_audit()})
  for label in ['unadapted','vae_only']:
   for case in validation:
    destination=a.out/'readouts'/label/case['id'];shutil.copytree(baseline/'readouts'/label/case['id'],destination,dirs_exist_ok=True)
  for case in validation:sample(case,'sampling_dc')
  live(status='complete',phase='sampling_complete',seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),**freeze_audit())
 except Exception as e:live(status='failed',exception=repr(e));raise
 finally:print(json.dumps(state),flush=True)

if __name__=='__main__':main()
