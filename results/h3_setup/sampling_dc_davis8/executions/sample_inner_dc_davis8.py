"""Fixed, prior-work-inspired sampling DC control; no training/parameter search."""
import argparse,ast,hashlib,json,os,shutil,subprocess,sys,time,types
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
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/h3_setup/sampling_dc_davis8'));a=p.parse_args();os.chdir(ROOT);a.out.mkdir(parents=True,exist_ok=True)
 gpu=os.environ.get('CUDA_VISIBLE_DEVICES','');assert gpu.isdigit()
 free,util=map(int,subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));assert free>75000 and util==0,(free,util)
 torch.set_num_threads(4);torch.manual_seed(1703);torch.cuda.set_per_process_memory_fraction(.9);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
 comfypath=ROOT/'external/ComfyUI_H3';sys.path.insert(0,str(comfypath));from comfy.cli_args import args as ca;ca.reserve_vram=8.
 import comfy.sd,comfy.sample,comfy.nested_tensor,comfy.utils
 protocol=json.loads((a.out/'protocol.json').read_text());protocol_sha=sha(a.out/'protocol.json')
 assert sha(a.out/'source_manifest.json')==protocol['source_manifest_sha256']
 for name,expected in json.loads((a.out/'source_manifest.json').read_text()).items():assert sha(ROOT/'experiments/h3_backbone'/name)==expected,name
 assert sha(a.out/'input_manifest.json')==protocol['input_manifest_sha256'] and sha(a.out/'reuse_sha256.json')==protocol['reuse_manifest_sha256']
 cases=json.loads((a.out/'input_manifest.json').read_text())['cases'];assert len(cases)==16 and len({c['sequence'] for c in cases})==8
 grid=torch.linspace(1,0,50);sigmas=12*grid/(1+11*grid);assert sigmas.tolist()==protocol['sigma_grid']
 assert sha(comfypath/'comfy_extras/nodes_minimax_h3.py')==protocol['guide_helper_sha256']
 state={'status':'running','pid':os.getpid(),'gpu':gpu,'admission_free_MiB':free,'admission_util':util,'protocol_sha256':protocol_sha,'script_sha256':sha(__file__),'completed':[]}
 def live(**kw):state.update(kw,epoch=time.time());write(a.out/'LIVE_STATE.json',state)
 model_path=ROOT/'external/h3_assets/minimax_h3_fl2va_pruned_bf16.safetensors';vae_path=ROOT/'external/h3_assets/minimax_h3_video_vae_fp16.safetensors'
 assert model_path.stat().st_size==40225724176 and vae_path.stat().st_size==5207808496
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
  if label=='sampling_dc':current.set_model_sampler_post_cfg_function(hook)
  t=time.perf_counter()
  try:
   with torch.no_grad(),native_training_math():samples=comfy.sample.sample(current,comfy.sample.prepare_noise(latent,101),49,1.,'euler','simple',positive,[],latent,seed=101,disable_pbar=True,sigmas=sigmas)
   torch.cuda.synchronize()
  finally:observer_handle.remove()
  video,audio=samples.unbind();assert calls==49 and torch.isfinite(video).all() and torch.isfinite(audio).all()
  if label=='sampling_dc':assert hook_calls==49 and [v['index'] for v in observations]==POINTS
  save_file({'video':video.cpu().contiguous(),'audio':audio.cpu().contiguous()},str(folder/'sampled_av_latents.safetensors'));shutil.copy2(case['guide'],folder/'guide.safetensors')
  result={'status':'complete','mode':label,'case':case['id'],'split':case['split'],'shape_hw':case['shape_hw'],'input_prefix':[0,21],'bounded_wait_frames':21,'observed_lr_sha256':case['observed_lr_sha256'],
   'guide_sha256':case['guide_sha256'],'context_sha256':protocol['context_sha256'],'protocol_sha256':protocol_sha,'steps_model_evaluations':calls,'sigma_points':50,'cfg':1.,'seed':101,'seconds':time.perf_counter()-t,
   'decoder_pending':True,'HR_condition':False,'gt_target_not_read_by_sampler':True,'correction_events':observations,'E_calls':sum(x['E_calls'] for x in observations),'D_calls_inside_sampler':sum(x['D_calls'] for x in observations),
   'guide_checks':guide_checks,'audio_inside_hook_unchanged':True,**freeze_audit()}
  write(folder/'sample_result.json',result);state['completed'].append(label+'/'+case['id']);print(json.dumps({'event':'sampling_complete',**{k:v for k,v in result.items() if k!='observed_lr_sha256'}}),flush=True)
  return {'video':video.cpu(),'audio':audio.cpu()}
 def decode(case,label):
  folder=a.out/'readouts'/label/case['id'];folder.mkdir(parents=True,exist_ok=True)
  target=folder/'decode_result.json'
  if target.exists():
   prior=json.loads(target.read_text());assert prior['status']=='complete' and all(sha(folder/k)==v for k,v in prior['output_png_sha256'].items());return
  latent=Path(case['guide']) if label=='vae_only' else folder/'sampled_av_latents.safetensors';key='guide' if label=='vae_only' else 'video'
  begin=time.perf_counter()
  with torch.no_grad():pixels=get_vae().decode(load_file(str(latent))[key]).float().cpu()
  if pixels.ndim==5:pixels=pixels[0]
  assert pixels.shape==(22,*case['shape_hw'],3) and torch.isfinite(pixels).all();files={}
  for i,frame in enumerate(pixels):
   path=folder/f'output_{i:02d}.png';Image.fromarray(np.uint8(frame.numpy().clip(0,1)*255+.5)).save(path);files[path.name]=sha(path)
  result={'status':'complete','scope':'Same canonicalFP16wrapper/math/PNG rounding asv2; noHRinput','case':case['id'],'mode':label,'shape':list(pixels.shape),'range':[float(pixels.min()),float(pixels.max())],'vae_precision':'FP16storageandcompute','torch':torch.__version__,'python':sys.version,'vae_sha256':SHA_VAE,'latent_sha256':sha(latent),'output_png_sha256':files,'seconds':time.perf_counter()-begin}
  write(target,result);print(json.dumps({'event':'final_decode_complete','case':case['id'],'mode':label,'seconds':result['seconds']}),flush=True)
 try:
  # Exact official preprocessing; test guides are LR-only, frozen before sampling.
  node_source=comfypath/'comfy_extras/nodes_minimax_h3.py';node=next(n for n in ast.parse(node_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_resize')
  namespace={'comfy':types.SimpleNamespace(utils=comfy.utils)};exec(compile(ast.Module(body=[node],type_ignores=[]),str(node_source),'exec'),namespace);resize=namespace['_resize'];guide_records=[]
  for case in cases:
   paths=[ROOT/'data/real_video'/p for p in case['lr_paths']];assert [sha(p) for p in paths]==case['observed_lr_sha256']
   guide=Path(case['guide'])
   if case['split']=='test':
    live(phase='LR_only_guide_encode',case=case['id']);guide.parent.mkdir(parents=True,exist_ok=True)
    low=torch.stack([torch.from_numpy(np.array(Image.open(p).convert('RGB')).copy()).permute(2,0,1) for p in paths]).float()/255
    up=torch.nn.functional.interpolate(low,size=case['shape_hw'],mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
    pixels=resize(up.permute(0,2,3,1),case['shape_hw'][1],case['shape_hw'][0],'center');t=time.perf_counter()
    with torch.no_grad():encoded=get_vae().encode(pixels).cpu().contiguous()
    save_file({'guide':encoded},str(guide));case['guide_sha256']=sha(guide);guide_records.append({'case':case['id'],'guide_sha256':case['guide_sha256'],'new_E_calls':1,'seconds':time.perf_counter()-t})
   else:assert sha(guide)==case['guide_sha256'];guide_records.append({'case':case['id'],'guide_sha256':case['guide_sha256'],'new_E_calls':0})
  write(a.out/'guides_frozen.json',{'frozen_before_sampling':True,'records':guide_records,'cases':cases,'helper_sha256':sha(node_source)})
  # Verify every reused byte against the completed pilot archive, then preserve it.
  reuse=json.loads((a.out/'reuse_sha256.json').read_text());source=ROOT/reuse['source'];assert sha(source/'protocol.json')==reuse['source_protocol_sha256']
  for case in cases:
   if case['split']!='val':continue
   for mode in ['unadapted','sampling_dc']:
    previous=json.loads((source/'readouts'/mode/case['id']/'sample_result.json').read_text());assert previous['guide_sha256']==case['guide_sha256'] and previous['observed_lr_sha256']==case['observed_lr_sha256'] and previous['context_sha256']==protocol['context_sha256'] and previous['seed']==101 and previous['steps_model_evaluations']==49 and not previous['HR_condition']
  for name,record in reuse['files'].items():
   origin=source/name;assert origin.stat().st_size==record['bytes'] and sha(origin)==record['sha256'],name
   destination=a.out/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,destination)
  for case in cases:
   if case['split']=='val':write(a.out/('ready_'+case['id']+'.json'),{'case':case['id'],'status':'complete','origin':'fullSHAverifiedpilot','protocol_sha256':protocol_sha})
  live(phase='load_frozen_foundation');patcher=comfy.sd.load_diffusion_model(str(model_path),model_options={'dtype':torch.bfloat16});model=patcher.model.diffusion_model.eval().requires_grad_(False).cuda();assert len(model.blocks)==50 and patcher.model.latent_format.scale_factor==1
  frozen=[(n,v,v._version) for n,v in model.named_parameters()]
  for case in cases:
   if case['split']=='val':continue
   sample(case,'unadapted');decode(case,'unadapted')
   sample(case,'sampling_dc');decode(case,'sampling_dc');decode(case,'vae_only')
   write(a.out/('ready_'+case['id']+'.json'),{'case':case['id'],'status':'complete','origin':'new_fixed_protocol','protocol_sha256':protocol_sha});live(phase='case_complete',case=case['id'])
  live(status='complete',phase='sampling_and_decode_complete',seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),**freeze_audit())
 except Exception as e:live(status='failed',exception=repr(e));raise
 finally:print(json.dumps(state),flush=True)

if __name__=='__main__':main()
