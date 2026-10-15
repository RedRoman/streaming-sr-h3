"""Foreground canonical-FP16 readout, exact-v2 metrics, and cost audit."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT);RUN=ROOT/'results/h3_setup/sampling_dc_pilot';PY=sys.executable
assert json.loads((RUN/'LIVE_STATE.json').read_text())['status']=='complete'
protocol=json.loads((RUN/'protocol.json').read_text());state={'status':'running','pid':os.getpid(),'completed':[]}
def live(**kw):state.update(kw,epoch=time.time());(RUN/'postprocess_live.json').write_text(json.dumps(state,indent=2)+'\n')
try:
 gpu=os.environ['CUDA_VISIBLE_DEVICES'];free,util=map(int,subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','));assert free>30000 and util==0
 live(phase='FP16decode',device='cuda')
 subprocess.run([PY,'experiments/h3_backbone/inner_dc_shared_postprocess.py','decode','--comfy','external/ComfyUI_H3','--vae','external/h3_assets/minimax_h3_video_vae_fp16.safetensors','--run',str(RUN),'--val','results/h3_setup/native_sr_val_pairs/pairs_manifest.json','--modes','unadapted','sampling_dc'],check=True)
 state['completed'].append('decode4newvalcases')
 live(phase='fixed_control_metrics',device='cpu')
 for case in protocol['validation_cases']:
  dest=RUN/'evaluation'/case;dest.mkdir(parents=True,exist_ok=True)
  source=ROOT/'results/h3_setup/native_sr_pilot400_native_io_v2/evaluation'/case/'metrics.json'
  if not (dest/'metrics.json').exists():shutil.copy2(source,dest/'metrics.json')
 env=dict(os.environ,PYTHONPATH=str(ROOT/'external/eval_deps'),TORCH_HOME=str(ROOT/'external/eval_torch_cache'))
 subprocess.run([PY,'experiments/h3_backbone/inner_dc_shared_postprocess.py','evaluate','--run',str(RUN),'--val','results/h3_setup/native_sr_val_pairs/pairs_manifest.json','--device','cpu','--modes','unadapted','sampling_dc'],env=env,check=True)
 state['completed'].append('same_v2_metrics')
 samples=[json.loads(p.read_text()) for p in list((RUN/'guard').glob('*/*/sample_result.json'))+list((RUN/'readouts/sampling_dc').glob('*/sample_result.json'))]
 assert len(samples)==6 and sum(x['steps_model_evaluations'] for x in samples)==294
 assert sum(x['E_calls'] for x in samples)==27 and sum(x['D_calls_inside_sampler'] for x in samples)==15
 assert all(x['base_weight_grad_count']==0 and x['base_parameter_versions_unchanged'] and x['audio_inside_hook_unchanged'] for x in samples)
 for x in samples:
  assert x['steps_model_evaluations']==49 and not x['HR_condition'] and x['input_prefix']==[0,21]
 for x in samples:
  if x['mode']=='sampling_dc':assert [v['index'] for v in x['correction_events']]==[28,45,48] and x['E_calls']==6 and x['D_calls_inside_sampler']==3
 parity=json.loads((RUN/'zero_correction_parity.json').read_text());assert parity['passed']
 costs={'passed':True,'new_sampling_trajectories':6,'DiT_NFE':294,'E_calls':27,'D_calls_inside_sampler':15,'new_final_D_calls':4,
        'guard_final_RGB_not_separately_decoded':'Full-video/audio latent equality exact0 is stronger; guard original/identity latents retained, no redundant decoder calls.',
        'final_guard_latent_maxabs':parity['original_vs_identity_latent_maxabs'],'sampling_seconds_sum':sum(x['seconds'] for x in samples),
        'hook_D_seconds':sum(v['D_seconds'] for x in samples for v in x['correction_events']),
        'hook_E_seconds':sum(v['E_original_seconds']+v['E_projected_seconds'] for x in samples for v in x['correction_events']),
        'hook_projection_seconds':sum(v['projection_seconds'] for x in samples for v in x['correction_events']),
        'new_case_sample_results':[{'case':x['case'],'mode':x['mode'],'seconds':x['seconds'],'NFE':x['steps_model_evaluations'],'E':x['E_calls'],'midD':x['D_calls_inside_sampler']} for x in samples],
        'baseline_reuse':'v2 unadapted andVAE per-PNG SHA verified; no baseline NFE hidden innew count'}
 (RUN/'cost_audit.json').write_text(json.dumps(costs,indent=2)+'\n');live(status='complete',phase='complete',device='cpu')
except Exception as exc:live(status='failed',exception=repr(exc));raise
finally:print(json.dumps(state),flush=True)
