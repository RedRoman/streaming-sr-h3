"""Fixed LR-flow temporal residual audit for the archived 8-source H3 controls."""
import argparse,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from pixel_consistency import BicubicConsistency
spec=importlib.util.spec_from_file_location('root_source_aligned_evaluation',ROOT/'experiments/real_video_evaluation/evaluate.py');shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 q=Path(str(p)+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def load(p):return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2,0,1).float()/255

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'results/h3_setup/sampling_dc_davis8');p.add_argument('--out',type=Path,default=ROOT/'results/h3_setup/sampling_dc_davis8_temporal');a=p.parse_args();os.chdir(ROOT);a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);start=time.perf_counter()
 state={'status':'running','pid':os.getpid(),'device':'CPU','CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'completed':[]}
 def live(**kw):state.update(kw,epoch=time.time());write(a.out/'LIVE_STATE.json',state)
 source=a.run/'input_manifest.json';cases=json.loads(source.read_text())['cases'];assert len(cases)==16
 frozen_png=json.loads((a.run/'output_png_sha256.json').read_text())['files'];flow_records=[]
 try:
  for case in cases:
   prefix=ROOT/'results/video_correspondence'/('native480' if case['resolution']=='native480' else '')/case['split']/case['sequence'];flow=prefix/'lr_correspondence.npz'
   if not flow.exists():
    assert case['id']=='soapbox_native480','Onlydeclaredmissingflowmaybegenerated';prefix=a.out/'flow/native480/test/soapbox';flow=prefix/'lr_correspondence.npz';live(phase='LR_only_missing_flow',case=case['id'])
    command=[sys.executable,'experiments/real_video_evaluation/cache_motion.py','--manifest',str(source),'--case',case['id'],'--out',str(prefix)]
    write(a.out/'missing_flow_command.json',{'command':command,'LR_only':True,'canonical_script_sha256':sha(ROOT/'experiments/real_video_evaluation/cache_motion.py'),'fixed_parameters':{'direct_interval':17,'photo_sigma':.05,'cycle_sigma':1}});subprocess.run(command,check=True)
   provenance=json.loads((prefix/'provenance.json').read_text());assert provenance['lr_sha256'][:22]==case['observed_lr_sha256']
   archive=np.load(flow);assert tuple(archive['pair_confidence'].shape[1:])==(case['shape_hw'][0]//4,case['shape_hw'][1]//4)
   assert archive['pair_confidence'].shape[0]>=22
   flow_records.append({'case':case['id'],'flow':str(flow),'flow_sha256':sha(flow),'provenance':str(prefix/'provenance.json'),'provenance_sha256':sha(prefix/'provenance.json'),'cached_frames':int(archive['pair_confidence'].shape[0]),'use_indices':[0,21],'LR_sha_verified':True,'existing_root_cache':case['id']!='soapbox_native480'})
  write(a.out/'flow_manifest.json',{'cases':flow_records,'HR_used_for_motion':False,'fixed_parameters':{'direct_interval':17,'photo_sigma':.05,'cycle_sigma':1}})
  protocol={'scope':'Frozen8DAVISsources x2FOVcontexts temporalresidual audit; raw+DC25 forbase/VAE/innerDC; no DiT/decoder/training/parameterselection','cases':[c['id'] for c in cases],'parent_protocol_sha256':sha(a.run/'protocol.json'),'input_manifest_sha256':sha(source),'parent_PNG_manifest_sha256':sha(a.run/'output_png_sha256.json'),'flow_manifest_sha256':sha(a.out/'flow_manifest.json'),'metric_source_sha256':sha(__file__),'root_warp_source_sha256':sha(ROOT/'experiments/real_video_evaluation/evaluate.py'),'pixel_operator_sha256':sha(ROOT/'src/pixel_consistency.py'),'correspondence_source_sha256':sha(ROOT/'src/video_correspondence.py'),'confidence_threshold':.5,'pair_indices':list(range(1,22)),'projection_iterations':25,'precision':'PNG/HR/LRloadfloat32; DC25internalfloat64thenreturnsfloat32; officialrootwarp usesfloat32grid; no extraPNGquantizationafterDC','metric':'RMSE[(out_t-warp(out_t-1))-(HR_t-warp(HR_t-1))] overconfidence>=.5; pairRMSEmean→case→equally8sources/context; empty masks excludedandcoveragecountreported','interpretation':'LR-flow residual relative toHR residual, not a direct flicker score. HR only evaluation, never motion/conditioning; project-test alreadyused, notsealedholdout','GPU_used':False,'DiT_NFE':0,'VAE_calls':0}
  write(a.out/'protocol.json',protocol);records=[];pair_records=[]
  for case,flow_record in zip(cases,flow_records):
   live(phase='temporal_metrics',case=case['id']);folder=a.out/case['id'];folder.mkdir(exist_ok=True)
   assert [sha(ROOT/'data/real_video'/q) for q in case['lr_paths']]==case['observed_lr_sha256'];assert [sha(ROOT/'data/real_video'/q) for q in case['hr_paths']]==case['evaluation_hr_sha256']
   low=torch.stack([load(ROOT/'data/real_video'/q) for q in case['lr_paths']]);hr=torch.stack([load(ROOT/'data/real_video'/q) for q in case['hr_paths']]);flow=np.load(flow_record['flow']);op=BicubicConsistency(tuple(hr.shape[-2:]),tuple(low.shape[-2:]));assert sha(flow_record['flow'])==flow_record['flow_sha256']
   references=[]
   for i in range(1,22):
    coords=flow['current_to_previous'][i];confidence=F.interpolate(torch.from_numpy(flow['pair_confidence'][i])[None,None],hr.shape[-2:],mode='bilinear',align_corners=False)[0,0];mask=confidence>=.5;delta=hr[i]-shared.warp(hr[i-1],coords);references.append((coords,mask,delta,float(mask.float().mean())))
   rows=[]
   for mode in ['unadapted','vae_only','sampling_dc']:
    outputfolder=a.run/'readouts'/mode/case['id'];paths=[outputfolder/f'output_{i:02d}.png' for i in range(22)]
    assert all(sha(q)==frozen_png[str(q.relative_to(a.run))] for q in paths)
    raw=torch.stack([load(q) for q in paths])
    for dc in [False,True]:
     begin=time.perf_counter();name=mode+('_dc25' if dc else '');output=op.project(raw,low,25)[0] if dc else raw;assert output.dtype==hr.dtype==torch.float32;pr=[]
     for i,(coords,mask,truth,coverage) in enumerate(references,1):
      predicted=output[i]-shared.warp(output[i-1],coords);error=predicted-truth;value=float(error[:,mask].square().mean().sqrt()) if mask.any() else None
      row={'case':case['id'],'sequence':case['sequence'],'split':case['split'],'resolution':case['resolution'],'method':name,'frame':i,'temporal_error_excess_rmse':value,'temporal_mask_coverage':coverage,'valid_pixels':int(mask.sum()),'total_pixels':mask.numel()};pr.append(row)
     values=[x['temporal_error_excess_rmse'] for x in pr if x['temporal_error_excess_rmse'] is not None];assert values
     summary={'case':case['id'],'sequence':case['sequence'],'split':case['split'],'resolution':case['resolution'],'method':name,'temporal_error_excess_rmse':float(np.mean(values)),'temporal_mask_coverage':float(np.mean([x['temporal_mask_coverage'] for x in pr])),'valid_pairs':len(values),'total_pairs':21,'seconds':time.perf_counter()-begin,'flow_sha256':flow_record['flow_sha256']};rows.append(summary);pair_records.extend(pr);print(json.dumps({'event':'temporal_metric_complete',**summary}),flush=True)
   records.extend(rows);write(folder/'metrics.json',{'rows':rows,'protocol_sha256':sha(a.out/'protocol.json'),'observed_lr_sha256':case['observed_lr_sha256']});state['completed'].append(case['id']);live(phase='case_complete')
  assert len(records)==96 and len(pair_records)==2016
  aggregates=[]
  for resolution in ['256','native480']:
   for split in ['all','train','val','test','trainval6']:
    for method in sorted({r['method'] for r in records}):
     rows=[r for r in records if r['resolution']==resolution and r['method']==method and (split=='all' or r['split']==split or (split=='trainval6' and r['split'] in ['train','val']))];assert len(rows)=={'all':8,'train':4,'val':2,'test':2,'trainval6':6}[split]
     aggregates.append({'resolution':resolution,'split':split,'method':method,'source_count':len(rows),'temporal_error_excess_rmse':float(np.mean([r['temporal_error_excess_rmse'] for r in rows])),'temporal_mask_coverage':float(np.mean([r['temporal_mask_coverage'] for r in rows])),'valid_pairs':sum(r['valid_pairs'] for r in rows),'total_pairs':21*len(rows)})
  write(a.out/'aggregate.json',{'scope':protocol['scope'],'aggregate':aggregates,'rows':records,'seconds':time.perf_counter()-start,'protocol_sha256':sha(a.out/'protocol.json')});write(a.out/'per_pair.json',pair_records);live(status='complete',phase='complete',seconds=time.perf_counter()-start,metric_rows=len(records),pair_rows=len(pair_records));print(json.dumps({'status':'complete','aggregate':[r for r in aggregates if r['split']=='all']}),flush=True)
 except Exception as e:live(status='failed',exception=repr(e));raise
if __name__=='__main__':main()
