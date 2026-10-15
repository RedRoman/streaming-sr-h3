"""Archive and report the frozen eight-source evaluation; no GPU work."""
import argparse,csv,hashlib,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
KEYS=['mean_frame_psnr_db','mean_ssim_rgb','mean_lpips_alex_v01','mean_lr_float_operator_rmse']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def mean(rows):return {k:statistics.mean(r[k] for r in rows) for k in KEYS}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'results/h3_setup/sampling_dc_davis8');a=p.parse_args();o=a.run
 protocol=json.loads((o/'protocol.json').read_text());cases=json.loads((o/'input_manifest.json').read_text())['cases'];guides=json.loads((o/'guides_frozen.json').read_text());live=json.loads((o/'LIVE_STATE.json').read_text());evaluation=json.loads((o/'evaluation/aggregate.json').read_text())
 assert live['status']=='complete' and evaluation['status']=='complete' and len(evaluation['rows'])==128
 assert sha(o/'input_manifest.json')==protocol['input_manifest_sha256'] and sha(o/'source_manifest.json')==protocol['source_manifest_sha256']
 for name,expected in json.loads((o/'source_manifest.json').read_text()).items():assert sha(ROOT/'experiments/h3_backbone'/name)==expected
 rows=evaluation['rows'];assert len({r['sequence'] for r in rows})==8 and len({r['case'] for r in rows})==16
 pngs={};samples=[];decodes=[];reused_files=json.loads((o/'reuse_sha256.json').read_text())['files'];reuse_output_checks=0
 for case in cases:
  assert (o/('ready_'+case['id']+'.json')).exists()
  for mode in ['unadapted','sampling_dc','vae_only']:
   folder=o/'readouts'/mode/case['id'];d=json.loads((folder/'decode_result.json').read_text());assert d['status']=='complete' and d['vae_sha256']==protocol['vae_sha256_previously_verified']
   for name,expected in d['output_png_sha256'].items():
    q=folder/name;assert sha(q)==expected;qrel=str(q.relative_to(o));pngs[qrel]=expected
    if case['split']=='val':assert reused_files[qrel]['sha256']==expected;reuse_output_checks+=1
   decodes.append({'case':case['id'],'split':case['split'],'mode':mode,'seconds':d['seconds'],'new':case['split']!='val'})
   if mode!='vae_only':
    q=json.loads((folder/'sample_result.json').read_text());assert q['steps_model_evaluations']==49 and q['seed']==101 and q['cfg']==1 and not q['HR_condition'] and q['observed_lr_sha256']==case['observed_lr_sha256']
    samples.append({'case':case['id'],'split':case['split'],'resolution':case['resolution'],'mode':mode,'NFE':49,'E':q.get('E_calls',0),'midD':q.get('D_calls_inside_sampler',0),'seconds':q['seconds'],'events':q.get('correction_events',[]),'new':case['split']!='val'})
 assert len(pngs)==1056 and reuse_output_checks==264 and len(samples)==32 and len(decodes)==48
 new=[s for s in samples if s['new']];assert len(new)==24 and sum(s['NFE'] for s in new)==1176 and sum(s['E'] for s in new)==72 and sum(s['midD'] for s in new)==36
 assert len([d for d in decodes if d['new']])==36 and sum(q['new_E_calls'] for q in guides['records'])==4
 # Old val images are SHA-identical. If path-sensitivecachemiss caused metricrecompute,
 # retain numeric equality proof rather than claiming its CPU work was reused.
 old=ROOT/'results/h3_setup/sampling_dc_pilot';numeric_keys=KEYS+['pooled_psnr_db','max_lr_float_operator_rmse','per_frame_psnr_db','per_frame_ssim','per_frame_lpips'];parity=[]
 for case in [c for c in cases if c['split']=='val']:
  prior={r['method']:r for r in json.loads((old/'evaluation'/case['id']/'metrics.json').read_text())['rows']}
  current=[r for r in rows if r['case']==case['id']]
  def difference(x,y):
   if isinstance(x,list):return max(abs(a-b) for a,b in zip(x,y))
   return abs(x-y)
  delta=max(difference(r[k],prior[r['method']][k]) for r in current for k in numeric_keys);assert delta==0,(case['id'],delta)
  parity.append({'case':case['id'],'numeric_metrics_maxabs':delta,'output_PNG_SHA_equal':True})
 write(o/'reused_val_numeric_parity.json',{'passed':True,'rows':parity,'note':'Allvalnumericmetricsexactlyequalprior. TORCH_HOMErelative/absolutepathcachekeymissrecomputedmetricsCPU; valGPUoutputsnotrerun.'})
 aggregation=[]
 for res in ['256','native480']:
  for split in ['all','train','val','test','trainval6']:
   for method in sorted({r['method'] for r in rows}):
    subset=[r for r in rows if r['resolution']==res and r['method']==method and (split=='all' or r['split']==split or (split=='trainval6' and r['split'] in ['train','val']))]
    assert len(subset)==({'all':8,'train':4,'val':2,'test':2,'trainval6':6}[split]);aggregation.append({'resolution':res,'split':split,'source_count':len(subset),'method':method,**mean(subset)})
 bysource=[]
 for case in cases:
  subset={r['method']:r for r in rows if r['case']==case['id']};current=subset['sampling_dc_dc25']
  item={'case':case['id'],'sequence':case['sequence'],'split':case['display_split'],'resolution':case['resolution'],'sampling_dc_dc25_psnr':current['mean_frame_psnr_db'],'sampling_dc_dc25_lpips':current['mean_lpips_alex_v01']}
  for base in ['unadapted_dc25','vae_only_dc25','bicubic_dc25']:
   item['dPSNR_vs_'+base]=current['mean_frame_psnr_db']-subset[base]['mean_frame_psnr_db'];item['dLPIPS_vs_'+base]=current['mean_lpips_alex_v01']-subset[base]['mean_lpips_alex_v01']
  bysource.append(item)
 write(o/'sourcewise_metrics.json',{'scope':protocol['scope'],'source_equal_weight_aggregate':aggregation,'case_deltas':bysource,'all_rows':rows})
 with (o/'source_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(bysource[0]));w.writeheader();w.writerows(bysource)
 for resolution in ['256','native480']:
  subset=[s for s in samples if s['resolution']==resolution]
  for mode in ['unadapted','sampling_dc']:
   ss=[s for s in subset if s['mode']==mode];assert len(ss)==8
 # Costs distinguish newextensionexecution and finaldataset materialized acrossoldruns.
 cost={'status':'passed','extension_new':{'trajectories':24,'DiT_NFE':1176,'guide_E':4,'inner_E':72,'inner_D':36,'final_D':36,'sample_seconds_sum':sum(s['seconds'] for s in new),'final_D_seconds_sum_includingPNG':sum(d['seconds'] for d in decodes if d['new']),'process_wall_seconds_includingloadGuidesReuseAndPNG':live['seconds'],'peak_allocated_bytes':live['peak_allocated_bytes'],'peak_reserved_bytes':live['peak_reserved_bytes']},
  'materialized_16case_dataset':{'sources':8,'contexts_per_source':2,'trajectories':32,'DiT_NFE':1568,'inner_E':96,'inner_D':48,'final_D':48,'guidance_E':16,'guidance_E_note':'12reusedtrain/valguides+4newtestguides; notHRtargetencoding'},
  'prior_parity_guard_not_in_dataset_cost':{'DiT_NFE':98,'E':3,'D':3,'source':'sampling_dc_pilot/guard','zero_delta_parity_sha256':sha(old/'zero_correction_parity.json')},'CPU_evaluation_process_seconds':evaluation['seconds'],'sample_rows':samples,'decode_rows':decodes,
  'time_note':'Offline fullblockwalltime,notlivearrival latency. Boundedinputprefix0..21 at24fps givesnominal21framewait=.875s beforecompute. MidD/E/projection cost is real additional work despiteequal49DiT NFE. finalD times includePNGwriting; processwallincludesinitialization. CPUevaluatoroverlapsGPU,soitswallincludescasewaiting; do notaddoverlappedwalltimes.'}
 cost['mean_sample_seconds_by_context_mode']=[{'resolution':res,'mode':mode,'source_count':8,'seconds':statistics.mean(s['seconds'] for s in samples if s['resolution']==res and s['mode']==mode)} for res in ['256','native480'] for mode in ['unadapted','sampling_dc']]
 for key,field in [('inner_D_seconds','D_seconds'),('inner_E_original_seconds','E_original_seconds'),('inner_E_projected_seconds','E_projected_seconds'),('inner_projection_seconds','projection_seconds')]:cost['extension_new'][key]=sum(ev[field] for s in new for ev in s['events'])
 write(o/'cost_audit.json',cost);write(o/'output_png_sha256.json',{'count':len(pngs),'reused_val_PNG_count':264,'new_PNG_count':792,'files':pngs})
 # Keep preregistration and frozen numerical sourcecopies in the archive.
 ex=o/'executions';ex.mkdir(exist_ok=True)
 for name in json.loads((o/'source_manifest.json').read_text()):(ex/name).write_bytes((ROOT/'experiments/h3_backbone'/name).read_bytes())
 (ex/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
 table=['|Context|Method|Sources|PSNR dB|LPIPS Alex|SSIM|','|---|---|---:|---:|---:|---:|']
 for r in aggregation:
  if r['split']=='all':table.append(f"|{r['resolution']}|{r['method']}|8|{r['mean_frame_psnr_db']:.4f}|{r['mean_lpips_alex_v01']:.5f}|{r['mean_ssim_rgb']:.4f}|")
 source_table=['|Source|Split|Context|ΔPSNR vs base DC25|ΔLPIPS vs base DC25|ΔPSNR vs VAE DC25|ΔLPIPS vs VAE DC25|','|---|---|---|---:|---:|---:|---:|']
 for r in bysource:source_table.append(f"|{r['sequence']}|{r['split']}|{r['resolution']}|{r['dPSNR_vs_unadapted_dc25']:+.4f}|{r['dLPIPS_vs_unadapted_dc25']:+.5f}|{r['dPSNR_vs_vae_only_dc25']:+.4f}|{r['dLPIPS_vs_vae_only_dc25']:+.5f}|")
 wins=[]
 for res in ['256','native480']:
  subset=[r for r in bysource if r['resolution']==res]
  wins.append(f"{res}: vs base DC25, PSNR improves {sum(r['dPSNR_vs_unadapted_dc25']>0 for r in subset)}/8, LPIPS improves {sum(r['dLPIPS_vs_unadapted_dc25']<0 for r in subset)}/8; vs VAE DC25, PSNR improves {sum(r['dPSNR_vs_vae_only_dc25']>0 for r in subset)}/8, LPIPS improves {sum(r['dLPIPS_vs_vae_only_dc25']<0 for r in subset)}/8.")
 report='''# Frozen H3 sampling-inner DC: 8 DAVIS sources\n\nComplete: 16 cases from 8 independent existing DAVIS sources, each in 2 spatial contexts; 1056 output PNG hashes verified. Four train sources (bear, bus, car-turn, flamingo), two val sources (cows, dance-twirl), and two previously used project-test sources (bmx-trees, soapbox). Project-test is not a sealed holdout. Sealed DIV2K was never accessed. No training or result-dependent parameter selection.\n\nFrozen BF16 MiniMax-H3 base, real neutral context from the NVFP4-AWQ text encoder, canonical FP16 VAE, seed 101, CFG 1, and the official 50 sigma points / 49 Euler steps. At the three preregistered indices 28, 45, and 48, apply the unit residual to post-CFG clean z0: E(P25(D(z0))) − E(D(z0)). Euler, audio carry, and guide remain unchanged, and the base has no gradients. This is a DDNM/FlowDPS-inspired engineering control, not an original memory method or exact paper reproduction.\n\nThe four val cases reuse SHA-identical prior outputs (264 PNGs), with exactly equal original numeric metrics. CPU metrics were recomputed because absolute/relative TORCH_HOME strings changed the cache key; val GPU inference was not repeated. The new 12 base and 12 inner-DC trajectories and 4 LR-only project-test guides ran after protocol, source, and input hashes were frozen. Matched VAE-only uses the same official guide latents.\n\nThe 256 crop [296,112,256,256] and native 832×480 crop [8,0,832,480] cover different fields of view. This is not a pure resolution ablation. Frame means are averaged per case, then equally across the 8 sources within each context; 16 cases do not count as 16 independent observations. Split-specific means and the shared 6-source train+val subset for external-baseline comparison are in sourcewise_metrics.json. Raw precedes final DC25; the _dc25 suffix applies the same final projection to each control. sampling_dc_dc25 means inner DC plus final DC25.\n\n'''+ '\n'.join(table)+'\n\n'+'\n'.join(wins)+'\n\n'+'\n'.join(source_table)+'\n\n'+f"New extension: 1176 DiT NFE + 76 E (4 guide / 72 inner) + 72 D (36 inner / 36 final). The complete 16-case dataset costs 1568 DiT NFE + 112 E (16 guide / 96 inner) + 96 D (48 inner / 48 final), excluding the prior 98-NFE parity guard. Extension wall time {live['seconds']:.2f}s, summed sampling {cost['extension_new']['sample_seconds_sum']:.2f}s, CPU metric process {evaluation['seconds']:.2f}s overlaps the GPU and includes waiting. Peak allocated memory {live['peak_allocated_bytes']/1e9:.3f}GB. Exact per-case, projection, E, and D costs are in cost_audit.json. Equal 49 DiT calls does not mean equal compute. A nominal 22-frame block needs a 21-frame wait (.875s at 24fps) plus compute; these are offline timings, not measured live ingestion.\n\n"+f"Protocol SHA `{sha(o/'protocol.json')}`. Input SHA `{sha(o/'input_manifest.json')}`. Source manifest and metric-loop equivalence are archived. The first preflight failed before model loading or GPU work because an import shadowed a Path variable. The variable-name fix and failed attempt are retained in preflight_attempt0; parameters and math never changed.\n"
 (o/'REPORT.md').write_text(report)
 files={str(q.relative_to(o)):{'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(o.rglob('*')) if q.is_file() and q.name!='full_file_sha256.json'}
 write(o/'full_file_sha256.json',{'status':'complete_archive','file_count':len(files),'bytes':sum(v['bytes'] for v in files.values()),'files':files})
 print(json.dumps({'status':'complete','PNG':len(pngs),'files':len(files),'source_count':8,'protocol_sha256':sha(o/'protocol.json'),'aggregate':[r for r in aggregation if r['split']=='all'],'wins':wins}),flush=True)
if __name__=='__main__':main()
