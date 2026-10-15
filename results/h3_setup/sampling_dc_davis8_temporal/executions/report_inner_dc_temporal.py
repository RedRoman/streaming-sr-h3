"""Report the preregistered LR-flow residual audit without new inference."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/h3_setup/sampling_dc_davis8_temporal');a=p.parse_args();o=a.out;d=json.loads((o/'aggregate.json').read_text());state=json.loads((o/'LIVE_STATE.json').read_text());protocol=json.loads((o/'protocol.json').read_text());assert state['status']=='complete' and len(d['rows'])==96
 assert sha(ROOT/'experiments/h3_backbone/evaluate_inner_dc_temporal.py')==protocol['metric_source_sha256'] and sha(ROOT/'experiments/real_video_evaluation/evaluate.py')==protocol['root_warp_source_sha256'] and sha(ROOT/'src/pixel_consistency.py')==protocol['pixel_operator_sha256']
 original=ROOT/'results/h3_setup/sampling_dc_davis8';assert sha(original/'output_png_sha256.json')==protocol['parent_PNG_manifest_sha256'] and sha(original/'input_manifest.json')==protocol['input_manifest_sha256']
 flows=json.loads((o/'flow_manifest.json').read_text());assert sha(o/'flow_manifest.json')==protocol['flow_manifest_sha256']
 for r in flows['cases']:
  for key,hashkey in [('flow','flow_sha256'),('provenance','provenance_sha256')]:
   remote=Path(r[key]);local=ROOT/str(remote).split('/streaming_sr_20261015/')[-1];assert local.exists() and sha(local)==r[hashkey],local
 pairs=json.loads((o/'per_pair.json').read_text());assert len(pairs)==2016 and all(r['frame'] in range(1,22) for r in pairs)
 metrics=json.loads((original/'evaluation/aggregate.json').read_text())['rows'];joined=[]
 for r in d['rows']:
  base=next(x for x in metrics if x['case']==r['case'] and x['method']==r['method']);joined.append({**r,'PSNR':base['mean_frame_psnr_db'],'LPIPS':base['mean_lpips_alex_v01']})
 assert all(len({r['temporal_mask_coverage'] for r in d['rows'] if r['case']==c})==1 for c in {r['case'] for r in d['rows']})
 source=[]
 for case in sorted({r['case'] for r in d['rows']}):
  rows={r['method']:r for r in joined if r['case']==case};r=rows['sampling_dc'];x={k:r[k] for k in ['case','sequence','split','resolution','temporal_mask_coverage','valid_pairs','total_pairs']}
  for suffix in ['','_dc25']:
   current=rows['sampling_dc'+suffix];x['innerDC'+suffix+'_temporal_RMSE']=current['temporal_error_excess_rmse']
   for ref in ['unadapted','vae_only']:
    base=rows[ref+suffix];x['delta_temporal_vs_'+ref+suffix]=current['temporal_error_excess_rmse']-base['temporal_error_excess_rmse'];x['delta_LPIPS_vs_'+ref+suffix]=current['LPIPS']-base['LPIPS']
  source.append(x)
 with (o/'source_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
 write(o/'sourcewise_metrics.json',{'source_rows':source,'joined_case_metrics':joined,'aggregate':d['aggregate']})
 table=['|Context|Method|Sources|Temporal excess RMSE ↓|Mask coverage|Valid pairs|','|---|---|---:|---:|---:|---:|']
 for r in d['aggregate']:
  if r['split']=='all':table.append(f"|{r['resolution']}|{r['method']}|8|{r['temporal_error_excess_rmse']:.7f}|{r['temporal_mask_coverage']:.4%}|{r['valid_pairs']}/{r['total_pairs']}|")
 details=[]
 for res in ['256','native480']:
  sub=[r for r in source if r['resolution']==res]
  for suffix in ['','_dc25']:
   for base in ['unadapted','vae_only']:
    key='delta_temporal_vs_'+base+suffix;delta=statistics.mean(r[key] for r in sub);wins=sum(r[key]<0 for r in sub);details.append(f"{res}, {'final DC25' if suffix else 'raw'}, innerDC versus {base}: mean temporal-RMSE change {delta:+.7f}, decreases on {wins}/8 sources.")
 text='''# Fixed LR-flow temporal residual audit\n\nAll 8 existing DAVIS sources ×2 FOV contexts, 22 frames each. Base, matched VAE-only and sampling-inner DC are evaluated at raw output and after identical final DC25. No DiT, decoder, training, parameter search, or GPU work. The original 8-source archive stays unchanged.\n\nFor every pair t=1..21, use the root evaluate.py warp and confidence upsampling verbatim: bilinear LR flow scaled to output size, zero padding, align_corners=False; confidence ≥0.5. The score is RMSE of (output_t−warp(output_t−1)) − (HR_t−warp(HR_t−1)) within that mask. All inputs and outputs are RGB float32 in [0,1]. DC25 internally uses float64 and returns float32, with no extra PNG rounding after projection.\n\nFlow and confidence use observed LR only. HR appears only in evaluation. The 15 existing root caches are reused by SHA; bmx native480 was copied from the existing local cache, and only missing soapbox native480 was computed using unchanged cache_motion.py and fixed DIS parameters (direct interval17, photo sigma0.05, cycle sigma1). Every flow provenance LR prefix is SHA-matched to the exact case. Pair RMSE is averaged within each source, then equally across the 8 sources per context. Source/split/common-trainval6 aggregates and all 2016 pair rows are retained. Empty-mask handling and valid pair counts are explicit.\n\nThis is flow-relative error after subtracting true motion/appearance residual. It is not a direct flicker score, and flow confidence restricts its coverage. Two spatial contexts are not two independent sources. Project-test was used earlier and is not a sealed holdout; sealed DIV2K remains untouched.\n\n'''+ '\n'.join(table)+'\n\n'+'\n\n'.join(details)+f"\n\nCPU process wall time {d['seconds']:.3f}s, including initial flow setup. The metric does not use LPIPS; the source table joins previously frozen LPIPS values only for comparison. Full metric source and flow hashes are in protocol.json and flow_manifest.json.\n\nInitial preflight attempts found one existing flow only local and the canonical cache helper absent remotely; these files were copied unchanged and the failed logs retained. No metric evaluation ran before the successful protocol freeze.\n"
 (o/'REPORT.md').write_text(text)
 ex=o/'executions';ex.mkdir(exist_ok=True)
 for name,path in [('evaluate_inner_dc_temporal.py',ROOT/'experiments/h3_backbone/evaluate_inner_dc_temporal.py'),('launch_inner_dc_temporal.py',ROOT/'experiments/h3_backbone/launch_inner_dc_temporal.py'),('report_inner_dc_temporal.py',Path(__file__)),('root_evaluate.py',ROOT/'experiments/real_video_evaluation/evaluate.py'),('cache_motion.py',ROOT/'experiments/real_video_evaluation/cache_motion.py'),('video_correspondence.py',ROOT/'src/video_correspondence.py')]: (ex/name).write_bytes(path.read_bytes())
 files={str(q.relative_to(o)):{'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(o.rglob('*')) if q.is_file() and q.name!='full_file_sha256.json'};write(o/'full_file_sha256.json',{'status':'complete','files':files,'file_count':len(files),'bytes':sum(x['bytes'] for x in files.values())})
 print(json.dumps({'status':'complete','cases':16,'sources':8,'metric_rows':96,'pair_rows':2016,'files':len(files),'aggregate':[r for r in d['aggregate'] if r['split']=='all'],'delta_summary':details}),flush=True)
if __name__=='__main__':main()
