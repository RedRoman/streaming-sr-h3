"""Controlled legacy/v2 diagnostic; neither selects a checkpoint nor trains."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[2]
old=root/'results/h3_setup/native_sr_pilot400'
new=root/'results/h3_setup/native_sr_pilot400_native_io_v2'
read=lambda p:json.loads(p.read_text())
legacy=read(old/'protocol.json');fixed=read(new/'protocol.json')
keys=['seed','updates_each','controls','lr','optimizer','training_sigma','train_cases',
      'validation_cases','validation_sigmas','validation_schedule','readout',
      'context_sha256','model_sha256','train_manifest_sha256','val_manifest_sha256','schedule']
assert all(legacy[k]==fixed[k] for k in keys)
a=read(old/'evaluation/aggregate.json');b=read(new/'evaluation/aggregate.json')
assert a['status']==b['status']=='complete'
index={(r['resolution'],r['method']):r for r in a['aggregate']}
rows=[]
for row in b['aggregate']:
    before=index[row['resolution'],row['method']]
    rows.append({'resolution':row['resolution'],'method':row['method'],
                 'legacy_fp32_io':{k:before[k] for k in ['mean_frame_psnr_db','mean_ssim_rgb','mean_lpips_alex_v01']},
                 'native_io_v2':{k:row[k] for k in ['mean_frame_psnr_db','mean_ssim_rgb','mean_lpips_alex_v01']},
                 'delta_native_v2_minus_legacy':{k:row[k]-before[k] for k in ['mean_frame_psnr_db','mean_ssim_rgb','mean_lpips_alex_v01']}})
baseline_pixels={}
for case in fixed['validation_cases']:
    for method in ['vae_only','unadapted']:
        path=Path('readouts')/method/case/'decode_result.json'
        baseline_pixels[f'{method}/{case}']=read(old/path)['output_png_sha256']==read(new/path)['output_png_sha256']
result={'status':'complete','scope':'Controlled IO correction diagnostic on two held-out val sources; final400fixed; no native-capacity/test/memory claim from legacy',
        'same_protocol_keys':keys,'same_baseline_PNG_SHA':baseline_pixels,'rows':rows,
        'protocol_sha256':{v:hashlib.sha256((p/'protocol.json').read_bytes()).hexdigest() for v,p in [('legacy_fp32_io',old),('native_io_v2',new)]}}
(new/'legacy_io_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
table=['# 相同预算：流精度修复诊断','',
       '旧版FP32 raw IO与新版官方BF16 IO，均从相同初始化固定各400次。仅两源val，不选择checkpoint或超参数；差值不能作为记忆方法收益。','',
       '| 分辨率 | 控制 | v1 PSNR | v2 PSNR | ΔPSNR | v1 LPIPS | v2 LPIPS | ΔLPIPS |',
       '|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    x,y,d=r['legacy_fp32_io'],r['native_io_v2'],r['delta_native_v2_minus_legacy']
    table.append(f"| {r['resolution']} | {r['method']} | {x['mean_frame_psnr_db']:.3f} | {y['mean_frame_psnr_db']:.3f} | {d['mean_frame_psnr_db']:+.3f} | {x['mean_lpips_alex_v01']:.5f} | {y['mean_lpips_alex_v01']:.5f} | {d['mean_lpips_alex_v01']:+.5f} |")
table+=['','基线PNG逐SHA是否相同见legacy_io_comparison.json；完整raw与DC25各列均保留，不能只选择较好的后处理视图。']
(new/'LEGACY_IO_COMPARISON.md').write_text('\n'.join(table)+'\n')
print(json.dumps({'status':'complete','baseline_PNG_identical':baseline_pixels,'rows':rows}))
