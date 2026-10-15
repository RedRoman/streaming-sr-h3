"""Verify the frozen sampling DC archive, source-wise metrics and real costs."""
import csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'results/h3_setup/sampling_dc_pilot';BASE=ROOT/'results/h3_setup/native_sr_pilot400_native_io_v2'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(RUN/'protocol.json');c=read(RUN/'cost_audit.json');m=read(RUN/'evaluation/aggregate.json');z=read(RUN/'zero_correction_parity.json')
assert c['passed'] and z['passed'] and m['status']=='complete'
checked=0;reused={}
for label in ['unadapted','vae_only','sampling_dc']:
 for case in p['validation_cases']:
  folder=RUN/'readouts'/label/case;d=read(folder/'decode_result.json');assert len(d['output_png_sha256'])==22
  for name,digest in d['output_png_sha256'].items():assert sha(folder/name)==digest;checked+=1
  if label!='sampling_dc':
   reused[label+'/'+case]=d['output_png_sha256']==read(BASE/'readouts'/label/case/'decode_result.json')['output_png_sha256'];assert reused[label+'/'+case]
assert checked==264
rows=[]
for case in p['validation_cases']:
 d=read(RUN/'evaluation'/case/'metrics.json')
 for row in d['rows']:
  assert row['split']=='val';rows.append(row)
assert len(rows)==32
sourcewise={'scope':'Fixedtwo independentval sources x2contexts; no test/hyperparameterselection','rows':rows}
(RUN/'sourcewise_metrics.json').write_text(json.dumps(sourcewise,indent=2)+'\n')
source_summary=[]
for case in p['validation_cases']:
 index={r['method']:r for r in rows if r['case']==case};new=index['sampling_dc_dc25'];base=index['unadapted_dc25'];vae=index['vae_only_dc25']
 source_summary.append({'case':case,'samplingDC_DC_PSNR':new['mean_frame_psnr_db'],'samplingDC_DC_LPIPS':new['mean_lpips_alex_v01'],
  'delta_PSNR_vs_baseDC':new['mean_frame_psnr_db']-base['mean_frame_psnr_db'],'delta_LPIPS_vs_baseDC':new['mean_lpips_alex_v01']-base['mean_lpips_alex_v01'],
  'delta_PSNR_vs_VAEDC':new['mean_frame_psnr_db']-vae['mean_frame_psnr_db'],'delta_LPIPS_vs_VAEDC':new['mean_lpips_alex_v01']-vae['mean_lpips_alex_v01']})
with (RUN/'source_summary.csv').open('w') as file:
 writer=csv.DictWriter(file,fieldnames=list(source_summary[0]));writer.writeheader();writer.writerows(source_summary)
names={'bicubic':'Bicubic','vae_only':'H3 VAE','unadapted':'H3 BF16 base','sampling_dc':'H3 内置投影'}
text=['# 冻结H3：采样内观测一致性强控制','',
 '本实验是DDNM/FlowDPS启发的工程强基线，不是原创记忆方法、逐行论文复现或精确后验采样。全部预声明协议保持：未训练H3基础主干、相同官方LR guide/真实中性text/seed101/正常50点49Euler；仅在index28/45/48（σ.9/.516129/.2）修正post-CFG clean video estimate，强度1。没有扫描sigma/强度或使用test挑结果。','',
 '修正为 z0 + E(P25(D(z0))) − E(D(z0))；原Euler、audio clean estimate和guide保持。内部用float RGB、canonical FP16 H3VAE原生22帧、scale1。P25是既有AA bicubic+RGB box的固定Dykstra25，不保证与数据的Pillow量化完全等价；E/D不是互逆，像素投影结果不能冒充最终真实D输出。完整raw与最终DC25同时报告。','',
 'train bear_256原版/identity-hook完整49步video与audio latent最大差均0，identity delta严格0；实际P25在可行RGB固定点误差<1e−12。4个val case全49NFE、guide逐检查max0、hook未改audio；530基础参数张量版本不变、basegrad0。非零video修正可能经之后联合DiT影响audio轨迹，不宣称最终audio与base相同。','',
 '| 空间上下文 | 控制 | PSNR↑ | SSIM↑ | LPIPS↓ | LR RMSE↓ |','|---|---|---:|---:|---:|---:|']
for resolution in ['256','native480']:
 for method in ['bicubic','vae_only','unadapted','sampling_dc']:
  for dc in [False,True]:
   name=method+('_dc25' if dc else '');row=next(x for x in m['aggregate'] if x['resolution']==resolution and x['method']==name)
   text.append(f"| {resolution} | {names[method]}{' + final DC25' if dc else ' raw'} | {row['mean_frame_psnr_db']:.3f} | {row['mean_ssim_rgb']:.4f} | {row['mean_lpips_alex_v01']:.5f} | {row['mean_lr_float_operator_rmse']:.6g} |")
text+=['',
 f"新运行共{c['new_sampling_trajectories']}条轨迹/{c['DiT_NFE']}次DiT、{c['E_calls']}次E、{c['D_calls_inside_sampler']}次采样内D、{c['new_final_D_calls']}次末端D。2条train轨迹只作严格latent/audio parity；不额外重复末端D。其余4条是val方法控制。v2的8个VAE/base控制case共176PNG逐SHA复用，额外NFE没有隐藏。成本含CPU投影和传输，详见cost_audit.json及各sample_result逐hook记录。",
 f"6条采样wall time合计{c['sampling_seconds_sum']:.3f}s，其中hook D {c['hook_D_seconds']:.3f}s、hook E {c['hook_E_seconds']:.3f}s、CPU projection {c['hook_projection_seconds']:.3f}s；cold load/总进程时间另见LIVE_STATE.json，不能相加重复或当实时吞吐。",
 '',
 '全部12个输出case/264PNG本地逐SHA通过；来源拆分在sourcewise_metrics.json与evaluation/*/metrics.json。仅2个独立val源×2空间上下文，单seed/22帧；两空间上下文FOV不同，不是纯分辨率消融。实际观测end21、名义块内等待上界21帧，没有实测采集/稳态流式延迟。',
 '',
 '经典依据：[DDNM](https://arxiv.org/abs/2212.00490)、[作者实现](https://github.com/wyhuai/DDNM)、[FlowDPS](https://arxiv.org/abs/2503.08136)。采样内projection/residual re-encoding是当前H3上的工程移植；它不更新历史memory，也不将生成结果写成新证据。']
(RUN/'REPORT.md').write_text('\n'.join(text)+'\n')
with (RUN/'REPORT.md').open('a') as file:
 file.write('\n两个独立source的native480均改善：\n\n')
 for row in source_summary:
  if row['case'].endswith('native480'):
   file.write('- %s：%.3fdB / LPIPS %.5f；相对同DC base ΔPSNR%+.3f、ΔLPIPS%+.5f，相对同DC VAE %+.3f/%+.5f。\n' % (row['case'],row['samplingDC_DC_PSNR'],row['samplingDC_DC_LPIPS'],row['delta_PSNR_vs_baseDC'],row['delta_LPIPS_vs_baseDC'],row['delta_PSNR_vs_VAEDC'],row['delta_LPIPS_vs_VAEDC']))
 file.write('\n256的PSNR仍低于VAE，且只有2source/单seed val，不能外推test或记忆收益。native480采样约70.3–70.6s，相对v2 base约30.6s约2.3倍wall time；同49NFE不代表同算力，因为额外VAE/CPU投影成本已实际发生。\n')
integrity={'passed':True,'checked_output_PNG':checked,'baseline_reuse_all_SHA_equal':reused,'protocol_sha256':sha(RUN/'protocol.json'),'metrics_sha256':sha(RUN/'evaluation/aggregate.json'),'cost_audit_sha256':sha(RUN/'cost_audit.json')}
(RUN/'archive_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
print(json.dumps({'archive_integrity':integrity,'costs':c,'aggregate':m['aggregate']}))
