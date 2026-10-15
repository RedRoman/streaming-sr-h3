"""Source-paired RGB/region summaries for the neutral native-H3 capacity audit."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);args=p.parse_args();root=Path(args.results);rows=list(csv.DictReader((root/'rgb_metrics.csv').open()));complete=json.loads((root/'complete.json').read_text());sources=sorted({r['source'] for r in rows});assert len(sources)==8
    metrics=['rgb_psnr','union_psnr','old_union_psnr','old_exclusive_psnr','new_only_psnr','overlap_psnr','current_psnr','current_far_rmse','past_near_rmse'];groups=defaultdict(list)
    for row in rows:groups[(row['family'],row['method'],int(row['prefix']))].append(row)
    summaries=[]
    for (family,method,prefix),rr in groups.items():
        d=dict(family=family,method=method,prefix=prefix,n_sources=8,n_orders=2)
        for metric in metrics:d[metric]=float(np.mean([float(r[metric]) for r in rr if r[metric]!=''])) if any(r[metric]!='' for r in rr) else None
        summaries.append(d)
    with (root/'region_summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    pairs=[('H3_decoded_atlas_full','H3_last_fullcode'),('H3_ED_pixel_inverse','H3_last_fullcode'),('H3_ED_pixel_inverse_matched_support','H3_last_fullcode'),('H3_ED_pixel_inverse_matched_support','H3_decoded_atlas_full'),('H3_ED_pixel_atlas_matched_support','H3_decoded_atlas_full'),('H3_decoded_atlas_full','H3_decoded_atlas_delta'),('H3_ED_pixel_inverse','pixel_inverse')]
    preconditioned=any(r['method']=='H3_preconditioned_last_fullcode' for r in rows)
    if preconditioned:pairs.extend([('H3_preconditioned_decoded_atlas_full','H3_preconditioned_last_fullcode'),('H3_ED_pixel_atlas_matched_support','H3_preconditioned_decoded_atlas_full'),('H3_ED_pixel_inverse_matched_support','H3_preconditioned_decoded_atlas_full'),('H3_ED_pixel_inverse','H3_preconditioned_last_ED')])
    rng=np.random.default_rng(20261015);bootstrap_indices=rng.integers(8,size=(10000,8));effects=[]
    for family in ['raw','DC']:
        for a,b in pairs:
            for metric in ['rgb_psnr','union_psnr','old_exclusive_psnr','new_only_psnr','overlap_psnr']:
                delta=[]
                for source in sources:
                    def mean(method):return np.mean([float(r[metric]) for r in rows if r['family']==family and r['method']==method and r['source']==source and r['prefix']=='4'])
                    delta.append(mean(a)-mean(b))
                delta=np.array(delta);ci=np.percentile(delta[bootstrap_indices].mean(1),[2.5,97.5]);effects.append(dict(family=family,comparison=a+' minus '+b,metric=metric,mean_difference_dB=float(delta.mean()),source_wins=int((delta>0).sum()),n_sources=8,CI95_source_bootstrap=ci.tolist(),source_differences=dict(zip(sources,delta.tolist()))))
    (root/'source_paired_effects.json').write_text(json.dumps(dict(unit='8 natural-photo sources; two observation orders averaged within source; no optimization seeds or independent videos',bootstrap='10000 paired independent source draws; neutral experiment uses no donor replacement',effects=effects),indent=2))
    def score(method,metric='union_psnr',family='DC',prefix=4):return next(x[metric] for x in summaries if x['family']==family and x['method']==method and x['prefix']==prefix)
    labels={'current_H3_LR':'当前 LR→H3 VAE','pixel_lastnear':'pixel last-near','pixel_atlas':'经典 pixel atlas','pixel_inverse':'完整历史线性 inverse','H3_last_fullcode':'H3 last-near full-code','H3_last_deltacode':'H3 last-near delta-code','H3_decoded_atlas_full':'逐packet H3 decoded atlas full','H3_decoded_atlas_delta':'逐packet H3 decoded atlas delta','H3_ED_pixel_atlas':'经典 atlas→H3 E/D','H3_ED_pixel_inverse':'经典 inverse→H3 E/D','H3_ED_pixel_atlas_matched_support':'atlas→H3 E/D，同历史支持读出','H3_ED_pixel_inverse_matched_support':'inverse→H3 E/D，同历史支持读出','oracle_H3_HR':'HR→H3 E/D，仅oracle容量'}
    if preconditioned:labels.update(H3_preconditioned_last_fullcode='同物理预处理 H3 last-near full-code',H3_preconditioned_last_ED='同物理预处理 H3 last-near，完整D读出',H3_preconditioned_decoded_atlas_full='同物理预处理逐packet H3 decoded atlas')
    text=['# B3 互补观测的真实 H3 容量审计','',
          '互补观测增益能穿过真实原生 H3 E/D。8 个自然验证源、两种固定相反顺序、4 个重叠 near2× 足迹，neutral appearance、无遮挡、已知相机；没有训练或test访问，没有H3 DiT。当前far固定同一真实LR packet，隔离表示容量，不声称已经实现18步光照/遮挡/改变协议或真实运动视频。HR只用于模拟传感读数、评估和单列oracle，所有有效状态只由实际LR构造。','',
          '| 第4次near后的相同far DC | 全图PSNR | union | 旧覆盖独占区 | 新覆盖独占区 | 重叠区 |','|---|---:|---:|---:|---:|---:|']
    for method,label in labels.items():text.append('| '+label+' | '+' | '.join(f'{score(method,m):.4f}' for m in ['rgb_psnr','union_psnr','old_exclusive_psnr','new_only_psnr','overlap_psnr'])+' |')
    text+=['','相同支持算子的匹配很关键：逐packet路径先分别E/D，再把各自历史near row-band残差合并；匹配控制先从全部真实LR融合，再E/D，最后施加完全相同的逐足迹平均row-band/current-far-null读出。两者都用同一当前LR锚点和DC，不把decoder输出加入真实观测ledger。另保留未加历史支持过滤的全E/D强控制，原始输出与DC后都在CSV。','',
          '| source内先平均两顺序，union配对差 | ΔdB | 源胜数 | 95% source bootstrap |','|---|---:|---:|---|']
    for a,b in pairs:
        e=next(x for x in effects if x['family']=='DC' and x['comparison']==a+' minus '+b and x['metric']=='union_psnr');lo,hi=e['CI95_source_bootstrap'];text.append(f'| {a} − {b} | {e["mean_difference_dB"]:+.4f} | {e["source_wins"]}/8 | [{lo:+.4f}, {hi:+.4f}] |')
    diag=json.loads((root/'diagnostics.json').read_text());timing=complete['native_timings_seconds'];n=complete['native_calls'];text+=['',
          f'经典 inverse 经H3 E/D相对其pixel结果丢失 {score("pixel_inverse")-score("H3_ED_pixel_inverse"):.4f}dB union；但相对未经物理预处理的H3 last-near full仍有 {score("H3_ED_pixel_inverse")-score("H3_last_fullcode"):.4f}dB互补收益。相同历史支持过滤后，先融合再E/D比未经预处理逐packet decoded atlas高 {score("H3_ED_pixel_inverse_matched_support")-score("H3_decoded_atlas_full"):.4f}dB；但单near prefix1已存在约1.14dB，必须先排除bicubic near与物理反演预处理的混杂，不能据此声称非线性多观测学习缺口。','',
          f'主匹配与强预处理控制实际{n["encode"]}次native22编码+{n["decode"]}次native22解码，{complete["seconds"]:.2f}秒，GPU peak {complete["peak_GPU_bytes"]/1e9:.3f}GB，native E均值{timing["native_encode22"]/n["encode"]*1000:.2f}ms、D均值{timing["native_decode22_lastframe"]/n["decode"]*1000:.2f}ms；48步matrix-free inverse累计{timing["pixel_inverse_prefix"]:.3f}秒/64prefix查询。初轮另52.15秒保留原样；全部物理审计合计{complete["seconds"]+52.15317130088806:.2f}秒，没有优化步数。脚本按source/context逐次记录runtime_operations.csv，包含同步计时；评估复用确定性native输出缓存，不能把这些均值冒充整套流延迟。','',
          f'物理核验：所有锚定路径prebox当前far观测差RMS最大{max(x["prebox_anchor_difference_rms"] for x in diag):.3g}。精度nativeE/D FP16，camera float64，DC float32，TF32关闭；同一观测4种footprint，与前轮CPU传感/噪声seed一致。22次重复图像只提供native时域context，不是22条观测。源级CI分母8，不把顺序/prefix当独立视频。','',
          '状态成本需区分payload、精度与临时缓存：one native code float32为172032B（原始FP16 encoder code可无损以86016B存储）；RGB residual sum+单通道count的float32 atlas为1048576B；4个64×64×3实际LR packet的float32 ledger只需196608B，或固定footprint的sum/count统计，不能刻意把经典baseline成本膨胀成无限ledger。所有方法另有当前LR49152B、4个packet/geometry的64B payload；实际float64审计数组为相应非native状态的两倍，详见state_cost.csv。共享camera算子655360B、当前native latent172032B、当前decoded RGB及native执行缓存另计，完整真实GPU峰值已报告。Python容器元数据不伪装成精确payload字节。decoder前融合的强控制仍需保留原始LR统计以继续更新，不能只报它查询时输出的一个code。','',
          '下一步先测冻结“当前D上下文→可见LR物理修订→E重编码”递归控制的累积遗忘；经典投影+重编码不命名原创。只有跨appearance/遮挡/实际内容变化出现旧区保留与新证据修订的可测缺口，才训练同预算anchor条件写入算子：旧detail、当前anchor latent、当前packet detail/geometry/confidence→canonical revision；真实D过去LR保留+当前可见LR纠错loss与无冲突顺序组合训练。与同输入GRU、非可逆/无swap、直接融合与完整LR inverse→E/D比较。此neutral容量审计本身不能证明可逆解耦，也不支持把经典预处理的摊销称原创。','',
          '完整源hash、输入packet hash、命令、模型SHA、版本、native次数/状态/时间/峰值在complete.json；rgb_metrics.csv含raw/DC每prefix的union/old/new/overlap及past-LR residual，state_cost.csv同时列float32部署payload与float64审计payload。输入packet.pt只在远端保留，明确不含HR。']
    if preconditioned:
        correction=f'最终公平性结论：对每个near单独施加同样的LR物理预处理，再执行原生E/D与相同支持读出，最后decoded atlas达到{score("H3_preconditioned_decoded_atlas_full"):.4f}dB，比同样加强的last-near {score("H3_preconditioned_last_fullcode"):.4f}dB高{score("H3_preconditioned_decoded_atlas_full")-score("H3_preconditioned_last_fullcode"):.4f}dB；batch atlas→E/D matched为{score("H3_ED_pixel_atlas_matched_support"):.4f}dB，仅差{score("H3_ED_pixel_atlas_matched_support")-score("H3_preconditioned_decoded_atlas_full"):+.4f}dB。最初约1.1dB差几乎全部由经典预处理解释，neutral已知几何下尚无明确必须学习的更新缺口。正面结果是强last-near之上的互补收益仍真实存在并被native表示保留。'
        text[2:2]=[correction,'']
    (root/'native_capacity_report.md').write_text('\n'.join(text)+'\n')
    plotted=['current_H3_LR','H3_last_fullcode','H3_last_deltacode','H3_decoded_atlas_full','H3_ED_pixel_inverse_matched_support','H3_ED_pixel_inverse','pixel_inverse'];fig,axes=plt.subplots(1,3,figsize=(15,4))
    if preconditioned:plotted=['current_H3_LR','H3_preconditioned_last_fullcode','H3_preconditioned_decoded_atlas_full','H3_ED_pixel_atlas_matched_support','H3_ED_pixel_inverse','pixel_inverse']
    for ax,metric,title in zip(axes,['union_psnr','old_exclusive_psnr','new_only_psnr'],['All observed near coverage','Old coverage outside current near','New coverage outside old near']):
        for method in plotted:
            xx=[i for i in range(1,5) if score(method,metric,prefix=i) is not None];ax.plot(xx,[score(method,metric,prefix=i) for i in xx],'-o',label=method,linewidth=1.3)
        ax.set(title=title,xlabel='Number of distinct near packets',ylabel='PSNR after same DC (dB)',xticks=[1,2,3,4]);ax.grid(alpha=.2)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=8);fig.tight_layout(rect=(0,.17,1,1));fig.savefig(root/'native_complementary_capacity.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(root/'native_capacity_report.md')


if __name__=='__main__':main()
