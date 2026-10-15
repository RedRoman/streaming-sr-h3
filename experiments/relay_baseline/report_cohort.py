"""Summarize the fixed cohort without selecting models or sources."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/relay_baseline/cohort'


def main():
    data = json.loads((OUT/'evaluation/evaluation.json').read_text())
    assert data['status'] == 'complete' and data['frame_metric_rows'] == 2112
    rows = data['source_metrics']; methods = ['h3_vae_dc', 'relay_official_dc', 'relay_causal_repeat_start_dc']
    lines = ['# RelayVSR与H3：固定输入的外部恢复对照', '',
             'Relay在6个独立视频源上均显著降低LPIPS，但只有bus的PSNR超过同最终DC的H3 VAE；六源的LR-flow掩码时序残差均更高。它提供明显更锐利的外观，同时存在纹理定位与保真取舍。不能把较低LPIPS等同真实身份恢复，也不能仅由时序残差断言可见闪烁。', '',
             '固定4train/2val来源、两个不同视野、22帧。所有输入LR SHA一致，24次模型运行、528PNG逐SHA归档。以下每行先按源内22帧平均再等权平均6源；两个视野不作为12个独立视频。所有方法均只做一次相同最终Dykstra25，raw指标完整保存在evaluation目录。', '',
             '| 视野 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 时序残差RMSE↓ |',
             '|---|---|---:|---:|---:|---:|']
    for res in ['256', 'native480']:
        for name in methods:
            r = next(r for r in data['aggregate'] if r['resolution']==res and r['split']=='all' and r['method']==name)
            lines.append(f"| {res} | {name} | {r['psnr']:.5f} | {r['ssim']:.5f} | {r['lpips']:.6f} | {r['temporal_error_excess_rmse']:.6f} |")
    lines += ['', '仅两val源的native480：H3 VAE 25.92030dB/0.399305，Relay官方24.78541/0.184628，因果变体24.72410/0.190595。这是同源不同已训练系统的比较，非同预训练数据、参数、训练算力或生成预算的架构消融。这里H3 VAE沿用Torch AA-bicubic guide，和H3采样内DC报告的官方AddGuide/Lanczos预处理略有不同，不直接拼接成同一VAE控制。', '',
              '## 来源差与等待/计算', '',
              '| native480来源 | official−H3 PSNR | official−H3 LPIPS | causal−H3 PSNR | causal−H3 LPIPS |',
              '|---|---:|---:|---:|---:|']
    paired = []
    sources = sorted({r['source'] for r in rows})
    for res in ['256', 'native480']:
        for source in sources:
            subset = {r['method']: r for r in rows if r['source']==source and r['resolution']==res}
            result = {'source': source, 'resolution': res, 'split': subset['h3_vae_dc']['split']}
            for name in methods[1:]:
                result[name] = {k: subset[name][k]-subset['h3_vae_dc'][k]
                                for k in ['psnr','lpips','temporal_error_excess_rmse']}
            paired.append(result)
            if res=='native480':
                a,b = result[methods[1]], result[methods[2]]
                lines.append(f"| {source} | {a['psnr']:+.5f} | {a['lpips']:+.6f} | {b['psnr']:+.5f} | {b['lpips']:+.6f} |")
    timing = []
    for protocol in ['official', 'causal_repeat_start']:
        for res in ['256','native480']:
            records = [json.loads(p.read_text()) for p in (OUT/protocol).glob('*/measurement.json') if p.parent.name.endswith('_'+res)]
            timing.append({'protocol': protocol, 'resolution': res, 'source_count':len(records),
                           **{k: float(np.mean([r[k] for r in records])) for k in ['model_load_seconds','stream_seconds','first_output_seconds']},
                           'peak_allocated_bytes_max': max(r['peak_allocated_bytes'] for r in records)})
    lines += ['', '官方源码commit e856773350f53036d825404c9ea9228e6dffadca保持clean；官方CLI默认K16、seed42、BF16，显式allocator上限0.8。观察器仅记录输入消费、CUDA同步和输出，不改预测。Wan权重储存5,033,156,288个标量（约5B，含缓存prompt张量），Flash与Wan完整文件均按官方SHA核验。', '',
              '官方模式22帧使用Wan位置[0,15,21]、最大前视15帧。因果repeat-start在[0,15]生成参考，Flash双端点重复最近参考，最大前视0；因此改变了条件和Wan调用数3→2，不称官方因果等价实现。H3 VAE使用完整22帧块，前视上界21帧。', '',
              '| 协议 | 视野 | 平均模型加载s | 平均22帧stream s | 平均首输出s（stream起算） | 最大allocated GB |',
              '|---|---|---:|---:|---:|---:|']
    for r in timing:
        lines.append(f"| {r['protocol']} | {r['resolution']} | {r['model_load_seconds']:.3f} | {r['stream_seconds']:.3f} | {r['first_output_seconds']:.3f} | {r['peak_allocated_bytes_max']/1e9:.3f} |")
    lines += ['', '时间来自单H100、文件输入，stream包含观察器同步与PNG写盘，但不含模型加载或真实采集等待；首样例还有首次执行开销，六源均值未剔除。输入标记24fps是输出时间基准，未验证源视频实际fps，不能由本表宣称直播端到端延迟或稳定实时吞吐。', '',
              '## 评价与归档', '',
              'PSNR为全RGB；SSIM为11×11/sigma1.5有效内区RGB均值；LPIPS Alex v0.1沿用已校验权重且matmul/cuDNN TF32均禁用。Raw统一8位保存，bicubic也量化8位；一次最终DC以float32评价，不额外量化。P25使用float64分离AA-bicubic与RGB box，记录实际残差，不声称等价Pillow整数观测或有限步已收敛。', '',
              '时序评价仅使用既有参数的LR-only DIS，confidence≥0.5；计算输出差分减去同warp的HR差分。六个native480 flow在本轮按相同接口补建，LR输入SHA全部匹配，没有使用HR估计运动。', '',
              '固定t=0/11/21全画幅面板已输出。人工检查cows_256与dance-twirl_native480的共同DC面板：Relay明显更锐利，牛身斑纹/地面纹理存在与HR不一致的局部边界；native480整幅缩略图不足以判断人脸细节。面板按预声明时刻生成，未按指标挑帧。', '',
              '第一次评价12674在读取手动复制LPIPS的dist-info时失败，尚未计算指标；源码存evaluation_attempts。仅将不可用版本metadata改为明确记录，并保留代码/权重hash；数学不变。第二次17464的SSH返回255，但独立17372核验远端12case/2112记录完整、PID退出、GPU5空闲，未重跑。', '',
              '该对照不训练新记忆。H3仍为主模型，后续方法须同时证明观测细节利用、旧细节保留及有意义的修订，而非只复制外部模型的锐化外观。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'paired_and_cost.json').write_text(json.dumps({'paired':paired,'timing':timing},indent=2))
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    selected = [r for r in paired if r['resolution']=='native480']
    for axis, metric, title in zip(axes,['psnr','lpips'],['PSNR difference (higher is better)','LPIPS difference (lower is better)']):
        y = np.arange(len(selected))
        for method,offset,color,label in [(methods[1],-.12,'#2867a2','Relay official, lookahead15'),(methods[2],.12,'#c7672d','Causal variant, lookahead0')]:
            axis.scatter([r[method][metric] for r in selected],y+offset,c=color,label=label,s=45)
        axis.axvline(0,color='0.4',lw=1);axis.set_yticks(y,sources);axis.invert_yaxis();axis.set_title(title);axis.grid(axis='x',alpha=.2)
    axes[0].legend(loc='lower left',fontsize=8)
    fig.suptitle('Native 480p: paired difference from H3 VAE + same final DC25\n6 development sources; sharper appearance does not imply higher fidelity',fontsize=12)
    fig.savefig(OUT/'source_paired.png',dpi=180);fig.savefig(OUT/'source_paired.pdf');plt.close(fig)
    payload = []
    for path in sorted((OUT/'evaluation').rglob('*')):
        if path.is_file():payload.append({'path':str(path.relative_to(OUT)), 'bytes':path.stat().st_size,
                                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT/'evaluation_local_manifest.json').write_text(json.dumps({'files':payload,'count':len(payload)},indent=2))
    print(json.dumps({'report':str(OUT/'REPORT.md'),'paired_rows':len(paired),'evaluation_files':len(payload)}))


if __name__=='__main__':main()
