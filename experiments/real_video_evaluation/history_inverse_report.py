"""Render the fixed real-video history-control results without selecting settings."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/real_video_history_inverse/matched_final_projection'
data = json.loads((OUT / 'summary.json').read_text())
diag = json.loads((OUT / 'diagnostic.json').read_text())
sources = data['sources']
aggregates = {}
for split in ['train', 'val', 'all']:
    rows = [s for s in sources if split == 'all' or s['split'] == split]
    aggregates[split] = {m: {k: sum(s['methods'][m][k] for s in rows) / len(rows)
                            for k in ['psnr', 'ssim', 'lpips']}
                         for m in rows[0]['methods']}
(OUT / 'aggregate.json').write_text(json.dumps(aggregates, indent=2))

text = '''# 真实视频：固定历史逆问题强控制（最终投影预算已配平）

固定past-4、LR-only DIS对应、光度/双向一致性权重、ridge=0.01、24次CG，在当前帧低清观测的null方向吸收历史观测，最后统一25次DC/box投影。无训练、无参数搜索、无验证集选择。6个DAVIS来源分别为4 train/2 val；每源22帧、256中心视野。输出使用同一冻结H3 VAE的22帧块，故历史修正虽只用当前及过去LR，完整方法仍有同样的块内等待，不能称零前视流式方法。

这是经典多帧反演控制，没有原创方法主张，也没有长期身份检索。相同源的帧不作为独立样本。

归档检查发现旧版基线少做了最终一次DC；有限迭代的box投影尚未完全幂等，t=0最大像素差0.03138。因此本版本为基线补上与历史逆问题相同的最终25次投影，历史修订输出保持不变；6源最终t=0四控制分别与其无历史版本逐位相同。旧版输出/指标/源码在父目录完整保留，不以其数字作最终比较。没有改动观测、历史预算、ridge、CG或任何模型参数。

## 按来源结果

所有数值均在相同DC之后；PSNR取逐帧均值，每个来源等权。

| 来源 | split | H3 VAE PSNR | 历史逆问题 PSNR | 差值 | H3 VAE LPIPS | 历史逆问题 LPIPS |
|---|---|---:|---:|---:|---:|---:|
'''
for row in sources:
    a, b = row['methods']['vae_dc'], row['methods']['inverse_vae_dc']
    text += f"| {row['source']} | {row['split']} | {a['psnr']:.5f} | {b['psnr']:.5f} | {b['psnr']-a['psnr']:+.5f} | {a['lpips']:.5f} | {b['lpips']:.5f} |\n"
text += '\n## 四控制均值\n\n| split | 方法 | PSNR | SSIM | LPIPS |\n|---|---|---:|---:|---:|\n'
for split, methods in aggregates.items():
    for method, metrics in methods.items():
        text += f"| {split} | {method} | {metrics['psnr']:.5f} | {metrics['ssim']:.5f} | {metrics['lpips']:.5f} |\n"
text += '''
H3 VAE上追加历史逆问题在4/6源提高PSNR，train平均+0.11956dB；val的cows提高0.11660dB、dance-twirl下降0.86470dB，val平均下降0.37405dB。全6源平均下降0.04498dB，LPIPS平均改善约0.0037。不能称稳定质量改进。

## 观测模型诊断

评价程序把当前GT经同一个估计对应/warp/downsample算子映射到过去LR，仅在评价时访问GT。该误差混合几何、遮挡、亮度和退化不交换性，不能直接称为纯光流端点误差。以下按每个past-current对计算加权MSE，再按对平均。

| 来源 | GT观测MSE | 原H3 VAE观测MSE | 修订后观测MSE | mean confidence |
|---|---:|---:|---:|---:|
'''
for row in diag['sources']:
    v = row['pair_mean']
    text += f"| {row['source']} | {v['truth_weighted_observation_mse']:.7f} | {v['base_weighted_observation_mse']:.7f} | {v['inverse_weighted_observation_mse']:.7f} | {v['mean_confidence']:.4f} |\n"
text += '''
在flamingo和dance-twirl上，GT本身比原H3输出更不符合估计的历史观测模型，优化继续降低该错误模型的残差，同时损害HR质量。即便只看置信度≥0.8像素，两源也有同样排序。因此既不能把更小历史重投影误差当作正确细节的证据，也不能把高photo/cycle权重当作可靠对应已成立。

人工查看固定t=0/11/21面板：bus窗框和车身部分边缘略清楚；dance-twirl衣物/身体与flamingo身上的新增条纹是明显伪影。与量化结果一致，不能仅凭高频能量增加称细节恢复。

后续可操作方向是区分“可由几何/外观误差解释的残差”和“由新采样相位支持的细节”，并让后来的独立观测修订此前的观测解释。需要与稳健反演、可回放factor cache、学习对应和不保留历史的控制比较；本实验尚未实现该新核心。

## 计算与数值

双精度独立小算子检查：adjoint内积误差2.22e-16；新观测MSE从0.00013149降至0.00003322，当前LR变化约6.3e-18；没有历史输入时求解器严格不改变输入。实际推理的24步CG均有限，最大相对normal residual 4.65e-6；最终DC还受RGB box约束及有限迭代影响，六源最大LR RMSE为1.19e-4，不能声称所有最终输出严格零残差。

4×22×6=528个float RGB输出，132个source-frame组合，两个历史控制各自求解；t=0无历史。6源原求解/DC计时约20.65s（含两个prior）；未包含flow、文件读写、VAE、模型加载以及公平修复时补上的基线最终投影，起始未逐次CUDA同步，不能作为端到端吞吐或在线延迟。GPU5执行，未载入DiT；H3 VAE输出复用已核验缓存。最终LPIPS与原控制统一关闭CUDA matmul及cuDNN TF32；没有用不同数值设置的评分选择方法。

`protocol.json`记录固定参数与LR来源，`geometry_provenance.json`记录每源78对LR-only映射，父目录`operator_audit.json`记录算子检查，`inference.json`记录原/新输出SHA与逐帧计算，`fairness_repair.json`记录零历史对齐，`summary.json`/`aggregate.json`/`per_frame.json`提供指标，`diagnostic.json`/`diagnostic_pairs.json`/`diagnostic_frames.json`提供模型误差评价。GT只用于评价；未打开项目封存的DIV2K确认图片。
'''
(OUT / 'REPORT.md').write_text(text)
print(OUT / 'REPORT.md')
