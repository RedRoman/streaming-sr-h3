"""Source-level long-stream summary; fixed >=60-frame gap stratification."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/long_video_vae'
DATA = ROOT / 'data/long_video'


if __name__ == '__main__':
    manifest = json.loads((DATA / 'derived_h3_manifest.json').read_text())
    summaries = [json.loads((OUT / v['sequence'] / 'evaluation.json').read_text()) for v in manifest['videos']]
    methods = ['bicubic', 'h3_vae', 'bicubic_dc25', 'h3_vae_dc25']
    rows = []; long_gaps = {}
    for video, summary in zip(manifest['videos'], summaries):
        sequence = video['sequence']; folder = OUT / sequence
        values = json.loads((folder / 'per_frame.json').read_text()); rows.extend(values)
        ann = DATA / video['split'] / sequence / 'annotations_evaluation_only'
        missing = np.loadtxt(ann / 'full_occlusion.txt', delimiter=',').reshape(-1).astype(bool)
        missing |= np.loadtxt(ann / 'out_of_view.txt', delimiter=',').reshape(-1).astype(bool)
        offset = video['start_frame0']; absent = missing[offset:offset + video['frame_count']]
        events = []; length = 0
        for i, hidden in enumerate(absent):
            if hidden:
                length += 1
            else:
                if length >= 60:
                    events.append({'absent_start_local': i - length, 'return_local': i, 'gap_frames': length})
                length = 0
        selected = {i for event in events for i in range(event['return_local'], min(len(absent), event['return_local'] + 17)) if not absent[i]}
        long_gaps[sequence] = {'events': events, 'methods': {}}
        for method in methods:
            chosen = [r for r in values if r['method'] == method and r['frame'] in selected]
            long_gaps[sequence]['methods'][method] = {
                'frames': len(chosen),
                **{key: float(np.mean([r[key] for r in chosen if key in r]))
                   for key in ['psnr', 'lpips', 'roi_psnr', 'roi_lpips', 'temporal_error_excess_rmse']
                   if any(key in r for r in chosen)}}
    aggregate = {}
    for split in ['all', 'train', 'val', 'test']:
        selected = [s for s in summaries if split == 'all' or s['split'] == split]
        aggregate[split] = {'sources': len(selected), 'methods': {}}
        for method in methods:
            aggregate[split]['methods'][method] = {
                key: float(np.mean([s['methods'][method][key] for s in selected if key in s['methods'][method]]))
                for key in ['psnr', 'lpips', 'roi_psnr', 'roi_lpips', 'temporal_error_excess_rmse']
                if any(key in item['methods'][method] for item in selected)}
    out = {'source_averaged': aggregate, 'long_gap_return': long_gaps,
           'long_gap_rule': 'Same >=60 consecutive absent-frame threshold used in source selection; first17 visible frames after return, evaluated posthoc stratification, no parameter tuning',
           'caution': '4 sources, frames not independent replicates; no learned persistent memory'}
    (OUT / 'aggregate.json').write_text(json.dumps(out, indent=2))
    lines = ['# 四条连续600帧长流：冻结H3 VAE基线', '',
             '全部2400帧已通过原生H3 E/D并完成固定指标评价。本实验验证长流底座行为；没有训练记忆、身份检索或跨遮挡细节更新模块。', '',
             '每源35个22帧块、5帧重叠；首次输出22帧，随后每块只输出17个新帧。历史画面不回写。每个输出等待其块末帧，首块最大21、后续最大16个输入间隔；源fps未独立核实，不将名义30fps或离线吞吐称作直播延迟。', '',
             '## 同一观测一致性投影后的主要结果', '',
             '| 来源/split | bicubic PSNR | H3 VAE PSNR | ΔdB | bicubic LPIPS | H3 VAE LPIPS | 目标ROI ΔdB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        a = s['methods']['bicubic_dc25']; b = s['methods']['h3_vae_dc25']
        lines.append(f"| {s['sequence']}/{s['split']} | {a['psnr']:.3f} | {b['psnr']:.3f} | {b['psnr']-a['psnr']:+.3f} | {a['lpips']:.4f} | {b['lpips']:.4f} | {b['roi_psnr']-a['roi_psnr']:+.3f} |")
    all_methods = aggregate['all']['methods']; a = all_methods['bicubic_dc25']; b = all_methods['h3_vae_dc25']
    lines += ['', f"按源等权平均：PSNR变化{b['psnr']-a['psnr']:+.4f}dB，LPIPS变化{b['lpips']-a['lpips']:+.5f}。train/val/test分别汇总于aggregate.json；小规模来源不作总体显著性或普适性结论。", '',
              '## 时间变化与长遮挡后重现', '',
              '时序误差先用相邻LR-only DIS对应，再减去HR自身的运动/外观变化；只在confidence≥0.5区域计算。对应错误与覆盖率仍影响这一指标，低误差不能独立证明纹理身份。', '',
              '| 来源 | bicubic+DC时序误差 | VAE+DC时序误差 | VAE+DC分块边界误差 | ≥60帧缺失事件数 |',
              '|---|---:|---:|---:|---:|']
    for s in summaries:
        a = s['methods']['bicubic_dc25']; b = s['methods']['h3_vae_dc25']
        lines.append(f"| {s['sequence']} | {a['temporal_error_excess_rmse']:.6f} | {b['temporal_error_excess_rmse']:.6f} | {b['boundary']['temporal_error_excess_rmse']:.6f} | {len(long_gaps[s['sequence']]['events'])} |")
    lines += ['', 'per_frame.json保留全部指标；evaluation.json中的return包括所有标注缺失→可见切换后的17帧，aggregate.json另按素材选择时既定的≥60帧缺失标准分层，防止短暂标注切换混同长回访。标注仅用于评价，目标bbox不是精确表面/身份真值；ROI LPIPS要求裁剪最短边≥32，逐指标有效帧数单列。', '',
              '### 长缺失后的目标区域', '',
              '| 来源 | 缺失帧数 | ROI PSNR变化 | ROI LPIPS变化 |',
              '|---|---:|---:|---:|']
    for sequence, item in long_gaps.items():
        if not item['events']:
            continue
        a = item['methods']['bicubic_dc25']; b = item['methods']['h3_vae_dc25']
        gaps = ', '.join(str(e['gap_frames']) for e in item['events'])
        lines.append(f"| {sequence} | {gaps} | {b['roi_psnr']-a['roi_psnr']:+.4f} | {b['roi_lpips']-a['roi_lpips']:+.5f} |")
    lines += ['', '这一子集也不等于记忆恢复实验：没有跨块身份状态。book-20的标注目标是人物身后书架上的书，不是手持封面书；目标位置预览见data/long_video/target_annotation_preview.jpg。标注框只用于定位评价对象，没有作为模型输入。', '',
              '## 运行、预览与限制', '',
              '140次原生E/D的循环耗时375.004秒，包含循环内I/O，不包含模型加载。它包含重叠上下文重复计算，不是端到端在线延迟。评价在共享SSH中断后保留已完成两个来源，仅重算未落盘部分；最终方法参数和指标协议不变。', '',
              '全部控制共用浮点bicubic观测右逆与RGB box的25轮Dykstra投影；报告实际Pillow uint8重投影误差，不称精确Pillow一致性。基础VAE使用float AA bicubic upLR，与官方AddGuide的额外uint8/lanczos处理分开。所有基础输出先量化至8bit，再加入同一DC。', '',
              '每源的event_panel_preview.jpg为元数据固定帧的全视野预览；对应原尺寸event_panel.png保留在运行目录。H3输出、输入LR/HR及各块SHA均在推理或评价阶段实际核验。', '',
              'LaSOT原始JPEG经仅下采样/裁剪生成HR，再构造4×LR；没有放大GT，且源JPEG已有压缩。固定revision第三方镜像通过ZIP CRC/逐文件SHA验证，未独立核验官方完整archive SHA。来源与使用范围见research/long_video/SOURCES.md。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(aggregate['all'], indent=2))
