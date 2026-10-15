"""Build the fixed three-seed B2 report from saved native-decoder evaluations."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--evaluation',default='val_three_seed');args=p.parse_args();root=Path(args.root);evaluation=root/args.evaluation
    raw=list(csv.DictReader((evaluation/'rgb_metrics.csv').open()));dc=list(csv.DictReader((evaluation/'projected/rgb_metrics.csv').open()));outcomes=json.loads((evaluation/'source_paired_outcomes.json').read_text());diag=json.loads((evaluation/'diagnostics.json').read_text());complete=json.loads((evaluation/'complete.json').read_text())
    seeds=sorted({int(r['seed']) for r in raw if int(r['seed'])>=0});sources=sorted({r['source'] for r in raw});assert seeds==[17,29,43] and len(sources)==8
    runtimes={stage:sum(json.loads(path.read_text())['seconds'] for path in (root/stage).glob('*/complete.json')) for stage in ['latent_stage1','rgb_stage2']}
    def score(rows,method,scenario):return np.mean([float(r['rgb_psnr']) for r in rows if r['method']==method and r['scenario']==scenario])
    def paired(group,control,family='DC'):return next(r for r in outcomes['results'] if r['group']==group and r['family']==family and r['comparison']=='factorized_swap minus '+control)
    labels={'H3_current_LR':'当前 LR→H3 VAE','factorized_swap':'B2 full','factorized_no_swap':'no swap','noninvertible_swap':'非可逆','mixed_memory':'mixed','fixed_B1_retained_detail':'固定 B1 retained','fixed_B1_direct_near':'固定 B1 direct near','B2_direct_observed_detail':'B2 direct observed detail'}
    text=['# B2 固定三 seed 验证：支持残差有效，学习状态尚未胜直接近景控制','',f'32 train / 8 val 自然照片 source；seed17/29/43；四模型各600步latent+200步真实冻结H3 decoder微调，batch4与损失权重不变。当前表为验证集机制结果，B2未缓存或评估16个test源。共{len(raw)}条raw RGB与{len(dc)}条相同DC记录。受控相机/appearance/局部替换不能称真实运动视频或长期回访效果；H3 DiT proposal明确为零。','',
          'B1同appearance实际decoder干预显示，far写入丢失62.64%的near支持细节增益；固定current-LR anchor读出干预提高回访0.457dB且far观测误差不变。B2因此将绝对latent状态改成零初始化的持久观测detail残差，far不覆写near支持细节。四模型共享经典历史支持/当前null投影与最终64轮DC；这些共享层解释结构增益，不是可逆chart的独立贡献。','',
          '| 同DC后 PSNR/dB | 回访 A0 | 替换后 far B2 |','|---|---:|---:|']
    for method,label in labels.items():text.append(f'| {label} | {score(dc,method,"return_A0"):.4f} | {score(dc,method,"appearance_B2"):.4f} |')
    text+=['','| full 相对控制的source内配对差 | 回访 ΔdB / 源胜数 | 替换后far ΔdB / 源胜数 |','|---|---:|---:|']
    for control in ['factorized_no_swap','noninvertible_swap','mixed_memory','H3_current_LR','fixed_B1_retained_detail','fixed_B1_direct_near','B2_direct_observed_detail']:
        a=paired('return',control);b=paired('postchange_far',control);text.append(f'| {labels[control]} | {a["mean_difference_dB"]:+.4f} / {a["source_wins"]}/8 | {b["mean_difference_dB"]:+.4f} / {b["source_wins"]}/8 |')
    text+=['',f'Raw回访：full {score(raw,"factorized_swap","return_A0"):.4f}dB，LR-VAE {score(raw,"H3_current_LR","return_A0"):.4f}dB，固定B1 direct near {score(raw,"fixed_B1_direct_near","return_A0"):.4f}dB。强控制的比较在DC前后都保留。近景即时读出由结构与当前LR-VAE相同，不把该恒等式计作学习贡献。','',
          '统计先在每source内配对、再在该source内平均三个训练seed。源胜率分母8，而非24个seed-source，更不能把事件/epoch当独立视频。5000次source环形长度2 block bootstrap只描述这些固定checkpoint上的探索性source差异；donor复用的残余相关性与小source数仍是限制。各seed差异、return/lighting/即时真实替换/替换后far及CI均保存在source_paired_outcomes.json。','',
          f'物理核验：near读出与当前LR-VAE的RGB最大差 {max(x["near_readout_baseline_maxabs"] for x in diag):.3g}；far读出前当前足迹锚点差RMS最大 {max(x["prebox_current_far_observation_difference_rms"] for x in diag):.3g}；所有{len(diag)}个source/seed/model状态均通过zero初始化、far不覆写near、只读query与重复ID幂等检查。实际原生D是FP16，chart/projector为strict FP32，cuDNN和matmul TF32均关闭。训练前已用真实batch确认appearance/chart/update非零梯度和参数更新。','',
          '固定B1 retained/direct probes使用已训练的B1 appearance/chart，所以是零**额外**训练控制，不是完全无学习。B2 direct控制绕过可学习recurrent update但保留同一B2 chart，直接将真实near−LR-only proxy输入detail读出；所有有效控制推理时无HR输入。HR-derived detail teacher单列为oracle容量诊断，绝不算有效方法或状态。','',
          '当前结果支持current-LR锚点与保留历史支持残差，但尚未证明学习更新、反事实swap或可逆性超过强直接证据控制。单near接近充分，容易让learned递归仅衰减证据；未做额外步数或权重细调。下一轮固定互补near协议已先在8val自然图像CPU确认经典融合优于last-near约2.37dB，因此经典融合/全history inverse/参数配平GRU必须保留，而不能把该多观测增益改称原创。','',
          '执行记录：seed29/43最后mixed43的前台SSH在100/200步被远端关闭，进程已确认终止。中断checkpoint/log完整保留在interrupted_attempts；仅该控制从同一stage1初始化、相同source-batch顺序重放完整200步。最终模型仍是有效200步，额外100步是故障物理重算成本，未用于延长优化。没有后台GPU任务。','',
          f'三seed实际decoder验证耗时{complete["seconds"]:.2f}秒。12个模型的有效训练循环用时合计：latent阶段{runtimes["latent_stage1"]:.2f}秒，RGB阶段{runtimes["rgb_stage2"]:.2f}秒；中断100步另计。各阶段根目录complete.seconds只是最后一次命令耗时，不是跨重启的总计。RGB固定50次真实D反向中，20次位于有历史细节的far阶段，另30次是初始/near读出，细节梯度由结构为零；所有控制使用同一安排。逐模型用时、命令与source-code hash在latent_stage1/rgb_stage2 config、complete、training.csv，完整会话注册在上层SESSION_REGISTRY.json。每个主模型参数约14.4万；persistent detail172032字节+RGB reference786432字节=958464字节，packet ID与共享观测矩阵元数据另计，不只报168KB latent。','',
          '可复现入口：experiments/h3_factorized/B2_PROTOCOL.md；下一轮协议B3_COMPLEMENTARY_PROTOCOL.md。完整CSV/统计/图/原生D容量诊断保留在本目录。']
    (root/'B2_three_seed_report.md').write_text('\n'.join(text)+'\n')
    checkpoints=[]
    for stage in ['latent_stage1','rgb_stage2']:
        for path in sorted((root/stage).glob('*/best.pt')):checkpoints.append(dict(stage=stage,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    (root/'checkpoint_manifest.json').write_text(json.dumps(dict(seeds=seeds,source_count=8,test_accessed=False,checkpoints=checkpoints),indent=2));print(root/'B2_three_seed_report.md')


if __name__=='__main__':main()
