"""Verify local archive and write a compact report for the fixed pilot."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'results/h3_setup/native_sr_pilot400'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=json.loads((RUN/'protocol.json').read_text());audit=json.loads((RUN/'protocol_audit.json').read_text());metrics=json.loads((RUN/'evaluation/aggregate.json').read_text())
    assert audit['passed'] and metrics['status']=='complete'
    checked=0;files={};gt={}
    for mode in ['vae_only','unadapted','lora_only','guide_only','joint']:
        for case in protocol['validation_cases']:
            folder=RUN/'readouts'/mode/case;d=json.loads((folder/'decode_result.json').read_text());assert d['status']=='complete' and len(d['output_png_sha256'])==22
            for name,digest in d['output_png_sha256'].items():
                assert sha(folder/name)==digest,(mode,case,name);checked+=1
    assert checked==440
    for mode in ['lora_only','guide_only','joint']:
        p=RUN/mode/'adapter_final400.safetensors';files[str(p.relative_to(RUN))]={'bytes':p.stat().st_size,'sha256':sha(p)}
        rows=list(map(json.loads,(RUN/mode/'train.jsonl').read_text().splitlines()));assert len(rows)==400 and sorted(r['update'] for r in rows)==list(range(400))
    cases=json.loads((ROOT/'results/h3_setup/native_sr_val_pairs/pairs_manifest.json').read_text())['cases']
    for case in cases:
        assert [sha(ROOT/'data/real_video'/p) for p in case['lr_paths']]==case['observed_lr_sha256']
        gt[case['id']]=[sha(ROOT/'data/real_video'/p) for p in case['hr_paths']]
    integrity={'passed':True,'checked_output_PNG':checked,'control_unique_updates':{'lora_only':400,'guide_only':400,'joint':400},'final_adapters':files,'validation_HR_evaluation_sha256':gt,
               'protocol_sha256':sha(RUN/'protocol.json'),'metrics_sha256':sha(RUN/'evaluation/aggregate.json'),'protocol_audit_sha256':sha(RUN/'protocol_audit.json')}
    (RUN/'archive_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    names={'bicubic':'Bicubic','vae_only':'H3 VAE guide','unadapted':'H3 BF16 base','lora_only':'LoRA400','guide_only':'Guide400','joint':'LoRA+Guide400'}
    text=['# H3 原生 SR 适配：固定预算 pilot','',
          '本目录保留 legacy_fp32_io 版本：三控制各400次更新，共1200次，但事后审计发现训练直接调用使用FP32流输入/输出，官方采样wrapper使用BF16，存在训练/推理精度混杂。零适配器前后 video/audio 最大误差均0仅指同一raw调用路径内；不能据此宣称训练与官方包装层完全一致。冻结基础参数版本不变且梯度为0，guide输入梯度全程非零。该旧版本不用于判断H3原生适配能力；修复后新实验另存 native_sr_pilot400_native_io_v2。','',
          '训练仅用 bear、bus、car-turn、flamingo 四个视频，各256方形/832×480两种空间上下文，每case50次。独立val为 cows、dance-twirl 两源，同样两种上下文。固定seed703、相同case/sigma/noise序列、AdamW学习率1e-5，不扫参数、不按val选择checkpoint；统一使用最终400权重。两种空间上下文取景范围不同，不能把它们当纯分辨率消融。','',
          '| 控制 | 可训练参数 | 初始val velocity MSE | 400次后 | 相对变化 |','|---|---:|---:|---:|---:|']
    for row in audit['rows']:
        delta=100*(row['fixed_validation_final400']/row['fixed_validation_initial']-1)
        text.append(f"| {names[row['mode']]} | {row['trainable_parameters']:,} | {row['fixed_validation_initial']:.6f} | {row['fixed_validation_final400']:.6f} | {delta:+.2f}% |")
    text+=['','这些控制参数量不同，不作等容量比较。velocity验证改善不能替代完整采样后的RGB恢复指标。','',
           '每个val case固定同neutral text、观测LR构造的guide、prefix0..21、seed101、CFG1和正常50个sigma点/49次Euler读出；guide projection是被比较的可训练模块。canonical FP16 VAE统一在H100解码。文本context来自真实Qwen3VL32B NVFP4-AWQ编码器/BF16计算，DiT精度与文本编码器量化独立标注。','',
           '| 空间上下文 | 控制 | RGB PSNR↑ | RGB SSIM↑ | LPIPS↓ |','|---|---|---:|---:|---:|']
    for resolution in ['256','native480']:
        for method in ['bicubic','vae_only','unadapted','lora_only','guide_only','joint']:
            row=next(r for r in metrics['aggregate'] if r['resolution']==resolution and r['method']==method+'_dc25')
            text.append(f"| {resolution} | {names[method]} + DC25 | {row['mean_frame_psnr_db']:.3f} | {row['mean_ssim_rgb']:.4f} | {row['mean_lpips_alex_v01']:.5f} |")
    text+=['','上表对所有控制统一施加固定25次输出空间LR一致性投影；投影结果不成为新证据。raw/逐帧/重投影残差指标完整保留在 evaluation/aggregate.json。部分适配器虽降低velocity误差，却放大raw输出的观测漂移；LPIPS与PSNR存在取舍，不能据此宣称稳定超过VAE强基线或建立记忆方法收益。','',
           '结论仅限4个训练源、2个验证源的单seed固定预算pilot，没有测试集结论。22帧是有效原生时间网格，但短于SGLang H3发布元数据的124帧质量工作负载；单块实际观测end21，名义输入等待上界21帧，没有实时采集或稳态流式延迟测量。','',
           '全20个输出case共440张PNG已逐文件SHA核验，三个最终adapter与全部1200条训练日志均已本地归档。中间step/optimizer恢复点保留在H100；SSH断连只发生在guide读出阶段，400次优化没有重放，详情见 execution_history.jsonl 和 executions/。','',
           '复现入口：experiments/h3_backbone/train_native_sr_pilot.py；完整protocol/schedule见 protocol.json；预算、冻结和条件梯度审计见 protocol_audit.json；完整基础主干SHA与精度见 asset_provenance.json。']
    math_path=ROOT/'results/h3_setup/native_math_alignment/result.json'
    if math_path.exists():
        math=json.loads(math_path.read_text());zero=math['conditions'][0]
        text+=['','固定bear_256单例数学审计（../native_math_alignment/result.json）：v目标epsilon−zHR、t=1000σ、video process_in/out scale1、24ch guide/layout/context均匹配。对齐官方BF16 stream IO和audio carry后，直接调用与官方wrapper预测maxabs0；官方一阶Euler与xσ−σv最大差2.38e−7。零adapter时原FP32训练路径相对BF16官方预测v RMSE %.6f、x0 RMSE %.6f，说明本版本存在实际数值差异；不把该差异与小数据训练问题混为一谈，也不把单点差异断言为全部raw漂移原因。审计仅用HR构造监督桥，不是可因果使用的SR输出。'% (zero['v_trainF32_vs_official_wrapperBF16']['rmse'],zero['training_x0_vs_official_oneEuler_output']['rmse'])]
    (RUN/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'archive_integrity':integrity,'aggregate':metrics['aggregate']},ensure_ascii=False))

if __name__=='__main__':main()
