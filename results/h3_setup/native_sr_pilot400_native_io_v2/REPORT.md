# H3 原生 SR 适配：官方BF16流IO修复后的固定预算pilot v2

本版本修复legacy_fp32_io的训练/推理精度混杂：训练stream输入/输出逐项对齐官方BF16包装层，audio严格在carried latent精度转换后还原physical audio，loss按FP32累计。零adapter和guide bias+0.01的实际官方前向预测均maxabs0，真反传非零、基础参数梯度0。旧版本完整保留，本版本不继续旧checkpoint，而从同初始化、同seed/case/sigma/noise/lr重新各400次，共1200次。

训练仅用 bear、bus、car-turn、flamingo 四个视频，各256方形/832×480两种空间上下文，每case50次。独立val为 cows、dance-twirl 两源，同样两种上下文。固定seed703、相同case/sigma/noise序列、AdamW学习率1e-5，不扫参数、不按val选择checkpoint；统一使用最终400权重。两种空间上下文取景范围不同，不能把它们当纯分辨率消融。

| 控制 | 可训练参数 | 初始val velocity MSE | 400次后 | 相对变化 |
|---|---:|---:|---:|---:|
| LoRA400 | 3,942,400 | 0.208730 | 0.193514 | -7.29% |
| Guide400 | 13,880 | 0.208730 | 0.204401 | -2.07% |
| LoRA+Guide400 | 3,956,280 | 0.208730 | 0.186821 | -10.50% |

这些控制参数量不同，不作等容量比较。velocity验证改善不能替代完整采样后的RGB恢复指标。

每个val case固定同neutral text、观测LR构造的guide、prefix0..21、seed101、CFG1和正常50个sigma点/49次Euler读出；guide projection是被比较的可训练模块。canonical FP16 VAE统一在H100解码。文本context来自真实Qwen3VL32B NVFP4-AWQ编码器/BF16计算，DiT精度与文本编码器量化独立标注。

| 空间上下文 | 控制 | RGB PSNR↑ | RGB SSIM↑ | LPIPS↓ |
|---|---|---:|---:|---:|
| 256 | Bicubic + DC25 | 25.097 | 0.7503 | 0.35850 |
| 256 | H3 VAE guide + DC25 | 25.388 | 0.7591 | 0.31226 |
| 256 | H3 BF16 base + DC25 | 24.948 | 0.7358 | 0.29908 |
| 256 | LoRA400 + DC25 | 24.943 | 0.7355 | 0.29901 |
| 256 | Guide400 + DC25 | 24.743 | 0.7251 | 0.30117 |
| 256 | LoRA+Guide400 + DC25 | 24.793 | 0.7282 | 0.30117 |
| native480 | Bicubic + DC25 | 25.641 | 0.6675 | 0.44550 |
| native480 | H3 VAE guide + DC25 | 25.922 | 0.6777 | 0.39854 |
| native480 | H3 BF16 base + DC25 | 25.892 | 0.6759 | 0.39208 |
| native480 | LoRA400 + DC25 | 25.871 | 0.6749 | 0.38853 |
| native480 | Guide400 + DC25 | 25.293 | 0.6481 | 0.37932 |
| native480 | LoRA+Guide400 + DC25 | 25.380 | 0.6524 | 0.37486 |

上表对所有控制统一施加固定25次输出空间LR一致性投影；投影结果不成为新证据。raw/逐帧/重投影残差指标完整保留在 evaluation/aggregate.json。部分适配器虽降低velocity误差，却放大raw输出的观测漂移；LPIPS与PSNR存在取舍，不能据此宣称稳定超过VAE强基线或建立记忆方法收益。

结论仅限4个训练源、2个验证源的单seed固定预算pilot，没有测试集结论。22帧是有效原生时间网格，但短于SGLang H3发布元数据的124帧质量工作负载；单块实际观测end21，名义输入等待上界21帧，没有实时采集或稳态流式延迟测量。

全20个输出case共440张PNG已逐文件SHA核验，三个最终adapter与全部1200条训练日志均已本地归档。中间step/optimizer恢复点保留在H100；本版本独立训练，不继续旧版400权重，前台PID/命令/日志见 execution_handles.json。

复现入口：experiments/h3_backbone/train_native_sr_pilot_native_io_v2.py；完整protocol/schedule见 protocol.json；预算、冻结和条件梯度审计见 protocol_audit.json；完整基础主干SHA与精度见 asset_provenance.json。

前置审计见../native_math_alignment_v2/result.json：v目标epsilon−zHR、t=1000σ、scale1、24ch guide与text/layout均匹配；零adapter和guide末层bias+0.01两条件的输入video/physicalaudio及v预测对官方包装层maxabs0，x0与官方单步Euler最大差2.38e−7。全部50层真实反传guide/LoRA/projection梯度非零，530基础张量版本不变、basegrad0。数学诊断仅用HR构造监督噪声桥，不是SR推理结果。协议审计另逐项核验v2与legacy的case/sigma/noise/lr/seed/400预算完全一致。
