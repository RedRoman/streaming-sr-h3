# B2 固定三 seed 验证：支持残差有效，学习状态尚未胜直接近景控制

32 train / 8 val 自然照片 source；seed17/29/43；四模型各600步latent+200步真实冻结H3 decoder微调，batch4与损失权重不变。当前表为验证集机制结果，B2未缓存或评估16个test源。共1080条raw RGB与1080条相同DC记录。受控相机/appearance/局部替换不能称真实运动视频或长期回访效果；H3 DiT proposal明确为零。

B1同appearance实际decoder干预显示，far写入丢失62.64%的near支持细节增益；固定current-LR anchor读出干预提高回访0.457dB且far观测误差不变。B2因此将绝对latent状态改成零初始化的持久观测detail残差，far不覆写near支持细节。四模型共享经典历史支持/当前null投影与最终64轮DC；这些共享层解释结构增益，不是可逆chart的独立贡献。

| 同DC后 PSNR/dB | 回访 A0 | 替换后 far B2 |
|---|---:|---:|
| 当前 LR→H3 VAE | 24.2119 | 23.8652 |
| B2 full | 24.6819 | 24.3339 |
| no swap | 24.6770 | 24.3468 |
| 非可逆 | 24.6785 | 24.3340 |
| mixed | 24.6778 | 24.3337 |
| 固定 B1 retained | 24.6689 | 24.0525 |
| 固定 B1 direct near | 24.7550 | 24.5027 |
| B2 direct observed detail | 24.7528 | 24.3626 |

| full 相对控制的source内配对差 | 回访 ΔdB / 源胜数 | 替换后far ΔdB / 源胜数 |
|---|---:|---:|
| no swap | +0.0049 / 5/8 | -0.0129 / 2/8 |
| 非可逆 | +0.0034 / 6/8 | -0.0001 / 5/8 |
| mixed | +0.0041 / 4/8 | +0.0002 / 4/8 |
| 当前 LR→H3 VAE | +0.4700 / 8/8 | +0.4687 / 8/8 |
| 固定 B1 retained | +0.0130 / 6/8 | +0.2814 / 8/8 |
| 固定 B1 direct near | -0.0731 / 0/8 | -0.1688 / 0/8 |
| B2 direct observed detail | -0.0709 / 0/8 | -0.0287 / 1/8 |

Raw回访：full 23.9816dB，LR-VAE 23.5659dB，固定B1 direct near 24.0429dB。强控制的比较在DC前后都保留。近景即时读出由结构与当前LR-VAE相同，不把该恒等式计作学习贡献。

统计先在每source内配对、再在该source内平均三个训练seed。源胜率分母8，而非24个seed-source，更不能把事件/epoch当独立视频。5000次source环形长度2 block bootstrap只描述这些固定checkpoint上的探索性source差异；donor复用的残余相关性与小source数仍是限制。各seed差异、return/lighting/即时真实替换/替换后far及CI均保存在source_paired_outcomes.json。

物理核验：near读出与当前LR-VAE的RGB最大差 5.96e-08；far读出前当前足迹锚点差RMS最大 2.9e-09；所有96个source/seed/model状态均通过zero初始化、far不覆写near、只读query与重复ID幂等检查。实际原生D是FP16，chart/projector为strict FP32，cuDNN和matmul TF32均关闭。训练前已用真实batch确认appearance/chart/update非零梯度和参数更新。

固定B1 retained/direct probes使用已训练的B1 appearance/chart，所以是零**额外**训练控制，不是完全无学习。B2 direct控制绕过可学习recurrent update但保留同一B2 chart，直接将真实near−LR-only proxy输入detail读出；所有有效控制推理时无HR输入。HR-derived detail teacher单列为oracle容量诊断，绝不算有效方法或状态。

当前结果支持current-LR锚点与保留历史支持残差，但尚未证明学习更新、反事实swap或可逆性超过强直接证据控制。单near接近充分，容易让learned递归仅衰减证据；未做额外步数或权重细调。下一轮固定互补near协议已先在8val自然图像CPU确认经典融合优于last-near约2.37dB，因此经典融合/全history inverse/参数配平GRU必须保留，而不能把该多观测增益改称原创。

执行记录：seed29/43最后mixed43的前台SSH在100/200步被远端关闭，进程已确认终止。中断checkpoint/log完整保留在interrupted_attempts；仅该控制从同一stage1初始化、相同source-batch顺序重放完整200步。最终模型仍是有效200步，额外100步是故障物理重算成本，未用于延长优化。没有后台GPU任务。

三seed实际decoder验证耗时52.41秒。12个模型的有效训练循环用时合计：latent阶段775.46秒，RGB阶段406.38秒；中断100步另计。各阶段根目录complete.seconds只是最后一次命令耗时，不是跨重启的总计。RGB固定50次真实D反向中，20次位于有历史细节的far阶段，另30次是初始/near读出，细节梯度由结构为零；所有控制使用同一安排。逐模型用时、命令与source-code hash在latent_stage1/rgb_stage2 config、complete、training.csv，完整会话注册在上层SESSION_REGISTRY.json。每个主模型参数约14.4万；persistent detail172032字节+RGB reference786432字节=958464字节，packet ID与共享观测矩阵元数据另计，不只报168KB latent。

可复现入口：experiments/h3_factorized/B2_PROTOCOL.md；下一轮协议B3_COMPLEMENTARY_PROTOCOL.md。完整CSV/统计/图/原生D容量诊断保留在本目录。
