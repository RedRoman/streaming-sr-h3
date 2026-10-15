# B2 首个 seed：锚点修复有效，可学习核心尚未超过直接近景控制

这是固定 B1 诊断后的一次结构修订：32 个自然照片训练源、8 个验证源，seed17，四个约 14.4 万参数控制均使用 600 步 latent 训练和 200 步真实 H3 decoder 微调。无 B2 test 访问。图片上的相机、光照、局部替换是受控模拟，不是实际运动视频结果；H3 DiT proposal 明确为零，实际使用的是冻结 H3 VAE。

B1 的同 appearance 查询显示，near 带来的观测支持细节增益在下一次 far 写入中丢失 62.64%。B2 改为零初始化的持久细节残差；far 只更换当前 LR 锚点与 appearance，不覆写 near 细节。四控制共享经典历史支持/当前观测零空间读出与同一个 64 轮 affine/box DC。该投影是控制条件，不是原创核心。

| 验证事件，相同 DC 后 PSNR/dB | 当前 LR→H3 VAE | B2 full | no swap | 非可逆 | mixed | 固定 B1 retained | 固定 B1 direct near | B2 direct observed detail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 回访 A0 | 24.2119 | 24.6852 | 24.6770 | 24.6797 | 24.6799 | 24.6712 | **24.7570** | 24.7520 |
| 真实替换后的 far B2 | 23.8652 | 24.3318 | 24.3417 | 24.3307 | 24.3334 | 24.0220 | **24.5044** | 24.3625 |

full 回访相对 LR-VAE 为 +0.4734 dB，8/8 源获益；但相对固定 B1 direct near 为 −0.0717 dB，0/8 源获益。full 相对 no swap 仅 +0.0083 dB（5/8 源），相对非可逆 +0.0056 dB（6/8），相对 mixed +0.0053 dB（3/8）。这些小差异不能建立可逆性或反事实训练的独立价值。固定 B1 probes 使用已训练好的 B1 appearance/chart，含义是零**额外**训练，不是完全无学习。

raw 回访 full 为 23.9846 dB，LR-VAE 23.5659 dB，固定 B1 direct 24.0446 dB；因此对直接证据控制的不足也存在于 DC 前。近景事件由结构保证回到当前 LR-VAE，不能将该恒等式当成学习进展。

物理核验：所有控制零残差真实 decoder 读出与 LR-VAE 的 RGB 最大差为 0；far 读出前观测锚点误差 RMS 最大 2.84e−9；验证 near 读出与当前 LR-VAE 最大差 5.96e−8；far/query/重复 packet 保留细节状态逐位相同。真实 decoder 的历史 near 方向导数 RMS 为 4.58e−6，而相同 far 足迹方向导数为 0。appearance/chart/update 分支均已在实际 batch 确认非零梯度和更新。

单个 near 已近乎充分：使用 HR 构造、仅作容量诊断的 detail teacher 回访 24.7583 dB，与直接 near 24.7570 接近。该 oracle 绝不作为有效方法或实际初始状态。下一轮应改变可辨识的 episode 结构，让多个互补 near、跨光照、局部遮挡与内容替换要求状态真正融合证据；继续调此单 near 套件的步数或损失权重无法验证核心。

统计单位是 8 个自然照片 source，先在每 source 内配对；不是把 epoch、事件或 seed 当独立视频。当前区间为 source 环形长度 2 block bootstrap 的探索性结果，donor 复用仍造成相关性。正在按已固定相同预算复制 seed29/43，未进行细调。

完整数据：`val_seed17/rgb_metrics.csv`、`val_seed17/projected/rgb_metrics.csv`、`val_seed17/source_paired_outcomes.json`、`val_seed17/diagnostics.json`、`protocol_checks.json`。真实主阶段耗时约 250.5 秒 + RGB 微调 134.0 秒，验证含强控制 25.80 秒；缓存 88.07 秒。精度：strict FP32 chart/projector、TF32 关闭，原生冻结 H3 decoder FP16。
