# 持续细节状态：可复现机制实验报告

日期：2026-10-15（Asia/Shanghai）。此报告由已完成的 CSV/JSON 结果生成；没有把 toy inverse problem 结果称为真实视频超分性能。

## 结论与下一步主线

支持继续推进的主线是：**把已证实的高清细节存成有证据出处的对象状态，让新的互补观测纠错；对矛盾观测先隔离，再分叉新状态，避免永久污染旧身份。** 本套件证明了这个方向的必要性和可操作切口，但普通的 G/b 累积本身就是经典 Gaussian MAP，不构成原创方法。

在 32 个 seed 的互补相位实验中，持续证据状态最终平均 35.29 dB，4 帧逆问题基线 29.70 dB，逐帧方法 20.94 dB。跨 seed 配对增益分别为 +5.59 [+5.44, +5.74] 和 +14.35 [+12.39, +15.87] dB（95% seed bootstrap CI）。

同时得到两个会影响架构设计的负面机制：重复同一观测不能纠正其不可见的假纹理；只看单对象 LR 创新时，持续错误身份关联和物体真实变化可以产生完全一样的观测历史。对应解法应是带证据子空间的更新，以及保留旧身份的多假设分叉/检索，而不是永久覆盖单一状态。

## 实验协议

- HR：24×24 灰度 canonical patch；16 个随机 Fourier 解析纹理 seed + 16 个自然照片 patch seed。自然 patch 均来自 Matplotlib 自带的 `grace_hopper.jpg`，先缩至 160×180，再确定性随机裁剪。单一照片来源限制外推范围。
- 观测：周期边界 Gaussian PSF（sigma=0.60）后，已知相位按 2 或 4 倍下采样；独立 Gaussian LR 噪声 sigma=0.01。共 16 个因果帧。
- 初始高清 proposal 完全由第一张 LR 的插值和与真值独立的随机高频纹理组成，再投影到第一张 LR 的观测约束。错误先验情境将随机高频振幅从 0.10 加到 0.20。HR 仅进入仿真器和评价指标，绝不用于恢复初始化、更新、门控或停止。
- 32 seeds 为 0…31；单个 seed 的各情境共用对象和噪声随机数流以便配对。主 suite 运行 256 个 seed×情境组合、16 帧、8 方法，产生 32,768 行 per-frame 指标。
- 最终帧、事件后半段和全时段分别聚合；5000 次按 seed 重采样计算 bootstrap 区间。跨帧先在 seed 内平均，避免将相关帧当独立样本。
- λ=0.0025，与固定 proposal 方差 τ²=0.04 和 sigma² 的 Gaussian MAP 对应。所有方法不做超参数搜索；额外噪声压力测试保持 τ=0.2，按 sigma² 改变 λ。

## 方法与公平性

1. `current_only`：仅当前 LR 的插值 prior + 当前已知算子 Tikhonov 恢复。
2. `fixed_reference`：固定第一帧 LR 构造的高清 proposal，不读取未来观测。这里的 reference 是生成 proposal，不能冒称外部真实 HR 参考。
3. `ema`：0.8 旧状态 + 0.2 当前单帧结果。
4. `residual_injection`：固定增益 0.35 的当前 LR 残差逆投影。
5. `short_window_map`：最近 4 个日历帧中的实际观测联合 MAP。遮挡期间会到期；不偷偷保留过去 4 个可见观测。
6. `all_history_map`：每个时刻独立重组全部过去实际 LR 观测的 G/b，再求一次 batch MAP。
7. `persistent_evidence`：递推维护 G_t=G_{t-1}+A_tᵀA_t、b_t=b_{t-1}+A_tᵀy_t；输出 (G_t+λI)⁻¹(b_t+λp)。proposal p 只计一次，从不把自己输出作为新证据。
8. `quarantine_restart`：用当前 posterior 的 LR 预测残差均值除以逐像素预测方差做创新分数；大于 4 的观测先隔离，连续 3 帧矛盾时用这 3 帧启动新 epoch，隔离帧不污染旧 G/b。分数仅用了预测方差对角项，阈值是预先指定的工程规则，不是精确卡方检验。

持续状态与 all-history MAP 使用同一初始化/正则/数据前缀，二者理论上等价。将两者比较的目的是避免把高质量结果错误归因于新优化器或偷偷复用生成图像。

## 主结果（最终帧，dB，32 seeds 平均）

| 情境 | 当前帧 | 固定先验 | EMA | 残差注入 | 4帧MAP | 全历史MAP / 持续证据 | 隔离重启 |
|---|---:|---:|---:|---:|---:|---:|---:|
| complementary | 20.94 | 17.92 | 23.09 | 23.82 | 29.70 | 35.29 | 35.29 |
| repeated | 22.11 | 17.92 | 22.50 | 17.95 | 17.95 | 17.95 | 17.95 |
| scale_reveal | 20.96 | 16.10 | 22.57 | 21.74 | 29.58 | 33.20 | 33.20 |
| revisit | 20.94 | 17.92 | 22.66 | 21.60 | 29.70 | 32.48 | 32.48 |
| wrong_prior | 20.94 | 14.34 | 23.01 | 19.72 | 29.49 | 35.23 | 35.23 |
| intermittent_mismatch | 20.94 | 17.92 | 21.60 | 21.29 | 29.70 | 22.67 | 34.64 |
| persistent_mismatch | 13.42 | 17.92 | 14.78 | 13.83 | 12.30 | 18.24 | 12.47 |
| content_change | 21.58 | 12.29 | 21.05 | 21.58 | 29.70 | 18.22 | 32.45 |

| 配对比较：持续证据减去基线 | 均值与 95% CI（dB） |
|---|---:|
| complementary / short_window_map | +5.59 [+5.44, +5.74] |
| scale_reveal / short_window_map | +3.61 [+3.49, +3.74] |
| wrong_prior / fixed_reference | +20.89 [+20.30, +21.45] |
| repeated / current_only | -4.15 [-5.62, -2.97] |
| intermittent_mismatch / quarantine_restart | -11.96 [-13.87, -10.03] |
| content_change / quarantine_restart | -14.23 [-15.69, -12.67] |
| persistent_mismatch / quarantine_restart | +5.76 [+5.38, +5.96] |

`persistent_evidence - quarantine_restart` 为负表示隔离重启更好。完整逐帧、事件后半段、均值误差、区间和所有配对比较见 CSV，不以单张定性图替代统计。

## 自然与解析纹理分组

| 数据组 | 情境 | 当前帧 | 4帧MAP | 持续证据 | 隔离重启 |
|---|---|---:|---:|---:|---:|
| analytic | complementary | 17.76 | 29.61 | 35.32 | 35.32 |
| analytic | repeated | 17.76 | 15.96 | 15.97 | 15.97 |
| analytic | intermittent_mismatch | 17.76 | 29.61 | 26.67 | 34.62 |
| analytic | content_change | 17.83 | 29.61 | 20.24 | 32.42 |
| natural | complementary | 24.12 | 29.79 | 35.26 | 35.26 |
| natural | repeated | 26.45 | 19.93 | 19.94 | 19.94 |
| natural | intermittent_mismatch | 24.12 | 29.79 | 18.68 | 34.65 |
| natural | content_change | 25.32 | 29.78 | 16.20 | 32.48 |

## 机制证据与可操作判据

**信息方向，而非帧数。** 重复情境的最终 G 秩为 144/576，覆盖互补相位的情境为 576/576。重复情境第一帧观测 nullspace 中的误差首尾差最大仅 1.08e-14，证明重复观测只能降可见方向的噪声，不能纠正不可见的初始幻觉。操作上应记录方向可靠度 q(v)=vᵀGv/(vᵀGv+λ)，只在受到新证据支持的方向纠错；未证实纹理可以持续保存，但不应被计为恢复成功或反复提升置信度。

**持久化确实跨过空白时段。** 回访情境 t=4…11 没有该对象的新观测，所有算法在间隔内冻结输入，逐帧方法也不获得幽灵观测。间隔末帧的平均 PSNR 为 current_only: 20.96 dB；fixed_reference: 17.92 dB；short_window_map: 17.92 dB；persistent_evidence: 29.70 dB。4帧窗口到期后退回初始 proposal，而持续状态保留间隔前已经取得的证据。

**证据污染与真实变化必须处理。** 两次分散的错误身份观测会长期污染累计 MAP；预测残差隔离在多数 seed 中能防止这种污染。真实内容改变时，永不遗忘的 MAP 会把两个状态平均，连续矛盾后重启可快速追上新内容。

**单对象残差不能决定身份。** `persistent_mismatch` 与 `content_change` 在 t≥8 的输入 LR 历史逐值相同，只是任务评价真值不同：前者应保留原身份，后者应恢复新内容。因此任何只依赖此历史的确定性恢复器输出相同，都不可能同时在两种任务上可靠选对。这个对照不是要否定长期记忆，而是指向一个更强架构：保持原 identity state 不销毁，将冲突证据组成新 epoch hypothesis，再由带轨迹/几何/独立身份线索的检索器选择状态。仅把 gate 分数当成身份置信度会失败。

## 噪声压力测试

噪声测试使用 seeds 0…15；先验方差固定 τ²=0.04，未按结果调参。

| LR噪声sigma | 情境 | 持续证据 | 4帧MAP | 隔离重启 |
|---:|---|---:|---:|---:|
| 0.003 | complementary | 45.56 | 39.81 | 45.56 |
| 0.003 | repeated | 18.07 | 18.07 | 18.07 |
| 0.003 | intermittent_mismatch | 22.70 | 39.81 | 44.94 |
| 0.003 | content_change | 17.93 | 39.80 | 42.58 |
| 0.030 | complementary | 26.46 | 22.31 | 26.46 |
| 0.030 | repeated | 17.98 | 17.93 | 17.98 |
| 0.030 | intermittent_mismatch | 20.68 | 22.31 | 25.88 |
| 0.030 | content_change | 17.10 | 22.25 | 23.91 |

## 最强可行方法建议

建议将下一阶段核心方法做成 **保留身份的证据子空间更新 + 矛盾证据分叉 + 可检验的重定位**，而不是给现成 MAP 加一个名字：

1. 在冻结的流式生成底座旁建立 canonical object/patch identity state，并分别存储生成 proposal、已证实残差信息和观测来源/精度。
2. 新 LR 到来时，由可训练对应模块提取观测约束；仅更新其可支持的 latent 方向，旧生成结果不再充当新观测。通过退化一致性或小型局部算子近似落实 G/b 思路。
3. 高创新观测进入隔离分支；持续冲突触发可逆的新 epoch，而非覆盖原 identity。返回旧视角/轨迹时，可以用新 LR 对各分支做预测验证后取回旧状态。
4. 把身份选择所需的独立信息显式加入架构，例如局部几何、对象轨迹、前后锚点与遮挡可靠度。用本套件构造的成对不可区分输入作为失败测试，检验新线索是否真正解决选择问题。
5. 真正的论文贡献应在学习到的对应/观测子空间、带不确定度的分支记忆，以及实际生成模型上的稳定改进；需和全历史/更大缓存/检索参考/参数量匹配等强基线比。

## 实验进度、适用边界与后续

已完成：pilot 找实现问题；32 seed 主 suite；自然/解析分组；跨 seed bootstrap；MAP 等价与统计量重建审计；补充相位、尺度、重复观测、错误先验、遮挡回访、错误身份和真实变化；两个额外噪声量级。

尚不能据此声称：真实视频效果、生成质感、人脸/文字身份保真、准确光流下的泛化、训练架构的原创性或实时速度。当前稠密 G 是机制原型，576维时单份 float64 G 约2.65MB；真实空间分辨率必须用局部块、低秩或 latent basis 压缩并测试其误差。proposal 由第一帧 LR 构造，因此 MAP 指给定该经验 prior 的二次目标最优解；预测协方差是门控用的近似量，不能当成经过频率学校准的置信度。

下一步优先做实际冻结生成器的局部状态注入，并针对遮挡后回访、近景新证据、错误重定位和真实变化四类片段安排有 GT 的训练/评价。应同时报告旧细节保留、新细节收益、身份串线率、分支恢复速度、前缀与额外计算预算。不要把 repeated 情境的失败靠换指标隐藏，也不要在真实模型评估前收窄为只会证明经典 MAP 的数学结果。

## 可复现记录

```bash
python3 experiments/mechanism/run_suite.py --seeds 2 --n 16 --steps 16 --tag pilot
python3 experiments/mechanism/run_suite.py --seeds 32 --n 24 --steps 16 --tag main
python3 experiments/mechanism/run_suite.py --seeds 16 --n 24 --steps 16 --noise .03 --lam .0225 --tag high_noise
python3 experiments/mechanism/run_suite.py --seeds 16 --n 24 --steps 16 --noise .003 --lam .000225 --tag low_noise
python3 experiments/mechanism/analyze.py
```

主 suite runtime=59.16s（含绘图与 CSV 统计，不是每帧推理耗时）；Python 3.14.5 (main, May 10 2026, 10:21:34) [Clang 21.0.0 (clang-2100.0.123.102)]；numpy 2.5.1；scipy 1.18.0；matplotlib 3.11.1；Pillow 12.3.0。

全历史MAP与持续状态的最大逐像素差 0；独立重组统计量与递推统计量最大差 0。源代码 SHA256、机器信息、命令、数据 seed 和来源在每个运行的 metadata.json / audits.json 中。

核心结果文件：`main/per_frame.csv`、`main/summary.csv`、`main/paired_contrasts.csv`、`main/subgroup_summary.csv`、`main/audits.json`、`main/metadata.json`。

![主实验学习曲线](main/learning_curves.png)

![最终帧定性对照](main/qualitative_panels.png)
