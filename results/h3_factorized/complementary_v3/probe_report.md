# 互补 near 协议的自然图像线性可辨性审计

8 个 BSDS 官方 val 源、256×256、两个相反的 near 顺序，已在 CPU 实际执行。四个 128 crop 左上角为 (32,32)、(32,96)、(96,32)、(96,96)，near2×，far4×，相同 Gaussian0.6 + bicubic；noise σ0.003。这里先保持中性 appearance、无遮挡，严格只使用 prefix LR，HR 仅用于合成传感读数与评估。没有 H3 执行、没有网络训练、没有实际运动视频结论。

| 四次互补 near 后 | 全图 RGB PSNR | 历史 near union ROI PSNR |
|---|---:|---:|
| 当前 far | 24.1773 | 23.9022 |
| last-near 直接支持残差 | 25.0263 | 25.4103 |
| last-near 线性 inverse | 25.0295 | 25.4051 |
| 对齐 residual atlas 均值 | 26.0950 | **27.7765** |
| 全历史线性 inverse | **26.1132** | 27.7532 |

按 source 内先平均两个顺序，atlas 比 last-near direct 的 union ROI 提高 2.3661 dB，8/8 源；全历史 inverse 提高 2.3429 dB，8/8 源。只有一次 near 时 atlas 和 last-near direct 完全相同。第2/3/4次 near 后，atlas 的 union ROI 分别领先 last-near 1.2942/2.1880/2.3661 dB。说明下一轮多 near 任务确实要求融合；大部分增益已由经典残差融合解释，不能作为学习核心的证据。

全历史 inverse 解固定当前 far 的零空间残差，48 步 matrix-free CG，ridge=.01；prebox far-anchor 差 RMS 最大 4.87e−10。初始 ridge=1e−4 的原始 val pilot 放大 near/far 相容性弱方向的噪声，已原样保留在 `linear_probe/`，不作为强控制。随后先写 `baseline_calibration_plan.json`，仅在4个 TRAIN 源上比较 [.001,.01,.1]，以第4步 union ROI 选择 .01：train分数27.5138/30.8872/29.9582 dB；再一次性用于8val，未用val选择ridge。三个校准结果均保留。该校准只强化经典 baseline，没有改变 B2 的结构、损失或预算。

实际用时：初始 pilot10.66秒，train校准三次合计13.91秒，固定选定配置 val11.36秒。代码 `experiments/h3_factorized/probe_complementary.py`，命令/来源hash/版本/逐prefix数据在各目录 `complete.json` 与 `rgb_metrics.csv`。下一步仍需原生 H3 E/D 容量检查，以及跨 appearance、遮挡、内容变化时的强控制；此处尚未运行新学习架构。
