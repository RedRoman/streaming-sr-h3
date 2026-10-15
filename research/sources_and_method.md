# 流式超分持续细节状态：源头核验与核心方法候选

核验日期：2026-10-15（Asia/Shanghai）。本文件区分作者报告、源码事实与本项目假设；没有运行以下大型模型，没有将摘要数字视为本项目结果。检索使用原始论文、作者项目页和官方仓库。

## 1. 源文档中的三项主要底座均真实

| 工作 | 原始来源与日期 | 已核验事实 | 对本项目的实际接口 |
|---|---|---|---|
| FastVR | [arXiv 2609.36757](https://arxiv.org/abs/2609.36757)，v1 2026-09-29；[论文全文](https://arxiv.org/html/2609.36757v1)；[官方仓库](https://github.com/chenxx89/FastVR) | 一步 diffusion、轻量 VAE、chunk causal attention。README 给出两阶段训练与推理工程；作者报告 H20 上 1080p 11 FPS。 | 适合完整底座、训练器与视觉参照。源码中的相邻 chunk 注意力并非可重定位的局部场景状态。README 的任意倍率输出不能自动等同于连续隐式场景解码。 |
| RelayVSR | [arXiv 2609.37850](https://arxiv.org/abs/2609.37850)，v1 2026-09-29；[论文全文](https://arxiv.org/html/2609.37850v1)；[官方仓库](https://github.com/kopperx/RelayVSR) | 稀疏大模型参考 latent + 小型 Dual-Memory Video Transformer，参考 KV 和近期视频 KV 分开。论文同时有 first-keyframe 与 dual-endpoint；README 当前发布推理。 | 最明确的外部细节读出接口。论文 dual-endpoint 的 29.29 FPS 在 A100 80GB、1080p、15 帧间隔；first-keyframe 报告 35.89 FPS。这些均不是本机测量。 |
| ReCaVSR | [arXiv 2609.37831](https://arxiv.org/abs/2609.37831)，v1 2026-09-29；[论文全文](https://arxiv.org/html/2609.37831v1)；[官方仓库](https://github.com/kopperx/ReCaVSR) | 前序预测 latent 回收、各层学习不同 KV scope、LR-conditioned FlashDecoder。当前仓库是推理工程，CPU 读写器缓冲全片。作者报告 A100 80GB 1080×1920 21.20 FPS。 | 适合“输出反馈 + 有限缓存”对照。需要额外实现可定位、可修订状态与实际流 I/O，不能把 recycled prediction 当新观测。 |

官方 README 已逐项查看：[FastVR](https://raw.githubusercontent.com/chenxx89/FastVR/main/README.md)、[RelayVSR](https://raw.githubusercontent.com/kopperx/RelayVSR/main/README.md)、[ReCaVSR](https://raw.githubusercontent.com/kopperx/ReCaVSR/main/README.md)。三者都要求 NVIDIA/CUDA 环境，不能依据代码可下载就认为能在当前 Mac 原样训练。

本次 GitHub API 返回的 main **tree SHA**（不是运行成功记录，也不是 commit SHA）：FastVR `b8ae45b9928b7f7e9ca90205687c738fca0edb3c`；RelayVSR `e856773350f53036d825404c9ea9228e6dffadca`；ReCaVSR `b98b1f2ec7fc550aed99fd7ec45a5e4b757e4306`。

### RelayVSR 接口已到源码层核验

[小模型源码](https://github.com/kopperx/RelayVSR/blob/main/refvsr_inference/models/flash/model.py) 中 `FlashVSRConditionNet.decode_step(lr_frame, state, condition_latent=..., end_condition_latent=..., total_frames=...)` 返回输出和下一状态。`prepare_condition` 将 `[B, condition_channels, H, W]` 端点编码成各层 KV，只允许 eval 且带 `no_grad`；`begin_stream` 合并两端点并建立新的 LR history。训练 adapter 时需要走可微路径或自行缓存 encoder 输出，不能直接期待 `prepare_condition` 回传梯度。

[运行器源码](https://github.com/kopperx/RelayVSR/blob/main/refvsr_inference/runtime/streaming.py) 的 `CollaborativeFrameStream.frames` 收齐 interval 后生成 end reference，再逐帧解码。默认公开运行器属于双端点有界等待；论文支持 first-keyframe 不代表直接运行默认脚本就没有未来输入。适配本项目时可以把已到达观测形成的状态渲染为参考 latent，但需要验证参考分布偏移；更换参考的时机、视频 KV 是否延续也必须作为机制消融。

## 2. 最接近前作与原创性边界

| 前作 | 已经覆盖什么 | 本项目需要另外建立的能力 |
|---|---|---|
| [Improving Video Super-Resolution with Long-Term Self-Exemplars](https://cqf.io/papers/Video_Super-resolution_ICRA2023.pdf)，ICRA 2023；早期版 [2106.12778](https://arxiv.org/abs/2106.12778) | 远隔帧同实例、跨尺度 patch 检索与参考融合。 | 单独提出“远处帧帮近处恢复”或“近景帮后续远景”不够；应更新同一局部生成解释，并区分已确认与未确认方向。 |
| [KEEP](https://arxiv.org/abs/2408.05205)，ECCV 2024；[全文](https://arxiv.org/html/2408.05205v1)；[官方训练代码](https://github.com/jnjaby/KEEP) | 用 Kalman 思想递归估计稳定的人脸生成 prior，结合 CodeFormer latent 和跨帧信息。 | “Kalman + 生成先验 + 时序稳定”不是新贡献。应比较对象一般化、长遮挡检索、可辨识性驱动的高频修订。 |
| [Video-to-Video Dynamic SR](https://link.springer.com/article/10.1155/ASP/2006/61859)，2006；[作者托管全文](https://people.duke.edu/~sf59/VideoEURASIP.pdf) | 动态 SR、递归观测融合、平移/共同模糊情况下的快速 Kalman 近似。 | 精度矩阵累积、在线最小二乘与 Kalman 递推是经典方法，必须作为强机制对照。 |
| [Robust Online Video SR Using an Efficient Alternating Projections Scheme](https://arxiv.org/abs/1909.00073)，2019/2020 | 在线视频 SR、创新残差鲁棒代价、保边正则、快速交替投影。 | 只用鲁棒残差门控或旧状态正则也不新；要给出稳定细节与可修订性的统一状态机制。 |
| [DDNM](https://arxiv.org/abs/2212.00490)，2022/ICLR 2023；[全文](https://arxiv.org/html/2212.00490v1)；[GAN Prior based Null-Space Learning](https://arxiv.org/abs/2211.13524)，2022 | 固定观测算子的 range/null 分解，数据一致性与生成细节分离。 | 不能把“只生成 null-space”当首创。可探索随时间新增观测而变化的局部可辨识子空间，以及生成样本跨后验的延续。 |
| [Sequential Posterior Sampling with Diffusion Models](https://arxiv.org/abs/2409.05399)，2024 | 以前序 diffusion 输出与学习到的时间转移初始化下一次 posterior sampling，减少采样步骤；应用于超声。 | warm start / 前序输出初始化不是新；需要持续的 canonical 细节状态、证据账本和跨后验样本耦合。 |
| [Video SR: All You Need is a Video Diffusion Model](https://arxiv.org/abs/2503.03355)，2025 | 无条件视频 diffusion 先验 + DPS 做 VSR。 | posterior sampling 用于视频 SR 本身不新。 |
| [LoCoVSR](https://arxiv.org/abs/2609.32742)，v1 2026-09-26 | 移动局部窗口 DPS，所有帧共享 noise trajectory；VFHQ 人脸。 | “共享噪声保证一致性”已经是直接前作。关键对照需要 shared seed / shared trajectory，而不只比独立随机噪声。 |
| [InfVSR](https://arxiv.org/abs/2510.00948)，v1 2025-10-01，当前 v3 2026-05-22 | 因果一步视频 diffusion、rolling KV、joint visual guidance、长视频评价。 | 单纯长视频稳定或更长 KV 不足以证明持久状态方法。 |
| [WorldMem](https://arxiv.org/abs/2504.12369)，2025 | 带 pose / timestamp 的记忆单元、state-aware attention，返回旧场景。 | 可定位场景记忆与回访能力也已有前作。区别应落在 LR 物理观测带来的细节证实/修订。 |
| [Latent Spatial Memory / Mirage](https://arxiv.org/abs/2606.09828)，v1 2026-06-08，v2 2026-08-27；[作者项目](https://microsoft.github.io/LatentSpatialMemory/) | 将 diffusion latent 深度反投影到持续 3D cache，直接 latent warp 读出。 | “latent 3D memory + read/write”不能单独主张原创；本项目可选 2D canonical patch 起步，但最终要比较状态更新的证据利用。 |
| [WorldTrace](https://arxiv.org/abs/2608.07408)，2026-08-07 | 给压缩 KV 分配训练域内虚拟位置，修复 RoPE 长时寻址；LoopBench 测回访。 | 检索失败与内容错误必须分开：更多记忆和地址可用不保证记忆内容能被新证据正确修改。 |
| [Keepsake](https://arxiv.org/abs/2610.06588)，2026-10-05 | 固定容量记忆按 pose/appearance 替代性动态保留与淘汰，training-free。 | 只做 novelty/redundancy 驱动 eviction 容易落入已有路线；本项目应研究内容本身的可修订性。 |
| [REEL](https://arxiv.org/abs/2610.16632)，2026-10-12 | 可训练检索/生成/更新 visual memory planner，并评价 identity 与预期状态变化。 | “retrieve/update memory”宽泛表述已很拥挤；物理观测似然、局部高频方向的证据支撑是更具体切口。 |

另有两篇相关理论工作：[A Posterior-Dynamics Framework for Imaging Inverse Problems with Pretrained Diffusion Priors](https://arxiv.org/abs/2608.15144)（当前题名与部分搜索索引旧题名不同）；[Posterior Information Dynamics of Diffusion Models for Linear Inverse Problems](https://arxiv.org/abs/2608.21709)。它们说明后验动力学、信息分配与 operator-prior alignment 都已有理论研究。以下公式是经典推导和本项目候选组合，不能仅凭检索未命中宣称首次。

## 3. 推荐主线：证据可修订的持久生成细节

积极假设：**一份细节状态可以同时维持未被否定的生成身份，并在新增信息真正约束某些方向时局部修订；其收益应超出固定参考、共享随机数和经典观测融合。**

这并不要求所有细节立刻正确。真正有吸引力的画面是：首次远景得到合理的织物结构；遮挡回来仍认得它；近景到来时只有原先不能确定的针脚得到修订；再拉远仍能读到被证实的同一结构。若真实内容发生变化，状态也应改写该区域，而不是永久锁定。

### 方向 A：跨后验的局部细节运输（首选）

状态条目：canonical 坐标、长期内容 key、已确认的观测信息 `J,h`、生成样本 `z`、局部不确定子空间 `U`、即时外观分支。LR 观测只在有对应且可见时写入；decoder 预测不新增 `J`。

已知线性映射原型可精确实现：`y_t=A_t T+epsilon_t`，先验 `N(mu0,C0)`，`J_t=C0^-1+Σ A_i' R_i^-1 A_i`，`h_t=C0^-1 mu0+Σ A_i'R_i^-1 y_i`，`mu_t=J_t^-1 h_t`，`C_t=J_t^-1`。同一物体回访关联到同一条目，而不是重新初始化。

不能只输出均值，因为正确均值可能抹平未知纹理。保留每个对象的一份生成解释，并把它从旧后验运输到新后验。高斯控制实验可以用 Bures/optimal-transport coupling：`x_new=mu_new+T(x_old-mu_old)`，其中 `T=C_old^-1/2 (C_old^1/2 C_new C_old^1/2)^1/2 C_old^-1/2`（数值上需正则化）。该运输和 Kalman 都是经典数学，作用是提供透明上界与学习目标。也应比较 `mu_t+chol(C_t)z` 的固定随机数耦合与精确随机条件更新，排除只是 random seed 带来的收益。

可学习架构：预测观测支持的低秩校正基与似然精度，先把新残差写入受支持方向，生成 decoder 只负责其余部分；通过短 rollout 蒸馏精确小 patch teacher，再用 HR 真值监督真实纹理。支持基、canonical correspondence 与 decoder 可逐段训练；冻结大生成器，通过 reference adapter 将当前状态送到 RelayVSR 类读出器。

主要可证伪预测：在相同前缀和生成调用预算下，A 同时降低回访高频漂移，并提高新近景之后的远景重建准确度；新增观测约束的方向改善，而剩余 nullspace 的无证据漂移下降。若只改善漂移却降低新信息利用，或精确 MAP/Kalman 同样达到全部收益，就尚未确立所需新核心方法。

### 方向 B：多解释状态与证据驱动的分支延续

大胆假设：初次低分辨率图案常有离散歧义，例如字符、编织拓扑或条纹相位。单一均值/协方差会把多种结构混合成模糊；只保留一个错误高频参考又会锁死。每个局部对象维护少量有共同低频解释的生成粒子，各自具有持续 ID；新真实观测通过可微退化似然更新权重，再对各分支做局部运输。输出分支延续用确定的选择规则，不能每帧重抽导致跳变。

先用 2–4 个程序纹理模式构造具有相同远景投影、但近景可区分的“歧义—揭示—回访”数据。比较固定单样本、单高斯状态、独立重生成、标准粒子滤波和 learned branch transport。评价正确模式命中率、揭示后切换延迟、纹理锐度及错误锁定率。不要把 particle filtering 本身当原创；潜在贡献是可检索局部生成分支、只用真实证据的运输以及流式读出的稳定机制。

若 B 在受控离散歧义中稳定优于 A，才进入真实图案和生成 latent；若 A 已能表达这种歧义，B 可并入 A 的非高斯扩展。该方向提供负机制结论的解法：若“单一持久样本会锁错”为真，给出多解释延续而非只报告失败。

### 方向 C：时间更新与尺度查询可交换的细节场

同一状态提供 `read(ROI,scale,pixel_footprint)`，查询只读、观测才写。用共享小波/多尺度系数与积分式像素 footprint decoder 起步，之后再训练连续隐式读出。核心实验是查询路径 `2×→8×→2×`、ROI 移开返回，以及同一批真实观测不同顺序的读取。输出降采样一致性必须与 HR 清晰度同时报告，防止靠模糊获得一致性。

强对照是每次独立 SR、固定共享金字塔、静态可连续解码状态，以及 A 在固定 4× 读出。只有 A 已建立“证据修订 + 持久身份”后才值得扩大 C；C 是可扩展的能力主张，不应在没有基础结果前替代 A。

## 4. 第一轮必须做的强对照与关键实验

### 共同对照

1. 单帧恢复，无状态，独立噪声。
2. 单帧恢复，共享噪声/共享 seed/共享 noise trajectory（LoCoVSR 风格机制对照）。
3. 固定高清参考与长期 LR self-exemplar；参考可检索但不更新。
4. 同预算 EMA/GRU 或更长 KV；相同真实观测、状态容量和大模型调用次数。
5. 精确累计 MAP、Kalman mean、Kalman posterior sample coupling；已知对应时使用精确算子，不人为削弱。
6. A 完整方法与去掉证据分离、去掉方向信息、去掉长时寻址、每次重抽样四个机制消融。
7. 参考读取 oracle：把已到达的正确参考送到同一 decoder，估计读出器能利用的上限；禁止未来/HR 真值作为正式方法输入。

### 数据与事件

| 场景 | 操作 | 判据 |
|---|---|---|
| 重复远景 | 同一有噪声独立观测/完全相同帧拷贝分别重复 | 区分独立测量的降噪收益与同一测量重复记账；重复拷贝不能虚增置信。 |
| 亚像素多相位 | 从只见一个 phase 到多个 phase | 根据堆叠算子谱验证真实新信息是否增秩；比较受约束高频方向误差。 |
| 遮挡后回访 | 10/50/200 帧间隔，干扰对象可相似 | 把 association failure 与内容状态漂移分开；先 oracle 对应，再自动。 |
| 远—近—远 | 近景真实 LR 包含更多世界细节 | 用近景到来前后**同一未来远景查询**评价 gain，输入前缀完全一致。 |
| 光照/高光变化 | 纹理固定、即时外观改变 | canonical 纹理身份保持，观测外观不冻结。 |
| 内容确实变化 | 局部图案替换、物体状态改变 | 更新延迟、错误旧纹理残留、未变区域保留；给 B 的分支/变更机制机会。 |
| 退化/匹配失配 | 模糊核、噪声、光流误差可控 | 获得稳定收益范围与失效曲线；不能只报完全已知算子。 |

评价至少包含：PSNR/局部误差、感知距离（未使用 LPIPS 就不得把其他距离标成 LPIPS）、对应后高频漂移、reveal 后 gain、观测残差、状态内存与额外时间。漂移与准确度分两轴，不能只合成一个有利分数。数据以对象/纹理源划分训练与测试，不把同一纹理的不同视频前缀分到两侧。

必须保留零等待或明确有界等待协议。审查因果性可用未来帧扰动：给定同样输入前缀，改变所有未来帧，检查已输出结果与状态一致；查询只读可用前后状态 hash。所有 HR 只作离线监督和评价，oracle 映射单独标明。

## 5. 从机制实验走向可投稿工作的决策

当前最有价值的是先跑 A/B 的受控实验，尽快看清“稳定身份”和“有证据时修订”能否同时成立。精确线性 Gaussian 只是机制锚点，不能以其成功宣称真实生成 SR 已解决，也不能因其属于经典统计而提前否定整个方向。

形成足够强主线应有：一个可学习且可部署的状态架构；真实视频上跨对象稳定效果；与上述强对照的同预算优势；A 或 B 的可辨识性机制证据；一个直观长回访/新近景演示；可复现训练与系统成本。若困难集中在非高斯锁定，转向 B；若集中在匹配，增加显式局部关联的联合训练；若状态收益完全能由经典融合解释，则继续寻找生成表示与后验运输之间的新机制，不能只给旧方法换名字。

本文件没有声称已证明原创性或已达到 CCF A 论文工作量。它给出了三条可以直接实现、被强对照证伪、并按证据扩展的路线。
