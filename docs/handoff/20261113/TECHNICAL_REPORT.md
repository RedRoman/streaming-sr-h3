# streaming-sr-h3 / Omni restoration 技术交接报告

日期：2026-11-13。范围：从 `archive/modelarts-20261103` 的 `43efbd4d090056365dc46f0ac6edb88e3d875761`，到本次交接前已归档的实际实验。本文不新增实验，不把计划、CPU fixture 或进程退出成功当作质量结果。

本文所有 `code/`、`configs/`、`preparation/`、`results/` 路径均相对仓库根目录。文末证据索引给出可直接检索的文件；表中 `[E..]` 是该索引编号。运行位置、最新资源与传输状态以同目录 `runtime_ops_snapshot.json` 和运维交接文档为准。`TAKEOVER.md` 是追加式历史记录，旧段落的 running / pending 已可能被后面的终态取代；也有部分早期 readiness JSON 仍保存历史阶段状态，不能覆盖最终验收。

## 1. 当前结论与必须保留的状态区分

项目已经形成可核验的预训练加载、源级数据合同、RGB 恢复器、固定生成器对照、完整三步 H3→VAE 反传和独立数组审计基础，但**没有完成具有质量优势的 omni video restoration 方法，也没有完成正式 guide-matched LoRA 训练**。

| 工作线 | 已实际完成 | 当前科学 / 工程边界 |
|---|---|---|
| 旧 archive SR | 旧 MiniMax-H3 + Stage-1 LoRA / Turbo / Rφ / innerDC 的多组历史实验 | 旧数据、旧模型和旧统计口径，不是当前 Open-H3 结果；部分理论措辞需收紧 |
| Open-H3 SR pilot | refiner + Turbo 底座，独立 rank8 LoRA，raw-guide **300 步**，8 来源配对开发评估 | 单独 sigma 的 latent 目标；LPIPS 改善，PSNR/时序非一致改善；不是正式三步 RGB 轨迹训练 |
| Rφ RGB | v1 三臂各 1000 步；stage2 三恢复器各 3000 步，另冻结恢复器的 1500 步方差拟合 | 去噪明显有效；去压缩、clean 保护和跨任务统一收益未通过 |
| 固定 H3 | 1 来源四任务 + identity/near-clean；raw/Rφ guide、固定 α=.25 soft anchor；VAE-only 与缓存补偿 | 生成造成明显保真损失；补偿无统一收益；没有扩成多来源质量结论 |
| RGB 可微工程 | 小尺寸 joint；全 22×480×832 三步 H3+VAE 同图反传；一次真实 TRAIN AdamW 更新及保存子集 CPU 审计 | 全尺寸梯度连通已验；真实 AdamW 原 run 超时失败，严格 reload + merge 补验尚未完成；不能晋升为部署可用 |
| P1 历史效用 | 新 mask-aware 恢复器、25 来源修订数据、teacher 1000 步、完整 headroom 和四 mask 诊断 | 原 headroom 门失败；原定三个 student 臂**没有启动** |
| 输入保留 | 1745 参数 raw-only head；matched / shuffled 各 1000 步；完整 2880 行开发评估；两次 TRAIN 内部诊断 | 两头全选 K1，screen 失败；balanced-Huber 后总体区分度接近随机，按预声明暂停此支线 |

计数不能混写：**历史 SR raw-guide LoRA=300 步；真实 TRAIN 单次 AdamW 工程更新=1 次且原 run failed；正式 guide-matched、多任务、完整 RGB 轨迹 LoRA=0 步**。[E05,E19,E29]

截至本轮运维快照，迁移包下载已失败，不能继续称 RUNNING；但原 31112 主机有新的空闲候选卡快照，原机补验可能比迁移更直接。资源快照不等于准入或已经启动。已有 generic runner 支持 saved-AdamW continuation，接续缺的是新鲜资源检查、具体 run plan/admission 和最终结果，不是从零实现 launcher。本文不根据瞬时空卡推断任务状态。[E30]

## 2. 三代系统与代码地图

### 2.1 旧 archive：以 SR 的观测约束分解为中心

旧路径大致为：低分辨率退化视频 → causal Rφ 预测较干净的低频/观测域目标 → H3 keyframe conditioning → 采样过程中 innerDC → VAE decode → 分块拼接。主要入口：

| 模块 | 文件 | 含义 |
|---|---|---|
| 低频恢复与描述子 | `code/range_branch.py`、`code/omni_stage1.py`、`code/operator_desc.py`、`code/operator_estimator.py` | 观测堆叠残差 CNN；后续增加 FiLM、operator one-hot/MTF/severity 条件 |
| 退化/目标生成 | `code/make_ops.py` | SR、噪声、模糊等合成协议；target 的定义随旧任务而异 |
| 生成与投影 | `code/h3_comfy.py`、`code/pixel_consistency.py` | Comfy MiniMax-H3，像素域约束经 encode/decode 回写采样的 x0 |
| 部署与评估 | `code/run_deploy.py`、`code/deploy_eval.py` | 分块生成、指标、数据集选择；必须显式绑定 EVAL_SET |
| 旧训练 | `code/train_stage1.py` 的 legacy 分支 | 冻结 DiT + LoRA，旧 velocity / latent 训练路径 |

旧 `REPRODUCE.md` 列出的主要配置为 20.1B DiT、rank32 Stage-1 checkpoint、旧 8-step Turbo 权重，但实际报告使用 2 步 sigma `[1, .923077, 0]`。这与接管后 Open-H3 refiner/Turbo/native3 是不同实验。旧说明中的“流式结构已验证”也不能直接转移到新 full22 block H3 入口。[E01]

### 2.2 接管后：显式 RGB Rφ + 冻结生成器因素对照

```text
原始观测（SR 先映射到目标 HR 网格）
     ├─ causal K3 RGB Rφ ── RGB R / Q4(R) / 可选误差方差
     ├─ raw guide 预处理 + VAE encode ── 固定 warm-start draft
     └─ raw 或 Rφ guide 预处理 + VAE encode ── H3 conditioning
                 冻结 refiner+Turbo、同 noise/seed/native3
                            ↓
                     H3 video/audio latent
                            ↓
                      冻结 VAE decode
                            ↓
             原生成 RGB / 固定 soft anchor / 缓存诊断
```

`RphiModel` 本身不调用 H3；`rphi_soft_anchor.py` 不调用生成器，也不使用 confidence。固定 H3 的 guide 与 warm start 是两个不同输入：换 Rφ guide 时 **raw warm start 保持不变**。这使 guide 对照有明确含义，但也意味着 VAE(Rφ) 和 H3-Rφ 不是完全相同的 latent 起点。[E06,E09,E11]

### 2.3 后续研究模块与可微接口

| 文件 | 实际用途 | 不应推断的能力 |
|---|---|---|
| `code/openh3_train_trajectory.py`、`code/openh3_differentiable_euler.py` | 保留部署三步 joint video/audio 语义，绕开只供推理的 no_grad，支持受控 block checkpoint | 不等于已经完成长期 LoRA 训练 |
| `code/vae_checkpoint_context.py` | decoder forward/recompute 分别进入 native math，退出后恢复 | 不保证任意尺寸/任意 offload 配置都适用 |
| `code/rphi_history_utility.py` | 11 通道 mask-aware Rφ、1745 参数历史 gate、LOO 效用 | 模块存在不等于三个 student 臂已训练 |
| `code/history_utility_revised_data.py` | 修订 19/6 cohort 与原数值协议之间的显式 authority 桥接 | 不把旧 32/8 research-ready 标志改为 true |
| `code/rphi_input_preservation.py` | raw-only 单图收益头，raw/K1 固定阈值硬选择 | 未校准为概率；实际 screen 已失败 |
| `code/h3_evidence_checks.py` | 成功断言不重复全量 JSON 刷盘，失败仍保存并抛错 | 未修复旧超时 run，也不代替补验 |

## 3. 旧结果还能支持什么

### 3.1 旧 77 realistic 子集成绩

以下是 archive 自身记录的均值，保留为历史基线，不与当前 8 来源 pilot 或 19/6 mixed cohort 横向相减：

| 旧 arm | LPIPS↓ | PSNR↑ |
|---|---:|---:|
| InfVSR | .2344 | 23.28 dB |
| FlashVSR | .2253 | 23.42 dB |
| FlashVSR+ | .2072 | 24.23 dB |
| 旧 ours 2-step | .1929 | 25.78 dB |

证据：`REPRODUCE.md` 的 “77 realistic clips” 表、`results/DELIVERY_20261103/README.txt`。[E01] 该批结果有价值，但需要连同以下约束使用：

- 77 子集均值、后续 96 样本报告、总体中位数差和逐来源配对中位数不是同一统计量。不得仅凭相同 clip 名合并。
- 旧 FlashVSR+ 桥接曾使用 CRF10/yuv420p 的 MP4 输入/输出，我方路径使用 PNG；需要无损桥接或共同 codec 控制后才是严格公平比较。旧 MVSR4x15 的 LPIPS 我方 .407777、FlashVSR+ .360262，是保留的负结果，不应被 synthetic SR 主表掩盖。[E02]
- RealVSR 的既有协议含对 LQ 额外 ×4 下采样，不等同原生真实 ×4。原始录制视频作为参考也不自动成为无压缩、无噪声的 pristine GT。[E02]
- 旧计算延迟记录为约 .733 秒/帧；每 22 帧块 12.99 秒对 .708 秒预算，约慢 18.3 倍。它不是实时系统，也不是新 native3 的性能结果。[E01]

旧 audio 结果同样应保留范围：交付文件记录 77 clip 的 mel 28.247→8.718 dB、视频 77/77 bit-identical；`REPRODUCE.md` 另有 12 clip 实验。它们不能互相替代，也没有重新证明当前 Open-H3 噪声 audio 路径具有音频恢复能力。[E01]

### 3.2 range/null 的数学边界

必须区分两个实现：

1. `band_energy.py`、`error_decomposition.py` 的低频为 `U(D(x))`：D 是 `bicubic, align_corners=False, antialias=True` 下采样；U 是 bicubic 上采样，未指定 antialias。一般 `D U != I`，所以 `U D` 不是幂等正交投影，`x-U D x` 也不能直接叫严格 null-space 分量。能量交叉项不必为零。
2. `pixel_consistency.py` 从同一个浮点 bicubic 算子矩阵构造右逆，仿射更新 `x + A†(y-Ax)`，再以固定 25 次 Dykstra 处理 `[0,1]` box。这是另一条明确的数值投影路径。有限迭代、dtype、Pillow uint8 resize 与浮点算子的差异仍需记录剩余残差。

因此，历史 `null_energy/null_cos/GT-range` 可以重述为**指定滤波器下的 detail residual / GT lowband substitution 诊断**；不能解释为正交能量分解或可观测/不可观测误差的严格比例。普通 RGB PSNR/LPIPS 并不因此自动失效。新的 metric v2 已同时保留兼容字段和较准确命名。[E03,E05]

固定 bicubic 也不天然排斥 omni：可以把目标明确为 canonical `D0(x)`，让 Rφ 从各种退化观测估计它；但这不是声称 `D0` 就是实际物理退化。旧 deblur 数据 target 保留模糊算子输出、部署仍走固定 bicubic 等接口差异，需要按具体任务解释，不能直接组成统一物理逆问题。新 RGB Rφ 统一预测原 RGB，再确定性输出 `Q4(rgb)`，解决的是目标定义的一致性，尚未建立未知物理算子的精确 DC。[E03,E06]

### 3.3 不要把局部否定结论扩成不可能性定理

- `FROZEN_mapper_feasibility.json` 的某个 latent-target mapper 相对 roundtrip control：PSNR −1.167 dB、LPIPS +.0559，75/76 clip PSNR 更差。它强烈反对该目标/mapper 配置，不能支撑文件中“disqualifies the whole class”这类泛化。直接对原始 RGB 的完整轨迹监督仍是不同假设。[E04]
- `FROZEN_mtf_boundary.json` 支持的是所测 descriptor、训练 operator 家族和插值/外推条件下的边界。per-sample blur MTF 的实现修复后仍未稳定优于 one-hot；one-hot 对词表外 x6 报错不是等预算盲恢复质量比较。不能据此认定连续条件无价值或 MTF 已实现未知混合退化识别。[E04]
- `PROTOCOL_mech.md` 已承认生成素材上“跨块无信息”的否定在真实素材不成立。文档仍有“运动补偿一律有害”等过强措辞；已测 flow/重采样/数据上的负结果，不排除更可靠对齐或不同历史融合。接手者应保留数据，不继承无条件结论。[E01]

## 4. Open-H3 迁移：已经接入什么、没有接入什么

主 recipe 是 **`refiner-turbo-comfy`**。配置 `configs/openh3_20261111/refiner_turbo.json` 固定完整 bytes/SHA：

| 角色 | 当前主路径中的资产 | 实际状态 |
|---|---|---|
| base | `minimax_h3_fl2va_pruned_bf16.safetensors`，40,225,724,176 bytes | 冻结 Comfy FL2VA 底座 |
| refiner | `open_regen_2k_refiner_rank384_bf16.safetensors`，5,868,409,512 bytes | 已转换、验证、合并进主底座 |
| Turbo | `openh3_turbo128_comfy.safetensors`，1,956,171,936 bytes | 已分离/验证、与 refiner 按声明顺序合并 |
| VAE | `minimax_h3_video_vae_fp16.safetensors`，5,207,808,496 bytes | 当前主路径；没有切到上游 fast VAE |
| text context | 固定 neutral restore context，299,557 bytes | 缓存上下文，非逐样本自由 prompt 搜索 |
| Sharp | release-components 可选 recipe，非主实验 | loader 只接受已证明 curve-B 全零的受限路径；未实现任意非零 curve runtime |
| upstream latent upscaler / fast H3 VAE | 资产库存存在 | 主线未接入、未验收为实际训练组件 |

预训练合并覆盖 50 DiT blocks + 2 token-refiner blocks 的 208 个 Linear；按 FP32 累加各预训练低秩增量，再 cast 一次。此后冻结，额外注入 **rank8 adaptation**，目标是 50 DiT blocks 的 200 个 Linear、400 个 FP32 A/B tensor，B 零初始化。旧 Stage-1 与旧 Turbo 不叠加，loader 会拒绝不符 recipe/shape/coverage/hash 的资产。[E07]

当前三步 schedule 是 `[0.909375, 6/7, 2/3, 0]`。sub-unit 起点使 raw draft 以 `(1-σ)` 的系数实际保留；若 σ=1，warm-start 内容会被完全消掉。video/audio 的官方 packing、audio sigma 变换及 NestedTensor noise 不能用一个简化 video-only 噪声公式代替。[E07,E16]

**这不是上游 Diffusers 等价复现。** 当前是 ComfyUI FL2VA port、neutral text、噪声 audio、已有 FP16 VAE、自有 raw draft/训练目标。主分支明确记录 `upstream_diffusers_parity=not_validated`。Open-H3 名称说明预训练权重来源，不代表采用了上游所有组件、调度细节或训练损失。[E07]

### 4.1 已完成的 300 步 SR pilot

`train_stage1.py` 的 pretrained 分支在三个 native sigma 上轮转，用 draft 构造 `x_t=(1-σ)draft+σε`，预测 `x0_hat=x_t-σv`，对 target latent 回归。它不是把三步 Euler 展开后对最终原始 RGB 监督；旧可选 perceptual 项也不能自动视为当前原始 RGB 目标。[E05,E07]

| 8 来源开发均值 | pretrained baseline | rank8 / 300 步 | 适配后−前 |
|---|---:|---:|---:|
| PSNR | 25.73653 | 25.79208 | +.05555 dB；仅 3/8 来源提高 |
| LPIPS | .290132 | .268046 | −.022086；8/8 来源改善 |

配对 PSNR 95% 区间 `[-.28528,.47943]`，配对中位数 **−.15206 dB**；LPIPS 区间 `[-.039013,-.009227]`。detail 与 temporal 诊断非一致改善，不能称保真/时序全面提升。这批输入是 existing ×4 development windows，7 个 realistic + 1 个其他类型；不是 77 旧子集，不是 native 四任务，不是冻结 test。[E05]

`draft-source=pixel_upsample_guide_fallback` 是显式 fallback，不能标成已经调用 pretrained latent upscaler。训练 manifest 排除了已知历史 eval 来源，但这些开发数据后来又用于设计，独立测试身份不会因目录名叫 val/heldout 而恢复。

## 5. Rφ RGB 与 stage2：受控结构变化及负结果

### 5.1 结构和损失

`RphiModel(variant, k=3, ch=64, blocks=8, condition_dim=6, observation_encoder=...)`：

- 输入 `[B,3,3,H,W]`，最旧帧在前、当前帧在后。因果性由 loader 的 frame indices/availability 保证，模型本身不能从像素识别“把未来帧塞进过去槽”的错误。
- 9 通道卷积 → 8 个 FiLM 残差块 → 零初始化 RGB residual tail；`rgb = current + residual`，训练输出不 clamp。
- known：公开 task one-hot（sr/deblur/denoise/compression）、`log2(scale)/3`、known mask；不含真实 severity。
- blind：原观测与相邻差分产生 16 维连续 code，不读 condition/隐藏退化 metadata。legacy 在卷积前 pool 到最多 64×64；native 保留原像素尺度，先两次 stride2 卷积再全局汇聚，参数 shape 相同。
- `anchor=Q4(rgb)` 是确定性 canonical bicubic antialias 下采样，不是独立学习的 range head。
- confidence：共享 feature detach 后的局部 log-variance head；Gaussian NLL 使用 detached 实际 unclamped RGB error，范围 `[-12,0]`，独立优化。没有用 learnable confidence 缩小 fidelity loss，也未接入 H3 DC。

fidelity 采用 Charbonnier ε=.001 + .1 edge-L1 + guard×相对输入的损伤约束。数据有 identity / near-clean 控制，各期望 12.5%。代码先构造共享主干再构造 variant 分支，以相同 seed 保证初始共享权重一致。[E06]

### 5.2 v1 的教训

known / blind / blind_confidence 各 1000 步均数值完成，checkpoint reload 通过，但 blind 与 confidence 的最终 fidelity 不一致：CPU 实物比对在 1000 步 relative-L2 约 6.9353%、max_abs .0433484。代码上没有确认 confidence 梯度流入主干，不能直接归因某个 NPU kernel，也不能把两臂 RGB 差异称作校准收益。stage2 改成从同一个**冻结** blind checkpoint 单独拟合 confidence。[E08]

known 的四任务变化主要来自去噪：相对输入 PSNR 为 +.0355/+ .0855/+8.6177/+ .0875 dB。blind 的 compression 为 −.3020 dB；这也是不把“统一接口”当作“所有任务恢复有效”的理由。[E08]

### 5.3 stage2 的可比性与结果

三恢复器均 final3000，32 train / 8 dev；legacy_g05→native_g05 仅改变条件编码像素尺度；native_g05→native_g20 仅把 guard .5→2.0。数据、来源、初始主干、crop、优化器、步数对齐。评估 stride4，每臂 8 来源×4 任务；不是相邻帧完整 flicker 结论。下表是各任务**相对输入**的来源均值：

| arm | SR ΔPSNR / ΔLPIPS | deblur ΔPSNR / ΔLPIPS | denoise ΔPSNR / ΔLPIPS | compression ΔPSNR / ΔLPIPS |
|---|---:|---:|---:|---:|
| legacy_g05 | +.13100 / −.002901 | +.22951 / −.003137 | +9.89805 / −.376838 | −.00464 / −.000143 |
| native_g05 | +.11927 / −.002336 | +.21118 / −.002562 | +9.74382 / −.375291 | −.01819 / −.000146 |
| native_g20 | +.02886 / −.000441 | +.05685 / −.000489 | +9.00964 / −.337680 | +.04488 / −.000179 |

证据：`rphi_stage2_scientific_review_v2.json/per_arm`。[E08] 原像素编码是有理由的接口修正，但当前质量假设没有获支持；不能因为设计更合理就默认晋升 native。强 guard 减轻损伤，也牺牲大部分 SR/deblur 纠正。

| 控制 | legacy_g05 | native_g05 | native_g20 |
|---|---:|---:|---:|
| identity MAE（0–1） | .00098777 | .00100341 | .00050998 |
| identity 受损来源 | 8/8 | 8/8 | 8/8 |
| near-clean ΔPSNR | +1.42101 dB | +1.39837 dB | +1.58149 dB |
| near-clean 任一 draw 受损来源 | 2/8 | 2/8 | 0/8 |

identity 输入严格等于参考，旧 PSNR=120 是数值 cap；不要用与120相减得到的巨大负 dB 夸张表示损伤，使用实际 MAE/MSE。

冻结 confidence 的 1500 步拟合与 native_g05 fidelity SHA 相同，全部已记录 RGB 指标差 0，这是隔离成功，不是恢复收益。控制上的 1σ 覆盖约 .9987–.9989，远高于高斯名义 .683；不能直接当可信动作概率。它拟合的是 Rφ 自身误差，不是“修改比不修改更好”的概率，也不是实际观测噪声方差。[E08]

后续 fixed-H3 选择可解释的 legacy_g05 final3000，checkpoint SHA `61ff3a2f696033791f2168410a15f02fc2f8b25274d8fb0c62a0e8cbf7a3f617`。这一选择不是宣称它是 omni 最优模型；8 dev 已用于决策。[E08,E09]

## 6. 固定 H3 因素对照、clean 保护与 VAE 表示瓶颈

### 6.1 为什么先不用旧 innerDC

旧 innerDC 的 49-step 落点 `[28,45,48]` 映射到三步后都落在最后一步，修正后不再有后续 DiT 融合。native HR 观测若仍套同尺寸 bicubic consistency，会退化为全帧 identity 型投影。因而新对照冻结 H3，先隔离 conditioning，再做显式 post-decode soft anchor。[E03,E09]

固定六臂为 input、Rφ-only、H3(raw guide)、H3(raw)+anchor、H3(Rφ guide)、H3(Rφ)+anchor。两次采样同 raw draft、seed101、权重、三步预算；anchor α=.25，在同一生成结果上 CPU 后处理：

- SR：`x + α A†(Q4(R)-Q4(x))`，Q4 与右逆数值对应；
- native：`x + α(R-x)`；
- 最终 clip/dtype 造成的残差另报。无 confidence、无物理观测精确一致性、无新 LoRA。[E09]

### 6.2 ordinary 与 controls 的实际结果

ordinary 只跑来源 `6UZVovkR3IU` 四任务，全22帧、stride1、21相邻 pair，共8次 sampler /24 NFE。Rφ guide 改善该例 denoise，相对 raw guide 的 SR/deblur 增益较小，compression 更差。最终 Rφ+anchor 的 PSNR 在四任务都低于 Rφ-only；不能用跨任务均值藏掉压缩损伤。[E09]

compression 同源示例：

| arm | PSNR | LPIPS |
|---|---:|---:|
| input | 40.38018 | .016127 |
| Rφ-only | 40.47587 | .015294 |
| H3 raw | 32.84482 | .032329 |
| H3 Rφ | 32.67219 | .032874 |
| H3 Rφ + α=.25 | 34.79369 | .026362 |

证据：fixed-H3 review，以及同 input/target 绑定的 native baseline 汇总 `per_task/compression/reference_arms`。[E09,E13]

接着跑了同一来源 identity/near-clean×四 task 标签，共8 records、16 sampler /48 NFE。identity 的四标签原输入/Rφ/无 anchor H3 输出相同，不能算4独立来源；near-clean 是同源四个 task-keyed σ=1/255 draw。全部 H3 arms 都损伤 controls；α=.25 只是缓解，不是保护通过。没有自动扩成8来源H3矩阵。[E10]

### 6.3 VAE-only 的两路诊断

为分清表示损失，固定同 VAE 分别做 `D(E(raw))`、`D(E(Rφ))`，12记录×两路，0 H3 sampling。实际 latent SHA 与先前 H3 guide 路径匹配，确保比较同一编码；只在新 VAE 实例 hook finalizer，保存真正 **pre_internal_clamp**，并硬验 `clamp(capture)==原decode返回值`。旧 H3 RGB cache 是 **as_returned_internal_clipped**，不能改名当作 raw decoder 输出。[E11]

identity（native anchor 口径）显示：

| 路径 | MAE | LPIPS |
|---|---:|---:|
| 原输入 | 0 | 0 |
| Rφ-only | .00114454 | .00155075 |
| VAE(raw)，内部 clipped 返回值 | .00915168 | .01913737 |
| VAE(Rφ)，内部 clipped 返回值 | .01032913 | .01908016 |
| H3-Rφ | .01523973 | .02882411 |
| H3-Rφ + native α=.25 | .01148615 | .02120403 |

这支持“当前 VAE 路径本身已有显著保真损失”，不支持把每项 MSE 差写成独立可加的归因百分比，也不证明 VAE 是任意方法不可突破的下界。H3-Rφ 用 raw warm-start，而 VAE(Rφ) 直接解码 Rφ latent；两者不同轨迹以及误差交叉项仍在。[E11]

### 6.4 缓存 RGB 补偿没有统一解决问题

固定全 grid β∈`{0,.05,.1,.25,.5,.75,1}`，比较：

- scalar：`R + β(G_R-R)`；
- compensation：`R + β(G_R-V_R)`，`V_R=D(E(R))`。

12 records×2 families×7 β=168条，0新H3/NFE/训练。β0两者严格等于R；scalar β1为G，compensation β1为`clip(R+G−V)`；native scalar β=.75对应旧α=.25 anchor，但SR不是这个等价关系。[E12]

denoise β=.25：scalar PSNR36.89489/LPIPS.079337，compensation35.84783/.113465，R为35.91716/.113663。去掉的并不全是无用失真。compression β=.75 的补偿比scalar好，却仍低于R；identity所有正β仍损伤原clean和R。相同β也不等于相同修改幅度。没有选“最好β”部署，没有把当前源的最优点变成下一默认参数。[E12]

## 7. 从 latent 训练到真实 RGB 轨迹：已闭合与未闭合的工程门

### 7.1 为什么不能直接在旧 train_stage1 加一个 decode loss

问题包括：推理 sampler 和 wrapper 的 no_grad；joint video/audio packing；真实 raw warm-start；σ / x0 语义；Comfy patcher/offload；native VAE 的内部 clamp；不同 grad-mode 的 kernel 分派；checkpoint backward 时上下文恢复；FP16 梯度下溢；完整模型同时驻留的显存。

新 `ResidentH3Trajectory` 保持模型完全驻留，使用真实三步轨迹，分别给 DiT 和 VAE forward/recompute 设置 math context；不让一个全局 VAE native-RoPE replacement 穿过整条 H3 backward。checkpoint closure 在进入时复制可变 `transformer_options`，避免所有重算误用最后一个 block_index；异常中途也必须恢复 forward wrappers、patcher state 和 math functions。[E16]

### 7.2 VAE 单模块验证历程

| 问题 / 实验 | 实际观察 | 正确结论 |
|---|---|---|
| 初始 grad/no-grad parity | max_abs 约 .002856，反传前失败 | no-grad in-place native RoPE 与 grad 路径被换成 eager，数学路径需绑定，不放宽门 |
| full22 非 checkpoint grad | allocated58.22GiB / reserved60.18GiB 附近 OOM，0 backward | 小尺寸成功不能外推全尺寸 |
| decoder block checkpoint |36个block；小尺寸与full尺寸分别验 | full480×832 的108 block调用来自3个**空间 tile 行**，不是3个 temporal chunks；fullH3驻留可能改变native tilebatch，必须实测 |
| full22 原始 unscaled 梯度 | 一次路径全零；同图 scale1024 后非零 | 固定 loss scale 是明确数值协议，不可静默变更 |
| A1/A2 fresh-process repeatability | 同输入/forward SHA相同，梯度不逐位相同 | 旧 strict repeatability 门保留 failed，不等于训练必定无效 |
| 固定 seed102 latent descent | 8 forward/3 backward，固定step RMS .001，三步loss下降 | 证明该局部方向可优化，不证明 adapter / Adam 稳定或恢复质量 |

A1/A2 unscaled 梯度 maxdiff `3.7252903e-8`、relative-L2 `.0019251851`、cosine `.9999981468`；256个共同非零坐标反号，能量占比很小，但这不能推出 Adam 的稳定性。原定 paired-history B 因严格门失败未运行。独立 seed102 descent loss `.1147418246→.1143500432→.1139582545→.1135659069`，峰 allocated9.13175GiB。[E14,E15]

### 7.3 H3 同图、非零与真实更新不是同一个门

1. **小 joint RGB**：5帧128²，zero rank8 adapter；reference与candidate latent/video/audio/RGB parity、一次RGB backward及冻结/恢复通过。synthetic target只用于数值检查。[E16]
2. **小 nonzero SGD**：12NFE/4decode/1backward/1SGD/保存/清空重载/merge全部执行并独立审计。lr=1e−4，200个B tensor确实变化，但zero/updated/reloaded/merged video/audio/RGB全byte-identical。FP32理想增量极小，BF16合并只有91个权重坐标变化。接受的是参数持久化与声明流程，不是功能影响；synthetic checkpoint不可作研究warmstart。[E17]
3. **全22 joint**：真实TRAIN来源`2BdYaGuVAKo`的compression工程样本，22×480×832；reference3NFE+candidate3NFE、2decode、1RGB backward、**0optimizer**。video/audio/RGB全byte-exact；400 adapter梯度对已审，200个B非零，unscaled B L2 `.00036848238`；峰 allocated55.43597GiB、reserved59.86328GiB，217次decoder context恢复。该源后来被确认有自然切镜，因此仍只用于工程，不是连续退化primary质量证据。[E18]
4. **真实 AdamW 一步**：同真实RGB目标，新的固定AdamW协议。zero→updated未合并RGB实际改变；已保存400adapter与1200optimizer state。原run在3600秒超时，停在clear/reload inventory，reload和merge输出没有生成。其后独立CPU审计3222实际数组，验证unscale/clip/AdamW方程/保存子集，不能追认整个原run成功。[E19]

真实更新同样本的 NumPy RGB MSE `.0030889882946→.0030982169629`，增加约 .299%；PSNR25.10184→25.08888dB。**参数更新有效但这一次输出没有质量改善**，不能用“梯度非零”代替质量判断。[E19]

超时的静态候选原因是旧成功 `require()` 也保存整份 pretty JSON，clear/reload与optimizer inventory约5600次成功检查导致大量重复I/O；没有逐行时间戳，不能证明这是唯一原因。新helper成功不save、失败先记录再raise，显式阶段进度仍由调用方写；旧run/source/pins保持不变。[E20]

### 7.4 当前精确下一门

已实现独立 saved continuation，复用真实已保存状态，**6NFE/2decode/0backward/0update**：fresh加载→strict unmerged reload输出与原updated输出逐字节相同→再merge并描述BF16舍入与端到端差。不能重新做一次更新来替代原保存点，也不能偷偷增lr使“效果可见”。unmerged FP32 branch与`W+BA`后castBF16具有不同舍入和乘法结合，非零merge不应预先要求或宣称全局bit-equivalence。[E20,E30]

只有补验闭合后才讨论有限正式训练。全尺寸单次反传近60GiB reserved，不证明optimizer长期内存、batch扩展、全多任务cache或多步稳定性已经通过；加载/首次编译耗时也不能当稳态step/sec。

## 8. 数据谱系、连续性筛选与信息预算

### 8.1 不要混淆这几套数据

| 数据阶段 | 来源 / 窗口 | 已做工作 | 限制 |
|---|---|---|---|
| archive heldout | 原96/realistic77等多manifest | 历史评估与反复诊断 | 已参与设计，不是新冻结test |
| SR takeover pilot |32train /8dev，120 latent windows | train96 /dev24 windows，RGB评估8来源 | 只SR raw-guide，不自动涵盖原生四任务 |
| 四任务RGB pilot |32train /8dev；每源22帧 |40clean +160observation NPZ，四任务同target | 原始录制proxy；source排除已知eval，仍是开发集 |
| 96帧工程2+2 |2train/2dev，3native profile，双replay | loader/codec/seed/crop/角色工程通过 | 发现2Bd多次自然cut，不可直接做连续退化primary |
| target-only40筛查 |原32/8角色不变 |38符合冻结source-frame协议，2不合格；全帧图及疑点审查 | 数值eligible不等于视觉连续 |
| 修订primary |**19train/6dev** |25primary、13diagnostic、2ineligible；100NPZ实际审计 | 显式新cohort，不补凑32/8；无新独立test |

修订cohort SHA `c4c3a1c044fa13e6af93c03259ca0e52ea18531ddb6a2c55bbd19abff81d44e1`。25来源的3组旧profile严格复用，22来源在已verified target上做metadata bridge及3profile×2replay=132codec，**0新MP4 decode**。source SHA、target file/pixel SHA、windowkey、数值合同与新execution authority分别绑定；原receipt的continuity/research-ready字段不被改写。[E21]

两个低FPS来源的不合格指冻结4秒窗口中实际 source frame数不足96；fps filter可以产生96输出，但不能假装有96原生独立帧。只保留这个协议资格结论，不称源视频普遍无效。视觉筛选先冻结policy，再看全部96缩略帧、top6候选pair和必要原分辨率邻帧；运动、闪光、遮挡不直接当cut。疑点仍不清时留diagnostic。target-only旧失败a与恢复b均保留，完成来源按file/pixel SHA复用，不重解码掩盖失败。[E21]

96帧混合profile为 native blur→noise→H264 的 static/smooth/abrupt 变化，另有identity和near-clean；当前没有完成退化顺序交换的因果研究。2Bd已知cut将可用连续片段上界压到3.9039秒，低于96帧24fps所需95/24秒span，不盲目换窗伪造primary。[E21,E28]

### 8.2 信息预算要单列

| 方法/阶段 | 时间信息 | operator/监督信息 |
|---|---|---|
| blind Rφ / P1 gate | 当前+最多两过去帧，无future；边界availability显式 | 只raw观测；GT仅训练/评分，不作gate输入 |
| known Rφ | 同K3 | task和scale公开条件，无GT severity |
| H3/VAE full22 | 整个22帧块，块内未来信息；不是严格逐帧因果 | 固定neutral context，raw/Rφ guide、同rawdraft |
| FaverNet/AverNet当前native smoke | offline双向22帧，最大future21 | pretrained zero-shot，未做同等训练/参数预算 |
| GT-range / mask oracle / candidate oracle | 离线目标访问 | 只能诊断，不可部署，不可用于test时选β/threshold |

当前所有96crop结果都不能直接外推到整帧接缝、跨块长视频、真实时序漂移。六dev已多次用于设计，任何泛化表述都需要独立source/group-disjoint冻结test；库存里存在未纳入当前名单的视频，不等于其历史、质量、去重和使用资格均已通过。

## 9. P1 历史边际效用：做了新模块，但原研究门失败

### 9.1 机制与合同

`MaskAwareRestorer` 从 legacy_g05 继承58旧tensor，把首层9RGB输入扩为11通道（加两个历史权重图），新增1152首层参数置零，保留初始输出。缺失历史以current填充且显式标记weight0；不能把任意zero/duplicate视为真实过去帧。模型继承实际CPU输出通过，不仅是shape测试。[E22]

teacher固定1000步、batch16、seed20261110，训练mask分布覆盖缺失/局部/软权重；AdamW和Charbonnier+.1edge+.5guard沿既有合同。持久化16000行crop/source/profile/control/mask日程，不让loss影响采样；student计划使用独立固定seed20261111的同等1000×16日程。teacher冻结后，某历史槽的LOO效用是固定另一槽、固定所有像素/模型，用删除该槽造成的tile MAE差监督；GT只在label路径。[E22]

计划三学生臂是无效用辅助、真实效用、打乱效用，同容量、同输入和budget；打乱donor需跨source与source_group，并保持预声明strata。teacher训练、label生成和审计成本都需额外计入，不能只说student相同步数就称端到端计算完全匹配。

### 9.2 teacher1000和headroom实际验收

teacher1000已完成，16000行、58参数/174AdamW状态finite、finalcheckpoint严格重载，约433.81秒，峰NPU1.049GiB。smoke checkpoint没有作为正式warmstart。[E22]

headroom在6dev×5profile×96中心96²crop上比较all/K1/remove0/remove1，2880行，720batch模型调用、11430实际crop forward（边界无效LOO跳过）。180NPZ×6数组约1.91GB保存，独立NumPy重算1080数组/指标，最大差约6.94e−17。**唯一失败门是source-balanced degraded history gain**：`MAE(K1)-MAE(all)=-.0002606927`；4/6来源正，两个负来源主导均值，不能按“多数来源正”改门。[E23]

| profile | raw unclamped MAE | all-history MAE | K1 MAE |
|---|---:|---:|---:|
| abrupt blur/noise/H264 | .02494421 | .01905371 | .01882529 |
| smooth blur/noise/H264 | .02377536 | .01882442 | .01850553 |
| static blur/noise/H264 | .02124642 | .01796547 | .01773069 |
| identity | 0 | .00202832 | .00205973 |
| near-clean | .00259297 | .00285482 | .00291299 |

这是“恢复器对退化输入有收益，但平均加入历史反而弱于K1，clean也损伤”的结果，不是teacher完全无用。原三学生没有准入，未追加teacher步数、未挑中间checkpoint。[E23]

### 9.3 既有四mask诊断为何转向输入保留

在同一批保存输出上，GT四候选tile oracle相对K1 MAE收益 `.0002609831`，约 **1.422%**；两槽GT-LOO符号策略达到该候选收益的 **93.249%**，有正regret的tile约12.249%。这个量级不支持优先增加复杂联合四状态router。[E24]

四mask最优仍有identity MAE `.00187999`、near-clean相对raw损伤 `.0001489354`；加入raw为第五候选的secondary oracle在near-clean约44.19%tile选择raw。因此优先单独测试输入旁路有依据，但oracle使用GT，只是给定候选集合诊断，不证明raw-only可预测性，也不是任意soft-gate上界。五候选结果也不能直接当作raw/K1两候选上界；后续cache另算了真正两候选诊断。[E24,E25]

## 10. 输入保留：完整训练、评估和停止依据

### 10.1 冻结接口

`InputPreservationHead` 精确复用P1的raw feature及1745参数网络，不加新feature。每历史槽的11维tile feature来自current/past/absolute difference RGB与lag/availability；共享两槽网络后对logits取mean，缺失两槽仍允许依据current做决定。head不输入K1输出、GT、task或severity metadata；K1只作为最终候选。[E25]

label为FP64 tile benefit：`MAE(raw,target)-MAE(K1,target)`，以`.25/255`归一化并clip[-4,4]。输出score>0选K1，score≤0（含±0）保raw；nearest16+`torch.where`严格逐字节选支，无平滑/羽化；另存unclamped与clipped结果。此score不是概率或已校准不确定度。

### 10.2 cache和matched/shuffle实验

- TRAIN19：固定student16000行，冻结teacher K1实际1000批forward，0恢复器更新；保存K1、raw-only features、标签，raw/target通过已认证原NPZ重建，避免重复巨大K3 payload。
- DEV6：2880行复用已保存K1；完整rawK3重建必须逐行 observed SHA相等，失败不换源或fallback；0新restorer。
- 两缓存各自经CPU独立实际数组重算。TRAIN7000arrays、DEV1440新arrays与1080旧RGB，标签FP64误差仅数值归约量级；不是只看manifest complete。[E25]
- 两head各固定final1000，CPU FP32；同initial、同1000输入batch、同loss/optimizer、标签唯一差异来自whole-map donor。donor按profile/phase/effective-availability分层且跨source/group，不能沿用旧history-lag slotmap。[E25]

### 10.3 完整开发screen失败

2880行×raw/K1/matched/shuffle完整评估，Alex LPIPS v0.1使用预置两权重SHA，720调用/11520距离，无网络fallback；source/profile等权，identity/near-clean独立；180批score/mask/LPIPS数组与原RGB保存供独立CPU审计。2850对stride1相邻mask，16tile边界与非边界梯度均记录，不冒称光流flicker。[E26]

**matched和shuffled的103680个tile全部选K1，2880行clipped/unclipped RGB都与K1逐字节相同。** 新head对K1或shuffle的收益严格0；flip0来自常数全1决策，不是时序稳定性胜利。

| source/profile宏均值 | raw | K1 = matched = shuffled |
|---|---:|---:|
| 三native profile PSNR | 29.09718 | 31.08496 |
| 三native profile LPIPS | .15249456 | .09686566 |
| identity clipped MAE | 0 | .00154814 |
| identity LPIPS | 0 | .00249051 |
| near-clean PSNR | 48.53855 | 47.97560 |
| near-clean LPIPS | .00288145 | .00373010 |

native恢复收益来自冻结K1，不属于新head。near-clean clipped MAE略降也不能覆盖PSNR/LPIPS损伤。`6rJrrfRO3cA / abrupt / frame44` 的MAE `.01234114→.01438985`、LPIPS `.03941832→.12937805`；这是混合blur/noise/H264，不是纯去压缩benchmark。所有完整性通过与screen失败同时保留。[E26]

### 10.4 TRAIN-only 两次诊断与支线暂停

TRAIN标签576000tile中正标签73.7436%、负26.2564%，无零；zero处SmoothL1梯度−.32029，全局常数最优+.606486。原两臂日志step2起999/999 batch全K1，支持“目标总体正先验主导”的候选解释，但不能证明原图没有信息。[E26]

第一次诊断按source_group冻结5fold，仅19TRAIN、不读6DEV；共享两slot rawfeature取mean得到11维tile-local输入，fold-train内标准化，Huber β1、ridge1e−3、固定200GD、解析1/L。source宏常数loss **1.14531581**，线性 **1.14741963**，8/19来源略好但总体更差；所有fold仍100%选K1。小线性模型不包含原3×3非线性上下文，且教师已见过全部19来源，因此不是整个恢复器的独立泛化测试。[E27]

第二次是严格单变量 **balanced-Huber**，不是logistic：保留fold/std/features/y/β/ridge/200步/zero阈值，只令fold-train正负tile各占总权重.5；先用旧系数重放并严格核score SHA，再新拟合，不用heldout标签估训练权重。

| source宏指标 | 旧Huber线性 | balanced-Huber |
|---|---:|---:|
| balanced accuracy | .500000 | .501422 |
| AUROC | .511037 | .510326 |
| 负标签误选K1 | 1.000000 | .530613 |
| 正标签K1召回 | 1.000000 | .533458 |
| 同一未加权Huber loss | 1.147420 | 1.245646 |

减少负样本误选几乎等量牺牲正召回；near-clean BA .574678 / AUC .631390，但三native BA .473815/.495428/.471488，均低于.5。identity单类AUC为null并报有效来源分母，不硬填.5。运行86.26秒、峰313.1MB、0DEV/恢复器/NPU；宏统计独立复算最大差4.44e−16。[E27]

决策是**暂停当前保护头支线**，不增加旧head步数、不调6DEV阈值、不把weight-only结果包装新方法收益。它否定的是已测试表示/模型/目标下的可用证据，不是否定所有raw信息或更强时空表示。未执行的balanced-logistic草稿不能计入实验数量。

## 11. 外部基线与公平比较缺口

FaverNet-TUDCommon和AverNet的单源native三任务 **6/6 已实际完成**，不是仍待部署。使用同`6UZVovkR3IU`原RGB/观测/target绑定，全22帧stride1、相同LPIPS权重和GT-proxy flow支撑口径；无损RGB bridge避免新的编码混淆。[E13]

| 任务 | FaverNet PSNR / LPIPS | AverNet PSNR / LPIPS |
|---|---:|---:|
| deblur | 34.49857 / .058969 | 34.95929 / .055747 |
| denoise | 39.94093 / .029446 | 40.73593 / .028487 |
| compression | 40.25884 / .016031 | 40.27300 / .015863 |

这是一来源zero-shot工程/质量smoke，不是多源排名。compression两者PSNR都略低于input40.38018，不能只说它们比当前H3好。Faver/Aver拥有offline双向future21，而Rφ-only是causalK3；H3同为full22但参数量/算力不同。CPU单次forward耗时受平台与首次运行影响，不是正式速度benchmark。[E13]

下一公平表应冻结方法×任务×来源×窗口×future budget×训练数据/更新×输出codec/clip×时间/显存；只复用input/target/window/hash完全一致的已有格子。FlashVSR+适合SR/效率；native基线不能改尺寸冒充×4SR。公开论文/仓库已在计划里做过定向调查，但本交接没有新增文献检索，不能据计划提到某模型就说它已运行。[E02,E28]

## 12. 创新性：当前有假设和否证链，没有已成立的新方法贡献

预训练替换、LoRA、FiLM、误差方差、soft DC、两阶段恢复/生成、因果输入或时变退化标签各自都不足以单独构成创新。既有计划已记录与相对修复风险、选择性历史传播、因果路由等方向的相关工作重叠；该记录不等于本项目完成全面novelty证明，更不等于SOTA。[E28]

目前最有价值的研究进展是把问题拆成可证伪链：

| 假设 | 实际检验 | 当前判断 |
|---|---|---|
| raw→Rφ条件能让冻结生成器普遍更好 | 同rawdraft六臂 + controls | denoise例有用，但compression/clean严重损伤；不能普遍推广 |
| VAE表示误差解释大部分损伤，直接相减可修复 | 两路roundtrip + 全β缓存对照 | 表示损失确实存在；补偿没有统一方向优势，误差非独立相加 |
| 历史边际效用值得学习 | mask-aware teacher→headroom→四mask oracle | 历史平均门失败、有限选择余量小；原三学生未启动 |
| raw-only小head能保留clean且保留纠正收益 | matched/shuffle1000 +完整screen→TRAIN5fold→weights-only诊断 | 全K1塌缩；线性权重平衡总体近随机，当前支线暂停 |
| 直接原RGB轨迹损失能训练H3 adapter | scoped可微三步→VAE反传→full22→一次AdamW | 工程连通成立，非零输出变化已实证；严格部署补验和多步质量尚未成立 |

P2“退化顺序非交换性”目前仍主要是计划，不应列为已训练模块。必须先证明固定blur/noise/codec顺序互换产生可恢复且有用的信息，再测试预测顺序；“知道顺序有用”与“盲观测能识别顺序”是两门。P3频带残差已有诊断/缓存比较，不能仅换成频带融合名称就声称新方法。[E28]

## 13. 可解释的下一步顺序

1. **完成已有真实AdamW保存点的reload/merge补验。** 复用已审保存子集，新namespace，0update/0backward；不重跑原failed run。generic bounded runner已存在；先决定原31112还是已完整验证的迁移环境，不能双机重复执行。通过前保持deployment_ready=false。[E19,E20,E30]
2. **冻结正式guide-matched训练合同与cache覆盖。** 已有train-role raw/Rφ guide、controls和单源compression VAE-only cache工程证据，不等于全部train/dev任务都已VAE编码。逐样本明确raw draft、Rφ guide、原始RGB target，target不得走restorer输入。固定pretrained/adapter/encode处理与最终三步轨迹，不能从旧单sigma latent目标冒名迁移。[E29]
3. **有限、同预算的真实RGB LoRA研究。** 固定raw-guide与Rφ-guide、同初始base/LoRA、同rows/noise/steps/损失/最终checkpoint；先测实际每步内存与吞吐，再决定有界步数，不按小smoke外推。必须同时评价Rφ-only、固定H3、identity/near-clean、压缩负例，报告clipping与更新/merge影响；不能只看训练loss下降。
4. **补多来源同输入基线和独立测试资格。** 当前所有主要质量结果仍开发集；旧6条native smoke只按完整hash复用。source/group历史、近重复、target质量、自然cut、真实退化与合成任务覆盖先审，不能把原dev改名test。
5. **保护/历史支线暂不循环调参。** 只有在新信息表示、不同恢复器候选质量或新source-disjoint数据构成明确新假设时，才冻结独立计划；保留当前失败门。不得因为新的CPU诊断好于常数就直接放宽dev保护条件。

## 14. 接手时的证据和执行纪律

- **读取优先级**：最终accepted/failed审计与原report → 实际manifest/数组SHA → 冻结execution contract/source snapshot → 设计plan → 时间线文字。历史plan的pending不能覆盖终态，反之文件存在也不能证明执行。
- **保存scope**：本地small archive通常只有JSON/log/小数组；大型NPZ/weights在远端。local_verification如果只核archive和metadata，就不能改写成“本地重算全部梯度/图片”。LPIPS独立审计常为保存distances重聚合，不是第二次模型forward。
- **复现scope**：source hash、实际文件bytes、array dtype/shape/pixel SHA、source_id/group、split、windowkey、frame index、crop、availability、seed、数值recipe、teacher/checkpoint、code/environment都需绑定。relative runtime path与私有权重不随GitHub自动提供。
- **数值scope**：unclamped MAE、clipped PSNR/LPIPS、内部VAE clamp、最终blend clamp分开；零MSE用null+exact或明确旧cap。相同输出SHA检查须区分float数值相等与bytes相等（包括signed zero）。
- **实验scope**：final-step固定；no threshold/β/checkpoint挑最好；identity重复标签去独立计数；source宏和profile宏先定义，不能按frame/tile海量数冒充来源量。
- **资源scope**：新run、旧pins不改、不覆盖、无自动OOM retry；卡健康/无PID/显存/共享lock/15秒二检；CPU限制包括owned process group，超时仅终止本任务组。CPUfixture通过不等于真实NPU或整数据已验。
- **历史失败保留**：source-count、路径绑定、OrderedDict重载、旧metadata I/O超时、strict梯度重复性等失败不是删掉重做；新版本必须说明单变量修复与复用证据。

## 15. 仓库内证据索引

以下路径是本文的主要事实来源；相邻目录中的原report、MANIFEST、archive receipt和local verification用于继续追溯。

| 编号 | 路径 / 入口 |
|---|---|
| E01 | `REPRODUCE.md`；`PROTOCOL_mech.md`；`results/DELIVERY_20261103/README.txt` |
| E02 | `TAKEOVER.md` 的“研究主张与基线范围复核”；`configs/openh3_20261111/omni_benchmark_plan_20261111.json` |
| E03 | `code/band_energy.py`；`code/error_decomposition.py`；`code/pixel_consistency.py`；`code/make_ops.py`；`code/h3_comfy.py` |
| E04 | `results/FROZEN_mapper_feasibility.json`；`results/FROZEN_mtf_boundary.json` |
| E05 | `results/takeover_20261111/sr_pilot_300_metrics_v2.json`；`sr_pilot_300_baseline_metrics_v2.json`；`sr_pilot_300_train_log.jsonl`；`sr_pilot_300_queue_status.json`（后三项同目录）；`configs/openh3_20261111/pilot_queue_30794.json` |
| E06 | `code/rphi_model.py`；`code/rphi_condition_v2.py`；`code/train_rphi.py`；`code/rphi_data.py` |
| E07 | `configs/openh3_20261111/refiner_turbo.json`；`code/openh3_assets.py`；`code/openh3_runtime.py`；`code/lora.py`；`code/train_stage1.py` |
| E08 | `results/takeover_20261111/rphi_completed/health_summary.json`；`results/takeover_20261111/rphi_stage2_first_pair_review.json`；`results/takeover_20261111/rphi_stage2_scientific_review_v2.json` |
| E09 | `results/takeover_20261111/fixed_h3_legacy_smoke_review_20261111.json`；`code/run_fixed_h3_rphi.py`；`code/rphi_soft_anchor.py` |
| E10 | `results/takeover_20261111/fixed_h3_controls_review_20261111.json`；`results/takeover_20261111/fixed_h3_controls_completed/engineering_review.json` |
| E11 | `results/takeover_20261111/vae_roundtrip_v2_completed/scientific_review.json` |
| E12 | `results/takeover_20261111/cached_rgb_compensation_completed_20261111/scientific_review.json` |
| E13 | `results/native_quality_completed_20261111/summary.json` |
| E14 | `results/takeover_20261111/vae_gradient_rope_dispatch_audit_20261111.json`；`results/takeover_20261111/vae_gradient_repeatability_20261111i/scientific_review.json`；同目录`local_verification.json` |
| E15 | `results/takeover_20261111/vae_latent_descent_20261111j_completed/scientific_review.json`；同目录`local_verification.json` |
| E16 | `code/openh3_train_trajectory.py`；`code/openh3_differentiable_euler.py`；`code/vae_checkpoint_context.py`；`results/takeover_20261111/h3_joint_rgb_probe_20261112a_completed/scientific_review.json` |
| E17 | `results/takeover_20261111/h3_nonzero_adapter_probe_20261112b_31112_completed/scientific_review.json` |
| E18 | `results/takeover_20261111/h3_full22_joint_rgb_probe_20261112a_completed/scientific_review.json`；同目录`root_acceptance.json`、`independent_review.json` |
| E19 | `results/takeover_20261111/h3_real_adamw_failed_20261112a/root_review.json`；`results/takeover_20261111/h3_saved_subset_completed_20261112b/root_acceptance.json` |
| E20 | `code/h3_evidence_checks.py`；`results/takeover_20261111/h3_evidence_checks_independent_review_20261112.json`；`code/check_h3_adamw_saved_continuation.py`；`results/takeover_20261111/h3_adamw_saved_continuation_independent_source_review_20261112.json`；`results/takeover_20261111/h3_adamw_saved_continuation_b_independent_source_review_20261112.json` |
| E21 | `configs/openh3_20261111/history_utility_revised_cohort_20261112.json`；`results/takeover_20261111/history_utility_target_screen_completed_20261112b/engineering_review.json`；同目录`root_visual_review.json`、`architecture_visual_review.json`；`results/takeover_20261111/history_utility_full_cohort_readiness_20261112/known_cut_coverage.json`；`results/takeover_20261111/history_utility_revised_dataset_completed_20261112a/root_acceptance.json` |
| E22 | `configs/openh3_20261111/history_utility_contract_20261111.json`；`code/rphi_history_utility.py`；`code/train_history_utility.py`；`results/takeover_20261111/history_utility_checkpoint_integration_20261112a/local_review.json`；`results/takeover_20261111/history_utility_teacher_completed_20261112a/root_acceptance.json` |
| E23 | `results/takeover_20261111/history_utility_headroom_completed_20261112a/root_review.json`；`code/eval_history_utility_teacher.py` |
| E24 | `results/takeover_20261111/history_mask_oracle_diagnostic_completed_20261112a/root_review.json`；同目录`report.json` |
| E25 | `code/rphi_input_preservation.py`；`code/export_input_preservation_cache.py`；`code/train_input_preservation_heads.py`；`configs/openh3_20261111/input_preservation_screen_20261112.json`；`results/takeover_20261111/input_preservation_cache_completed_20261112b/local_verification.json`；`results/takeover_20261111/input_preservation_train_cache_audit_completed_20261112a/root_acceptance.json`；`results/takeover_20261111/input_preservation_dev_cache_audit_completed_20261112a/root_acceptance.json` |
| E26 | `results/takeover_20261111/input_preservation_eval_completed_20261112a/root_acceptance.json`；`code/eval_input_preservation_heads.py`；`preparation/input_preservation_eval_actual_audit_20261112a/audit_actual.py` |
| E27 | `results/takeover_20261111/input_preservation_train_feature_probe_completed_20261112a/payload/report.json`；同目录`split.json`；`results/takeover_20261111/input_preservation_balanced_huber_probe_completed_20261112a/root_acceptance.json`；同目录`payload/report.json`；`preparation/input_preservation_balanced_huber_probe_20261112a/scientific_plan.json` |
| E28 | `configs/openh3_20261111/parallel_research_plan_20261111.json`；`configs/openh3_20261111/mixed_temporal_data_plan_20261111.json`；`configs/openh3_20261111/input_preservation_followup_20261112.json` |
| E29 | `configs/openh3_20261111/guide_matched_lora_interface_readiness_20261111.json`；`results/takeover_20261111/h3_next_training_gates_20261112.json`；`results/takeover_20261111/rphi_train_guides_legacy_smoke_completed/engineering_review.json`；`results/takeover_20261111/rphi_train_controls_legacy_smoke_completed/engineering_review.json`；`results/takeover_20261111/train_vae_cache_portable_source_review_20261112.json` |
| E30 | `docs/handoff/20261113/runtime_ops_snapshot.json`；本目录运维交接；`preparation/omni_followup_runtime_20261112b` |

本文没有改变任何已冻结实验的判据，也没有以新文档授权重跑已完成任务。下一 agent 应从实际保存点和未闭合的最小门接续，而不是从早期 README 的成功摘要重新开始。
