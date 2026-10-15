# 流式超分：持续且可修订的细节状态实验报告

更新日期：2026-10-15（Asia/Shanghai）。研究仍在进行。本文件区分已完成实验、正在运行的训练，以及尚待检验的方法假设。原始目标保持不变：确立原创性足够强、效果稳定的新核心方法，并推进到有充分实验支撑的论文工作量。当前尚未达到这个目标。

## 用户指定的主底座

用户于本轮明确要求“尽量用h3基础模型完成”。现有工作区权重清单确认H3指MiniMax-H3；主方法转为优先使用H3基础模型，FastVR/RelayVSR作为对照。优先复用BF16 FL2VA基础权重、FP16视频VAE及已缓存条件；已有量化/加速变体需单独标记，不能代替基础模型质量结论。小模型训练仍用于机制诊断，不能当作H3训练已完成。

## 当前判断

**积极主线：将细节状态拆成真实观测证据、持续生成解释和可逆的身份分支。新信息负责修订有证据支持的方向；身份冲突暂存为新分支，避免破坏旧身份。** 正在检验这种结构是否能超越经典历史融合、共享噪声与同预算循环网络。

截至本轮最新结果，H3采样内固定观测约束在8源native480上相对VAE的PSNR有7/8源改善、LPIPS有8/8改善，但平均时序残差略高。全50层BF16主干、真实反传与三控制各400次适配已完成；适配仍存在PSNR/LPIPS取舍。A的条件响应压缩尚未建立有效的早压缩、晚消歧任务。B4七控制和B5四控制均已完成各自固定actual-D训练与终评：B5把实测细节直接送入像素读出后平均提升2.678dB，但学习部分没有超越相同零训练底图，移除神经历史输入几乎不影响结果。最新真实视频延迟证据诊断用预先固定的4帧后来LR判断，6源有5源判对更新利弊，冻结状态平均PSNR从25.522升到25.702dB；对应失配的舞蹈片段仍误接受。这个正面信号支持继续研究可重读的历史证据，尚不构成原创核心。Relay全部24case/2112条评价也已完成。下文保留实验演进背景，最新状态以本节及后面的新结果为准。

已完成实验支持长期证据积累的价值，并找到了会改变架构的两种失败：不可辨识的错误细节不会因重复看见同样低清画面而自动纠正；单条低清创新残差不能可靠区分身份串线与真实内容变化。对应方案已经进入实验，而非只留下负结论。普通累计 G/b 就是经典 MAP，不作为新方法。

已生成基于完整JSON的结果总览图：`results/progress_overview/evidence_overview.png`（PDF同目录）。长流四源的固定事件面板全部已目视复核；提升较小，细字与小目标仍明显低于HR，不能仅凭平均分数推断丰富细节恢复。

## 进度与证据

| 工作 | 状态 | 可核查证据 |
|---|---|---|
| 文献与官方接口核验 | 已完成首轮 | research/sources_and_method.md |
| 32-seed 机制主实验及两个噪声压力测试 | 已完成 | results/mechanism/mechanism_report.md；main/per_frame.csv、audits.json |
| 全局平移、模糊、下采样的精确 Fourier 账本 | 实现与数值核验完成 | src/fourier_ledger.py；results/fourier_ledger_audit.json |
| BSDS 自然图像全局算子实验 | 已完成首轮，修正窗口重复包处理后复跑 | experiments/fourier_stream.py；results/fourier_stream/ |
| 六种小模型、三训练种子与扩展评价 | 已完成18,000次optimizer step；属于前期机制诊断 | results/learned_core/FINDINGS.md；main/与extended/ |
| 非破坏性多解释状态与身份线索 | 32-seed实验与容量/旧版本回访完成 | results/branch_memory/branch_memory_report.md |
| H3基础模型接入 | BF16完成正常49次forward真实22帧视频；官方采样工作流逐值一致性通过 | results/h3_setup/dit_bf16_neutral_davis22_normal49/ |
| 原生H3 teacher与候选A | 严格FP32禁TF32缓存及两轮三seed完成；current-only已修复，学习transport未胜MAP；实际DiT guide五步接口通过但未胜经典near | results/h3_core/FINDINGS.md；LIVE_STATE.json |
| 内容／外观候选B | B1/B2三seed均完成；B2未胜直接near；B3互补收益可穿过H3，配平预处理后暂无额外非线性融合缺口 | results/h3_factorized/factorized_report.md；residual_v2/ |
| 完整H3主干SR适配 | v1各400更新与440帧评价完成；v2对齐官方BF16 IO后从同初始化完成全部1200更新、固定49采样及440帧评价/归档 | results/h3_setup/native_full50_gradient/；native_sr_pilot400_native_io_v2/ |
| 长流真实视频基线 | 四源×600连续帧HR/LR远端SHA全部通过；35块/源的原生H3 VAE与2400帧完整质量评价均完成 | data/long_video/；results/long_video_vae/ |
| 真实运动视频与对应接口 | 8源695帧校验完成；8×39帧LR-only flow控制已跑 | data/real_video/remote_verification.json；results/video_correspondence/REPORT.md |
| 真实视频历史反演强控制 | 6源×22帧、4方法固定参数完成；H3历史修订4/6源PSNR改善，val平均−0.37405dB；最终投影预算配平，GT观测误差诊断完成 | results/real_video_history_inverse/matched_final_projection/REPORT.md |
| B5历史信息直通与神经anchor四组结构实验 | 4×672实际D反传及8val终评完成；直通信息有效，学习历史仍无独立增益 | results/h3_factorized/factorial_v5/summary_seed17/statistics.json |
| 历史频带约束方向诊断 | 8val×7query纯CPU完成；硬near-row约束滤除大量真实高频方向，GT方向仅作诊断 | results/h3_core/historical_band_CPU_audit/FINDINGS.md |
| 后到低清证据撤回早期错误细节 | 冻结prefix21、未来22–37；固定前4帧gate完成，6源5判对，平均+0.18066dB；不是零等待输出 | results/real_video_history_inverse/delayed_lr_validation/REPORT.md |
| RelayVSR 官方推理与因果参考变体 | 6源×2视野×2协议24case、528PNG及2112条统一评价完成；LPIPS改善与保真/时序取舍明确 | results/relay_baseline/cohort/；experiments/relay_baseline/ |
| 原创方法在真实视频上的稳定优势 | 尚未验证 | 必须继续完成，不能由受控实验替代 |
| 连续尺度与流式时空读出 | 尚未开展方法训练 | 保留为原项目的后续实验路线 |

## 已完成：持续细节机制

协议为已知对应和退化的 24×24 canonical patch、16 个因果时刻、8 情境、8 方法、32 seeds，产生 32,768 条逐帧记录；另有两个噪声强度实验。16 seeds 为解析纹理，16 seeds 是同一张 Matplotlib 自带照片的不同 patch，因此不能把这部分统计解释为多场景真实视频结论。

| 情境 | 四帧窗口 MAP | 持续证据 / 全历史 MAP | 隔离重启 | 解释 |
|---|---:|---:|---:|---|
| 互补相位 | 29.70 | 35.29 | 35.29 | 新方向的信息可积累；持续状态相对四帧 +5.59 dB，seed bootstrap 95% CI [5.44, 5.74] |
| 重复观测 | 17.95 | 17.95 | 17.95 | 持久化任意幻觉可能变差；当前帧方法为22.11 dB，未知细节不会被重复观测证实 |
| 错误生成先验 | 29.49 | 35.23 | 35.23 | 互补真实观测能纠正错误先验；固定先验14.34 dB |
| 间歇身份串线 | 29.70 | 22.67 | 34.64 | 永久累积容易污染；隔离可有效防止分散错误写入 |
| 真实内容变化 | 29.70 | 18.22 | 32.45 | 永不遗忘的账本将新旧内容混合，需要状态变更机制 |
| 持续身份串线 | 12.30 | 18.24 | 12.47 | 直接重启可能把错误对象当成新内容，不能只靠残差决定身份 |

可操作结论：保存观测出处和方向可靠度；对高创新观测先隔离，以可逆分支吸收；加入独立几何/身份线索后再决定合并、替换或取回。机制报告包含完整负结果、分组、置信区间与数据来源。

## 已完成：更真实的全局观测算子

Fourier 账本处理周期边界下的全局 Gaussian 模糊、整数亚像素相位和平移后下采样。它允许邻近像素耦合，区别于独立4×4单元的简化点观测。每个频率别名组维护16维信息矩阵，状态大小不随视频长度增长。独立构造完整像素空间算子进行核验：前向最大误差2.37e-16，与稠密MAP求解最大误差3.90e-14；重复包不增加证据，读取不写入证据。

BSDS 从官方 Berkeley 下载并校验：500张图，训练200、验证100、测试200；归档 SHA256 为 `97e49d31764f3912f0c4122707d53062ac9e783ba0f095e447a4d53c1a41af8e`。初轮全局算子实验用训练200图拟合频谱先验，测试前24张独立图的64×64中心patch，6种协议、16时刻、7方法。它是自然图像上的受控多帧逆问题，不是真实运动和匹配实验。

修正后结果：互补相位下全历史Gaussian MAP为33.289 dB，四帧窗口27.624 dB，按源图像配对增益+5.664 dB（bootstrap 95% CI [4.754,6.563]）；当前帧24.286 dB。Bures后验样本30.036与固定噪声样本30.018接近，配对差0.018 dB、区间[-0.064,0.102]跨零。每次重采样的平均平方变化为固定噪声的5.68倍，但该量也包含有意义的证据修订，不等于真实运动视频质量。后验均值PSNR更好，高斯随机样本未被证实为逼真细节。

完全重复packet时三个MAP控制均为24.425 dB，状态不变；同位置独立噪声观测积累后为24.527 dB，符合两类重复应分开的预期。真实内容改变时全历史17.00 dB、四帧窗口27.62 dB，再次指出必须处理状态变更。

**仅采用经典高斯运输或共享随机数不足以确立新核心方法。** 新方向继续由可学习的非高斯细节修订与身份分支承担。完整结果见 `results/fourier_stream/REPORT.md`。旧错误版本保留在 `results/fourier_stream_v1_window_arrival_bug/`，不得引用其中有限窗口重复包数字。

## 已完成：三种子可学习核心诊断

训练图像严格按BSDS源图像划分，真实高分辨率只用于监督与评价。首轮使用4倍、已知canonical point footprint的简化协议；6种模型参数约6万、相差小于3%，包括当前帧CNN、ConvGRU、证据账本CNN、持续修订模型、Gaussian MAP加学习残差、Gaussian MAP加持续残差运输。另有从训练图像拟合的3×3邻域144维Gaussian全历史MAP作为强基线。

模型以3个独立训练seed、固定训练预算运行；同一测试测量集用于配对。评价区分新相位、完全重复包、混合重复、近景揭示、错误对应及程序纹理。完全相同数据包与同位置的独立噪声测量必须分别处理；前者不能重复提升置信，后者有降噪信息。记录准确度、高频能量、观测误差和重复稳定性，不能靠模糊获得稳定分数。

三种子固定预算训练已结束：6模型×3seeds×1000步=18,000 optimizer steps，H100训练合计351.4秒，独立测试每个源图像一个48×48 patch。

| 条件 | 强Gaussian MAP | 无状态学习残差 | 持续学习残差 |
|---|---:|---:|---:|
| 一个已观测相位后重复 | 23.369 | 24.506±0.047 | 24.726±0.021 |
| 四相位后重复 | 27.783 | 28.201±0.071 | 28.303±0.017 |
| 八个新相位 | 31.520 | 31.160±0.257 | 31.246±0.032 |
| 完整近景揭示 | 40.336 | 40.336 | 40.336 |

单位dB，±为训练seed标准差。欠观测条件的非高斯修正收益在三个seed复现；八相位时强MAP更好，持续运输相对无状态残差的独立优势不稳定。不能把弱循环基线输掉的结果当作新方法已成立。后续扩展测试已运行：高噪声、独立重复packet、真正未训练checker纹理、去掉投影/冻结细节消融，以及只用validation拟合的风险判据；详见learned_core结果。风险判据只作架构诊断，不包装为H3核心训练。

由此保留积极主线：让H3提供非高斯生成细节，用证据支持与风险估计决定其注入方向和力度，而不是固定融合。正在搭建H3条件记忆adapter；H3 BF16与量化主干真实forward均已完成，核心方法稳定效果仍待验证。

## 已完成：非破坏性分支与旧细节取回

32seeds、4个具有严格相同远景LR的纹理身份、8次对象访问，产生147,456条packet记录。无独立线索时远景选择为25%；微弱独立descriptor提升至51.56%，累计12次弱线索达70.90%；近景揭示后精确MHT与共享分支均100%。

在旧内容版本返回、当前只揭示3/48系数时，保留旧版本可取回未再观察的细节：42.70 dB，对比快速适应单状态23.10 dB。该收益也由标准版本MHT取得。分支与精确MHT+去重在本线性协议中数值等价（最大差8.1e-15），不是原创概率算法。真实变化下registry污染的平均改善置信区间跨零，不称稳定统计优势。

可操作解法是保留已确认身份和旧版本、将未确认episode暂存；身份信息充分后再合并。这部分提供H3记忆架构的必要控制与容量预算，不能替代新核心。完整报告见 `results/branch_memory/branch_memory_report.md`。

## H3接入的具体进展

现有完整BF16 donor实际文件40,225,724,176 bytes，safetensors头/末tensor范围核验通过；FP16视频VAE为5,207,808,496 bytes。源代码已搬至H100隔离目录，原服务器工作未被中断。

已直接在已有5090运行真实FP16 H3视频VAE：128²单帧→24×1×8×8 latent→128²重建；encode0.684秒、decode0.240秒，峰值reserved5.40GB，官方模型完整SHA匹配。这个实验证明接口和尺度约定可用，尚不证明恢复质量：RGB roundtrip只有21.46dB，目视有明显格纹。已完成固定权重、标准comfy.sd.VAE wrapper的对照：同一自然图像256单帧25.67dB、256×5帧26.61dB、256×22帧28.53dB；512单帧27.21dB、512×5帧28.61dB；128×5帧22.04dB。原生视频上下文明显优于极小单帧，因此继续采用H3、将teacher切到原生视频上下文，而非据低分辨率结果否定底座。此为同一内容重复成clip的接口/协议诊断，不是真实运动视频或多视频SR质量统计；跨分辨率PSNR还受到输入采样频率变化影响，同分辨率的1/5/22帧对照更能隔离时间上下文作用。峰值reserved6.80GB。

早期3图单帧32方向teacher缓存仅作接口pilot，没有据此扩大训练：半步有限差分相对差约39%–54%。后续已用原生22帧上下文与严格FP32修复并重建缓存，详见后文。可学习rank8基、非线性transport、强MAP/shared-noise/Bures及同参数GRU控制随后均已实际训练。MiniMaxH3AddGuide将guide编码为独立条件token并保留target去噪行，可以研究LR guide与历史细节状态。完整BF16/VAE已传至H100并通过全文件SHA验证，早期下载不再是当前阻碍。

新增自然图像模拟视频生成器 `src/h3_data.py`：39帧、256×256 HR/64×64 LR、远近变化/遮挡回访/光照；H3以22帧block、5帧重叠处理，最大21帧块内等待加计算时间。第一/最后同视角GT逐像素一致，输入LRPacket不含HR，源图像train/val/test交集均为0。39帧约1.6秒，只用于接口和机制调试，不能称长视频回访。真实DAVIS数据现已全部完成：8个独立源、695帧，train4/val2/test2；project test为官方val子集，并非隐藏官方test。原JPEG保留、GT仅modcrop、LR为明确Pillow bicubic；本地逐帧像素检查和远端3404个payload SHA/源JPEG CRC均通过。另准备8×39帧固定256/64窗口。原clip约2–4秒，不是长回访证据。数据许可按档案CC BY-NC4.0记录，具体来源见research/real_video_sources.md。

H3观测投影的线性性与adjoint核验通过，brightness nuisance消元残差<1.3e-13。另实测了数据仿真与可微算子的实现差异：20个validation源patch，PIL与torch退化LR RMS均值0.001746、最大0.002503，须计入观测误差或统一退化实现；不以过小噪声假设虚增precision。记录见 `results/h3_data/operator_audit.json` 与 `degradation_mismatch.json`。

## 新完成：真实H3主干与teacher修复

**BF16基础主干已实际执行。** 使用40,225,724,176-byte官方pruned BF16权重、固定中性恢复prompt，分为VAE编码、DiT采样、VAE解码三个前台进程。真实50层主干的4次video/audio forward全部finite，5×256×256输出已保存并目视检查。DiT采样5.89秒、含加载/卸载阶段总25.01秒，峰值allocated7.59GB；decode阶段另7.29秒。DiT无量化/LoRA/VDN，文本encoder为NVFP4-AWQ、条件缓存BF16，VAE为FP16，精度分别记录。这仍是静态重复图的4步接入实验，正常50-sigma/49-Euler静态基线随后完成，49次forward全部finite、采样36.54秒、该阶段总55.06秒；仍不能据此称实时SR或稳定画质提升。完整证据位于results/h3_setup/dit_bf16_neutral_guide5_256/。 正常步数输出位于同名_normal49目录，已目视检查；该单源首帧bicubic25.76dB、BF16四步24.80、正常49次25.09。直接给基础生成器LR guide尚无SR增益，这给后续学习记忆/guide适配提供明确基线，不能解释为H3最终方法效果。真实运动22帧块基线已开始接入。

在现有5090实际完成50层MiniMaxH3Model的4次forward及5帧256²引导采样：video/audio输出全部finite，真实Qwen layer50条件协议核对通过；Euler4步采样10.37秒，脚本总15.16秒，峰值allocated9.37GB。精度明确为NVFP4-all DiT、BF16 compute/text、FP16 VAE；该结果只是主干接入证据。5帧由单一已观测LR重复构成，4帧等待只是名义block上界，未测真实采集等待或吞吐。实际源图为三只猪，而复用文字cache描述单只羊，语义不一致，不能将这次样例用于SR画质比较。输出已目视检查，细节偏软；已单独实际编码固定中性恢复prompt：29×5120条件，NVFP4-AWQ文本encoder、BF16输出，不包含GT物体语义；后续各控制使用同一条件。正常基础模型采样与BF16主干结果单列，不从4步量化接通实验推断底座质量。

native256×22同一存储权重的精度审计给出可操作修复：FP16下epsilon0.1/0.3的LR导数差约34%–37%；改为FP32计算后差为5.16%，epsilon0.1的真实局部LR预测RMSE为7.95e-6。于是固定FP32 teacher、FP16实际读出评价的协议进入训练。该结果仅来自一个方向/源图，更新幅度较大时仍要实际重投影审核；不能将局部线性化称精确nullspace。

随机32方向在旧3图cache对latent细节差异的能量覆盖分别约26.5%、24.2%、7.8%，因此加入仅train拟合的PCA span和学习rank8坐标，并保留固定PCA、随机基与同预算GRU控制。 native22 FP32的3源缓存已经实际完成，半步导数差降至0.0096–0.0128；25步basis及25步更新器训练已执行，并复用保存checkpoint进行真实解码评价。当前只算训练链路证据；Bures强控制的小协方差平方根曾触及错误绝对数值floor，正在修复，不能把它的假失败计为方法优势。同时启动候选B：可逆内容/外观表示与反事实交换监督，直接检验近景修订与光照变化能否解耦。候选B已完成56源缓存（32train/8val/16test）：672次真实native256×22编码、56次真实解码审计，95.49秒。修复了条件可逆分支的双零初始化问题，并实测两步后chart/appearance参数均更新；四种控制参数相差<1%，600步匹配预算训练已在H100 GPU7实际运行，随后增加各控制相同预算的真实RGB decoder监督。两条候选均需真实decoder和DiT读出验证，尚无稳定原创优势结论。

## 新完成：真实运动对应与读出控制

8条39帧窗口已实际跑LR-only双向DIS对应，当前代码不读取HR或标注。平移方向/坐标审计EPE为0.02484 LR像素；同前缀独立重算、重复packet与只读查询检查通过。可信区域确有更低的HR重投影误差，例如heldout bmx-trees为0.0400，对照不可信区域0.2683；但该clip跨帧canonical可信覆盖均值只剩5.1%。这是后续H3记忆需要处理的对应失效，不能以光流代替身份重定位。

固定权重的时序颜色融合在8条视频均略低于bicubic，两个heldout test分别为27.251 vs27.425、28.378 vs28.465dB。它只作为强一致的接口控制，没有恢复增益。积极方向是让学习状态在可信几何区域吸收新证据、在失配区域回退当前观察，并测试多anchor和可逆身份状态；单纯延长RGB平均不足。所有方法可复用相同LR对应缓存，以免把额外几何信息误归为H3状态架构收益。完整逐帧结果与图见results/video_correspondence/。

## 新完成：真实H3运动视频、感知质量与一致性控制

BF16基础H3已完成真实DAVIS/bmx-trees前22帧的正常49次forward及解码，全部finite；采样41.72秒，含加载卸载64.83秒，峰值allocated7.67GB。输入LR与实际编码文件SHA逐一吻合。原始输出RGB PSNR22.011dB、LPIPS0.3195；bicubic为27.232/0.2821，说明基础生成器需要恢复任务适配。

为定位结构漂移，给bicubic、H3 VAE-only、H3 DiT全部加入相同的经典像素观测一致性投影（25步Dykstra，无HR拟合）。三者分别为28.104/0.2304、28.229/0.1926、27.938/0.1944（PSNR/LPIPS）。H3先验在这个单片段的感知指标上有积极信号，但VAE-only与DiT相当或更好，尚未证明DiT或新记忆的独立贡献。投影后时序误差增量分别0.03464、0.03453、0.03577，尚无时序优势。详细协议、真实Pillow重投影误差和公平图见 `results/real_video_evaluation/bmx_trees22/REPORT.md`；不能把一个片段的22帧当22个独立视频做显著性结论。

训练侧已定位第二个数值问题：即使FP32，默认cuDNN TF32也会破坏小幅PCA方向导数。禁用TF32后同一真实H3 decoder、epsilon0.05/0.1的LR导数差从46.4%降至0.242%，0.1/0.3降至0.141%；旧缓存保留为legacy，新strict缓存重新构造。官方eager RoPE数学实现已使真实冻结decoder可反传，forward差为0、latent梯度非零、底座权重无梯度。候选A的非线性状态更新器和候选B的内容/外观结构均在此基础上进行同预算训练，尚未确立稳定新核心。

## 新完成：内容／外观候选B的三seed独立来源评估

四种约14.4万参数模型，各三seed，累计7200步latent训练与2400步微调，其中600次真实H3 decoder RGB反传。16张独立test源照片组成受控远近／光照／内容替换episode，不能称16条真实视频。所有模型共用冻结H3 E/D及相同当前观测、去重和query协议。

相同DC后，full/no-swap/noninvertible/mixed分别24.761/24.785/24.815/24.557dB；当前LR→H3 VAE+DC为25.639dB。full相较no-swap在return事件有+0.093dB、13/16来源改善，但总体SR优势与可逆性独立贡献均未成立。源码、全数据、按源配对结果和局限见 `results/h3_factorized/factorized_report.md`。

已有正确初始HR身份的诊断也会在后续LR mixing中丢失细节；直接替换成真实appearance参数并未修复结果。这提示瓶颈涉及内容写入与LR/HR表示域，而非简单的appearance估计不准。下一结构版本保留当前真实LR为读出锚点，让历史状态承载有证据的detail残差，并约束decoder实际支持方向；继续使用相同强控制，不靠增加稳定性权重或选择test样例。该修改尚待验证。

## 新完成：8条真实视频上的H3 VAE强基线

8条DAVIS真实运动视频均完成22帧的两套固定视野：256中心ROI与原生832×480；共16次实际H3 VAE E/D、352帧输出，逐SHA转移校验通过。没有DiT或记忆训练。HR仅用于预先制作合成4×LR和评价，推理阶段只打开LR。

|视野|bicubic+相同DC PSNR / LPIPS|H3 VAE+相同DC PSNR / LPIPS|按源平均增益|
|---|---|---|---|
|256中心ROI|25.833 / 0.3796|26.071 / 0.3416|+0.238dB、LPIPS−0.0380；两项均8/8源改善|
|原生832×480|26.930 / 0.4167|27.159 / 0.3732|+0.229dB、LPIPS−0.0434；两项均8/8源改善|

这是可复现的积极底座信号，也给学习状态设定更强门槛。八源包含4train/2val/2project-test，实验不拟合参数；split分别报告。两种视野内容不同，不把均值差视为分辨率因果效应；小规模同域结果不能证明跨域普适性。完整数据、源码版本、逐帧指标和可视化见 `results/real_video_vae/REPORT.md`。后续方法仍需证明额外历史证据、持续身份和更新能力，不能把冻结VAE与经典DC的收益包装成新方法。

## 原生空间尺度与新状态结构的进展

真实832×480、22帧BF16基础主干正常采样也已完成，49次forward全部finite；采样88.54秒，峰值allocated8.23GB。同一官方量化guide及同一decoder后端下，DC后的bicubic／VAE-only／DiT分别为32.123／32.150／32.071dB，LPIPS为0.21177／0.17292／0.16994。DiT相较VAE-only只有小幅感知差，尚无稳定独立优势；对应实际输入与公平图见 `results/real_video_evaluation/bmx_trees22_native480/REPORT.md`。这是固定text-only条件，额外官方LR首帧视觉上下文正在另作能力诊断。

候选A原局部chart已完成三seed固定预算训练，test ROI平均transport25.911dB、强MAP26.059，transport相较stateless/GRU没有稳定独立收益。原方案在小span中过早把所有方向视为首帧可辨，且raw局部HR latent差分移植会破坏真实decoder上下文。此结构已明确记录，详见 `results/h3_core/FINDINGS.md`。

下一版先将当前观测决定的粗结构与持久细节分离。验证集真实算子检查显示：投影后同footprint对detail响应接近0，新footprint才使方向逐渐可辨；投影MAP相较anchor在8个val来源的ROI提高0.261dB。B的固定checkpoint诊断则发现near确有新增细节，但return写入丢掉约62.6%；保持当前LR锚点、只读历史支持残差的干预，在8val来源较当前LR-VAE提高0.457dB。这些是经典控制与结构机制证据，不是新学习方法的最终成绩；A/B已据此进入固定预算结构修订，保留这些无需额外训练的强控制，并计入其已训练chart成本。

长回访素材也已扩展：LaSOT固定revision镜像取得4条连续600帧原始片段（共2400帧），包含真实物体转动、近远变化、遮挡与再现；原JPEG逐ZIP CRC/SHA核对通过，原始720p三条、360p一条。按论文30fps约20秒/条，但尚未核对每源真实fps，因此同时报告原始帧数。来源与许可证据、官方/project split、选择规则和预览保存于 `research/long_video/SOURCES.md` 与 `data/long_video/manifest.json`。现已生成固定HR/LR：720p源仅下采样至480高后center mod32，360p源仅crop到640×352，随后Pillow bicubic 4×得到LR；不放大GT。没有把数据准备计为长视频方法实验；后续按冻结协议评价。

## 四条真实长流的完整底座结果

LaSOT四源各600连续帧已完成140个原生H3 VAE块；全部2400输出逐SHA在本地归档验证通过。相同DC后的按源平均PSNR由31.659升至31.978dB（+0.318），LPIPS由0.18192降至0.15225，4/4来源两项均改善；目标ROI平均+0.280dB。三个有≥60帧标注缺失的来源，返回后17帧ROI改善分别为−0.033/+0.202/+0.362dB，感知距离均改善，不能将全片均值当成一致的回访收益。

时序误差增量在3/4源降低，book-20几乎持平；源级表、分块边界、目标区域和长缺失分层均见 `results/long_video_vae/REPORT.md`。block协议是22帧、5重叠，首次22帧、此后17新帧一次提交，无历史回写；最大输入等待首块21、后续16个间隔另加计算。离线E/D循环375.004秒，不含加载，也不等于实时延迟。这是冻结codec基线的积极结果，尚无持久记忆或身份检索。

可视化复核明确book-20标注的是人物身后书架上的书，手持书并非该跟踪目标。所有目标文本与框均只用于数据选择/评价，未输入模型；定位图为 `data/long_video/target_annotation_preview.jpg`。本轮共享SSH断连导致未落盘的部分评价重算，已完成输出、训练检查点和来源指标全部保留，恢复记录单列。

## 完整H3主干训练与第二轮结构结果

40.23GB BF16主干全文件SHA为 `a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4`。实际原生50层、22帧256输入的两步LoRA+guide梯度审计完成，3,956,280个可训练参数，底座梯度计数始终0，guide输入和LoRA梯度均非零，峰值allocated41.394GB。原生math与可微math、零LoRA与identity-guide的video/audio forward差均为0。此处证明训练接口，不能把两步loss变化当恢复效果。

随后固定多源SR适配pilot：4条train真实视频×2分辨率，每case50次，共400更新；LoRA-only、guide-projection-only、联合三个控制共用case顺序、sigma/noise序列和AdamW学习率1e-5。验证只用cows/dance-twirl两源×两分辨率，最终400步权重统一正常49次forward读出，不选择最佳checkpoint、不看test调参。LoRA-only400步已完成，固定validation velocity MSE从0.208553降至0.193380（约7.27%）；三个控制均已完成400更新和固定49步读出；guide-only最终velocity为0.204384、联合为0.186553（共同初始0.208553），最终RGB评价和440帧逐SHA归档均已完成。三控制可训练参数量不同，单独报告，不能解释为等容量架构消融。

候选A2使用观测商空间、真实D反传、共享固定PCA32与RGB训练prior，三seed共6000次主run更新。val8 bounded ROI为transport24.81474、MAP24.82167、shared24.82291、Bures24.82191、stateless24.79479、GRU24.80471、anchor24.57074dB；transport−MAP为−0.00693，按来源95%CI跨0。相对stateless/GRU亦没有稳定独立增量。发现current-only控制将coarse重base后的累计h差误当当前packet，已排除错误分数并仅对该控制按原400步预算重训；其它full-history结果不受影响。相同真实D和prior、仅过去LR目标的固定40步非线性逆控制也已完成，结果见后。

候选B2 seed17在val8修复了历史支持细节被far写入覆盖的问题。同DC return事件full24.685、当前LR-VAE24.212，增益约0.473dB；但B2无recurrent的直接observed-detail为24.752，B1直接near控制24.757，学习模块仍落后约0.072dB。因而该增益归于当前LR锚点和历史残差读出，不能归于可逆chart或学习更新。29/43随后按原600+200预算完成验证；互补实验预先设计多个near footprint，保留last-near、对齐残差融合、全history逆问题及配平GRU控制，先做train/val可辨性审计。

候选B2已进一步完成三seed：回访相较LR-VAE平均+0.47003dB，8/8源改善；但相较固定B1直接near为−0.07307dB，8/8源均落后，source区间[−0.09039,−0.05748]。相较no-swap仅+0.00491、相较noninvertible+0.00335，不能建立核心增益。多个互补near的新协议已先用8val来源做线性审计：union ROI last-near25.410、对齐融合27.776、全history inverse27.753，说明新的判别空间存在，而大部分增益首先能由经典融合解释。inverse正则仅在4个train来源的预声明grid选择，详情见 `experiments/h3_factorized/B3_COMPLEMENTARY_PROTOCOL.md` 与 `results/h3_factorized/complementary_v3/`。

A2 current-only公平修复已完成，三seed平均24.60394dB，transport较其+0.21080，但不改变未胜MAP/GRU的结论。真实D非线性LR逆控制曾因大sum/σ²目标穿FP16导致NaN，该尝试保留为legacy；保持40步、学习率及相对项权重，仅统一缩放整目标后，8val全部finite且LR目标下降，8val ROI相较线性MAP平均提高0.27875dB，全图提高0.02928dB；每源额外40次backward、42次真实D调用，计算预算明确另列。不能把数值失败当作该控制失效。

下一步的主假设更贴近原问题：真正H3 DiT生成的细节样本具有欠观测的多种解释，而VAE(upLR)加MSE拟合prior接近条件均值。将真实生成proposal作为持续身份初态，检验近景新证据能否修订错误细节、同时保存尚未观察到的身份；经典投影、sharedseed和非线性求解继续作为强控制。此方向尚未取得新方法结果，不预先宣称优越。

补查已确认，泛化的posterior-to-posterior flow与历史生成热启动已有直接先例，例如 [Sequential Flow Matching](https://arxiv.org/html/2602.05319v1) 与 [Sequential Posterior Sampling](https://arxiv.org/html/2409.05399v1)。下一H3候选必须用局部新证据纠错和未约束细节身份保留的联合实验建立具体差异；见 `research/sequential_generation_prior_art.md`，不把宽泛概念或换底座作为原创性依据。

## 本轮收尾：原生适配、互补融合与真实生成细节

H3原生SR pilot v1的验证结果如下（训练/推理流IO存在下述数值混杂）。训练为4个真实视频源，验证为另2个源；所有方法使用相同25步DC，最终400步权重统一正常49次采样读出。数值按来源等权，不能把44帧或两个空间视野当独立来源。

| 方法 | 256 PSNR / LPIPS | 原生480高 PSNR / LPIPS |
|---|---:|---:|
| H3 VAE | 25.388 / 0.31226 | 25.922 / 0.39854 |
| BF16 H3基础主干 | 24.948 / 0.29908 | 25.892 / 0.39208 |
| LoRA400 | 24.939 / 0.29920 | 25.869 / 0.38836 |
| Guide400 | 24.738 / 0.30101 | 25.289 / 0.37928 |
| LoRA+Guide400 | 24.773 / 0.30090 | 25.309 / 0.37470 |

联合适配的velocity验证损失降低10.55%，却未带来稳定SR优势；raw PSNR分别仅14.656/15.892dB，存在明显观测漂移。原生480的LPIPS改善与PSNR下降形成取舍。当前结论限于这个固定预算pilot，既不能证明大规模适配无效，也不能当作记忆机制失败。完整训练、采样、raw/DC与冻结检查见 `results/h3_setup/native_sr_pilot400/REPORT.md`。

末轮固定bear_256单case审计实际确认v1训练流IO为FP32，官方BaseModel采样会将输入及最终v输出转为BF16。零adapter下两路径v RMSE为0.02575、x0 RMSE为0.01130，最大差约0.1985，不能声称训练/采样数值等价。对齐BF16 IO后，raw native与官方wrapper预测逐元素差为0；目标符号epsilon−zHR、sigma/time、video scale=1、24ch guide及Euler转换均无误（理想v回到HR误差约2.38e−7）。已要求保留v1全部结果，v2从相同初始化、同case/sigma/noise/lr序列重新跑三组各400步，先做真实梯度与官方wrapper一致性检查。这是接口修复后的独立重跑，不是给旧模型追加训练或按验证集调参；v2结果未出前不将v1的质量不足只归因于数据/预算。

B3互补近景审计进一步配平了每个packet的LR右逆预处理。8val同历史支持区域中，last-near经H3读出为25.1789dB，逐packet经H3再融合为26.9930dB（+1.8141）；经典融合后统一经H3并限制相同支持为26.9900dB，二者仅−0.0030。此前约1.1dB的表面非线性缺口主要是bicubic与右逆预处理混杂，不能据此声称神经融合必要。积极证据是互补观测增益确实能穿过原生H3；学习问题转向光照、遮挡、错误历史及真实变化下的可修订证据。

真实BF16 DiT三seed proposal与原生guide记忆适配也已执行。在单个train源上，正常49步生成的三个初始proposal几乎同样柔软，seed间MSE约1.42–1.93e-5，尚未建立丰富多解释。48,280参数guide-state接口通过全50层真实梯度；首次训练target越过RGB上界，完整保留为legacy，修正为共同far/box投影后按原5步预算重跑。最终reference/learned在共同约束下为28.83685/28.83826dB，原始身份X0为28.89489；经典near创新修订达到30.02720且near LR误差显著更低。五步结果只是接口证据，没有学习质量优势。身份保护来自所有控制共同的经典约束，不能归于adapter。

这个诊断还暴露了建模问题：若将所有像素观测null方向都锁死，就同时禁止了先验相关的新细节修订。下一候选应区分“与新证据相关、应当改变的解释”与“条件独立、应当保留的身份”，并比较相同H3初态下的经典反演、精确多假设重加权和已有sequential-flow控制。实际可学习缺口必须由新实验建立，不能用这一假设代替结果。

确认集来源补充：早期小模型已评价BSDS全部200 test，不能把未进入H3缓存的其余BSDS图当项目全新来源。现已按文件名hash预先从官方DIV2K validation选32张，全部ZIP CRC/文件SHA与PNG尺寸校验完成，方法和协议冻结前暂不观看、训练或评价。该集合是项目新数据来源，不称官方隐藏test，也不保证基础模型预训练未见；来源与规则见 `data/confirmation_div2k/SOURCES.md`。

## 官方数值路径修复后的H3原生适配结果

v2从同初始化重跑三控制各400次更新、固定49步采样与统一FP16解码/评价均已完成。440张PNG逐SHA、3个最终adapter、1200条唯一更新和573文件manifest均已在本地核验。初始validation velocity MSE为0.20872975，LoRA/Guide/联合分别降到0.19351371/0.20440091/0.18682107。零adapter与guide bias干预的原生/官方BF16路径预测max0，基础530张量没有更新；v1/v2的VAE和base八个case输出逐SHA完全一致。

| 方法 | 256 PSNR / LPIPS | 原生480高 PSNR / LPIPS |
|---|---:|---:|
| H3 VAE | 25.388 / 0.31226 | 25.922 / 0.39854 |
| BF16 H3基础主干 | 24.948 / 0.29908 | 25.892 / 0.39208 |
| LoRA400 | 24.943 / 0.29901 | 25.871 / 0.38853 |
| Guide400 | 24.743 / 0.30117 | 25.293 / 0.37932 |
| LoRA+Guide400 | 24.793 / 0.30117 | 25.380 / 0.37486 |

数值修复后，native480联合适配的LPIPS相对VAE下降约0.02368，但PSNR下降约0.54224dB；质量取舍仍在。256没有独立适配收益。这个4train/2val、单seed pilot不能排除更充分的学习，但已不能把当前失败仅归因于训练/采样精度错位。下一步应改变生成过程接受观测约束的方式，而非继续增加同一控制的步数或在val扫权重。完整证据见 `results/h3_setup/native_sr_pilot400_native_io_v2/REPORT.md`，旧v1仍单列。总览图已更新为v2，并保留旧图。

## 新完成：真实运动下的历史证据与连接模型

在现有4 train/2 val真实视频、256视野上，固定past-4 LR-only DIS、光度/循环权重、ridge0.01和24次CG，沿当前LR的null方向修订同一个冻结H3 VAE输出，再做共同25次DC。没有训练或参数选择。H3 VAE到历史修订的PSNR变化为bear+0.06988、bus+0.43664、car-turn+0.48337、flamingo−0.51166、cows+0.11660、dance-twirl−0.86470dB；train平均+0.11956，val平均−0.37405，6源平均−0.04498。LPIPS全6源均值改善0.00368，val近乎持平，不能称稳定恢复优势。所有方法使用相同22帧VAE上下文，修正只读过去LR不代表整体零前视。

独立小算子adjoint误差2.22e−16、没有历史时求解器严格不变，实际CG最大相对残差4.65e−6，排除了未收敛作为主要解释。评价时将当前GT送入同一个估计warp/downsample算子，flamingo与dance-twirl的GT观测残差反而高于原VAE输出，置信度≥0.8区域仍如此；求解器继续降低错误观测模型的残差，同时损害HR。固定面板中两源出现明显条纹。该量混合几何、遮挡、亮度和退化不交换性，不能称纯光流端点误差。可操作方向是让后来证据修订此前的对应/外观解释，而非把低重投影误差或current-LR一致性当作细节真实的保证。全部结果见 `results/real_video_history_inverse/matched_final_projection/REPORT.md`；该控制为经典反演，不作为原创方法。归档时发现有限box投影不完全幂等，旧版无历史控制少做最终25次投影；现已补齐，六源t=0输出逐位相同，历史逆问题输出不变，旧版全量保留。

A新的真实H3条件响应实验完成56次正常49步生成，4train/2val、四头各300真实D训练更新，每头18,888参数。两val global-query PSNR分别为loop25.38085、无loop25.38444、GRU25.36952、set25.38816；loop未胜无loop或普通集合控制。loop的路径MSE比无loop约低14%，但GRU更低、set严格0；privateHF漂移亦未优于set，不能单凭一致性项下降称成功。经典联合LR反演全图27.09748/global25.41857，优于loop全图25.41770/global25.38085。

撤去任意X0-PCA私有方向锁定的结构诊断使H3响应移植全图从25.48845提高到26.0333，但相同full24 VAE-guide响应为26.0650，仍未证明DiT独立先验贡献。直接实际H3 C_AB为26.2423，而直接VAE当前guide为26.8679；这里不因失败缩减固定实验，所有四头已完整训练，结果保留并停止同一损失的小幅调节。下一机制必须具备相对classic inverse、DeepSets与可回放证据的具体优势。

B3R已完成40源原生E/D缓存及六控制真实梯度检查。缓存RGB与部署saved-query解码路径有RMS5.87e−5差，已只重解码baseline到相同路径；观测、latent标签和训练协议未变。零状态RGB对齐后max0，closed group rebase/withdraw与逐factor replay误差1.86e−9/9.31e−10，VAE权重梯度0。随后六控制已按600latent+200RGB完成训练和预声明按val选best的pilot评价。这种选择符合原协议；我此前把固定预算概括成“只读最终权重”过窄，另做统一stage1-last600初始化、fresh RGB200仅读last200的补充对照，额外物理算量单列，不覆盖原pilot。

B3R原selection-pilot的8val、7query union DC平均如下：

| 控制 | PSNR |
|---|---:|
| 当前LR-H3 VAE | 23.84270 |
| closed-action full | 23.67001 |
| no-equivariance | 23.62371 |
| cache MLP | 23.68954 |
| nonlinear group | 23.67710 |
| online GRU | 23.65372 |
| replay GRU | 23.66819 |
| LR-photo retro atlas | 27.51398 |
| H3 decoded atlas | 26.79536 |
| H3 E/D inverse | 26.51293 |

full较当前LR平均−0.17269dB，0/8源获益；较cache MLP−0.01953，1/8获益。移除实测detail反而提高0.09277，打乱跨源历史仅下降0.0109且来源区间跨0；这说明模型主要依赖anchor，不能归因于成功学习了证据写入。精确rebase不等于有用的表征。下一结构诊断将固定单packet分离pre-lift24ch、learned lift/readout、same/cross appearance和像素域photo bridge，区分压缩/目标学习失败与native latent不等变；不先调整损失权重。

统一端点补充实验也完成：六模型都从各自stage1-last600出发，以相同schedule/lr新跑RGB200并仅评价last200，额外1200更新、300 actualD backward，训练208.85秒。full为23.65541，仍比当前LR低0.18729dB，比强LR atlas低3.85857dB，结论不变；原best-selection与endpoint分别保存在 `revision_v3/summary_seed17/` 和 `summary_endpoint_seed17/`。

8val单packet无训练诊断进一步定位问题：同appearance时pre-lift24ch直接D为26.2379，完整nativecode经像素photo bridge为26.7445；learned lift/action/readout仅23.5511，绕过action几乎相同。跨appearance时前lift仍有收益。因此主要损失在learned lift/readout与目标学习，不能只怪latent光照不等变。原RGB200实际每4步仅首个source/单query做一次真D反传，每模型只有50次，对32source×7query很稀疏；这些结果不证明4ch容量不足或更充分的decoded-RGB训练无效。下一固定单TRAIN源实验用相同模型/初始化，latent-L1与每步actualD-RGB两臂各300，直接区分目标/优化能否传递已观测细节，单源拟合不当作方法收益。

A的统一浮点读出感知评价也完成，按来源和AB/BA等权：loop LPIPS0.339365、basic0.339333、set0.342628、经典联合反演0.246842。loop对set的均值差不是两源一致，对basic正反各一；SSIM亦未胜强控制。新假设转向压缩状态中尚未消歧的content–operator联合关系，预声明与同容量joint-set/CNP、dense joint MHT和联合稳健逆问题比较；按固定完整训练预算推进，不把普通重积分或joint Bayes改名当原创。

## 新进展：生成过程约束与解码监督

冻结H3采样内观测一致性控制完成了固定pilot。正常49步、相同seed101与LR guide，在预声明sigma index28/45/48修正post-CFG clean estimate：`z0 ← z0 + E(P25(D(z0))) − E(D(z0))`，强度1，audio/guide/Euler接口不变。零创新hook对实际官方video/audio轨迹逐位一致，基础530个参数张量未更新；非零video修改可通过后续联合DiT影响audio，不把“hook没有直接改audio”说成最终audio相同。该控制源于现有观测约束生成思路，是强基线，不是新的记忆方法。

| 2个val源，同最终DC25 | 256 PSNR / LPIPS | 原生480高 PSNR / LPIPS |
|---|---:|---:|
| H3 VAE（官方guide预处理） | 25.38764 / 0.312261 | 25.92184 / 0.398536 |
| 原始H3生成 | 24.94762 / 0.299084 | 25.89222 / 0.392079 |
| 采样内观测约束H3 | 25.06172 / 0.268426 | 26.01898 / 0.352357 |

native480两源均同时改善：cows相对VAE +0.12372dB/LPIPS−0.05461，dance-twirl +0.07057/−0.03775。256的PSNR仍低于VAE。6条新轨迹294次DiT包括两条TRAIN/noop核验；4条val的额外投影和VAE使native480采样约70.3–70.6秒，旧base约30.6秒，不能用相同NFE宣称相同算力。4个case只代表2个独立来源，且没有证明长期身份。完整报告 `results/h3_setup/sampling_dc_pilot/REPORT.md`。现按原参数扩展到8DAVIS源×2视野，复用已核验的4val；新增12case各跑base和innerDC，共1176次DiT，project-test已用于既有基线，不称全新隐藏测试。

B的单TRAIN目标诊断已完成：固定train100075、相同4通道encoder/readout和初始化、同appearance条件，冻结action/appearance/core，latent-L1-only与每步actual-D RGB-only各300次。HR near PSNR初始25.8597；latent-only终点26.0128，RGB-only终点28.2840。RGB训练时latent L1从0.1706恶化到0.2560，latent-only则降到0.1160；优化latent距离不能替代最终图像目标。RGB终点移除实测delta降到26.5359、换另一训练源delta降到27.5601，说明拟合使用了观测内容，但单源拟合仍可能记忆anchor，不能当作泛化。

由此预声明B4：32train/8val、七个参数接近的控制，224个(source,query)组合均衡三遍，每模型672次actual-D反传，共4704次；移除latent-L1主项，其余相对权重沿用旧协议。所有控制从stage1-last600初始化，只评last672，使用因果LR控制器。新增当前图像神经SR控制，神经输入不含历史图像/code；共同后处理仍依赖历史控制器和几何support，另给几何-only读出，明确其信息边界。实际梯度、未来信息毒化和无证据一致性核验通过后已开始完整训练。协议见 `experiments/h3_factorized/B4_DECODED_PROTOCOL.md`。

A的未约束生成联合响应函数实验也已完整结束：48次官方49步轨迹、三模型各300更新，每更新两次actual D，共1800次反传。2val×2carrier先源内平均，mixed/additive/joint-CNP分别25.1033/25.1706/25.1491，原carrier25.2246；dense H3 family26.1642，dense VAE26.2075，联合稳健逆问题26.9100。mixed私有高频变化更大且误差更高，不称更丰富细节。新的单独版本使用与实际Gaussian0.6/reflect/AA观测相符的采样内P25；独立算子与noop官方轨迹核验均通过，所有12carrier+12center+48offset共72条轨迹重新生成，再做同一3×300预算。不混用raw与DC family，不从held coordinate监督训练，不按val选择版本。

Relay官方源码commit `e856773350f53036d825404c9ea9228e6dffadca`，Wan文件实际存储5,033,156,288个标量（约5B，包含缓存prompt张量），不是此前资源备注中的1.3B。既有6个train/val源、各22帧、两个空间视野，官方与仅过去参考变体共24case/528PNG已完成并逐文件核验。官方K16消耗未来端点，实测max-lookahead15帧、Wan位置[0,15,21]；因果repeat-start变体前视0、Wan位置[0,15]，改变了参考条件和计算量，不当作官方等价实现或同算力消融。H3仍是主底座，Relay用于暴露外部恢复质量和等待代价。

## 最新完成：8源基线与两条学习路线的完整结果

固定H3采样内约束的8源扩展已全部完成，新增24条49步轨迹共1176NFE、76E、72D；全部1056PNG、本地及H100共1344个文件逐SHA一致。原4val输出复用，重算指标与原值最大差0。native480的同最终DC25结果：H3 VAE27.160790dB/LPIPS0.372247，原H3生成27.094806/0.362725，采样内约束H327.245685/0.326539；后者对VAE的PSNR在7/8源改善、LPIPS在8/8改善。bmx是PSNR反例：对VAE−0.09031dB，但LPIPS−0.02053。256相对VAE仍有PSNR取舍，仅2/8源PSNR改善，LPIPS8/8改善。

该结果为H3生成先验提供了可复现的积极基础，不是新的记忆方法。native480平均采样base28.700秒→innerDC72.774秒，额外编解码和投影成本不能省略。共同6个train/val源的native480 innerDC为26.30625/0.348980，便于和Relay的同输入集合比较；保持各自预处理和等待差异说明。完整报告 `results/h3_setup/sampling_dc_davis8/REPORT.md`，八源差值图 `results/progress_overview/sampling_dc_source_paired.png`。独立LR-only时序评价也已完成：native480同DC25时序残差RMSE为base0.029250、VAE0.028410、innerDC0.028546，有效覆盖94.49%；innerDC较base改善6/8源，较VAE仅4/8。256较VAE的时序残差8/8更高。故LPIPS收益伴随相对VAE的时序取舍，不能声称全面时序改善，也不能由此直接断言可见闪烁。全量96行/2016对、各来源记录在 `results/h3_setup/sampling_dc_davis8_temporal/REPORT.md`。

B4的七控制各672次actual-D反传已全量完成，共4704次、844.94秒；8val终评及无当前观测t15查询也完成。全部为last672，不使用val挑checkpoint。union区域、同DC的源等权结果如下，不能和全图或真实视频指标混为一张排名表：

| B4控制 | 8源union PSNR |
|---|---:|
| 当前LR-H3 VAE | 23.84270 |
| closed-action full | 24.11443 |
| no-equivariance | 24.07241 |
| cache MLP | 24.15909 |
| nonlinear group | 24.12149 |
| online GRU | 24.05545 |
| replay GRU | 24.08248 |
| 当前图像神经SR + 共同controller | 24.10618 |
| LR-photo revision atlas | 27.51398 |

full相对当前LR +0.27173dB、8/8源改善，来源bootstrap95%区间[0.12581,0.48068]；但相对同容量当前图像神经模型仅+0.00825、3/8、区间[-0.01824,0.04658]。full相对replay +0.03194、7/8，但cache MLP均值更高。移除实测detail只下降0.00112、打乱历史只下降0.000535，冻结光照修订或可靠度权重也几乎不改变结果。t15没有新当前观测时full22.94224、anchor22.94015，未建立回访优势。因此本轮支持“真实解码目标修复了恢复训练”，仍不支持新的记忆/action贡献。完整结果在 `results/h3_factorized/decoded_v4/summary_seed17/`。

下一结构诊断针对明确的信息通路：原readout同时读取history h与当前anchor，即使h近乎常量，`f(h,anchor)−f(0,anchor)`也可能变成纯当前图像恢复。需将这种shortcut与随机4通道压缩造成的信息损失分开。计划检查各级history扰动传播，并比较证据细节直通初始化、去掉神经readout的anchor条件；最终H3解码仍可使用同一anchor。先让已有观测内容能可靠传递，再判断新的可修订结构是否优于普通cache和replay，不继续小幅扫损失权重。

A的独立DC版本也已完成全部72条49步轨迹和三头900更新。mixed/additive/CNP为25.09237/25.22918/25.21257，dense H3 family26.02019、dense VAE26.08861、稳健联合逆问题26.86590。mixed相对CNP两val源都更低。固定PCA8诊断保留训练H3响应约91.7%–91.9%能量，支持坐标的native MSE约0.013，学习头约0.067–0.095；但full24三支持坐标中心插值依然明显低于真实中心响应，说明同时存在学习拟合和响应非线性的缺口，不能只归因于低rank。

这轮A还有重要边界：所有压缩都在A+B到达后，属于条件响应表示pilot；不是“先A压缩、后来B再修订”的因果记忆试验。完整prior→A→AB审计显示两个版本所有源在仅A时几何条件分布已收敛到center，u与g的互信息为0（数值误差约1e−16），B只重新选择carrier，未形成要测试的内容—算子联合歧义。final onehot本身不能否定这种机制；真正的限制是早期已无联合歧义且压缩时点太晚。不能据此宣布联合记忆假设被证伪，也不能把本轮当作该机制的有效验证。后续必须先确认因果干预与实际不确定性存在，再分配更大生成预算。

Relay统一评价已完成：同最终DC25、6来源等权，native480 H3 VAE26.20265dB/LPIPS0.399531，Relay官方25.53930/0.169708，因果变体25.49202/0.174107。Relay两版本在6/6源降低LPIPS，但只在bus提高PSNR；六源LR-flow掩码时序残差都更高。固定面板显示明显锐化与局部纹理/边界偏差，这种取舍不能被单一平均分替代。官方最大等待15帧、3次Wan；因果变体等待0、2次Wan；H3VAE块上界21帧。完整协议、成本、来源差和固定面板见 `results/relay_baseline/cohort/REPORT.md`，图 `source_paired.png`。远端评价虽遇SSH255，独立进程/完整输出核验已确认完成，未重复运行。

A的统一感知评价补充：raw mixed/CNP LPIPS为0.274054/0.278940，但两源正反各一；DC mixed/CNP为0.276546/0.293787，两源分别改善0.01192/0.02256。mixed因此有感知—PSNR/SSIM取舍，不能笼统称所有指标更差；但同DC稳健逆问题LPIPS0.223707、dense H3 family0.235929，仍更好。这不改变早期联合歧义未被激活和post-arrival压缩的协议边界。96个浮点读出全部评价完成，输入文件与脚本SHA本地复核，记录于 `results/h3_core/operator_perceptual/evaluation.json`。部分经典控制保存为float64，LPIPS统一显式转float32匹配预训练网络；PSNR/SSIM沿用保存精度，首次dtype不匹配尝试留档且未产生最终指标。

## 最新结构定位：实测细节直通、频带限制和延迟证据

B4分级扰动诊断表明，多源训练后的history特征跨源RMS变化仅约0.00221；移除实测delta使最终DC图像变化约0.000359，而仅替换神经reader中的当前anchor、保留物理D的原anchor，就使最终图像变化约0.00793。单源拟合的history扰动明显更强。这个诊断定位了多源lift塌缩与anchor捷径，但不以敏感性本身代替画质收益。

B5预声明2×2结构实验随后完整完成：有／无神经readout的当前native-code条件，有／无实际full24解码photo atlas直通。四臂同为32train、8val、seed17、672次真实D反传；固定主权重last672，共2688次、482.50秒，峰值训练allocated约10.815GB。公式为 `x = B_S + H_common[D(z_current+delta_theta)−D(z_current)]`，不用E重编码。B/H使用相同外部因果LR控制器，所有臂保留并实际解码相同full24/proxy24 archive，故不作压缩声明。零残差逐像素等于各自底图、一次共同DC后也max0；第一步零head的encoder梯度为0属预期，第二步actual-RGB梯度非零，基础VAE始终无梯度。

| B5控制 | 8源union PSNR，同DC | 相对本臂零训练底图 |
|---|---:|---:|
| 有神经anchor、无直通 | 24.12894 | +0.28624 |
| 无神经anchor、无直通 | 24.07015 | +0.22745 |
| 有神经anchor、有直通 | 26.79023 | −0.00513 |
| 无神经anchor、有直通 | 26.76572 | −0.02964 |
| 共同零训练H3 decoded atlas | 26.79536 | — |
| 强LR-photo revision atlas | 27.51398 | — |

直通因子平均+2.67844dB，8/8源改善，来源bootstrap95%区间[2.46350,3.00023]；但有anchor直通臂相对自身atlas只有1/8源改善，平均−0.00513，区间跨0。四臂的神经history移除／打乱只改变约1e−5～3e−4dB。相反，移除实际像素bridge下降2.93057dB，打乱bridge下降4.97050dB，均8/8。结论是观测细节可通过正确路径保存，而去掉神经anchor仍未让压缩history得到有效学习；不能把bridge收益记成新状态方法。完整统计、query15和真实前缀状态审计位于 `results/h3_factorized/factorial_v5/`。

独立纯CPU方向审计还发现，原H算子除了far-null与空间support，还将改变量限制在near观测的行空间。以GT误差作为仅评价用方向，active crop的near-null能量占比均值40.64%；同DC union底图23.84270，原hard-band方向27.71309，移除near-row但保留相同far-null、权重和support后为46.09212，8/8源增加13.82～21.77dB。它不是可部署收益或优化oracle上界，H含重叠混合也不是正交投影；它说明现结构机械排除了许多正确的高频方向。后续生成分支应允许这些方向并用真实观测软约束，而非继续在原硬频带内调参。此审计未改变冻结的B5。

真实视频延迟验证提供了可操作的积极信号。原H3 VAE已看过0–21帧，因此固定第21帧的base与历史inverse候选，只用真正未见的22–37帧LR评分；先保存所有LR-only判断及SHA锁，再单独读取HR。预声明主规则只累计后来4帧：若同一LR-only warp/camera下candidate比base降低加权观测误差，则在第25帧到达时保留candidate，否则撤回base。主结果为base25.52171、全接受25.54811、固定gate25.70237dB，SSIM分别0.71818、0.70564、0.71890。gate正确撤回bear/flamingo，保留bus/car-turn/cows，误保留dance-twirl，共5/6源符号正确。舞蹈有效覆盖仅30.53%，低清收益接近0，仍不足以验证其细节。2–3帧时碰巧6/6正确，但只是次要曲线，未替换4帧主结果；长等待结果不单调。这里评价的是后来读回的第21帧状态，非第25帧实时重建，也不能推广为误差校准概率。无新训练或GPU/E/D/DiT；评分与GT评价各约0.60秒CPU。完整固定协议、96个未来观测分数、来源结果和图在 `results/real_video_history_inverse/delayed_lr_validation/REPORT.md`。

延迟规则的来源划分还须单独看：TRAIN4 base／全接受／gate为25.51388／25.72679／25.95818dB；VAL2为25.53737／25.19075／25.19075，仍比base低0.34662dB。因此六源正均值没有证明验证集稳定优势；后续必须解决模型失配下的误接受，不能直接推广这个符号规则。

下一核心候选转向“未来cue到达前，历史已经序列化成有限字节code；后来只能用code和新观测重新解释”。新增机制必须集中于按未来物理读出的可混淆性分配code，并与相同输入、监督、码率的conditional VQ/神经分布式编码、量化atlas比较。Wyner–Ziv、binning、functional compression、CNP与null-space residual都有直接先例，不把这些概念重命名为原创。当前A/R正在落实自然near/far下的因果可辨识任务；简单方向／XOR任务仅作为接口检查，不据此宣称有新的记忆能力。研究备忘录见 `research/next_core_decoder_memory_memo_20261015.md`。

## 后续实验与方向调整

1. 将多解释/可逆分叉与学习到的证据修订合成同一核心架构，比较精确Bayesian多假设跟踪，避免给标准MAP或粒子滤波改名。
2. 官方RelayVSR与因果参考变体已完成固定自退化视频对照；下一步将已验证的H3局部状态接入参考latent/adapter，测试回访、远近、光照、真实变化与错误检索。输入等待、模型加载和计算分别报告。
3. 将轻量训练从点footprint迁移到全局blur/shift，再引入估计对应、非刚性运动与真实压缩退化；判断收益是否依赖oracle mapping。
4. 连续放大路线要求同一状态、只读ROI/尺度查询、抗混叠与路径一致性，同时保持细节；普通resize不能作为此能力完成的证据。
5. 流式时空超分先建立SR→插帧和插帧→SR两种有限等待基线，再检验共享状态读出。生成视频帧率与实际吞吐分开报告。

若非Gaussian学习模块不能稳定胜过强MAP，将优先改结构和状态表达，而非继续小幅调参。若身份分叉收益完全可由标准多假设推断解释，则以此作为强基线，继续研究可学习对应、受支持latent方向和真实生成读出之间的新机制。原始积极主线保持开放，当前没有提前把它收窄为仅证明经典融合。

## 资源和复现

本机M5/32GB执行受控数值实验；可用空闲H100执行训练与公开模型推理。使用显式GPU编号，避免已有VLLM等任务。远端AGENTS要求所有命令前台运行，所有本项目长任务均通过可追踪的前台SSH/exec会话执行；不使用nohup/tmux或后台detach。未停止其他任务。公网DNS/出口故障通过仅限项目命令的SSH代理转发解决，没有修改服务器全局网络配置。

研究来源、命令、数据清单、源码和逐帧指标位于各子实验目录；最终报告将继续补齐训练结果、真实视频、方法消融、成本和后续方向。
