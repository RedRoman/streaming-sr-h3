# MiniMax-H3 方法独立审查与两条可训练架构

2026-10-15。已读 `sources_and_method.md`、三seed learned-core及extended结果、两套mechanism/branch结果；已与learned_core直接核对H3注入接口。以下是待训练、可证伪的架构候选，不是已完成的原创性或H3效果声明。

## 现有证据指向什么

| 当前证据 | 对下一步的含义 |
|---|---|
| learned repeat：persistent 24.726±0.021 vs Gaussian MAP 23.369；8phase new：31.246±0.032 vs MAP 31.520 | 非Gaussian先验有价值；继续学习生成先验如何随新证据变动，不能只扩大ledger或增加稳定性权重。 |
| stateless/persistent在new上的差仅0.086dB，独立持久化收益不稳定 | 新架构必须在真实生成latent与有遮挡、外观变化的rollout上建立贡献，不能把CNN先验收益归给transport。 |
| fork与精确MHT+去重最大后验差8.1e-15 | 分叉/去重是正确系统协议，不是新核心概率算法。 |
| 部分观测取回旧版本42.70dB vs快速适应单状态23.10；标准版本MHT同效 | 可逆身份状态确有用；原创切口在如何学习、压缩、寻址并读出生成细节。 |
| true-change全registry误差均值改善，但32seed配对区间跨0 | 不能以此宣称稳定优势；保留这个挑战，继续做H3真实表示实验。 |

H3官方规格是f16t4d24视觉latent，进入DiT前patchify为1×2×2；Base FL2VA/Ref2VA是已发布BF16检查点。**H3-Regenerate-2K尚未开源，下载Base不等于取得官方SR模型。** [官方源码与架构说明](https://github.com/MiniMax-AI/MiniMax-H3)、[官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)。

本项目已核对的原生入口：`cond_video_latents [B,24,T,H/16,W/16]`→`video_patch_proj`，condition rows每个denoise step保留；keyframe metadata中的`resolved_frame_index/latent`需同步。第一阶段仅冻结E/D训练小状态模块，第二阶段必须实际接入冻结H3-DiT；只有VAE实验时应称“H3表示实验”。原生Comfy包装器可能采用inference/no_grad及latent格式转换，梯度与scale/shift必须用一次实际前后向确认。

## 候选A：学习可辨方向的 canonical latent transport（优先）

**核心假设**：能学习一个紧凑、可重定位的生成细节坐标系，使“新观测支持哪些方向”与“哪些旧细节可继续保持”在同一坐标中可计算；其非线性转移应同时胜过固定基+经典更新和同参数GRU。

- 每个canonical ROI保留anchor `z0`、固定身份基`U_phi(z0)`、系数`a`、真实packet证据、未确认episode引用。先用4×4 latent-token ROI：24×4×4=384维，rank8；1–4个活跃ROI起步。全帧global-rank8容易欠表达，逐token24维首轮代价过大。
- 观测函数 `F_t(a,q)=A_t W_t S_q(D_H3(z0+U a))`。`q`为当前亮度/色温/低频shading等临时外观变量；`A/W`先已知，随后训练对应。生成输出永远不增加证据精度。
- 局部线性化 `B=∂_a F, M=∂_q F`；在白化空间去除appearance nuisance：`P=I-Mw(Mw' Mw+eps I)^-1 Mw'`，`G=Bw' P Bw`。这一步是经典nuisance消元原语，贡献不能落在Schur complement本身。
- 学习transition `delta=T_psi(a,innovation,G,precision_increment)`。谱内校正先约束到受支持方向；另以同一个持久identity变量耦合新旧未定细节，训练跨后验的非线性延续。对照“只做谱投影/精确小维MAP”以分离真正的transport收益。
- packet新鲜度与信息增量分开：同方向独立测量也增加精度，不能继续使用旧core的`counts==0`作为唯一novelty。用fresh UID及`Δlogdet(I+C_old B'R^-1B)`/precision谱增量描述新信息；原样packet副本不更新。

**最短训练实现**：固定32个局部channel×DCT原子`P0`，离线缓存`K0=A W J_D(z0)P0`；训练`C_phi∈R32×8`并QR，令`U=P0 C_phi, B=K0 C_phi`。这样basis可训练，算子缓存不会因C更新失效。anchor/内容明显变化时刷新线性化；若直接训练U却不刷新旧B，会产生错误投影。8维transport用2层MLP或小卷积；先不反传33B DiT。

**训练batch与损失**：同一照片/短视频的两条共享prefix，只在一个ROI多给一次近景；4–8步unroll，监督新增信息后的同一远景查询，用RGB/latent重建损失，同时保留其它已证实ROI与未被新证据否定的旧identity。光照扰动单独生成`q`监督。用冻结D的真实输出计算LR残差，不能把学习operator自报一致性当通过。线性化只是一阶局部支持，trust region外需重算；D跨ROI作用必须实测。

**最有判别力的消融**：

| 实验 | 能排除的解释 |
|---|---|
| learned U vs同rank固定DCT/random U；相同参数/内存/训练步数 | 只是多给了一个压缩缓存。 |
| learned transport vs同U、同精确B的MAP/Kalman/fixed-seed coupling | 只是经典融合或共享随机数。 |
| decoder-Jacobian支持 vs像素mask/对角gain；同样最终像素一致性 | 只是多做了一次观测投影。 |
| 有/无appearance消元；纹理不变但亮度/高光变，再恢复原照明 | 把外观变化错误写成永久内容。 |
| 真新相位、同相位独立噪声、原样副本三者分开 | 把“有新帧”误当成“有新信息”。 |

主判据是**近景后再拉远的真实细节增益+未约束区域旧身份漂移**两轴同时改进。重定位失败、状态错误、H3读出不利用reference必须分别测，不能只报最终平均PSNR。

## 候选B：可逆 content/appearance code 与证据条件编辑

**核心假设**：直接学习H3 latent的可逆因子化，比在纠缠latent中加时序gate更容易同时做到“光照可变、身份可留、新细节可改”。它与A的显式局部Jacobian路线不同，首轮可不依赖昂贵JVP。

- 在24-channel latent上放4个小型affine-coupling层，构造可逆`F_phi(z)=(c,a)`；`c`为canonical内容场，`a`为每chunk重估的低容量appearance分支。appearance需空间平滑/低容量约束，防止它偷存整张纹理。
- 持久状态只保存`c`及必要的旧epoch；`a_t`由当前LR推断。`R_psi(c,LRinnovation,geometry,evidence)`产生内容编辑，当前读出`z_t=F_phi^-1(c_updated,a_t)`，经原生H3 guide注入。没有真实可见证据时query只读，分叉/合并协议沿用标准MHT控制。
- 训练用**反事实交换**而非只做latent自重建：同一内容不同光照交换a应复原对应光照；相同光照不同局部内容交换c只改变该内容；远→近→远时近景应更新c而不锁死a；真实局部换纹理需新epoch且未变c保留。自然图像+有监督photometric/局部替换可开训，真实视频再检验。
- 首轮E/D冻结，训练coupling、LR encoder和小edit head；latent L1+decode RGB/梯度重建+counterfactual swap+prefix query保持。RGB训练所用HR是监督，正式推断仍只读LR。第二阶段将学习code作为native guide输入H3-DiT，固定大模型调用次数。

**杀手消融**：同参数非可逆双head、没有swap训练、单一混合memory、只更新appearance、只更新content、A与B组合。最关键一组是“纹理A/光照1→纹理A/光照2→近景揭示A→光照1回访”，再配“同样光照变化但局部内容确实变B”；必须前者不永久改身份、后者及时改内容。若只在已知亮度增益下有效，应扩展材质/阴影/高光，而非立即缩成亮度校正claim。

## 最可能重叠的前作与怎样超越

| 前作（本次核验primary source） | 已覆盖／本项目必须多做 |
|---|---|
| [SILO, ICCV2025](https://arxiv.org/abs/2501.11746) | 已学习latent degradation operator减少E/D调用；“可训练operator”本身不新。A要建立canonical细节坐标+跨新证据的可修订状态，并超出一次恢复。 |
| [ReSample](https://arxiv.org/abs/2307.08123) | 已解决非线性latent硬一致性及回到噪声流形；解码后像素校正不新。 |
| [HDPS, ACM MM2026](https://arxiv.org/abs/2608.00537) | 已讨论decoder Jacobian盲区并混合像素/latent；Jacobian测量不是独立贡献。A需证明受支持方向学习和持续身份运输的共同收益。 |
| [Sequential Posterior Sampling](https://arxiv.org/abs/2409.05399)、[KEEP](https://arxiv.org/abs/2408.05205) | 学习时间转移、Kalman生成prior已存在；warm start/递归融合不足。 |
| [LoCoVSR](https://arxiv.org/abs/2609.32742) | 已共享noise trajectory；H3 shared-seed/trajectory必须是强控制。 |
| [NeRF-W](https://arxiv.org/abs/2008.02268) | 已将appearance、transient与静态场分开；B不能只声称内容/外观解耦。新增问题是冻结视频生成latent中的因果证据编辑与可逆身份回访。 |
| WorldMem/Mirage/REEL等，见原sources文件 | 可定位/检索/更新生成memory已拥挤；多解释记忆本身也不是空白。 |

建议先训练A的局部basis+transport最小实现，保留B的反事实appearance训练作为第二条强候选。遇到H3读出吞掉状态时，先用同一读出器的已到达reference oracle定位接口问题，再训练小native-guide adapter；不能退回纯像素MAP并把它称为H3新方法。两条架构都尚未获得H3实验证据，当前目标仍是确立稳定、原创的新核心，而非以toy成功或已知负机制结束。
