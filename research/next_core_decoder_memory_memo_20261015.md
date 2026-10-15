# 下一核心独立研究 memo：把“修订记忆”收敛为可在未来读出的证据编码

日期：2026-10-15。性质：只读研究与可执行实验建议；未运行训练、GPU、采样或修改 A/B 文件。先读了本项目 `sources_and_method.md`、`h3_method_review.md`、`memory_and_revision_prior_art.md`、`sequential_generation_prior_art.md`、最新 RUN_LOG；再核查原始论文、作者仓库及 B5 实际计算图。以下“候选”是待验证的机制，不是已经获得的原创性结论。

**建议仅推动一个新核心候选：在未来 cue 到达前完成量化、按未来物理读出的可混淆性分配代码的证据 codec。** 其状态不是一张当前重建图，也不是已选定的 appearance-corrected latent；它保留将来 cue 可能需要区分的历史。先把问题改为“同一严格字节预算下，保留多少未压缩历史的可读收益”，再谈生成身份或更复杂更新器。B5 的像素 bridge 应保留为信息路径诊断和强底座，不以 bridge/skip 作为论文贡献。H3 innerDC 则保留为强冻结读出参照，暂不把约72秒/22帧的采样链放进新核心的首轮训练循环。

## 1. 本地结果现在支持什么

| 已有证据 | 可推出的判断 | 不能推出的判断 |
|---|---|---|
| H3 native480 innerDC+DC25，8源均值27.2457dB / LPIPS .32654；base27.0948/.36273；matched VAE27.1608/.37225。PSNR7/8、LPIPS8/8改善 | 真实H3先验在正确物理约束下有增量，不能因旧低分辨率/错误IO实验否定H3 | 已解决长期记忆，或所有指标全面胜VAE |
| native480时序残差DC25：inner .0285458，base .0292499，VAE .0284098，覆盖94.49% | 对base的平均时序误差也改善；相对VAE仍有小的时序代价 | 直接把该残差称为闪烁，或称时序全面优于VAE |
| native480 base平均28.70s，inner72.77s，均49NFE | 新增E/D/投影是实成本，等NFE不是等计算 | 把短clip离线walltime当流式吞吐 |
| B4 4704次实际D训练后 full≈24.114、anchor≈24.106；移除历史约.0011dB，配对区间跨0 | 真实D梯度通了，但当前网络几乎没有形成有用历史通路 | 压缩历史信息本身无用；更不能以非零梯度证明记忆学会了 |
| 强LR atlas≈27.514；B5开始检查 full24 bridge 与 anchor shortcut | 存在可用历史细节，编码/表示/读出链路是首要瓶颈 | 需要再增加递归层或训练步数 |
| A的压缩位于cue之后，early q已onehot | 当前结果不能验证“先压缩后被未来cue解开”的联合歧义 | 可以把当前A的正负结果外推到真正pre-cue状态压缩 |

主要本地证据：`results/h3_setup/sampling_dc_davis8/{REPORT.md,sourcewise_metrics.json,cost_audit.json}`；`sampling_dc_davis8_temporal/{REPORT.md,sourcewise_metrics.json}`；`results/h3_factorized/decoded_v4/history_dependence/summary.json`；A的 `operator_function_dc_v1/{association_prefix_audit.json,readout_truth_audit.json}`。B4干预不是“历史完全无效”的信息论证明，只是固定架构/预算下的经验事实。

## 2. 原始前作把哪些空间占住了

| 原始来源与核查深度 | 已覆盖的关键机制 | 本项目需要增加的东西 |
|---|---|---|
| [Predictive-State Decoders, NeurIPS2017](https://arxiv.org/abs/1709.08520)，作者摘要 | 给RNN状态增加预测未来观测的监督，将未来预测与充分状态联系起来 | 不能把“未来损失训练状态”本身称新。需要明确迟到cue、实际字节预算、物理可读性和强codec控制 |
| [Conditional Neural Processes, ICML2018](https://proceedings.mlr.press/v80/garnelo18a.html)，原始论文页 | 从少量context观测形成表示，预测不同query | 不能把set encoder + query decoder、可查询状态本身称新 |
| [DSIN, ECCV2020](https://arxiv.org/abs/2001.04753)及[官方代码](https://github.com/ayziksha/DSIN)，作者摘要/README | 编码器看不到decoder侧相关图像，decoder联合使用code和side information | “future cue只给reader”与“先编码后看side information”已有直接图像先例 |
| [Neural Distributed Source Coding, 2021预印本/2024修订](https://arxiv.org/abs/2106.02797)及[官方代码](https://github.com/acnagle/neural-dsc)，作者摘要与实验开关 | conditional VQ-VAE处理decoder-only side information；官方明确有joint/distributed/separate三配置和codebook bits | 这是最强直接学习基线；不能只对无cue的AE或GRU赢 |
| [Neural Distributed Compressor Discovers Binning](https://arxiv.org/abs/2310.16961)，作者摘要；[早期ISIT版](https://arxiv.org/abs/2305.04380) | 神经量化能学到Wyner–Ziv式binning与side information解码 | 不把“保留歧义的离散bin”或普通neural Wyner–Ziv当新理论。OpenReview全文被验证页挡住，未据此声称完整复现 |
| [Generative Diffusion Model-based Compression of MIMO CSI, 2025](https://arxiv.org/html/2503.03753v1)，原文 | 固定长度code结合decoder侧信息，使用条件diffusion恢复目标 | 给conditional codec换生成decoder也不是自动新颖；本项目需要物理观测/未来读出特有的码分配与实验增量 |
| [A Decoder Suffices for Query-Adaptive Variational Inference, UAI2023](https://proceedings.mlr.press/v216/agarwal23a.html)，PDF摘要/方法入口 | 直接依赖生成decoder，针对任意缺失/观测结构做非摊销后验推断 | “decoder-aware + 新query再优化”已有直接先例。保留全历史/实际D优化为强上限，不能把它的算力省略 |
| [Hilbert Coresets, JMLR2019](https://jmlr.org/beta/papers/v20/17-613.html)，原始论文页/PDF摘要 | 在log-likelihood函数空间压缩数据；随机投影与稀疏加权 | 仅把记忆改叫likelihood function、packet coreset或小factor graph，仍不足以形成新机制 |
| [The risk of KV cache compression, 2026](https://arxiv.org/html/2607.01520v1)，§3.2和摘要 | 可压缩性取决于未来query可见的response profile；普遍压缩保证受到限制 | 它是attention上的结果，不是H3定理；但提示我们测未来读出的可区分性，而不是仅测latent MSE |
| [Learned Primal-Dual](https://arxiv.org/abs/1707.06474)及[官方代码](https://github.com/adler-j/learned_primal_dual)，作者摘要/README；[SILO](https://arxiv.org/abs/2501.11746)，作者摘要 | 物理算子进入学习迭代、学习latent degradation以减少E/D | 学一个创新更新器或latent operator、把72秒链摊销为小网络，首先应视为已有方法的适配控制 |

原来的 Sequential Flow Matching、Sequential Posterior Sampling、KEEP、SparkVSR、ReCaVSR、WorldMem/Mirage/REEL 等已有综述继续适用；本次不把warm-start、轨迹复用、reference传播、检索与生成记忆重新列成新方向。另一个跨领域边界是 [TierMem, 2026](https://arxiv.org/abs/2602.17913)：压缩summary不够时取回原始日志已经有明确先例；保存全archive再回读不能冒充有限状态压缩。

## 3. 推荐机制：按未来物理读出的可混淆性学习离散证据代码

### 3.1 需要改变的状态对象

当前B把历史压成4-channel再要求一个以当前anchor为主的decoder用它补细节；这容易把当前重建误差的下降误当作历史学习。建议让writer的主任务改为：**历史从内存删除之后，未来cue仍能从一份很小的实际bit code中读回此前观测支持的细节。** Reader的RGB残差形式可以完全普通，全部方法共享；候选贡献集中在code如何分配，而不是skip。

因果图应为：

`past actual-LR packets → writer → serialized code m_t → [past packets不可访问] → reader(m_t, arrived LR cue/current LR/operator) → restoration`。

实际未来cue不能输入writer。训练时可知道未来query分布和监督目标，这与推理时看见该次cue不同。Code hash在cue到达前落盘；同一code分别接受正确cue、暂时错误cue、修正后的cue。Read不改code，因此“解释能恢复”首先是**同一记忆可被重新解释**，不是从已删除的资料凭空恢复信息，也不是对任意观测撤回有常数内存保证。

### 3.2 具体量化规则，而非只换训练名称

普通VQ按输入/latent距离选最近prototype，可能保留高能量外观，丢掉低能量但后续重要的区别。建立一个小的TRAIN-only decoder response bank，以未压缩历史的物理atlas/相同H3 prior为teacher，记作 `F(H,c)`：同一历史H在未来cue/operator c下可读出的历史残差。

训练时形成历史对 `(H,H')` 和未来cue c。只有c在两个历史下都物理可成立，而且 `F(H,c)` 与 `F(H',c)` 在后续观测支持区域显著不同，才把它们记为**不能共用code的冲突边**。若c本身已能区分两段历史，不必强迫code再保存相同区别。Writer用离散code的冲突分离损失和实际读出失真联合训练，避免把当前输出相同、未来输出不同的历史压成同一状态：

`L = L_readout(F(H,c), reader(Q_phi(H),c)) + lambda*rate + beta*future-confusability margin`。

有限枚举smoke可先用64bit；自然图像主试验则采用每个4×4 native ROI **896bytes**，与4ch×7×4×4 FP16状态等字节，而不是一开始压到8bytes。可用8组256-way product-code、每组112个时空token，各存uint8 index，合计896bytes。重点是code partition由未来物理读出的冲突决定，不是最近latent中心。对照也必须获得相同past pixels/operator、当前anchor、reader和全部监督，不能靠多输入取胜。连续软code训练的临时workspace与最终序列化bit数分开统计。

**这不是新Wyner–Ziv定理。** Binning、side-information coding、functional compression及图着色都有既有理论，例如 [On Network Functional Compression](https://arxiv.org/abs/1011.5496) 明确研究图着色和多个函数/side information；不能把本候选的冲突图定义当成新信息论原理。如果普通conditional VQ加同样future-pair训练就达到相同结果，合理结论是任务/训练修复有效，不能把新名字当新算法。潜在可发表增量需要落在：实际H3非线性读出/不同尺度与nuisance下，所提冲突量化比直接conditional codec更有效地保留可被新cue解开的观测差别，并支持固定code的多次重新解释。

### 3.3 首轮不要承诺的事

- 不承诺任意query下的有限充分统计量；若未来会随机查询所有独立细节，所需信息会增长。
- 只有当前/未来side information与历史有可利用相关性时，才可能有Wyner–Ziv式rate收益。一个与纹理独立的随机旋转标签可以测试pre-cue读出，却不会凭空降低纹理熵。
- 不承诺胜过保留全部LR的atlas。首先争取在小字节预算下保留它的大部分收益；同等字节下击败强conditional codec才是核心判据。
- 不把已到达cue之后再运行encoder叫pre-cue memory。A当前response bank只能作为renderer/teacher表，不构成新的因果证据。
- 不在所有arm都可取回full24/proxy24或RGB archive时报告“压缩状态”。需要独立reader进程，输入只含bitstream、已到达cue和共享权重。

## 4. 用现有缓存和少量新数据的正面验证路径

### 第0步：先等B5回答读出是否真的通了

保持其4arm预算。若所有有bridge的arm都只等于各自零训练base，且在**bridge固定**时删掉神经history仍无影响，先暂停更复杂group/transport训练。把full24解码证据直接读出的线路设为teacher；用最简单pixel residual codec证明观测细节可从被保存的code走到输出。否则继续优化状态代数没有对象。

### 第1步：CPU可完成的因果与可压缩性小审计

复用B的32TRAIN/8VAL自然图像源、actual-LR near/current缓存；A的48/72 response仅用作冻结renderer bank。另用这些自然纹理构造少量独立prefix：早期近景看不见方向/关联标记，未来远景LR才显露标记。标记需真实出现在LR像素中，不能以GT标签/文件名/group ID把q传给writer。可先四种离散方向/两种关联，全部枚举检查：相同早期prefix必须兼容至少两个不同未来解释；early cue classifier不能已onehot；writer输入中的pose、camera/group描述不得暗中消除该歧义。

这个方向/关联任务只验证“先压缩后读”的因果能力，不承诺side-information rate收益。真实压缩效益另用相关near/current-far配对：current far保留公共低频，历史code负责它不能读出的局部细节。若当前真实缓存已经泄露全部q，只能用来建立信息通路，不能为联合歧义作证。

先在cached pixel/response表上训练writer+reader，不加载DiT/VAE；自然图像主预算预声明896bytes/ROI，448/1792bytes作完整次要rate曲线；64bit只用于有限枚举smoke，不用VAL选一个最好点。Writer前端与普通conditional VQ/CNP/GRU共享；新增机制仅为confusability quantizer及其显式训练对象。一次固定的小预算，例如32TRAIN、8VAL、seed17、各600updates，全部case都报。这里的预算是建议，正式启动前应整体冻结，而不是在VAL失败后改损失/增步。

### 第2步：先验证能力，再一次真实H3读出

必须同报：cue前重建/歧义、cue后受支持区域误差、其它区域漂移、code-only恢复率、错误cue→正确cue的恢复、同prefix重复query的code SHA不变、history code zero/shuffle的影响。空记忆/仅当前cue是强消融；若它与full code相同，该任务仍没有检验历史。

直接baseline应为：

1. 未压缩actual-LR photo atlas / decoded atlas，以及full24+proxy24缓存：不限制容量的强参照，时间和字节真实计入。
2. 相同总字节的RGB残差DCT/PCA/量化atlas，以及可用的学习图像codec；不只压H3 latent。
3. [neural-dsc](https://github.com/acnagle/neural-dsc)式**distributed conditional VQ**（encoder无future cue，decoder有cue）；再给它相同future-pair loss，分离损失与量化规则的贡献。
4. 相同字节、相同前端/reader的DeepSets/CNP、GRU；不能把相同维数的float向量与bitstream称等容量。
5. cue预先给encoder的joint codec：标明违反线上协议的容量/信息上限，不能混进eligible方法。
6. 小枚举任务的精确posterior表/显式factor与简单图着色/量化：避免把本可由极小经典统计量完成的任务当学习贡献。

首轮主问题是“同码率的conditional VQ是否仍丢掉未来可读的区别，而所提code保留了它”。如果只比无side-information AE好，机制尚未成立。若码率优势成立，再在固定TRAIN/VAL上各抽预声明case，用相同FP16真实D检查cached-readout的正信号是否还存在；只做1次D/query，native22上下文与已审计精度一致。不要在这之前新开一轮49步H3教师采样。

## 5. 实际新增资源与状态核算

| 阶段 | 新计算 | 状态/缓存 | 新观测与等待 |
|---|---|---|---|
| prefix与ambiguity审计 | CPU变换/枚举/缓存读写，0 DiT、0 VAE | 复用现有TRAIN/VAL图像和response bank；新cue帧及source hash | 未来标记必须真实在LR中到达；writer前看不到它 |
| cached codec pilot | 小CNN/MLP训练；固定预算，0真实D反传 | 自然图像896bytes code/ROI为主；几何、置信度、UID、codebook和reader参数分别计数 | reader只用已经到达的current far/cue；不能回读过去archive |
| 真D能力核验 | 建议8VAL×2cue×4主要控制=64次FP16 D，0 DiT；真实时间待测 | 相同native22 current latent与共享模型；不隐去其buffer/anchor成本 | 与所有控制同prefix、同可用cue，不增加HR或未来信息 |
| 后续H3生成接入 | 只有前两级明确成功后才考虑；固定同seed/同native49/同innerDC | 统一生成初态与proposal数；不把reference缓存免费 | 记录实际latest distinct observation与block等待，不用静态重复图代替实采等待 |

896bytes不是系统总内存。一个4×4 native latent ROI的24×7×4×4 FP16 anchor就有5,376bytes；full256 native current slab有86,016bytes。若要保留初始H3身份，anchor必须算进持久状态。每个past full24+proxy24 ROI约10,752bytes，两个packet约21,504bytes，另有RGB/controller/ID等。即使历史payload缩到896bytes，共享anchor也会限制总压缩倍率。Codebook与reader是共享权重，需同时报单流resident bytes及多流摊销，不能只宣传896bytes。

对“可修订”的第一步仅要求：同一pre-cue bitstream在不同已到达cue下可重读并恢复，不重写旧历史。如果进一步要求撤回历史packet，必须保留可删除的packet code或其它额外结构，明确O(N)与误差；不能从一个丢失信息的定长state免费实现任意撤回。

## 6. B5 shared bridge独立判定

已读 `src/h3_factorized/revision_bridge.py` 和 `experiments/h3_factorized/B5_FACTORIAL_PROTOCOL.md`。实际图为：

`x = B_S + H_common[ D(z_current + delta_theta) − D(z_current) ]`。

S=1的B是用actual-LR full24经native D得到的共同decoded atlas；S=0是current D。H由独立、因果LR controller给定；learned appearance不进入B/H；没有再E编码。因此它确实绕开了“观测细节必须穿过4ch压缩并被H3D重建”的路径，适合定位信息丢失。A开关只移除神经readout的current-native conditioning，物理D仍有current anchor；不能称完全去anchor。

但这条式子属于强重建底座上的受约束残差学习。与 [Deep Null Space Learning](https://arxiv.org/abs/1806.06137)、[RegNets](https://arxiv.org/abs/1812.00965)、MoDL/learned primal-dual的已有思想直接相交；B的H含因果支持/权重，并不能直接继承理想正交nullspace网络的收敛定理。NoE bridge或decoder-residual subtraction是合理工程选择，不是单独论文贡献。

B5最有价值的正结果应为：在固定正确bridge下，神经history removal/shuffle显著破坏额外读出，而正确history稳定补足bridge不能表达的误差；gain必须相对本arm的零训练B报告。若收益全由S开关带来，只证明信息路径修复成功。所有arm保留并使用full24+proxy24，协议自己也明确不作压缩claim；独立评审赞同这点。

## 7. 给root的具体建议

先完成B5四arm，不中途换目标。若bridge成功、神经history仍无用，把它当强teacher/信息通路，不继续扩大现有closed-action state。随后只开一条小的pre-cue code实验：独立reader、实际bit-pack、future cue只给reader、conditional VQ和量化pixel atlas齐备。其正面研究目标是**较少历史字节保留未来可以解开的细节**，不是在易物理模型上击败全部raw evidence，也不是为现有group代数换名字。

原创性仍有一个硬闸门：若普通neural-dsc / conditional VQ、其相同future-pair loss版本或简单物理量化atlas就给出相同rate–utility曲线，候选只能作为强baseline/任务修复。若在严格causal commitment、相同总字节和计算下，decoder-confusability code对未见source/未见cue组合保持更好的未来读出，再考虑把它发展为主机制；此时新东西是可复现的码分配/状态能力，不是“Bayes、充分统计或未来信息”几个词。

H3 innerDC的8源正信号应继续放在报告主表，它是当前最可靠的H3强底座；同时保留VAE时序优势与72秒成本。第一轮新核心不需要额外49步采样。该建议允许大胆替换“先学记忆更新器”的顺序：先学并证明历史信息能在严苛容量下跨未来cue存活，再研究如何在长流中增量维护它。
