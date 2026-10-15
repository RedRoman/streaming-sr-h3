# B3R：真实 H3 原生表示中的可撤回压缩证据状态

## 目前结论

本轮没有证明独立的学习方法收益。六个参数配平控制均已完成原定 600 latent + 200 RGB 阶段；原协议合法地按 val 指标选 best，随后另做了统一 last600 → fresh RGB200 → last200 的补充实验。两者不能混比，也不把补充实验称为原实验的数值修复。

核心问题已超出 group closure：实际观测进入 lift 前，24 通道 H3 表示能恢复细节；经过现有 4 通道 lift/readout 后，质量低于 current-LR anchor。相同 appearance 时绕过 action 也不能恢复，因而不能仅归因于 latent 光照非等变。单源固定目标诊断另行记录，不把它当跨源方法效果。

## 数据、模型和固定协议

- 32 train / 8 val 个自然图像源，源与 donor 不跨 split。BSDS 这些源曾在本项目更早实验中出现，不能称为全新封存测试集。本轮未访问旧 16 test 或封存 DIV2K。
- 自然图像上的已知合成相机与 18 步机制 episode：4 个互补 near footprint、两个已知 nuisance group、延迟 far 光照线索、暂态前景、真局部替换、精确重复 packet、只读查询。group tag 由模拟器给定；所有控制共享，不能称为从真实 LR 自动发现分组。HR 仅监督或评价。
- 原生 MiniMax H3 VAE FP16，SHA256 `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`；冻结 E/D，非 DiT 实验，proposal 明确为零。256² 输入重复 22 帧只提供原生 VAE context，并非 22 个观测；latent `[1,24,7,16,16]`，取实际 D 最后 RGB 帧。Torch2.10.0+cu128，H10080GB；cuDNN/matmul TF32 均关闭。
- 六控制 used parameters 146028–146576，差异不足 0.4%：closed action、no equivariance、same-compression factor-cache MLP、nonlinear group、online GRU、可重放 GRU。共同的 causal LR-only IRLS 光照/可靠性控制器、current-LR anchor、history-supported far-null readout，以及相同 64 步 affine/box DC。
- 原 fixed budget：Adam .002/.0005，batch4，600/200 更新，appearance10、equivariance.5、RGB5。**RGB 阶段只在每四步的 batch 首源单 query 反传真实 D：每模型 50 次 actualD backward，覆盖固定 schedule 中 27/32 train 源，而 latent L1 每步存在。** 这是一项实质限制，不能据此否定 4ch 容量或整体学习假设。val actualD 用于原协议 checkpoint selection；不能把原 best 模型误称为最后一步模型。

## 原 selection-pilot 与补充 endpoint

数字是 DC 后 union PSNR，先在每源内平均 7 个 query，再对 8 源平均；一个训练 seed17。完整 raw/DC、区域、query、源配对 CI 在两个 summary 目录中。

|方法|原 val-selected best|补充 last600→last200|
|---|---:|---:|
|current LR-VAE|23.84270|23.84270|
|closed action|23.67001|23.65541|
|no equivariance|23.62371|23.61307|
|factor-cache MLP|23.68954|23.67501|
|nonlinear group|23.67710|23.67125|
|online GRU|23.65372|23.65337|
|replay GRU|23.66819|23.65584|
|LR-only photo/reliability revision atlas|27.51398|27.51398|
|H3 preconditioned decoded atlas|26.79536|26.79536|
|LR inverse → actual native E/D|26.51293|26.51293|

原 pilot 的 full−current 为 −0.17269 dB，8 源全负，95% source bootstrap CI [−0.22616, −0.12343]；full−factor-cache 为 −0.01953，1/8 胜；full−replay 为 +0.00182，4/8 胜且 CI 跨零；full−强 LR atlas 为 −3.84397，8 源全负。补充 endpoint 对应 −0.18729、−0.01960、−0.00043、−3.85857 dB，结论一致。

原 pilot 仅 appearance correction 的固定 checkpoint 干预收益 +0.000033 dB；可靠性 revision +0.00313 dB。移除 measured detail 的 delta 反而提升 0.09277 dB；跨源打乱历史仅下降 0.01091 dB，CI 跨零。故当前网络没有把真实历史细节转换成有用的新增证据。full 比 no-equiv 高 0.04630 dB 不足以支持该核心，因为两者都输 current LR 和标准 factor cache。

原 checkpoint steps：stage1 full best500，其余600；stage2 full150/noeq150/cache1/nonlinear150/online200/replay100。原 `rgb_stage2/val_seed17` 完整保留。补充 `rgb_endpoint_stage2/val_endpoint_seed17` 全部明确读取 stage1 last600 和 RGB last200；各 checkpoint 路径、SHA256、initial path 在各 evaluation `complete.json`。

## 不调参的单 packet 实际 D 诊断

使用补充 endpoint 同一 checkpoint；8 val 源的 packet0，t3 同 appearance、t4 中性 appearance。仅保留该 packet 的支持区域。以下为该 near footprint 的 DC PSNR；HR teacher 行是明确不具备有效 LR-only state 资格的评价 ceiling。

|固定读出|同 appearance t3|跨 appearance t4|
|---|---:|---:|
|current LR-VAE|23.78588|23.47430|
|pre-lift 24ch delta 直接 D|26.23795|25.95068|
|pre-lift full native code D + pixel photo bridge|26.74451|26.40058|
|pre-lift delta+proxy 重组 D + pixel bridge|26.74417|26.40022|
|LR-only 物理/photo bridge 后 native E/D|26.81799|26.51593|
|现 checkpoint lift→action→readout→D|23.55108|23.24233|
|绕过 action，其余相同|23.55107|23.24242|
|仅把 measured delta 置零|23.69032|23.38463|
|仅跨源打乱 measured delta|23.55158|23.24661|
|HR teacher delta，evaluation only|26.82612|26.57158|

这排除了“原生 H3 完全没有保存该观测细节”和“仅仅因为 appearance action 不等变”作为主要解释：同 appearance 已损失约 3.19 dB；action/bypass 的 latent RMSE 仅 2.05e−7。同一单 packet 情况下，latent target L1 从零输出的 .24001 降到 .16953，实际 RGB 却比 anchor 更差，表明 latent surrogate 改善不等于可用细节恢复。

该结果定位到**学习 lift/readout 与当前训练目标/优化过程的联合失败**，不证明 4ch 的信息论容量不够。完整 24ch 和相同 pixel bridge 提供可行的真实观测控制。初始化永久死分支也已被训练前真实数据与 nativeD 梯度检查排除。跨 appearance 的 24ch 直接 delta 和 pixel bridge 仍有差异，不能把 native latent 加法视作严格光度等变。

### 固定单训练源 300+300 目标诊断

已实际完成，41.372s。固定 train_100075 / packet0 / t3，同一 endpoint checkpoint，原4ch lift/readout、Adam.0005、clip5，各300更新。VAE/core/appearance全部冻结，双方给同一个模拟器正确 nuisance q；这是结构诊断的已知 nuisance 输入，不能套用为主实验的 LR-only appearance 估计。目标是**实际 LR-derived pre-lift full24 native code / nativeD RGB**，两个臂都没有 HR teaching；HR 只评价。0/50/150/300固定读出，不选 best。

|固定步数|latent-only：HR near DC|每步 actualD RGB-only：HR near DC|
|---|---:|---:|
|0|25.85974|25.85974|
|50|25.63969|26.70694|
|150|25.64332|27.61269|
|300|26.01277|28.28402|

共同 current LR-VAE 为26.30350，实际full24D teacher为29.11052。RGB臂最终比初始恢复+2.42428dB、比latent臂+2.27125dB；对实际full24D teacher的近景PSNR达到34.73761（latent臂28.64813）。反过来，RGB臂的latent L1从.17061变为.25599，而latent臂降到.11603。目标超出readout±2范围的比例仅2.325e−5，不能解释主要差距。

首步真实梯度均非零：latent臂encoder/readout L1梯度8.624/14.104，RGB臂.1582/.3205；frozen VAE/core/appearance0grad。因而4ch与当前readout至少**局部可以**表达大量已观测细节，旧结果不能据以否定其容量；decoder-aware目标与稀疏RGB监督是明确需要控制的因素。一个训练源可记忆anchor，不能把单源恢复当作跨源证据写入成功。固定saved300输入干预进一步表明：RGB臂actual28.2840、delta0为26.5359、替换另一TRAIN源delta为27.5601；latent臂为26.0128/26.3550/25.7104。局部恢复确实依赖输入delta，但仍不能由单源拟合声称泛化。后续32/8全actualD实验已另获授权并冻结为decoded_v4，独立预算/结果归档，不混入本报告。

## 状态、计算和可撤回代数

真正 incremental `RevisionRuntime` 在首次 near 到达时 lift 一次，之后保留 h/weight/id/group，不保留 native delta/proxy。对 vectorized 18-step 训练 rollout，六控制 maximum latent difference 为 0（前四种）和 ≤2.51e−6（GRU）；duplicate/query 只读均通过。full group rebase 与逐因子 replay 相差 1.86e−9，单因子撤回 9.31e−10。它们是结构正确性检查，不能替代图像质量收益。

8 packet、2 group 的实际未优化增量 tensor+metadata state 372000B；controller 的 reference LR/descriptor/group fields 55440B；当前 LR+native anchor+decoded RGB scratch 1007616B，总计 1435056B（约1.37MiB）。另有共享 camera matrices 524288B、full FP32 adapter weights584896B、冻结 VAE weights5207737968B；Python 开销、activations、完整 GPU peak 分列。eval 实测 peak7168429056B。

每个 4ch h 为28672B，但还需要 weight1024B、8×8 RGB descriptor768B、ID/group/geometry约24B。8 条 raw-LR inverse archive 为393216B，连同同类控制器/metadata约457040B；当前 learned persistent+controller427440B，只省约29600B，而额外 NN 权重更大。不能把 4ch 和 dense RGB atlas 的比值当完整状态压缩优势。共同 decoded anchor786432B 本身就大于 detail code。

固定2 group，N=8→512 的 isolated full appearance action约0.503→0.502ms，factor-cache非线性重解释约0.241→2.506ms。但是包含 ID/weight scan/allocation 的 whole revision，full约1.438→39.802ms，factor-cache约1.178→37.361ms，replay-GRU约2.560→140.129ms。当前完整实现并未比强 factor-cache 更快；nonlinear group 也有固定组数下常数成本的普通控制。LR controller 重新评估本身还可能随 N 增长。**只对 group action 子步骤声称固定 N 无关，不对整个流水线声称 O(1)。** N>8 仅重复一个真实 feature 做复杂度压力测试，不扩大独立样本数。

## 计算与可复现性

原 cache255.026s；saved-query baseline decode-route 对齐17.333s；六控 latent600共365.865s，RGB200共206.517s；原 val72.406s。补充六控 RGB200共208.848s，val73.480s，stream audit3.428s；单 packet信息诊断15.504s；固定单源300+300诊断41.372s。原预算3600 latent+1200 RGB更新，补充额外1200 RGB更新/300actualD backward，单源诊断额外600更新/300actualD backward，明确单列而不隐藏。两次正式 val 各168 E、848 D。

cache 原 E-output D 路线与 saved half-code runtime D 有约5.87e−5 RMS 的小差异，而相同 runtime D 重复完全一致。仅将 baseline RGB 从实际保存的 query tensor 重解码，未改观测或 latent labels；不对未隔离的差异来源作猜测。对齐后 zero-state actual RGB maxabs0。

完整命令/实际 sessions 在 `SESSION_REGISTRY.json`、logs、各 config；数据源与模型 SHA 在 manifest/complete 文件；代码 snapshot 与 `SOURCE_HASHES.json` 随结果保存。metadata、全部 best/last/Adam checkpoints 与图已归档本地；较大 cache `.pt` 保留 H100。统计使用10000次 source bootstrap，先平均同源 query；不能将 packet、epoch 或 seed 当独立视频。

## 与前作重叠及下一判断

仿射 action、可加充分统计与 factor replay 都是已有结构。[Deep Sets](https://arxiv.org/abs/1703.06114)、[Homomorphism Autoencoder](https://proceedings.mlr.press/v202/keurti23a/keurti23a.pdf) 直接覆盖 sum representation 与学习群作用方面；appearance/memory adaptation 还与 RSDN、RECAP-Forcing、Multi-Memory Streams 重叠。本轮只能作为受控候选和失败定位，不把它称为新原创方法。

下一学习核心须先在 decoder-aware 的真实 LR 约束下恢复单条新增细节，再证明跨源/组合更新优势；不继续依靠 closure 代数、projection/reencode 摊销或更多相同 toy 宣称进展。是否需要更完整 actualD 跨源训练，应由固定目标诊断决定，并另预声明预算。
