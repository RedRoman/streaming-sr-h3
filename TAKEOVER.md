# Open-H3 restoration 接管记录（2026-11-11）

本分支从 `archive/modelarts-20261103` 的 `43efbd4d090056365dc46f0ac6edb88e3d875761` 建立。
远端工作目录为 `huawei-modelarts-31984:/cache/zihaohe/omni_openh3_20261111`。
原项目 `/cache/zihaohe/streaming_sr0` 的权重、数据、运行时通过链接复用，所有新输出写入本工作目录。
原项目未提交的音频代码和结果已有快照。旧 `queue_sharp.sh` 在仅等待 `sleep 240` 时终止，证据在 `evidence/old_queue_takeover.json`；没有终止其他人的 NPU 任务。

端口排查后，实际 NPU 执行迁至 **30794 的卡6**，同名隔离目录 `/cache/zihaohe/omni_openh3_20261111`。31984 保留数据准备资产。30794 复用已有底座/VAE/Comfy/运行时，迁入 refiner、Turbo、120 latent 与8个RGB NPZ；130个文件逐一哈希校验。该主机原项目没有 `.git`，因此采用从本分支打包、按199个运行文件 SHA 校验的隔离副本。

30794 使用 `pilot_queue_30794.json`、`refiner_turbo_30794.json` 以及重写路径后的 train/dev manifests，CANN环境为 `/cache/yunfeng/envs/minimax-h3-npu`。仅选择卡6：卡7存在既有评测队列。另一自动队列的剩余任务被已有 review hold 暂停；本队列每步前核验此 hold 及卡的空闲状态，条件变化就停止。没有修改其他队列、hold或工作负载。

## 当前路线

首轮冻结 H3 底座以及 Open-H3 refiner rank-384、release Turbo rank-128，另加 rank-8 LoRA。各预训练增量在 FP32 中累加一次后转回模型精度，不再叠加旧 Stage-1 或旧 Turbo。208 个 linear 模块必须完整匹配；新增 LoRA 从零输出开始。权重路径、大小和完整 SHA-256 位于 `configs/openh3_20261111/refiner_turbo.json`。

训练保留独立 FP32 LoRA 参数。推理严格验证 checkpoint 来源后，将新增 LoRA 的增量在 FP32 中合并并转回底座精度，再移除 wrapper，避免 Comfy partial offload 将 LoRA 留在 CPU。该推理合并存在 BF16 舍入，不承诺与未合并训练前向逐元素相同；真实 RGB 开发评估按实际部署形态进行。

采用原生三步 sigma 网格 `[0.909375, 6/7, 2/3, 0]` 和显式 draft warm start。这里是现有 ComfyUI FL2VA 后端的组件迁移，仍使用原视频 VAE、缓存的中性文本上下文；本轮推理报告记录 `audio_init=zeros`。**没有完成上游 Diffusers 数值对齐，也不是 Open-H3 官方完整管线复现。**

[Open-H3 上游说明](https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/) 的完整路径包含 latent upscaler、refiner/Turbo/Sharp、特定音频初始化与可选 fast VAE。当前首轮 draft 明确为 `E(U(LR))`，没有冒充 pretrained latent upscaler；Sharp 暂不加入，以便独立判断其纹理和真实性影响。fast noncausal VAE 也不直接替换，否则已有 latent、质量和流式延迟会同时改变。

首轮新增 LoRA 使用 `warmstart_x0` 监督：在三个推理噪声点将 draft 与噪声混合，再监督预测的 clean target。它是有限预算的恢复适配实验，不是上游 refiner 的原始 flow-matching 训练复现。进入大规模训练前，应与 clean-target flow matching 对照，并检查三步实际 RGB 输出，不能仅凭 latent loss 下降选路线。

## 数据与证据边界

`results/takeover_20261111/data_audit_31984.json` 记录来源清单。历史 heldout 96 个来源中，有 87 个在 Stage7 train、9 个在 Stage7 val；这意味着不能拿它们验证新的 Stage7 训练。原先 Stage-1 pilot 使用另一套生成视频，本次证据不能据此反推其旧主表泄漏。

新 SR 工程 pilot 排除所有历史评估来源：32 train 来源/96 windows，8 dev 来源/24 windows。两侧来源互斥；已有 latent 均为 `[1,24,7,30,52]`，`target/guide_raw/guide_oracle` 类型与来源已在 CPU 核验。实际训练仅用 `guide_raw`。开发评估每个来源取 w0，以对应 NPZ 的真实 RGB `hr` 为目标，比较固定 seed 的 pretrained 与 adapted 输出。

另有同一 clean target 的 SR、deblur、denoise、compression 四任务计划：非 SR 任务保持分辨率，模糊只施加到观测，目标保持原视频帧。原视频是高质量记录的代理参考，并非保证无损的理想真值。manifest、实际 materialization 和 VAE encoding 是不同状态；只有生成产物并核验后才能提升状态。

四任务 materialization 已在31984完成：40个target、160个observations，共200个NPZ、2,205,237,727 bytes。已独立核验文件与像素SHA、22帧长度、SR/native尺寸、四任务共用target及来源互斥。文件在 `datasets/omni_pilot_v1/`，小型证据在 `results/takeover_20261111/materialized/` 与 `cpu_materialization_validation.json`。尚未做这批数据的 VAE encoding，多任务训练未启动。

**目前没有新的 frozen test。** 这 8 个来源是开发集，不能用于泛化结论。进入论文/发布结论前须增加从未用于训练、模型选择或退化调参的来源，包含真实退化与未见组合。

## 有限实验队列

`configs/openh3_20261111/pilot_queue.json` 固定以下顺序：

1. 实际 H3 一步小裁剪训练：零 LoRA 输出对齐、仅新增 LoRA 有梯度、底座冻结、B 矩阵确实更新、checkpoint 清空后重载输出一致。
2. 新进程载入该 checkpoint，完成三步采样与 VAE decode。
3. 8 来源 pretrained RGB 开发集基线。
4. rank-8、学习率 `5e-5`、300 step 训练，每 50 step 保存和开发验证。
5. 同来源、同输入、同 seed 的微调 RGB 开发评估与逐源配对差值。

任何一步失败即停止；不自动扩展大规模训练，不把小裁剪 smoke 当作图像质量实验。结果位于 `runs/openh3_pilot_20261111/`。队列要求卡健康、无现存进程、空闲至少 50,000 MB，并二次确认；文件锁只对遵守同一锁的任务有效，不能替代集群调度器。

启动命令：

```bash
cd /cache/zihaohe/omni_openh3_20261111
python3 -u code/takeover_queue.py configs/openh3_20261111/pilot_queue.json
```

实际30794运行的计划是 `configs/openh3_20261111/pilot_queue_30794.json`，tmux session `openh3-pilot-20261111`。2026-11-11 11:13（Asia/Singapore）已启动第1步；训练与RGB预检、tiny CPU contract均退出0。真实H3执行结果以运行日志为准，不能把已启动记为通过。

11:22 更新：第1步真实NPU训练 smoke 已通过。37,273,600个新增参数、200个LoRA模块；零初始化输出误差0、finite非零梯度、B更新、底座冻结、checkpoint清空重载输出误差0均通过。峰值分配41.142 GB；这只是2×8×8 latent裁剪的一步工程检查。证据为 `results/takeover_20261111/h3_npu_train_smoke_30794.jsonl`。队列继续执行独立进程重载推理；300步pilot和质量比较尚待完成。

11:35 更新：第2步也已通过，独立进程重载新LoRA、三步采样、VAE解码成功，输出5帧128×128 RGB。证据为 `results/takeover_20261111/h3_npu_reload_smoke_30794.json`。队列11:31进入第3步8来源pretrained RGB基线，300步训练及配对质量比较仍待完成。11:34重新检查：31051卡6/7、32193卡4/7健康且无NPU进程，每卡约60.7 GiB显存可用；30794卡6由本项目使用，卡7留给既有队列。空闲状态只是检查时快照。

13:03 更新（后续只读确认）：五步队列全部完成。完整尺寸训练约7.78秒/步、峰值47.215 GB。8来源开发集上，300步LoRA相对pretrained的LPIPS平均变化−0.022086（8/8改善）；PSNR平均+0.05555 dB（3/8改善，配对bootstrap区间跨零）。低频时序误差4/8改善、细节时序误差2/8改善，不能宣称统一的时序收益。原始指标、基线、训练日志与队列状态已归档至 `results/takeover_20261111/sr_pilot_300*`。这是raw-guide、mode=base实验，完全没有训练或接入R_phi；不能作为新omni R_phi方案的验收。

只读检查使用 `--inspect-only`。单个计划只允许一个队列进程；已经 running/failed/completed 的目录不能无意重跑。查看 `runs/openh3_pilot_20261111/queue_status.json` 与对应 logs。全部实验完成后停止，不会循环训练。

## 评估修正

`restoration_metrics.v2` 修复光流拉取方向、边界与前后向一致性 mask、静态视频的绝对时序误差、指标优劣方向。频带拆分明确为互补诊断而非正交投影；误差分解保留交叉项。旧 JSON 不重命名为 v2，也不与重算后的 warp 数值混合。

首轮 RGB 对比保留 RGB PSNR、LPIPS、时序误差以及频带诊断。能量比没有通用的“越大越好”，不能用纹理更多替代恢复正确性。帧采样 stride 和每源输入 hash 写进结果合同。

## 后续推进门槛

- SR pilot 必须先证明实际 RGB 指标、运动稳定性和可见细节的收益；保留失败样例，不能只挑最好视频。
- 四任务实物数据完成后编码并训练统一小样本 pilot；随后加入混合、未知、随时间变化的退化。观察算子/一致性约束需与任务相符，不能始终强制 bicubic x4。
- 在相同数据上分别比较 pretrained latent upscaler、当前像素上采样 draft，以及 Sharp 的增益/失真。IC/Ref2VA 的 clean reference 条件是值得验证的结构分支，不能仅因新版本存在就同时换全栈。
- 视频 omni 与音视频联合恢复分开验证。音频分支需要匹配的独立基线、同步/错位消融和跨模态收益证据。
- 真正流式需要显式未来帧预算、状态传递/重置、长视频漂移和实际首帧延迟；固定短窗和非因果 fast VAE 不自动满足它。

## R_phi 主线调整（2026-11-11）

本节是明确的下一阶段方案，尚未实现或启动。以 `configs/openh3_20261111/rphi_v2_plan.json` 区分设计状态和已完成实验；SR LoRA pilot只提供预训练组件与训练基础设施的对照。

旧R_phi有可保留的证据：历史77视频对照将innerDC目标从raw LR换为预测R_phi后，LPIPS为0.32458→0.22334、PSNR为23.8829→25.3180（`results/FROZEN.json`）。但这是旧底座、旧数据上的收益，不可直接转移给Open-H3。K3对低分辨率干净目标的realistic增益+3.2501 dB，pure bicubic输入却由58.1724降至44.5718 dB（`results/range_branch/r5_k3.json`）；需优先解决轻退化保护。

旧 `make_ops.py` 的deblur监督是blur(x)，不是sharp x。作为干净观测D(x)估计它有意义，但不能把该分支成绩解释成最终去模糊成绩。旧 `--op` 只改变FiLM条件，H3 innerDC仍构造BicubicConsistency；MTF是给定算子的径向功率描述，不是部署中已实现的盲退化估计。有限分类器AUC不能作为所有盲恢复路线的上限。

下一版保留K3因果残差主干作为可比较起点，明确区分两种输出：`range_pred` 是定义清楚的观测域/规范低频锚点，`rgb_pred` 是朝向同一原始clean-reference proxy的恢复图，另预测局部不确定度。SR的range与RGB几何不同；native deblur/denoise/compression必须对原始清晰RGB验收。压缩不得强行解释为具有固定线性range/null分解。已知物理算子、规范低频算子和插值算子须分开记录，不能共用一个含义不明的D。

实施顺序：

1. 直接读取已生成的四任务RGB数据，先做R_phi独立训练和geometry/target/source检查，无需等待VAE。补identity/near-clean样本，使用RGB重建、边缘和identity guard；新增结构的收益必须与同数据、同预算的旧主干比较。
2. 比较已知条件与观测推断条件；后者只读当前及过去帧，条件缺失mask/dropout，合成参数仅作训练辅助。先K3，再验证有状态的因果融合；不把精确估计全部物理退化参数作为恢复的前置条件。
3. 用误差监督校准局部不确定度，报告覆盖率/误差关系，防止置信度退化为全零。由置信度控制软约束，未知算子时不向带噪原图硬投影。原观测高分辨率信息另行保留，不能只把一个低分辨率anchor交给H3。
4. 固定Open-H3比较raw-guide、旧R_phi、改进R_phi及有/无软约束；另报告R_phi-only。最后才训练以相同R_phi输出为输入的LoRA，避免raw训练/R_phi部署失配。当前300步adapter是raw-guide对照，不直接充当新组合的匹配训练结果。
5. 加入有明确顺序及逐帧参数的混合/时变退化，并建立新冻结测试集。只有分任务原RGB保真、时序、干净输入保护和跨来源结果齐全，才能决定是否扩大模型或训练。

验收安排相应以R_phi独立恢复、R_phi与冻结/适配H3的因素对照为主；Sharp、额外LoRA和latent upscaler消融不得替代这些实验。此前四任务时间为条件目标，不能解释为已启动R_phi训练。

2026-11-11 接管时 31984 的 8 张卡均有既存训练进程。CPU 测试通过不代表实际 H3 smoke 或 pilot 已完成；以队列状态和运行产物为准。
