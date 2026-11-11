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

四任务 materialization 已在31984完成：40个target、160个observations，共200个NPZ、2,205,237,727 bytes。已独立核验文件与像素SHA、22帧长度、SR/native尺寸、四任务共用target及来源互斥。文件在 `datasets/omni_pilot_v1/`，小型证据在 `results/takeover_20261111/materialized/` 与 `cpu_materialization_validation.json`。这批数据尚未做 VAE encoding；13:41起已用于不依赖VAE的独立RGB R_phi训练。

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

本节区分整体目标与下面已落地的第一阶段。以 `configs/openh3_20261111/rphi_v2_plan.json` 区分设计状态和已完成实验；SR LoRA pilot只提供预训练组件与训练基础设施的对照。

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

### 13:35 开始执行独立 RGB 原型

新增 `rphi_data.py`、`rphi_model.py`、`train_rphi.py`、`eval_rphi.py` 和 `compare_rphi.py`。200个NPZ及物化receipt/manifest共246个文件已从31984校验迁入30794。最终47项CPU单测全部通过。真实NPU小步训练与整帧评估队列在13:37完成：RGB/anchor/variance/condition重载误差均为0，四任务480×832全帧chunk检查通过，本地LPIPS权重成功载入。证据为 `results/takeover_20261111/rphi_smoke_*`，只证明工程运行正确，不能解释成质量收益。

三组1000步计划 `rphi_pilot_queue_30794.json` 已在13:41由 tmux `rphi-pilot-20261111` 启动于30794卡6，输出 `runs/rphi_20261111_pilot/`。首臂known前325步约0.41秒/步、训练峰值分配1.112GB；训练后自动进行完整8来源/32任务样本评估，再顺序运行两个blind臂和配对比较。三臂实际完成状态及收益仍须读最终产物，不把队列启动等同验收通过。

三组为known（公开task/scale）、blind（只读三帧观测）、blind_confidence（同一blind恢复器+误差方差校准）。共享主干在相同seed下逐tensor相同，训练crop序列和预算相同。SR先把观测映射到目标HR网格，native三任务保留原网格；全部对原始RGB参考监督。25%期望比例的identity/near-clean控制在线生成，独立汇报损伤，不能混进普通任务成绩。

本阶段输出是RGB预测、由它确定性下采样得到的canonical Q4 anchor、以及局部log-variance。**尚未实现独立物理range head，也未将软约束接入H3或训练有状态时序模型。** 不确定度头读detach后的特征和误差，使用独立优化器；其作用是测量校准，按设计不应改变blind分支的RGB训练轨迹。将比较最终fidelity权重hash，不把第三组相同RGB成绩冒充结构提升。known臂使用同一FiLM残差结构家族的新RGB骨干，当前ch64/8blocks；它不是历史ch96/12blocks checkpoint的直接重用或数值复现。

训练使用RGB Charbonnier、边缘L1和相对几何对齐输入的identity guard。最后一个checkpoint用于开发评估，不按开发集挑选最好checkpoint。评估覆盖32个来源/任务样本，报告各任务原RGB PSNR/MAE/本地LPIPS、时序误差、干净输入保护、置信度校准与来源配对比较；8个来源仍是开发集，不能支持未见域泛化结论。原始源视频的hash由31984物化证据提供，30794严格重验当前NPZ文件、像素和metadata绑定，未声称重新读取源视频。

独立代码复核发现的限制：当前blind编码器在卷积前池化至最多64×64，96×96训练crop与480×832整帧评估的缩放比例不同，会把噪声/模糊的尺度泛化混入known/blind差值。本轮保留固定实现并如实报告，后续候选改为先在原像素尺度卷积再汇聚；不得据此轮结果断言盲恢复能力上限。当前三臂lr/warmup/优化器默认一致，但v1合同未保存这些字段，后续版本需补齐。比较程序未自动拒绝不同flow模式，验收另行核对每臂flow模式与有效支撑，不能只看 `complete=true`。

H3接入也需单独验证：现有49步设计的innerDC落点映射到3步采样后全部落在最后一步，修正后没有后续DiT融合；native HR若直接送BicubicConsistency会成为全帧identity投影。下一步先固定冻结H3和raw warm start，比较raw/R_phi guide，再在相同解码RGB上做显式post-decode soft anchor；SR使用canonical Q4，native任务使用完整RGB软融合。该后处理不是物理观测的精确逆投影，置信度尚未用于权重。

为上述下一阶段新增四个隔离模块，未修改已固定hash的pilot runtime：`rphi_condition_v2.py`（原像素尺度条件特征，尚未训练）、`rphi_soft_anchor.py`（CPU解码后软锚点）、`export_rphi_guides.py`（全部22帧float32输出及来源/权重/像素hash）、`audit_rphi_results.py`（光流支撑、LPIPS、chunk和三臂匹配审计）。最终75项CPU测试通过。soft anchor还在合成480×832数据上通过四任务数值检查：alpha=0保持原tensor，alpha=0.25且不clip时锚点残差约为原来的0.75；会分别记录clip与dtype舍入导致的残差。该合成检查不是H3质量结果。

known臂1000步于13:48:48完成，训练/验证/保存共402.89秒，重载误差0。按约19秒/任务样本的实际整帧评估速度，三臂完整开发结果的条件预计为当日14:30–15:00（Asia/Singapore）；卡可用状态改变或验证失败会使队列停止，此时间不覆盖后续H3或冻结测试验收。

13:59更新：known臂完整8来源/32任务评估完成，结果归档为 `results/takeover_20261111/rphi_known_1000_metrics.json`。相对bicubic/identity输入，SR、deblur、denoise、compression的PSNR平均变化分别为+0.0355、+0.0855、+8.6177、+0.0875 dB，LPIPS变化分别为−0.000703、−0.000864、−0.291926、−0.000176。四任务PSNR均8/8来源改善，但实际收益主要集中在去噪；SR光流对齐误差略增6.35e−6，不得宣称所有时序指标都改善。denoise标签下identity MAE为0.003053，near-clean PSNR相对直接输入下降2.4625 dB，仍有轻退化损伤。不能据此宣称omni完成或将弱去模糊/去压缩增益包装为强恢复。队列已进入blind训练。

另启动 `rphi-completion-audit-20261111`，执行一次有90分钟超时的 `audit_rphi_queue_completion.py`。它只等待上述有限队列完成，再在CPU运行严格结果审计，写 `runs/rphi_20261111_pilot/audit.json` 和独立 `audit_status.json` 后退出；上游失败则停止。该依赖步骤不使用NPU、不新增训练、不修改上游状态、不发送通知。其失败/超时/异常状态处理已在本地验证，上游状态与结果保持不变。

2026-11-11 接管时 31984 的 8 张卡均有既存训练进程。CPU 测试通过不代表实际 H3 smoke 或 pilot 已完成；以队列状态和运行产物为准。

### 14:17 研究主张与基线范围复核

新增 `configs/openh3_20261111/omni_benchmark_plan_20261111.json`，记录主基线、可执行资产状态、公平比较合同及待验证假设。它是计划，不代表基线已经部署或新结构已经训练；正在运行的三臂代码与队列未变。

FlashVSR+保留在SR和效率比较中。native去噪/去模糊/去压缩优先FaverNet-TUDCommon和AverNet；二者为离线双向模型，不能忽略未来信息差异，也不能把same-resolution resizing blur当作×4 SR。RONIN是因果盲统一恢复的重要论文参照，但官方仓库当前只有README。SeedVR2作为生成式SR/真实恢复补充，任务覆盖逐项核实。天气扩展才引入ORCANet/SEUD；TDM作为使用任务提示的生成式多任务先例，不混入相同信息条件的blind主表。

旧FlashVSR+脚本经CRF10/yuv420p输入并从MP4读回输出，而我方使用PNG；需要无损RGB桥接或共同编解码控制后重跑。原生RealVSR与本项目额外×4降采样协议分开。旧MVSR4x15的LPIPS我方0.407777、FlashVSR+0.360262，负结果保留。旧预测range替换在FlashVSR+上令LPIPS平均变差0.01322，也不能支撑跨骨干普遍受益。

候选主张缩到两项：一是预测相对基准的修复风险，决定哪些改动值得接受；二是根据观测识别退化变化后历史证据的有效性。当前variance只校准R_phi自身误差，不能直接当作修复动作有益概率或原观测噪声方差。先冻结R_phi比较输入/R_phi候选、固定混合、variance门控、相对风险、打乱风险与oracle，再扩到固定H3；oracle仅诊断。历史有效性先在K3内做，与K1、固定K3、固定衰减、无重置和打乱历史比较。额外校准来源与最终冻结测试来源分离，拒绝“几乎不修复所以不损伤”的伪收益。

这些是研究假设，尚未确认新颖性或质量优势。预训练组件替换、LoRA、FiLM、soft DC、两阶段恢复/生成及因果/时变标签本身不能作为新贡献。完整H3的未来帧预算、三步末端DC与post-decode对照、clean保护、真实混合退化和长期漂移均需实测。

### 16:04 健康检查：队列完成，质量与对照一致性未通过验收

30794上的三臂各1000步、三次32样本整帧评估及配对比较于14:40:07完成；CPU审计于14:40:20结束。检查时本项目无活动训练进程，卡6健康为OK；未改动其他进程。三臂记录中的loss/梯度有限，日志未见OOM或异常退出，checkpoint重载各输出最大误差均为0。训练峰值分配1.112/1.119/1.196 GB。前5到后5个已记录batch的平均loss均下降，但不是全量epoch均值。原始日志、最终metrics、comparison和audit保存在 `results/takeover_20261111/rphi_completed/`，归纳报告为 `health_summary.json`。

blind四任务PSNR相对输入变化为SR +0.0839、deblur +0.1016、denoise +6.9441、compression -0.3020 dB；SR/deblur/compression的LPIPS均变差。near-clean平均PSNR下降约0.916 dB，5/8来源受损。confidence恢复结果接近blind但不能解释为有效结构收益；其1σ覆盖率92.1%–99.2%，偏离高斯名义68.3%，不能直接用作已校准的门控。

审计发现blind与confidence的fidelity SHA不同。独立CPU检查确认：250步主干相对L2差3.2867%，1000步6.9353%，后者max_abs 0.0433484；不是仅SHA字符串不同。两臂训练参数（除variant/out）、合同、初始主干SHA、代码SHA相同。静态检查未发现confidence梯度进入fidelity参数，后端数值差异尚是候选原因，未确认。下一版应从同一个冻结blind checkpoint单独拟合校准头，直接保证恢复器相同；同时修正blind原像素尺度条件编码并优先验收去压缩/轻退化损伤。本次仅审计归档，没有重启或新增训练。

### 第二阶段接续（2026-11-11）

用户要求继续训练并指出有限试验结束后没有及时接续。原队列按1000步×三臂预定预算正常退出；上一阶段未接后续验收回访，现补齐。`rphi_stage2_plan.json` 记录本轮完整预算，三个3000步恢复器为legacy_g05、native_g05、native_g20；分别隔离编码器像素尺度和guard权重，其他数据/seed/crop/网络/优化器/控制样本比例相同。旧1000步结果保留，不能把不同训练预算的差异归因为结构。

`RphiModel` 新增可选native编码器，旧checkpoint默认legacy；训练合同补录AdamW/LR schedule/确定性设置，保存初始完整fidelity SHA。新的 `calibrate_rphi.py` 只从一个固定blind checkpoint复制并冻结恢复器，单独拟合1500步置信度头，每次保存验证完整fidelity SHA与固定训练probe输出；这属于训练集上的不确定度拟合，不冒称独立留出校准。`eval_rphi.py` 增加校准来源链的元数据记录，指标含义不变。

两条正式队列安排为：native_g05训练/评估→冻结校准/评估；legacy_g05训练/评估→native_g20训练/评估。最多并行两张卡，实际选卡通过健康/空闲/锁及二次检查；不终止其他进程。所有正式步骤须先通过独立native训练/冻结校准/整帧评估smoke。各组最终checkpoint评估32个任务样本、stride4筛查；后续候选需要stride1相邻帧及长视频验证。单独CPU依赖步骤 `finalize_rphi_stage2.py` 最多等3小时，自动审计并打包 `evidence.tar.gz`；运行成功仍不等于质量通过。

16:36，部署包SHA `bc5e5002c610ee5ae97c4d67aa563fb1e8a42bac3f819301d4949d4fe0881593` 的33文件在30794逐项校验通过；107项Rphi CPU测试通过，耗时12.188秒。测试使用CPU张量，torch_npu可能初始化运行时，未声称完全不初始化设备。16:37启动NPU smoke tmux `rphi-stage2-smoke-20261111`；正式训练状态须读实际队列，不以配置文件存在充当启动。

本线程已创建20分钟一次的验收与接续回访 `omni-r`（Omni Rφ 训练验收与接续）。它读取真实状态，不重启健康任务；无变化时安静，完成/失败/重要结论时再报告，并依项目计划推进下一阶段。后续先审计本轮的去压缩、轻退化保护和校准，再进入固定H3的guide/soft-anchor因素对照及匹配guide的LoRA；统一恢复基线、混合/时变数据和冻结来源测试按研究计划并行准备。

16:41:01，四步NPU smoke全部通过：冻结校准前后fidelity SHA相同，固定probe的RGB/anchor/code差全0，重载logvar等输出差全0；两个模型的四任务480×832评估与chunk检查通过，全部task/identity/near-clean RGB指标差为0。启动证据与107项测试日志已按归档SHA `186446683552a634581c7e10237c8c3ee461edbb9b287a800b65da6bdfd6e004` 验证下载到 `results/takeover_20261111/rphi_stage2_start_evidence/`。同一代码部署包也已在31984保存并核验SHA。

16:42:04 native正式队列在30794卡6开始，16:42:08 controls队列在卡1开始，CPU finalizer已等待两队列；tmux分别 `rphi-stage2-native-20261111`、`rphi-stage2-controls-20261111`、`rphi-stage2-finalize-20261111`。16:44:03实际检查，native_g05与legacy_g05均150/3000步，loss与梯度有限，初始完整fidelity SHA均为 `fd2a51b3de4023c87cdba25af5fc66cb4907d78ca482c52a69f028d9afec78e2`，25步约10.25/10.07秒，峰值分配1.127/1.119GB。两卡健康OK。按当前速度预计首组完整开发评估17:15–17:30，三组及冻结校准完整验收17:50–18:20（Asia/Singapore）；这是卡资源与验证持续正常时的估计，不包括后续H3/混合退化验收。

### 17:20 回访：前两臂完成并自动接续，下一阶段入口和基线资产已准备

legacy_g05 与 native_g05 的3000步训练分别于17:03前后完成，完整8来源/32任务评估于本次回访内完成；两条队列已自动进入 native_g20_train 与 frozen_confidence_fit，CPU finalizer仍等待全矩阵。原始双臂指标/训练日志/状态一次打包下载并核验，归档SHA `6fc0a3ea2c53ccc9087146b47c4104bca716a3138fb335ec5c9603c870745ff7`，本地目录 `results/takeover_20261111/rphi_stage2_first_pair/`。这不是四臂最终审计，未选定进入H3的候选。

初步完整任务指标（相对输入）为：legacy/native 的SR PSNR +0.1310/+0.1193 dB，deblur +0.2295/+0.2112，denoise +9.8981/+9.7438，compression −0.0046/−0.0182；LPIPS分别 −0.002901/−0.002336、−0.003137/−0.002562、−0.376838/−0.375291、−0.000143/−0.000146。原像素编码的质量假设目前未获支持，不因结构更换就默认采用native。两者near-clean平均改善约1.42/1.40 dB，但native仍有2/8来源损伤；clean identity的MAE为0.000988/0.001003，8/8出现非零损伤，不能声称无损保护。identity的输入误差为0，其PSNR被封顶120 dB，应以MAE/MSE理解损伤。stride4仍不是相邻帧flicker证明；冻结校准尚未完成，不报告覆盖率或完整fidelity一致性结论。

独立新增 `export_rphi_guides_stage2.py`、`run_fixed_h3_rphi.py` 和测试，未修改运行中锁定的训练代码。新入口验证最终checkpoint、blind K3连续历史、原始源/观测像素/导出文件绑定及native编码器代码SHA，统一float32指标。两次H3采样保持相同raw warm-start/seed/weights，仅换raw/Rphi conditioning；两份解码结果复用固定alpha=0.25软锚点，与input/Rphi-only组成六臂。SR采用canonical Q4，native采用RGB软融合，无innerDC、无confidence gating、无新LoRA。每任务2次sampler调用/6次denoiser评估；全32任务是64次sampler/192 NFE。主评估默认stride1，全22帧块处理不称严格因果。

隔离30794 `preparation/fixed_h3_bridge_v3` 的最终10项CPU测试通过；v2实际用此前tiny两步checkpoint完成1来源×4任务×22帧480×832的CPU导出，v3严格输入检查通过。这是数据桥接验证，没有执行H3/VAE。小证据归档SHA `6587bff0d108b0b4f2b68686b21541fecb4a0001711b90ae2126189b93a2e1af`，本地 `results/takeover_20261111/fixed_h3_bridge_preparation/`。执行计划在 `fixed_h3_rphi_plan_20261111.json`：等stage2验收、选可解释候选、空闲卡预约与二次检查、真实1来源H3 smoke后再扩全量。当前导出仅ordinary dev tasks；clean/near-clean H3控制与训练集guide导出仍须另实现，不能拿当前入口直接宣称clean保护或启动guide-matched LoRA。

FaverNet `14769f22685f4850c5e203d493c99ad2eba5b68b`、AverNet `c4eb019f71fe6a2ccb6807dca92451fd8d8b8cd3` 官方源码、两模型与SPyNet共223,811,280 bytes权重已实际下载。无损RGB adapter的4项NumPy测试及两个input-only preflight通过；未完成任何模型forward/严格权重加载/Ascend parity。MMCV 2.2.0存在NPU DCN分派，不应称完全不支持；仍需隔离环境构建、算子和全模型一致性验证。代码/权重已同步31984新目录 `preparation/native_baselines_20261111`，两commit与三权重SHA核验通过；归档SHA `5ddaa8097d16ae7eadf45b13435a9093d26981aaaeea67e2d68e1329dce8c807`。基线scope为native、offline双向、zero-shot，不能充当×4 SR或严格流式同信息对照。

31984只读库存发现1068个原始候选视频，778个ID未匹配到已枚举训练/验证/评估清单，但全部已有harvest元数据记录，不能视作未被看过。46个通过旧video sharp heuristic而仅因audio门槛被淘汰，是video-only校准/test资格审计的优先候选；不等于已合格或pristine GT。32个确定性抽样完成源SHA与首中尾解码检查，独立calibration/test当前获批数量仍为0。详见混合时变数据plan与inventory；先审计46来源历史/去重/质量和源组，再CPU物化小试验，不把现8dev改名为冻结测试。

前两臂独立配对审计已通过：仅encoder为声明差异，源/记录/输入像素哈希、初始fidelity、基准指标和LPIPS一致，384项flow支撑记录检查通过，chunk差全0；未归档baseline像素与flow mask，不能声称对它们逐像素独立重验。native−legacy的SR/deblur/denoise/compression PSNR为−0.01173/−0.01833/−0.15423/−0.01354 dB，LPIPS为+0.000565/+0.000574/+0.001546/−0.00000304。去噪6/8来源比legacy好但均值为负、置信区间跨0，保留来源异质性。最差去压缩来源 `EBmROluwLkc` 的native相对输入−0.33293 dB、相对legacy−0.06433 dB。报告 `results/takeover_20261111/rphi_stage2_first_pair_review.json`，不代替全矩阵及冻结校准验收。

### 17:34 创新性复核与并行假设收紧

用户询问当前创新性及其他并行设计。定向原论文核对发现直接重合：VetoNet（2026-10预印本，https://arxiv.org/abs/2610.21335）已有冻结恢复/生成候选、局部反事实相对收益和风险拒绝；LaverNet（https://arxiv.org/abs/2512.16313）已有all-in-one视频选择性传播，TURTLE已有因果历史路由。因此原H1降为控制基线，通用H2门控不能直接作为主创新。已同步修正benchmark plan，并补LaverNet/TURTLE文献与机制对照；没有确认LaverNet可运行官方资产。

`configs/openh3_20261111/parallel_research_plan_20261111.json` 记录两个方法假设和一个低成本诊断，均尚未实现/开训：P1用当前帧恢复损失的有/无历史差监督历史边际效用，比较同容量普通gate、真实效用与打乱效用，需先验证缺失历史干预不会制造分布外假标签；P2交换噪声与压缩顺序，先验证oracle顺序信息有恢复价值，再做blind小头，不能把顺序感知本身称为新颖；P3复用待生成的固定H3全22帧float缓存，正交分解G−R残差，判定是否存在可保留的正确细节，再决定生成接口及LoRA。频带融合本身也已有先例。阶段预算、依赖、失败条件及公平对照均写入计划；现有stage2 runtime/队列未修改，也未额外启动NPU任务。

当前Rphi本来只读原始K3观测，没有生成回灌bug；未来保持该接口属于约束，不冒充新贡献。退化改变不能简单等于清空历史，突然变噪时此前清楚帧可能更有用。源级数据、stride1/长片段、clean保护、连续损伤与恢复延迟必须一并评估，现32train/8dev不能支持泛化或风险认证。

新增CPU诊断 `results/takeover_20261111/rphi_selection_headroom_20261111.json`，使用已归档native_g05指标（metrics SHA be0d2a0e9ccc96afbca8db89621a7a3d7717f1a772eb7571bf24455852f463bc）：8个压缩dev来源中，整段GT-PSNR oracle在input/Rphi二选一时，相对input仅+0.06649dB，always-Rphi为−0.01819dB，oracle额外+0.08467dB，选择恢复5/8。它仅是这两个已有整段候选的诊断，不是patch门控/H3/未来模型的上界，也不能部署或替代全矩阵验收。
