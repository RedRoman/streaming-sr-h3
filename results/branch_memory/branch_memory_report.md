# 多解释持续细节状态：机制、可操作解法与原创性审计

日期：2026-10-15（Asia/Shanghai）。32 seeds 的所有实验均已实际完成。

## 结论

**可操作解法成立，但本线性条件下它与精确 Bayesian MHT 等价，不能作为独立原创核心方法。** 当远景观测无法确定对象身份时，不覆盖已证实条目；把新观测存为独立 episode 的证据，借助独立弱身份线索或后续近景揭示重新选择/合并。旧内容版本实际保留为独立数组，可在回访时用少量可辨观测取回未重新观测的隐藏细节。

无额外 cue 的远景正确身份率为25%；弱独立descriptor下为51.56%，延长到12次弱cue后为70.90%。后续近景揭示后，fork与精确MHT都达到100%身份正确率。

普通单状态在静态任务上也可达到同等结果；真正可见的收益是变化造成暂时错配时对其它身份的污染：single MAP 的全 registry 系数 MSE 为0.001449，fork/MHT为0.000313，而当前访问对象最终PSNR相近（43.46 vs 43.45 dB）。这说明应评价记忆库整体健康，而不只看当前输出。

## 数据与因果协议

- 4个16×16周期纹理对象。每个4×4块有3个正交、块均值严格为零的Haar式纹理模式，共48个系数；所有对象具有完全相同的4×4远景LR。程序逐值断言远景相同，不用视觉近似冒充不可区分。
- 2×2 box近景观测能揭示这些模式。脚本在其正交充分统计量中恢复48维系数，噪声sigma=0.03；这等价于近景LR像素噪声sigma=0.015。每个近景packet只揭示12个系数，4个packet逐步完整揭示。
- 已证实registry完全由3次独立近景LR注册，系数方差sigma²/3；HR图像仅用来生成观测与评价。没有用HR真值初始化记忆。
- 每seed有8个对象访问episode，4个对象各返回2次。episode边界作为已知轨迹分段输入；本实验未训练分段与对应模块。各时刻只读当前及过去packet。
- 弱descriptor来自四面体位置prototype（坐标±0.30）加sigma=1.0独立噪声；与纹理退化观测独立，均值来自注册身份。没有把真值identity标签输入推断器。
- 6情境：无cue→近景、有弱cue→近景、真实局部内容变化、同packet重复6次、延迟近景揭示、始终没有可辨观测。改变发生在episode 4，仅12/48系数发生变化。
- 32 seeds为0…31，统计先在seed内聚合再作5000次seed bootstrap；各方法共用同一观测历史。

## 比较方法

1. `single_map`：每次依当前身份分数硬分配，并立即更新唯一对象状态。允许变化分量扩大观测方向的更新方差。
2. `hard_reject_reset`：低置信拒绝；置信度超过0.95后，从当前局部观测重启单一状态。每个episode首次接受或身份切换时才重置，其后递推更新。
3. `hard_reject_preserve`：强化控制，拒绝低置信观测但保留已知状态，确认后只更新当前观测方向。其静态性能与fork相同，避免把弱重置策略当唯一基线。
4. `particle_mht_4/32`：4或32粒子的bootstrap多假设跟踪，包含身份与稳定/变化两类分支，ESS低于一半时重采样。该实现可能在揭示前丢失正确分支；更强rejuvenation particle算法未在此测试。
5. `exact_mht`：枚举4身份×2内容模式，在线Gaussian分支后验；每次packet视为独立。
6. `exact_mht_dedup`：同一精确MHT，加packet UID去重，是最强等价控制。
7. `fork_merge`：保留原registry及版本；每episode只存共享g/b/c/count与descriptor证据，必要时计算各身份条件后验。packet UID去重。episode结束且身份概率>0.95、有可辨观测时才写入registry，内容变化时保留旧版本。

两种精确方法在episode内都输出当前前缀后验的最可能状态；正式registry合并延迟到episode结束。当前渲染可以先正确，但正式写入比立即更新晚最多3个唯一近景packet，不能把这项延迟隐藏为免费收益。

## 身份与恢复结果

| 情境 | 方法 | 远景身份正确率% | 最终正确率% | 最终PSNR | 持续恢复延迟 |
|---|---|---:|---:|---:|---:|
| no_cue | single_map | 25.00 | 100.00 | 44.30 | 0.000 |
| no_cue | hard_reject_reset | 25.00 | 78.91 | 25.38 | 1.562 |
| no_cue | hard_reject_preserve | 25.00 | 100.00 | 44.30 | 0.000 |
| no_cue | particle_mht_32 | 22.66 | 99.61 | 44.08 | 0.016 |
| no_cue | exact_mht_dedup | 25.00 | 100.00 | 44.30 | 0.000 |
| no_cue | fork_merge | 25.00 | 100.00 | 44.30 | 0.000 |
| weak_cue | single_map | 51.56 | 100.00 | 44.29 | 0.000 |
| weak_cue | hard_reject_reset | 51.56 | 93.75 | 29.78 | 0.512 |
| weak_cue | hard_reject_preserve | 51.56 | 100.00 | 44.29 | 0.000 |
| weak_cue | particle_mht_32 | 49.61 | 98.83 | 44.01 | 0.047 |
| weak_cue | exact_mht_dedup | 51.56 | 100.00 | 44.29 | 0.000 |
| weak_cue | fork_merge | 51.56 | 100.00 | 44.29 | 0.000 |
| true_change | single_map | 47.66 | 100.00 | 43.46 | 0.020 |
| true_change | hard_reject_reset | 47.66 | 91.80 | 29.73 | 0.508 |
| true_change | hard_reject_preserve | 47.66 | 100.00 | 43.47 | 0.020 |
| true_change | particle_mht_32 | 48.18 | 95.70 | 42.74 | 0.188 |
| true_change | exact_mht_dedup | 47.66 | 100.00 | 43.45 | 0.012 |
| true_change | fork_merge | 47.66 | 100.00 | 43.45 | 0.012 |
| duplicates | single_map | 51.56 | 100.00 | 41.83 | 0.000 |
| duplicates | hard_reject_reset | 51.56 | 79.30 | 31.16 | 1.273 |
| duplicates | hard_reject_preserve | 51.56 | 100.00 | 41.83 | 0.000 |
| duplicates | particle_mht_32 | 49.61 | 96.09 | 41.44 | 0.156 |
| duplicates | exact_mht_dedup | 51.56 | 100.00 | 44.29 | 0.000 |
| duplicates | fork_merge | 51.56 | 100.00 | 44.29 | 0.000 |
| delayed_reveal | single_map | 70.90 | 100.00 | 44.34 | 0.000 |
| delayed_reveal | hard_reject_reset | 70.90 | 99.61 | 35.74 | 0.055 |
| delayed_reveal | hard_reject_preserve | 70.90 | 100.00 | 44.34 | 0.000 |
| delayed_reveal | particle_mht_32 | 68.59 | 97.27 | 43.77 | 0.109 |
| delayed_reveal | exact_mht_dedup | 70.90 | 100.00 | 44.34 | 0.000 |
| delayed_reveal | fork_merge | 70.90 | 100.00 | 44.34 | 0.000 |
| no_reveal | single_map | 25.00 | 25.00 | 21.26 | 4.000 |
| no_reveal | hard_reject_reset | 25.00 | 25.00 | 21.26 | 4.000 |
| no_reveal | hard_reject_preserve | 25.00 | 25.00 | 21.26 | 4.000 |
| no_reveal | particle_mht_32 | 23.83 | 23.83 | 21.01 | 4.000 |
| no_reveal | exact_mht_dedup | 25.00 | 25.00 | 21.26 | 4.000 |
| no_reveal | fork_merge | 25.00 | 25.00 | 21.26 | 4.000 |

持续恢复延迟：首个近景packet为0；要求此后所有packet均选择正确身份；最终未恢复计为4。无近景情境的4表示未获可辨证据，并不是实际等待4帧后恢复。首次正确与持续正确两种延迟均在per_episode.csv中。

| 关键统计 | 均值与95% seed CI |
|---|---:|
| 弱cue远景正确率% / fork | 51.56 [46.22, 56.77] |
| 12次弱cue远景正确率% / fork | 70.90 [67.19, 74.38] |
| 弱cue后最终正确率% / particle32 | 98.83 [97.27, 100.00] |
| true_change / single_map-minus-fork_merge / registry_mse | +0.001136 [-0.000062, +0.002777] |
| true_change / hard_reject_preserve-minus-fork_merge / registry_mse | +0.001086 [-0.000062, +0.002726] |
| duplicates / fork_merge-minus-exact_mht / final_psnr | +3.496729 [+3.189704, +3.827407] |
| weak_cue / fork_merge-minus-particle_mht_32 / final_correct | +0.011719 [+0.000000, +0.027344] |
| weak_cue / fork_merge-minus-exact_mht_dedup / final_psnr | -0.000000 [-0.000000, +0.000000] |

注意：真实变化下registry误差的配对差区间跨零，平均改善由少量困难seed驱动。零旁路写入是本实现的存储不变量，但当前样本不足以把整体误差改善称为稳定的统计优势。

## 未变内容保留与记忆污染

| true_change方法 | 首近景未变区域MSE | 全registry MSE | 非当前身份每episode写入漂移 |
|---|---:|---:|---:|
| single_map | 0.000759658 | 0.00144924 | 0.00027038 |
| hard_reject_reset | 0.0152783 | 0.0146101 | 0.000374465 |
| hard_reject_preserve | 0.000759543 | 0.00139912 | 0.000237457 |
| particle_mht_4 | 0.0132237 | 0.0345299 | 0.0102279 |
| particle_mht_32 | 0.0022316 | 0.00683199 | 0.00262868 |
| exact_mht | 0.000534247 | 0.000312826 | 0 |
| exact_mht_dedup | 0.000534247 | 0.000312826 | 0 |
| fork_merge | 0.000534247 | 0.000312826 | 0 |

这里首近景的输出仍可能选错身份，因此不能声称fork在身份未确认时总能显示正确纹理。它保证的是已确认registry在合并前不被新episode覆盖；实际合并后非当前身份漂移在本套件中为零。

## 旧版本回访：少量揭示恢复隐藏细节

另做32seed回访实验：对象0先改变12个系数，随后返回旧外观；registry有新版本，archive保留旧版本（均来自LR）。每packet只揭示3个系数。比较只保留最新状态、允许快速变化的单状态、局部重置，以及枚举旧版本的精确MHT/共享证据检索。

| 方法 | 第1次部分揭示PSNR | 第1次隐藏区域MSE | 第1次选对旧版本% |
|---|---:|---:|---:|
| single_latest_map | 22.60 | 0.00594867 | 0.00 |
| single_adaptive_map | 23.10 | 0.00594867 | 0.00 |
| hard_reset_partial | 17.36 | 0.0199129 | 0.00 |
| exact_version_mht | 42.70 | 5.50631e-05 | 100.00 |
| fork_version_retrieve | 42.70 | 5.50631e-05 | 100.00 |

首部分揭示时，版本检索相对允许快速变化的单状态，配对PSNR增益为 19.61 dB，95% seed CI [18.72, 20.47]。该收益来自保留旧版本的额外记忆，容量与存储预算必须纳入真实系统对比。

单状态的correct_old_version栏为0表示它没有旧版本选择能力，并不等同于身份分类错误。该实验给出的积极可操作机制是：独立分支确实使少量可辨观测取回尚未重新观测的旧细节；同样的收益由标准版本MHT实现，不能归为新的概率算法。

## packet去重

| duplicates方法 | 最终PSNR | 标称95%区间的实际覆盖率% |
|---|---:|---:|
| single_map | 41.83 | 71.77 |
| exact_mht | 40.79 | 72.71 |
| exact_mht_dedup | 44.29 | 94.88 |
| fork_merge | 44.29 | 94.88 |

未去重的packet不是新独立证据，反复累积会过度自信。精确MHT加同样去重后与fork完全相同；去重不是fork特有的理论优势。UID只在同一episode中去重；跨episode身份确认后的相同物理packet需要全局provenance索引，本原型尚未实现。

## 长时间未回访与容量

实际分配并测量numpy数组nbytes，缓存0/16/64/256/1024个未解决episode；全分支保存、共享证据保存、保护已证实registry并将未解决episode元数据限制为64三种表示。所有对象的远景纹理证据严格为零，可以只存descriptor充分统计量而不复制纹理分支。

| 1024未解决episodes的表示 | 数组字节 | 仍保留已证实身份数 | 旧弱cue回访正确率% | 近景后正确率% |
|---|---:|---:|---:|---:|
| exact_mht_materialized | 6491136 | 4 | 65.62 | 100.00 |
| shared_statistics | 166912 | 4 | 65.62 | 100.00 |
| reserved_bank_budget64 | 13312 | 4 | 53.12 | 100.00 |

共享表示可无损减少本线性模型的分支数组复制；精确MHT也可以用同一共享表示，所以这是可压缩架构切口，不是推断原创性证据。预算化保护了已证实身份，但被丢弃的旧descriptor证据会降低近景揭示前的弱cue检索率，容量约束并非免费。上表不含Python对象管理开销，不是RSS或GPU时延benchmark。

## 理论判据与可训练架构切口

可操作判据：**身份后验未确认时不破坏已证实条目；新数据只进入独立episode；当候选之间的预测分布被新的独立观测区分后才merge；若同身份旧、新内容均有证据，则split并保留旧epoch。** 身份是否可辨应由候选预测分布的似然比/距离与独立cue决定，不能靠时间重复生成提高置信度。

本套件的共享充分统计量为 g=Σ1/σ²、b=Σy/σ²、c=Σy²/σ²、count；每个候选只需既有身份的mean/variance引用，无需复制整份分支特征。若实际生成模型能学习近似条件线性的局部latent观测头，则可以共享观测证据、保留多个身份解释，并在重新定位时只解码被支持的候选。这是需要实际训练检验的切口。

研究需要比较：学习到的对应/观测头能否在不同对象、非刚体、真实光流误差下维持这种共享性；压缩以后是否仍保留真实身份分支；真实生成模型是否获得比强MHT/检索缓存/更大状态容量更稳定的细节收益。仅给MHT换成fork或episode名字没有原创性。

## 原创性与完成状态

精确MHT+去重与fork在所有因果packet上的最大mean/variance/probability差为8.1e-15；旧版本检索实验的精确与压缩后验最大差为6.55e-15。因此此阶段并未确立一个超过标准MHT的新核心推断算法。

已取得的积极结果是：将不可区分性负机制转为可实施的非破坏性存储/揭示检索协议；在真实变化时避免旁路污染；少量可辨观测恢复旧版本隐藏细节；并给出可训练与压缩的具体接口。下一步应优先在实际冻结生成器上训练slot观测/对应与稀疏证据头，不能继续仅靠扩大toy种子数量把已知推断规则包装成论文核心。

## 可复现记录

```bash
python3 experiments/branch_memory/run_branch_suite.py --seeds 32 --episodes 8 --tag main
python3 experiments/branch_memory/analyze_branch.py
```

运行时间14.96s，含统计、容量分配和绘图；Python 3.14.5 (main, May 10 2026, 10:21:34) [Clang 21.0.0 (clang-2100.0.123.102)]，numpy 2.5.1，scipy 1.18.0，matplotlib 3.11.1。命令、seed与源码SHA256见main/metadata.json，执行源码快照见main/source_snapshot.py。

输出：per_packet.csv、per_episode.csv、summary.csv、paired_contrasts.csv、archive_return.csv、capacity.csv及三张图。

![身份揭示曲线](main/branch_accuracy.png)

![共享远景的不同纹理身份](main/aliased_identities.png)

![重复packet与容量](main/dedup_capacity.png)
