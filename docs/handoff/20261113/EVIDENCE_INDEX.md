# 实验与证据索引（2026-11-13）

本索引按最后验收状态整理。`completed`只说明限定执行/工程通过；`diagnostic`是已完成诊断，可能明确否定方法；`failed`原样保留；`ready_notrun`尚无对应实际结果。机器可核验清单见[EXPERIMENT_LEDGER.json](EXPERIMENT_LEDGER.json)，每个`local_artifacts`项均有完整SHA256和字节数。远端大文件SHA仅引用已有验收，不冒充本次重哈希。

## 现在应接什么

- 优先补齐已保存H3 AdamW一步的 **reload+merge：6 NFE、2 decode、0 backward、0 update**。现成 runner 为 `preparation/omni_followup_runtime_20261112b` 的 `saved_adamw_continuation`；仍需新run plan/admission及即时资源、协作锁和15秒二检。不要重跑原超时一步训练。
- 原raw-guide SR300、Stage2、native六格、固定H3/controls、缓存诊断、P1教师1000与两输入头1000均已终态。新的guide-matched正式LoRA研究训练仍 **0步**。
- history headroom及输入保护screen均失败；三history学生未准入，旧1745保护头分支按预声明暂停，不能靠加步/改阈值续跑。
- 最新只读状态见[runtime_local_snapshot](runtime_local_snapshot.json)与[runtime_ops_snapshot](runtime_ops_snapshot.json)。迁移session为FAILED(exit255)，partial仅105,492,480/718,719,172 bytes，未上传/安装；omni-r已PAUSED。31112卡4–7、31984卡2/3/7当时无PID，**不是准入或持续空闲承诺**。

## 分阶段结论

| 实验 | 状态 | 关键结论 | 首要证据 |
|---|---|---|---|
| 历史来源与旧评测口径审计 | `diagnostic` | 旧heldout96来源全部出现在Stage7 train87/val9；不能用于新Stage7训练的独立测试。不能据此反推使用另一生成数据集的旧Stage1主表已泄漏。v2 flow方向/边界/遮挡、配对统计修复后不得把旧JSON直接改schema混表。 | [记录](../../../results/takeover_20261111/data_audit_31984.json) |
| OpenH3 raw-guide SR 300步pilot | `completed` | 8来源LPIPS全部改善，但PSNR配对均值CI跨零，时序并非一致改善。这个mode=base/raw-guide实验没有R_phi，不能作新guide-matched或omni验收。 | [记录](../../../results/takeover_20261111/sr_pilot_300_metrics_v2.json) |
| R_phi首轮三臂1000步 | `completed` | 训练/评测终态不等质量通过；blind与joint confidence臂fidelity不相同，差异不能归因于confidence收益。随后改用冻结恢复器单独拟合confidence。 | [记录](../../../results/takeover_20261111/rphi_completed/audit.json) |
| Stage2 legacy_g05 | `completed` | 仅选作固定H3工程参考，非质量验收模型；compression平均PSNR仍负，identity8/8受损。 | [记录](../../../results/takeover_20261111/rphi_stage2_completed/legacy_g05/eval/metrics.json) |
| Stage2 native_g05 | `completed` | 同预算native encoder未整体优于legacy，保留负结果。 | [记录](../../../results/takeover_20261111/rphi_stage2_completed/native_g05/eval/metrics.json) |
| Stage2 native_g20 | `completed` | 更强guard改善clean MSE/压缩平均值，但牺牲前三任务且near-clean LPIPS/flow恶化；已撤回主候选地位，仅guard消融。 | [记录](../../../results/takeover_20261111/rphi_stage2_completed/native_g20/eval/metrics.json) |
| Stage2 frozen_confidence | `completed` | 1500步只训练variance head；父native_g05的fidelity SHA及全部记录RGB/control/temporal指标一致。覆盖率不等于独立风险校准或可部署收益。 | [记录](../../../results/takeover_20261111/rphi_stage2_completed/frozen_confidence/eval/metrics.json) |
| 固定H3 ordinary四任务六臂 | `diagnostic` | 一开发来源6UZVovkR3IU。ordinary压缩明显过修复，最终H3+anchor四任务PSNR均低于Rphi-only；controls所有H3臂均损伤input和Rphi-only，alpha.25仅缓解。 | [记录](../../../results/takeover_20261111/fixed_h3_legacy_smoke_review_20261111.json) |
| 固定H3 clean/near-clean六臂 | `diagnostic` | identity四标签是同一输入重复；nearclean四噪声draw也只来自一来源。所有H3臂相对input及Rphi-only损伤，不能自动扩普通dev或称无损保护。 | [记录](../../../results/takeover_20261111/fixed_h3_controls_review_20261111.json) |
| 固定VAE往返诊断v2 | `diagnostic` | 24个往返用真pre-internal-clamp捕获并验clamp后等原decode；clean/nearclean/压缩显著损伤。VAE(Rphi)与H3-Rphi的warm-start路径不同，不能将MSE差解释成独立可加因果贡献或VAE不可突破下界。 | [记录](../../../results/takeover_20261111/vae_roundtrip_v2_completed/scientific_review.json) |
| 缓存scalar残差固定7点诊断 | `diagnostic` | 12样本×固定7点、0新H3/训练；GT pooled-MSE oracle仅事后诊断不部署。t=.05相对Rphi-only降低nearclean PSNR不等相对原input降低；SR与native anchor不同。 | [记录](../../../results/takeover_20261111/fixed_h3_residual_mix_review_20261111.json) |
| 缓存VAE补偿对照12×7×2 | `diagnostic` | 168评分完整、同beta两family比较；R+β(G−V)没有统一优势，去噪损失scalar有益修复，中大β压缩仍比Rphi差，clean仍受损；不选部署β。 | [记录](../../../results/takeover_20261111/cached_rgb_compensation_completed_20261111/scientific_review.json) |
| favernet deblur 单源官方基线 | `completed` | 官方权重CPU完整离线22帧推理/评分已完成；deblur/denoise空间指标强于Rphi，但deblur flow略差；两模型compression三指标均略差于Rphi。六格均勿重跑。 | [记录](../../../results/native_quality_completed_20261111/summary.json) |
| avernet deblur 单源官方基线 | `completed` | 官方权重CPU完整离线22帧推理/评分已完成；deblur/denoise空间指标强于Rphi，但deblur flow略差；两模型compression三指标均略差于Rphi。六格均勿重跑。 | [记录](../../../results/native_quality_completed_20261111/summary.json) |
| favernet denoise 单源官方基线 | `completed` | 官方权重CPU完整离线22帧推理/评分已完成；deblur/denoise空间指标强于Rphi，但deblur flow略差；两模型compression三指标均略差于Rphi。六格均勿重跑。 | [记录](../../../results/native_quality_completed_20261111/summary.json) |
| avernet denoise 单源官方基线 | `completed` | 官方权重CPU完整离线22帧推理/评分已完成；deblur/denoise空间指标强于Rphi，但deblur flow略差；两模型compression三指标均略差于Rphi。六格均勿重跑。 | [记录](../../../results/native_quality_completed_20261111/summary.json) |
| favernet compression 单源官方基线 | `completed` | 官方权重CPU完整离线22帧推理/评分已完成；deblur/denoise空间指标强于Rphi，但deblur flow略差；两模型compression三指标均略差于Rphi。六格均勿重跑。 | [记录](../../../results/native_quality_completed_20261111/summary.json) |
| avernet compression 单源官方基线 | `completed` | 官方权重CPU完整离线22帧推理/评分已完成；deblur/denoise空间指标强于Rphi，但deblur flow略差；两模型compression三指标均略差于Rphi。六格均勿重跑。 | [记录](../../../results/native_quality_completed_20261111/summary.json) |
| VAE梯度工程链 a | `failed` | 首个decoder探针在反传前失败；2forward/0backward，不能使用planned预算冒充完成。 | [记录](../../../results/takeover_20261111/vae_decode_grad_smoke_20261111a_failed/local_verification.json) |
| VAE梯度工程链 b | `completed` | 5×128² synthetic输入梯度/forward parity通过，只是小decoder。 | [记录](../../../results/takeover_20261111/vae_decode_grad_smoke_20261111b_completed/local_verification.json) |
| VAE梯度工程链 c | `completed` | 22×128² temporal chunk decoder梯度通过，未建立full geometry。 | [记录](../../../results/takeover_20261111/vae_decode_grad_temporal_smoke_20261111c_completed/local_verification.json) |
| VAE梯度工程链 d | `failed` | 22×480×832 no-grad成功，grad forward OOM，未完成backward。 | [记录](../../../results/takeover_20261111/vae_decode_grad_fullres_failed_20261111d/local_verification.json) |
| VAE梯度工程链 e | `completed` | 36块checkpoint小尺度与c梯度SHA完全相同，未证明fullH3。 | [记录](../../../results/takeover_20261111/vae_decode_grad_checkpoint_smoke_20261111e_completed/audit.json) |
| VAE梯度工程链 f | `failed` | full geometry 2forward/1backward完成但输入梯度missing/nonfinite/zero门失败，旧记录不能区分三者。 | [记录](../../../results/takeover_20261111/vae_decode_grad_checkpoint_fullres_20261111f_failed/audit.json) |
| VAE梯度工程链 g | `diagnostic` | 同图双backward：unscaled全零，scale1024后大部分非零。只建立配对缩放诊断，不是单反传训练内存门。 | [记录](../../../results/takeover_20261111/vae_decode_grad_loss_scale_pair_20261111g_completed/audit.json) |
| VAE梯度工程链 h | `failed` | single scaled backward finite/nonzero，但严格梯度SHA不等g。失败保留，不能用两条max标量差当逐坐标max梯度差。 | [记录](../../../results/takeover_20261111/vae_decode_grad_scaled_fullres_20261111h_failed/audit.json) |
| VAE独立进程A1/A2重复性 | `failed` | A1/A2 forward一致但梯度byteexact失败，条件B未执行。保留严格失败，不把小数值差升格为通过。 | [记录](../../../results/takeover_20261111/vae_gradient_repeatability_20261111i/scientific_review.json) |
| VAE固定latent三步下降 | `diagnostic` | seed102预声明rho=.001/scale1024，8F3B及三次下降通过；不修复i严格重复性失败，不是LoRA质量实验。 | [记录](../../../results/takeover_20261111/vae_latent_descent_20261111j_completed/scientific_review.json) |
| 真实三步H3 latent-only零adapter | `completed` | 6NFE/1backward/0VAE/0update，400梯度与200B非零，reference/candidate video/audio byteexact；小5×128²工程门。 | [记录](../../../results/takeover_20261111/h3_trajectory_probe_20261111b_completed/scientific_review.json) |
| 小尺寸H3+VAE同图RGB反传 | `completed` | 5×128²、6NFE/2decode/1backward/0update工程通过。不能代表full22或有用更新。 | [记录](../../../results/takeover_20261111/h3_joint_rgb_probe_20261112a_completed/scientific_review.json) |
| H3 latent探针a审计导入失败 | `failed` | load阶段0NFE/0backward失败：审计误把Torch类级伪源码名当项目路径；b修复并单独验收，a保留。 | [记录](../../../results/takeover_20261111/h3_trajectory_probe_20261111a_failed/local_verification.json) |
| 单条TRAIN compression VAE-only cache | `completed` | 2encode raw/Rphi+1decode_R，train source2BdYaGuVAKo，真实latents/RGB cache已CPU审计。只查已用VAE/context，unused base/refiner/turbo metadata-only，不能称全recipe已验。 | [记录](../../../results/takeover_20261111/train_vae_cache_portable_20261112e_31112_completed/science_review.json) |
| 小synthetic非零SGD保存重载合并 | `completed` | 200个B真实更新、400key持久化、reload严格相等；60个base矩阵/91坐标在BF16合并后改变，但video/audio/RGB四模式输出全相同，功能影响未证明。synthetic checkpoint禁止作realtrain warmstart。 | [记录](../../../results/takeover_20261111/h3_nonzero_adapter_probe_20261112b_31112_completed/root_acceptance.json) |
| 真实TRAIN full22联合RGB梯度 | `completed` | 22×480×832同图真实TRAIN RGB梯度通过，native tilebatch4固定；6NFE/2decode/1backward/0update。不是正式LoRA训练或恢复质量验收。 | [记录](../../../results/takeover_20261111/h3_full22_joint_rgb_probe_20261112a_completed/root_acceptance.json) |
| 真实TRAIN AdamW一步原run | `failed` | 3600秒guard在clear_and_strict_reload终止；实际已1backward/1AdamW/save，但reloaded/merged输出为0。原run永久failed；不能用clear_reload计数或generic审计文字声称严格重载通过。 | [记录](../../../results/takeover_20261111/h3_real_adamw_failed_20261112a/root_review.json) |
| 已保存AdamW一步CPU子集审计 | `completed` | 3222实物arrays、400adapter/1200 optimizer状态、unscale/clip/AdamW方程通过。zero/reference四流byteexact，updated RGB确实改变但MSE更差；只接受已存一步子集，不接受原run/reload/merge/部署。 | [记录](../../../results/takeover_20261111/h3_saved_subset_completed_20261112b/root_acceptance.json) |
| 已存AdamW reload+merge补验b | `ready_notrun` | 真实CPU10/10及输入inspect通过，但没有实际NPU结果。复用现成generic bounded runner的saved_adamw_continuation；只需新具体run plan/admission及实时锁/15秒二检，不需重造launcher。 | [记录](../../../results/takeover_20261111/h3_adamw_saved_continuation_b_independent_source_review_20261112.json) |
| P1修订来源/25源100NPZ | `completed` | 原40角色不变；38数值合格、25primary19train/6dev、13diagnostic、2ineligible。3来源复用+22新物化/132codec/0MP4decode，100NPZ远端file/pixel审计通过。旧research_ready=false与新pair_ready=true含义不同。 | [记录](../../../results/takeover_20261111/history_utility_revised_dataset_completed_20261112a/root_acceptance.json) |
| P1持久化1000×16 schedules及history-slot donor | `completed` | 两份1000×16 schedule仅元数据计划，不代表已完成训练；36strata/25125有效槽一一cross-source AND group donor可行。新input-preservation whole-map donor是另一份映射，禁止混用。 | [记录](../../../results/takeover_20261111/history_utility_schedules_completed_20261112a/local_review.json) |
| P1一次CPU teacher/student接口 | `completed` | 7恢复器forward/1gate、2backward、teacher1AdamW/student0update；379实际数组独立审计，仅接口工程。 | [记录](../../../results/takeover_20261111/history_utility_one_step_completed_20261112a/root_acceptance.json) |
| P1真实backend两步smoke | `completed` | 两步实际optimizer/严格checkpoint reload通过；smoke产物禁止作为研究warmstart。 | [记录](../../../results/takeover_20261111/history_utility_backend_completed_20261112a/root_acceptance.json) |
| P1固定1000步教师 | `completed` | fresh legacy初始化，1000updates/16000rows，58参数/174AdamW tensors及1000行日志验收；训练工程成功不等headroom通过，也未授权三学生。 | [记录](../../../results/takeover_20261111/history_utility_teacher_completed_20261112a/root_acceptance.json) |
| P1六源headroom完整评估 | `diagnostic` | 完整评估/真实数组复算通过，但七项门中的平均history gain失败：K1-minus-all MAE均值为负，4/6略正而2源大负；identity6/6损伤，nearclean5/6 MAE损伤。三history学生未准入。 | [记录](../../../results/takeover_20261111/history_utility_headroom_completed_20261112a/root_review.json) |
| 已存四mask及额外input oracle诊断 | `diagnostic` | 仅既有180六路RGB缓存、0模型；四mask GT-oracle/LOO sign同时删除与interaction作诊断，五候选另加input仅单列保护上界。不能称任意soft-gate理论上界或部署收益，不解除headroom失败。 | [记录](../../../results/takeover_20261111/history_mask_oracle_diagnostic_completed_20261112a/root_review.json) |
| input-preservation train/dev cache与实物审计 | `completed` | TRAIN16000行1000cache/7000array，DEV2880行180cache/1440array及旧1080array已实核；raw特征/标签FP64误差门通过。不能重新扫全cache或把元数据资格当科学收益。 | [记录](../../../results/takeover_20261111/input_preservation_train_cache_audit_completed_20261112a/root_acceptance.json) |
| 单图benefit标签donor映射 | `completed` | 全16000row/36strata跨source AND group双射，含1815个00缺历史行。与旧25125有效槽映射不同；不得重采样或丢00。 | [记录](../../../results/takeover_20261111/input_preservation_donors_completed_20261112a/payload/runs/input_preservation_donors_20261112a/independent_audit.json) |
| 输入保留头 matched 1000步 | `completed` | 1745参数CPU头完整1000步/16000行；same initialization/same features，唯一标签对应差异为whole-map donor。保存/清空/重载、全部1000行配对已审；固定final1000，不按DEV选checkpoint。 | [记录](../../../results/takeover_20261111/input_preservation_heads_training_completed_20261112a/payload/runs/input_preservation_head_research_matched_20261112a/output/report.json) |
| 输入保留头 shuffled 1000步 | `completed` | 1745参数CPU头完整1000步/16000行；same initialization/same features，唯一标签对应差异为whole-map donor。保存/清空/重载、全部1000行配对已审；固定final1000，不按DEV选checkpoint。 | [记录](../../../results/takeover_20261111/input_preservation_heads_training_completed_20261112a/payload/runs/input_preservation_head_research_shuffled_20261112a/output/report.json) |
| raw/K1/matched/shuffled完整DEV四臂 | `diagnostic` | 数值/执行验收通过，但科学screen失败。两头所有103680tile都选择K1，clipped/unclipped逐行RGB等于K1；没有有效输入保护。禁止加步、改旧阈值或用DEV选checkpoint。 | [记录](../../../results/takeover_20261111/input_preservation_eval_completed_20261112a/root_acceptance.json) |
| TRAIN标签分布与常数Huber诊断 | `diagnostic` | 576000tile正标签73.7436%，零点Huber梯度为负，常数最优+.606486；identity全负但仅12.5%row，nearclean并非全负。支持正先验主导假设，不证明特征无信息。 | [记录](../../../results/takeover_20261111/input_preservation_eval_completed_20261112a/payload/runs/input_preservation_train_label_diagnostic_20261112a/report.json) |
| TRAIN-only原权重5fold线性Huber | `diagnostic` | head内部5个source_group fold/每折200固定GD，标准化只用fold-train；19source宏常数1.14531581优于线性1.14741963，8/19略改善、全部tile仍选K1。教师见过全部19源，不能称restorer held-out。 | [记录](../../../results/takeover_20261111/input_preservation_train_feature_probe_completed_20261112a/independent_acceptance.json) |
| 同特征同fold仅改正负权重 | `diagnostic` | 先精确重放旧score SHA，只改fold-train正负权重各0.5；BA/AUC近随机。负误选降低几乎同幅牺牲正召回，三native profile BA均<.5，nearclean局部较好。按事前规则暂缓保护头支线，不再加步/调阈值。 | [记录](../../../results/takeover_20261111/input_preservation_balanced_huber_probe_completed_20261112a/root_acceptance.json) |

## 首要数值与科学负结果

- **SR300**：LPIPS配对均值−0.0220859，8/8改善；PSNR+0.0555519dB但95%区间跨0。时序低频4/8、细节2/8改善。没有R_phi，不能据此验收omni接口。
- **Stage2**：legacy/native g05去噪收益明显，compression平均PSNR仍负；native未整体优于legacy。g20保护MSE的代价是前三任务及nearclean LPIPS/flow退步。frozen confidence父fidelity完全同，但只在训练集拟合variance，不是独立风险证书。
- **固定H3**：单开发源compression Rphi40.4759dB→H3-Rphi-anchor34.7937dB，LPIPS/flow亦恶化；controls所有H3臂相对input与Rphi-only损伤。VAE诊断不是不可突破的质量下界。
- **Faver/Aver**：单源三native任务6/6已完成。deblur/denoise空间质量更好，deblur flow略差；compression两模型均略差于Rphi。offline22帧/未来21帧不是causal K3同预算SOTA。
- **H3工程**：full22仅1反传/0更新通过。synthetic SGD虽然200B非零/60base矩阵改变，输出却全等，功能影响未证明。真实AdamW已存子集输出改变但MSE0.0030889883→0.0030982170（+0.299%），不是质量改善；strict reload/merge待补。
- **P1教师**：1000步训练工程验收，headroom主门K1−all MAE=−0.000260692687；4/6来源正但2源大负，clean/nearclean保护也有负例。不能把complete当gate pass。
- **输入保护两头**：两臂全部103680tile选K1；native PSNR31.08496/LPIPS0.0968657均等K1；identity MAE0.001548，nearclean输入48.53855→47.97560dB。screen失败。
- **5fold诊断**：原Huber线性源宏1.14741963比常数1.14531581更差，仍100%选K1。仅改class权重后BA0.501422/AUC0.510326，负误选0.530613但正召回0.533458，三native profile BA均低于0.5。不是可部署保护。

## 如何核验或复用

1. 从对应root acceptance/scientific review读范围，再读其指向的原report、execution contract和auditor结果。`local_artifacts`全SHA只核本地小文件；大数组是否曾被真实重算写在原local/remote审计边界中。
2. 所有列出的旧run都设置`do_not_rerun_existing_run=true`。缺少大checkpoint/NPZ时，先恢复既有资产及核SHA，不要重新训练或物化来填证据。`ready_notrun`只允许另建明确的新run。
3. 只复用source/target/window/seed/weights/metric schema完全相同的格子；v2 temporal metric不能给旧JSON改名后混表。mean-per-frame PSNR、pooled MSE PSNR、配对中位数不能互换。
4. 本地11月13状态覆盖旧时间线中的RUNNING/pending；旧失败保留，并不因为新版本成功而改成通过。

## 大资产与Git边界

完整模型recipe在[refiner_turbo_30794.json](../../../configs/openh3_20261111/refiner_turbo_30794.json)：base约40.23GB、VAE5.21GB、refiner5.87GB、Turbo1.96GB、neutral context299557B。精确大小/SHA见账本`large_assets_not_in_Git`。路径依赖host，不能仅复制manifest就假定资产存在。

必须另外保留：原视频与NPZ数据/receipts，stage2最终checkpoint（30794及31984已验镜像），H3已存AdamW adapter/optimizer及zero/updated数组，teacher/head最终checkpoint，native预测、固定H3/VAE缓存、P1头特征与评估NPZ。它们不随小Git仓库完整发布；恢复位置与SHA从对应合同/审计获取。

stage2镜像记录：[stage2_checkpoint_backup_verified_31984.json](../../../results/takeover_20261111/stage2_checkpoint_backup_verified_31984.json)。Turbo精确重建记录：[turbo_exact_reconstruction_review](../../../results/takeover_20261111/turbo_exact_reconstruction_review_20261112.json)。旧15下载/8恢复传输已被精确重建替代，不能恢复那些旧失败session。

打包路径快照见账本`handoff_packaging_paths_snapshot`，它是历史记录而非待办或重跑清单。唯一剩余的迁移run_once.sh已纳入发布，下载partial不入Git。大型archive保留远端，Git保留小manifest/receipt、文本证据与资产位置；private transfer材料、token/私钥、模型/NPZ不进公共仓库。

## 容易误读的历史状态

- TAKEOVER是追加日志，旧“下一只准备/尚未运行/研究0步”后面已有完成记录；本索引按最终root acceptance。
- `rphi_stage2_scientific_review_v2.json`取代v1候选意见。guard2不是默认最优。
- frozenconfidence指标相同仅支持固定fidelity隔离，不是训练/DEV独立风险校准。
- 原AdamW失败terminal审计的generic成功路径说明不能代替后来的3222数组子集审计。clear_reload计数也不等严格输出通过。
- nonzero SGD模型/参数有变化不等实际输出有变化；大BF16权重舍入后影响需另量化。
- VAE g是双backward诊断，h/i严格重复性失败不被j下降或joint成功追认。
- original decoded已经内部clamp；后续blend clamp不是true pre-VAE-clamp。native RGB anchor与SR Q4 anchor不同。
- 同源identity四label不等四个独立样本；19TRAIN-fold只是head内部诊断，teacher已见19来源。

## 本次整理的验证范围

本账本整理仅本地小JSON/源读取与SHA、关键summary之间的数值一致性核对；没有SSH、模型/训练/评分、NPZ或权重重扫。11月13日远端状态由并行运维只读采集，引用两份独立快照。各条具体验证在账本`key_metric_consistency_checks`。已有原始科学负结果未改写，交接不创建新实验准入。
