# Omni Video Restoration 接手入口

更新日期：2026-11-13，Asia/Singapore。接手对象是 `RedRoman/streaming-sr-h3` 的 `codex/omni-openh3-takeover-20261111` 分支，起点是 `archive/modelarts-20261103` 的 `43efbd4d090056365dc46f0ac6edb88e3d875761`。本次交接发布代码、实验记录和技术汇报，不代表方法已完成或质量门已通过。

**目前完成的是预训练组件接入、多个 Rφ 与历史信息实验，以及真实 RGB 训练链路的逐项验证。当前输入保护头没有取得可用增益；新的 guide-matched H3 LoRA 正式训练仍为 0 步。下一步优先完成已经保存的一次 H3 AdamW 更新的重载与部署合并补验，再推进多来源同输入比较。**

## 从哪里开始

| 阅读顺序 | 文件 | 用途 |
|---|---|---|
| 1 | [技术汇报](docs/handoff/20261113/TECHNICAL_REPORT.md) | 架构、数据、实验结果、负结果、研究判断和下一阶段设计 |
| 2 | [运行接续手册](docs/handoff/20261113/OPERATIONS.md) | 主机、资产、锁、已有 launcher、准确命令、待完成前置条件 |
| 3 | [实验与证据索引](docs/handoff/20261113/EVIDENCE_INDEX.md) | 从结论定位原始报告、合同、检查点记录和证据 |
| 4 | [机器可读实验账本](docs/handoff/20261113/EXPERIMENT_LEDGER.json) | 各阶段真实状态、证据路径与 SHA，不要从 run 名推断完成情况 |
| 5 | [本地/传输快照](docs/handoff/20261113/runtime_local_snapshot.json)、[远端快照](docs/handoff/20261113/runtime_ops_snapshot.json) | 本次交接时的新鲜观察及采集时间 |
| 历史追溯 | [TAKEOVER.md](TAKEOVER.md)、[阶段计划](configs/openh3_20261111/rphi_stage2_plan.json) | 逐次决策和详细历史；旧日志中的 running、pending、0 步可能已经被后面的终态覆盖 |

## 接手时最重要的实际状态

- **H3 的失败与已验收子集必须同时保留。** `h3_real_adamw_update_20261112a` 达到 3600 秒上限，在重载阶段失败。其保存的 adapter、AdamW 状态和更新前后 RGB 已通过独立 CPU 子集审计；这不把原 run 改成成功，也不代表 strict reload 或 BF16 部署合并已经通过。
- **补验源码和通用有界执行器已经存在。** `h3_adamw_saved_continuation_preparation_20261112b` 只允许 6 NFE、2 次 decode、1 次 merge、0 次反传和更新。`omni_followup_runtime_20261112b` 已支持 `saved_adamw_continuation`。接手需要为新的隔离 run 绑定具体 plan/admission，并完成实际输入和资源检查，无需重新训练，也不应直接复用失败 run 的旧 queue。
- **原主机可能已能继续。** 2026-11-13 约 10:42 SGT 的快照中，31112 卡 4–7 没有 PID，31984 卡 2、3、7 没有 PID。这不是预留或启动许可；新进程前必须重新核实健康、显存、进程、共享锁，并进行 15 秒二次空闲检查。优先考虑 31112 原地补验，避免把跨机迁移当作必经步骤。
- **迁移下载已失败。** 唯一 session `h3-saved-continuation-missing-download-20261112a` 返回 exit 255。本地文件为 105,492,480 / 718,719,172 bytes，不能解包或安装；完整预期 SHA 为 `b26949c1b9c8b9fbd98d94249146c4135a19dda16ec824a465c3987124fd3f2a`。源包已成功打好，不要重打包。仅在仍需要迁移时查失败日志并处理传输；未启动目标上传或安装。
- **保护头支线已经按预声明暂缓。** matched/shuffled 两头各 1000 步后，6 个开发来源的 2880 行输出全部等于 K1。五折权重对照仅达到 source-macro BA 0.501422、AUROC 0.510326。保留这些负结果，不加步数、不修改旧开发集阈值、不把它包装成有效的新模块。
- **FaverNet/AverNet 的 6 个单来源任务早已完成。** 这是同一个来源、三个 native 任务、22 帧的 offline smoke；不是多来源 SOTA 主表。已有预测只能在输入、target、窗口和指标合同完全匹配时复用。
- **自动回访当前为 PAUSED。** `omni-r` 的暂停是本次交接前已有状态，本次没有修改它，也没有新启训练、迁移或安装。接手 agent 可以在用户已授权的项目范围内继续工作；确认唯一执行责任后再决定是否恢复调度，避免两个 agent 同时启动同一实验。

## 不能混淆的结果

| 结果 | 目前能说明什么 | 不能据此声称什么 |
|---|---|---|
| 旧 raw-guide SR LoRA 300 步 | 8 个开发来源的 LPIPS 平均改善约 0.02209，已有逐源证据 | 新 Rφ-guide 匹配训练完成；统一时序收益；omni 恢复完成 |
| Rφ / history teacher / 输入保护实验 | 有真实训练、相同预算对照和可追溯负结果 | 新结构稳定优于原恢复器，或跨来源独立泛化已验证 |
| 真实 H3 RGB 反传与已保存 AdamW 子集 | 新 LoRA 的更新、保存状态及部分数值链条已得到实物验证 | 原超时 run 成功、重载与部署等价、恢复质量有增益 |
| 输入保护两头与两轮线性诊断 | 当前方案没有提供可用选择能力，权重变化主要改变取舍 | 所有原图特征都不含信息，或任何保护模块都不可学 |
| FaverNet / AverNet 单源比较 | 官方权重、CPU 前向及该来源的推理评分可用 | 在相同未来信息预算下优于现有 SOTA，或对全部 native 任务泛化 |

训练和开发划分存在多个阶段：早期 32 train / 8 dev 与后来 history 路线的 19 train / 6 dev 不可混写。后者内部五折只把诊断 head 的拟合按来源隔离；其固定教师已见过全部 19 个训练来源。现在没有足以支撑最终论文结论的新独立 frozen test。

## 第一轮接手建议

1. 阅读上述三份人类可读文件，再核验账本中的关键证据。不要从历史日志复制“启动”命令后直接执行。
2. 优先检查 31112 的保存子集、continuation b 合同和通用 runner，绑定一个新的 run plan/admission。资源重新通过后只补 reload + merge；失败则保留终态并定位问题，不扩大预算或盲目重训。
3. 补验通过后，完成与实际部署路径一致的 RGB 数值/质量评估，再冻结 guide 与新增 LoRA 的匹配训练合同。旧 raw-guide adapter 不能直接充当新组合的匹配训练结果。
4. 并行冻结多来源、相同输入与 target 的 Rφ-only、固定 H3 raw-guide / Rφ-guide / 固定 soft-anchor、FaverNet / AverNet 比较矩阵。标清未来帧预算；只补缺失格，保留压缩和 clean/near-clean 负例。
5. 后续模块设计应增加可验证的新信息或修正明确失效机制，例如更有辨识力的局部退化/残差与运动证据。另立单变量计划和验收标准，不继续围绕这次失败 head 做开发集阈值搜索。

## 获取代码与资产

```bash
git clone --branch codex/omni-openh3-takeover-20261111 \
  https://github.com/RedRoman/streaming-sr-h3.git
cd streaming-sr-h3
git rev-parse HEAD
```

本地原工作区为 `/Users/hezihao/Documents/New project/omni-openh3-takeover-20261111`。远端工程根目录为 `/cache/zihaohe/omni_openh3_20261111`，主要涉及 30794、31112、31984。连接凭证不写入仓库。

Git 中保留代码、合同、小型报告和部分既有证据归档；模型权重、数据、RGB/梯度大数组及未完成下载保留在远端或原工作区。克隆仓库不等于已拥有这些资产，按 [OPERATIONS.md](docs/handoff/20261113/OPERATIONS.md) 和证据索引核实位置与 SHA。不要因 Git 中没有检查点或大数组而重跑已完成训练。

## 给接手 agent 的任务文本

> 接手这个分支的 omni video restoration 项目。先完整阅读 HANDOFF.md、技术汇报、OPERATIONS.md 和 EXPERIMENT_LEDGER.json，再核实最新远端状态。优先在 31112 完成已保存 H3 更新的 strict reload / merge 补验，使用现有 continuation b 与有界 runner，新建隔离 plan/admission，严格执行资源、共享锁及二次空闲检查。不要重启已完成或健康作业，不改冻结源码，不杀他人任务。保留输入保护与历史 headroom 的负结果，暂缓保护头支线；之后推进相同输入和预算的多来源比较、匹配 guide 的 LoRA 与新数据。每一步写清实际完成、未完成及科学结论边界。当前 omni-r 已暂停，先确认唯一执行责任。用户已有项目推进授权，无需为常规必要检查再次索取确认。
