# Streaming SR / Omni Video Restoration

本分支记录从 `archive/modelarts-20261103` 接管后的 Open-H3 组件迁移、Rφ 恢复器、生成器适配、数据与对照实验。最新交接日期为 **2026-11-13**。

**接手 agent 请从 [HANDOFF.md](HANDOFF.md) 开始。**

| 文档 | 内容 |
|---|---|
| [完整技术汇报](docs/handoff/20261113/TECHNICAL_REPORT.md) | 架构与改动、各阶段结果、负结果、创新性判断和后续路线 |
| [运行接续手册](docs/handoff/20261113/OPERATIONS.md) | 远端资产、执行器、合同、资源检查与准确接续步骤 |
| [证据索引](docs/handoff/20261113/EVIDENCE_INDEX.md) | 结论对应的报告、检查点身份和证据位置 |
| [实验账本](docs/handoff/20261113/EXPERIMENT_LEDGER.json) | 机器可读状态、路径与 SHA |
| [接管历史](TAKEOVER.md) | 按时间保存的决策和实验记录 |
| [原 archive 复现说明](REPRODUCE.md) | 接管前的历史方法与结果，不能当作当前方法成绩 |

当前正式 guide-matched、多任务 RGB 轨迹 LoRA 训练为 **0 步**；已有旧 raw-guide SR300 步结果和真实 AdamW 单步保存子集。下一步是补验该保存点的严格重载与部署合并。输入保护头的当前方案未通过质量筛选，其负结果保留。

Git 保存代码、合同与小型证据。大模型、数据、RGB/梯度数组和未完成的迁移下载不随克隆提供；资产位置、SHA 和获取前置条件见运行手册。
