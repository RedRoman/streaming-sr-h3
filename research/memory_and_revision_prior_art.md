# 持久记忆与修订：新增直接前作核查

2026-10-15。以下依据论文原文或出版方摘要；用于确定强控制，不能视为完整新颖性检索。

| 原始来源 | 已核实的机制 | 对当前实验的约束 |
|---|---|---|
| [RECAP-Forcing](https://arxiv.org/html/2608.26671v1)，§3 | 按新出现内容保存原始KV；光流决定novelty bank准入，初始sink保留首印象，固定容量按新颖程度选择 | 单纯按内容而非时间组织记忆、保留首次出现patch，不构成新贡献 |
| [SparkVSR](https://arxiv.org/html/2603.16864v1)，§3 | 稀疏HR keyframe与连续LR双编码，latent/pixel两阶段训练；reference dropout和guidance；可编辑关键帧传播 | 稀疏参考注入、历史HR条件与LR结构锚点本身已有明确先例 |
| [Recurrent Structure-Detail Network](https://arxiv.org/html/2008.00455v1)，§3.3 | 隐状态适配用于应对外观变化和错误累积 | 结构/细节拆分与选择历史状态本身不是新机制 |
| [Multi-Memory Streams](https://ieeexplore.ieee.org/document/11153883/)，出版方摘要 | 曝光纠正、动态/静态对齐和多记忆融合 | 不以曝光归一化、多个memory或融合门控作为单独新颖性依据；尚未阅读全文 |
| [D-Flow SGLD](https://arxiv.org/html/2602.21469v1)，§2与§4 | 固定flow映射，推断source空间的后验；在既有D-Flow优化上加入prior与SGLD | 固定生成器、更新source噪声并保持先验映射已有先例，必须比较相应优化/采样控制 |

结合此前Sequential Flow Matching与Sequential Posterior Sampling核查，当前待测差异收敛到具体行为：新增证据可修订相关解释；独立身份不漂移；错误证据后来被撤回或重解释时，状态能恢复；同一证据集合的不同有效到达路径不会任意改变身份。

这些是本项目的实验假设，不是对所有前作能力的否定，也不是已经证明的新方法。需配平观测、生成初态、经典预处理、状态容量和实际计算，再判断是否有独立增量。B3 neutral条件已显示预处理配平后经典融合足够，不能把该结果改名为学习贡献。
