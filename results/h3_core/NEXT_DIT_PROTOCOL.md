# 下一实验：固定真实 H3 proposal，学习新证据触发的条件记忆修订

**假设而非结论。** 前两轮的 VAE(upLR)+RGB 均值 prior 几乎没有多模态身份细节；这轮直接使用真实 50-block BF16 MiniMax-H3 的49步样本。要检验的是：学习到的非高斯条件更新，能否在相同物理保护下，比 shared seed、经典投影和真实非线性 LR 优化更准确地修正细节，并保留无关身份。

**两个同时可测的目标。** 同一 far LR 使用预注册 seeds 101/202/303 得到多个可能身份。新 near cue 能证伪的局部纹理/边缘必须向真实观测修正；cue 之外仍不可辨的高频身份必须相对各自 X0 保留。逐 seed 报告局部真实误差与未观察查询跨 prefix 变化，并报告种子间距离是否在已辨区域收敛、在仍欠观测区域保留。单纯把所有样本推到同一个条件均值，即使 MSE 下降，也不能算成功。

**输入与时间。** 先8个 train /4个 val 做接口，随后固定32/8；不碰新 test。t=0 只把第一张 far LR 重复为 native22-frame guide（重复不算新观测），固定 neutral Qwen context、seed、official49-step grid，实际运行 DiT 得到 X0；同一源的所有方法共享同一 X0。保存原始采样 latent、RGB、同 DC 后参考及 SHA。只在预定新 near cue 到达时写 detail state；相同 footprint 的独立噪声只更新 coarse，原 packet 副本不更新。return 只读取 state。先 oracle 几何/可见性，之后单列 LR-only 对应失败。

**可训练部件。** 固定 H3 基座与初始 X0，不重新抽身份。记忆含只读 reference features 与32维可修订 code；瞬时外观从当前 LR 单独估计。小网络读取新 near 的真实 LR innovation、物理可辨方向与旧 code，产生 canonical update；native condition adapter 将 reference/current-LR/updated-code 合为一段 guide latent（24×7×16×16，448个 condition tokens），直接通过冻结50层 DiT 训练。它不是仅给 VAE 像素后处理调 scalar：可学习的是 reference/cue 依赖的 **native conditioning transport**。资源组已验证 guide-input 梯度能穿全部50层、base gradient为0。

**约束与训练。** 所有方法共用 decoded-LR DC、相同 near-support/unknown-identity 保护，经典投影不计新贡献。无新 cue 时 code恒等；有 cue 时真实 near LR 与过去 witness约束修正错误，cue之外的 reference高频查询保持。用共享初始 X0 的成对 prefix/cue rollouts：相同 prefix，只改变局部可见新 cue。训练集另构造严格同 far LR、局部 near 可区分的成对高频扰动，检查更新是否响应真实 cue 而非内容类别。新 cue 支持区用 GT 重建和 native flow-matching 监督；仍欠观测区用各自 X0 的多尺度身份特征/高频查询保持，不用统一 HR-MSE 强行抹除种子差异。HR只进 train loss。appearance变化不得写 content state。直接使用 native22 做2–5步梯度与时间接口检查，再固定主预算；不以5帧边界行为指导主结构，不把小接口试验称主结果。

**配平与判别。** 每法1次共同初始化+near/return各1次 native49调用，噪声、context、guide tokens、adapter参数量一致；额外 VAE/优化调用另报。控制必须有：LR guide；共享 X0+shared seed；共享 X0+经典 near-support/DC；同 prior真实D40步 LR inverse；等容量 stateless/GRU；learned transport；zero-update严格退化检查。不能只胜 current-only 或 window baseline。

**已有先例边界。** [Accelerated Sequential Flow Matching, §3.3–3.4](https://arxiv.org/html/2602.05319v1) 已经学习 previous-posterior→current-posterior flow，并用生成轨迹、重加噪训练。因此 warm start / posterior transport 不是本轮可主张的新意。另保留同 X0、同观测、同参数与调用数的基本 sequential-flow/native-adapter 控制，仅去掉局部可证伪修订与各 seed 未约束身份成对训练。候选贡献须由同一对象长回访中“纠错且保身份”的独立增量支持；不靠改名成立。

**成功/失败。** 同时报告新 cue ROI及return误差、LPIPS、梯度/高频能量、全部真实LR残差、未受新cue约束查询相对 X0 的漂移及延迟。必须胜等容量历史控制并超出经典投影/非线性逆控制的解释，且不是模糊换低漂移；否则记录无独立收益，不扫gain/loss。初始生成样本固定、identity lock只读；更新是否正确由新观测支持而非生成结果自证。确认结构后使用从未参与选择的新 source和真实长流复验。

**首个实际 proposal 接口结果（train100075，非收益结论）。** 三个预定种子均完成 native22×49次真实50层 BF16 forward，11.09/12.25/12.61秒主采样。相同首LR的经典DC后PSNR28.895/28.898/28.895dB；相同DC的VAE-only28.903，三seed差异MSE1.42–1.93×10⁻⁵，确认非同一确定性VAE重建。该源未证明H3优于VAE，后续必须检验细节错误是否被新near cue有效证伪并保留未约束身份。文件见dit_proposal_train100075/initial_proposal_evaluation.json。
