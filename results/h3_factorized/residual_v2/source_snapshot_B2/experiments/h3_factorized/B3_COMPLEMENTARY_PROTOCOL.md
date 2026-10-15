# 下一轮固定协议草案：互补观测下的可逆细节状态

## 由实际结果提出的机制假设

B1 绝对 latent 写入在 far 回访丢失 62.6% 已证实 near 细节；B2 current-LR 锚定修复此问题，但单 near 的直接保留已经达到约 24.757 dB 的回访读出容量，学习状态尚无独立收益。下一轮不通过调权重解决这个问题，而要求多个互补传感足迹共同支持一个内容身份，并检验跨 appearance 更新是否可组合。

待验证假设：**在当前 LR 锚点条件下，学习到的可逆 appearance chart 能将不同光照、部分遮挡下的局部 LR 证据写入一个固定大小 canonical detail 状态，使无冲突证据近似可交换，并让新冲突只修订其可辨别部分。** 可逆耦合、解耦、观测投影、经典信息融合都不是新贡献；只有在固定预算和强控制之上，非线性 decoder 读出质量、可组合性或状态压缩有独立收益，才能支持这个候选。

## 数据与时间前缀

仍用官方 BSDS source-disjoint 32 train / 8 val；新协议设计与选择只看 train/val，16 test 保持封存。source/donor hash 沿用 B1 manifest；donor 不跨集合。自然图像的已知相机模拟用于机制定位，不称真实视频或真实长回访。

256×256 canonical 内容，far 为相同 blur σ0.6 + 4× bicubic；near 为 128×128 crop 经相同 blur + 2×，输入仍是实际 LR。四个 near footprint 左上角固定为 (32,32)、(32,96)、(96,32)、(96,96)，相邻重叠 64 像素。先只用已知真几何隔离融合问题；后续真实 flow 协议使用相同 LR-only 估计，不引入 HR correspondence。

单 episode 的 18 个时间步固定如下：

| t | 观测与用途 |
|---:|---|
| 0 | far A0，零细节状态初始化 |
| 1 | near footprint0 / appearance1 |
| 2 | far A2，不能冲掉 fine band |
| 3 | near footprint1 / appearance2 |
| 4 | far A0，第一次互补融合检查 |
| 5 | near footprint2 / appearance1，局部前景遮挡 |
| 6 | 无可见证据的只读查询，不写状态 |
| 7 | near footprint3 / appearance2 |
| 8 | far A0，完整覆盖检查 |
| 9 | t7 的重复 packet ID，必须幂等 |
| 10 | 新 packet ID、与 t7 相同足迹的独立 noise，允许改进均值 |
| 11 | far A1 |
| 12 | 仅在 footprint0 中发生真实局部内容替换后 near 回访 |
| 13 | far B2，检查改变处纠错、未变处保留 |
| 14 | 另一个局部遮挡 near，遮挡下无证据区域不可清空 |
| 15 | 无可见证据只读查询 |
| 16 | 可见 near0 回访，检查内容修订是否保留 |
| 17 | far B0 最终回访 |

光照沿用 B1 三种受控 RGB gain/bias/平滑 shade，appearance 权重不变。遮挡作为独立前景/不可见像素，不把遮挡值当背景真实观测；固定 noisy visibility mask 是传感元数据，各方法共用，并另测其误报。内容变化没有 oracle change mask；变化仅在当前可见且有 LR 证据冲突的部分可被辨别。离开后的无观察区不评价即时已知改变恢复为合理目标。

每源配两种顺序：0→1→2→3 和 3→2→1→0；最终 evidence 集一致。prefix 严格因果；训练可监督每个 prefix 的 HR，推理数据结构仅含 LR、几何、visible mask、packet ID。同一 source 的顺序变体不是独立样本。

## 必须先运行的便宜机制审计

1. 在实际 8 val 自然图像上比较 last-near、对齐残差平均融合、全历史线性 inverse；中性 appearance/无遮挡版本先隔离互补信息量。报告前三次 near 的旧区保留、新区获得、重叠区噪声与最终 union ROI。若 full-history 不胜 last-near，先修 episode，而不训练神经网络。
2. 用真实 H3 E/D 在相同 prefix 缓存基线上核验：直接 near 能力、读出锚点、不同足迹是否被 native latent/decoder 保留。重复22帧只是 native VAE context，不算22条观测。
3. 重复 ID 与独立同足迹 noise 必须区别；遮挡只读、顺序置换、zero-state RGB 恒等式逐项审计。缓存不生成大的 decoder Jacobian；共享 A 的 matrix-free 观测/商空间工具或自己的小 separable camera matrix。

步骤1已完成中性 appearance/无遮挡的8val自然图像CPU审计：第4次 near 后 union ROI，last-near direct25.4103dB，atlas27.7765（+2.3661，8/8源），all-history inverse27.7532（+2.3429，8/8源）；全图分别25.0263/26.0950/26.1132。inverse ridge仅以4train预声明grid[.001,.01,.1]校准选.01，保留初始失配pilot；没有因此调整B2学习配置。详见 `results/h3_factorized/complementary_v3/probe_report.md`。这仅建立互补信息量，同时说明经典融合是必须击败的强控制，尚不是学习结果。

## 强控制与预算

共同使用当前 LR→H3 VAE anchor、相同历史支持/当前 null 与最终 DC，proposal 初轮全零；接入 H3 DiT 时同一固定 LR-only proposal 共享给所有控制，不把它当观测。

- 当前 LR→H3 VAE。
- last-near：只保留最近有效 near，含 B1 full-code 与 B2 delta-code 两种，避免弱选一个。
- 对齐 residual atlas 融合：footprint/visible 加权、noise precision 均值，重复 ID 去重。appearance 用共同 LR-only 光度拟合；不使用真实 q 作为有效控制输入。
- 全历史线性 inverse/MAP：相同 LR 几何与 noise、过去可见证据全保留；同位独立 noise 累计，发生改变时用相同 LR 残差变点规则；另报不允许检测改变的静态 inverse 失配。经典方法明确标注，测运行时间与完整 ledger 成本。
- 参数配平 ConvGRU：状态与 full 相同 24×7×16×16，输入同样的当前 anchor latent、实际 observed-detail delta、LR-only appearance、footprint/visibility/confidence，参数误差≤1%。不可只给候选当前 anchor 上下文。
- 四 chart 控制：full / no-swap / noninvertible / mixed，状态大小与训练 source-batch 顺序一致，预算仍 600+200、batch4、appearance10/swap0.5/RGB5。若 GRU 改动输入宽度，四模型共同调整宽度并重新记录实测参数，不比较旧 checkpoint。

## 最小候选实现，不预先声称有效

让观测编码器在 z_currentLR 上下文中解释 E(near)−E(LR-only far proxy)，因为 decoder 非线性使相同 latent delta 随 anchor 改变。chart 将其送入 appearance-neutral canonical 状态。write network 同时看到旧状态、当前 anchor 与 packet geometry，并输出 revision/confidence；共享的物理约束层处理当前可见 footprint，一致的旧证据由真实 decoder 的 past-LR loss 约束。只存固定状态与有界 footprint/confidence 元数据；若保留 RGB atlas 或 packet ledger，其全部字节必须计入，不把压缩收益凭空归给 latent。

重点消融是去掉 anchor 上下文、去掉反事实组合约束、取消可逆性、去掉 past-LR preservation、同输入 GRU。损失预算固定；可交换性约束采用两种顺序共享末端 observation 集的无冲突部分，冲突/变化不强制可交换。过去 LR loss 只采真实过去有效观测，不把生成或 decoder 输出当真实观测。

## 预注册判据与停止解释

主要指标先按 source 内平均再比较：第4/8/13/17步 union-support RGB/ROI PSNR、旧未变区域误差、新覆盖区恢复、重叠观测残差、遮挡错误写入、恢复时延、顺序差、完整状态字节与 wall time。HR 只评估/训练监督；decoded LR consistency 单列。DC 前后均报，oracle q/HR teacher 仅作独立容量诊断。

真正支持学习候选的证据需同时包括：超过 last-near、超过同预算 GRU、超过简单对齐 residual 融合，且与全 ledger inverse 的性能/压缩权衡有实质价值。只胜 current LR 或利用更多观测不足以建立新机制。若改善等价经典融合，应如此报告，再寻找非线性 appearance transport、复杂遮挡/几何或 full-H3 proposal 失配中的可测缺口，不靠改名。
