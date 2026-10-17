# 第 2 步协议：把配准后的记忆细节写进 FL2VA 低清关键帧通道

日期：2026-10-16。本文件在任何第 2 步模型输出产生之前写定；之后只追加，不修改已定义的条目。

## 背景与问题

第 0 步：冻结 H3 从低清生成的细节只有 GT 的 36%。第 1 步：冻结 Ref2VA 参考通路不能搬运细节。但 H3 的输出紧跟低清关键帧（guide）通道。本步要回答：把消失前的高清内容配准到当前帧、只把它的高频部分写进 guide，冻结 H3 能否把这些细节保留到输出里，并且比直接用经典方法贴回去更好。

## 数据与块

与第 1 步相同：32 个 LaSOT 回访事件，只处理回来后的块 kpost（22 帧），评价帧为新输出帧中目标可见且框短边至少 16 像素的帧。参考帧 f_ref 与第 1 步相同（pairs_revisit_a 的中间帧，在消失之前）。

所有变体都使用 GT 目标框做定位（目标位置是 oracle），这一点对部署是乐观的，在报告中写明。

## 记忆写入（每个 kpost 帧 t，目标可见时）

1. 区域：ref 帧以 GT 框外扩 10% 裁剪，缩放到 t 帧同样外扩区域的大小。
2. 光流：在两区域的灰度图上算 OpenCV DIS（MEDIUM），t→ref 与 ref→t 两个方向，前后向一致性误差小于 1 像素的位置保留。
   - oflow：在 GT 灰度图上算（光流是 oracle）；
   - lrflow：在 LR 经 bicubic 放大后的灰度图上算（可部署）。
3. 用光流把 ref 内容扭到 t 的区域，放进一张以 U(LR_t) 为底的整帧图 W。
4. 门控（只用 LR，可部署）：e = 高斯模糊(σ=2) 的 mean_c |U(D(W)) − U(LR_t)|，保留 e < 0.04 的位置。
5. 掩码 m = 一致性通过 ∧ 在 t 的目标框内 ∧ 门控通过；软掩码 s = 高斯模糊(m, σ=2)。
6. guide 像素：G_t = clip(U(LR_t) + s · N(W), 0, 1)，N(x) = x − U(D(x))，D 为抗混叠 bicubic ×4，U 为 bicubic。目标不可见的帧 G_t = U(LR_t)。
7. 之后与原流程相同：官方 _resize、VAE 编码成 guide latent，FL2VA 49 步 + innerDC（seed 101、CFG 1、修正点 28/45/48、同事缓存的中性文本条件）。

## 变体

H3 变体（每个 32 块）：
- `none_rerun`：guide 不写记忆，走本步新代码重跑；用于检查与第 0 步输出是否一致，并量化噪声底。主比较的基线。
- `mem_gt_oflow`：内容为 f_ref 的 GT，oracle 光流。上界。
- `mem_gt_lrflow`：内容为 GT，LR 光流。
- `mem_self_lrflow`：内容为第 0 步 innerDC 在 f_ref 的输出，LR 光流。除目标框外完全可部署。
- `mem_wrong_oflow`：内容为下一个事件（循环）在其 f_ref 的 GT 目标区域，使用与 mem_gt_oflow 完全相同的光流与掩码（不重算门控）。内容无关、能量与位置相同的对照。

经典对照（不用 H3，CPU）：`classic_*` = 对同一 guide 像素 G 做与 innerDC 相同的 25 次 Dykstra 投影（浮点 bicubic ×4 观测 + [0,1]），对应以上每个 guide；`classic_none` 即 bicubic + 投影。

另列 `fl2va`：第 0 步 innerDC 在 kpost 的输出，不重跑。

## 指标

与第 1 步相同，在目标框内计算 PSNR、LPIPS、生成频带能量比、与 GT 余弦，另算整帧 PSNR 与 LR RMSE。另在 `mem_gt_oflow` 的硬掩码 m 内计算 PSNR（所有条件用同一掩码，掩码为空的帧跳过）。先帧内平均再事件等权平均；配对差给 bootstrap 95% 置信区间（10000 次，种子 0）和改善事件数。报告各变体每帧掩码在目标框内的覆盖率。

## 判读（预先写定）

1. 噪声底：`none_rerun` 与 `fl2va` 的事件级 PSNR 差的绝对值分布；若均值超过 0.05 dB，所有结论需要和该量级比较并在报告里写明。
2. 主比较 `mem_gt_oflow − none_rerun`：目标框 PSNR 提高至少 0.3 dB 或 LPIPS 降低至少 0.02，且置信区间不跨 0，判为"guide 通道能承载记忆细节"（上界为正）；否则判为该通道在冻结 H3 下也不能承载记忆，停止"冻结 H3 + 记忆"路线。
3. 内容检查：`mem_gt_oflow − mem_wrong_oflow` 的 PSNR 区间必须在 0 以上；否则收益不是来自正确内容。
4. H3 是否有价值：比较 `mem_gt_oflow` 与 `classic_gt_oflow`。若 H3 不优于经典贴回（PSNR 与 LPIPS 都不更好），写明"H3 在记忆利用上没有超过经典方法"，这直接影响新颖性。
5. 可部署性：`mem_gt_lrflow` 和 `mem_self_lrflow` 相对 `none_rerun`，阈值同第 2 条，只做报告，不改变第 2 条的结论。

## 冒烟测试

全量前先用 bicycle-1、bottle-14 跑 `none_rerun` 和 `mem_gt_oflow`，只检查流程、输出是否正常、耗时；不据此修改任何参数（包括门控阈值 0.04 和一致性阈值 1 像素）。冒烟输出保留并计入全量。

## 已知局限

- 目标框来自 GT 标注（定位 oracle）。
- 只取一帧参考；长时间缺失后姿态变化大时，掩码会很小，记忆不起作用，这是方法的真实情况，按事件报告覆盖率。
- 合成 bicubic 退化，不含压缩。

## 追加（20:55，任何 H3 输出和结果指标产生之前）

冒烟事件只构建了 guide（CPU），没有跑 H3，也没有算任何结果指标。发现：
- bottle-14 的 oracle 光流前后向一致性通过率 90%，但最终记忆掩码只覆盖目标框的 14%。原因是 LR 门控：配准后参考的低频与 U(LR_t) 的误差中位数为 0.06–0.12，超过阈值 0.04。
- 门控实现已核对：把当前帧自己的 GT 当作 W 时，误差中位数约 0.002，100% 通过。所以门控没有 bug，它挡掉的是参考与当前帧之间真实的低频差异（光照、颜色或配准偏差）。
- bicycle-1 的回来前后尺度相差 8 倍，一致性通过率只有 7%，覆盖率几乎为 0，这是该事件的真实情况。

修订（不改原有任何变体和阈值）：
- 增加两个次要变体 `mem_gt_oflow_nogate`、`mem_wrong_oflow_nogate`：掩码只用光流一致性 ∧ 目标框，不用 LR 门控；wrong 版本使用与 gt nogate 完全相同的光流和掩码。对应经典对照 `classic_gt_oflow_nogate`、`classic_wrong_oflow_nogate`。理由：写入的只是高频 N(W)，低频始终来自 LR，且 innerDC 会把输出投影回 LR 观测，所以低频差异不一定意味着高频细节错误。
- 判读第 2 条修改为：门控版和无门控版的上界只要有一个满足阈值且对应的内容检查（gt − wrong）区间在 0 以上，就判为"guide 通道能承载记忆细节"，并写明是哪一个；两个都不满足才停止该路线。
- 运行顺序按变体优先级：none_rerun、mem_gt_oflow_nogate、mem_wrong_oflow_nogate、mem_gt_oflow、mem_wrong_oflow、mem_gt_lrflow、mem_self_lrflow。
