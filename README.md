# Method-2：因果 latent upsampler 自蒸馏（独立方案，进行中）

## 最新阶段：2026-11-08 真实 226 帧 / 67-latent 全长评价与视频交付

[完整进展与验收说明](results/2026-11-08/v4g-full226/README.md) · [分析：单就 upscaler 本身，是否已是正式管线中已确定的可交付流式因果 upsampler](results/2026-11-08/v4g-full226/ANALYSIS_UPSCALER_STATUS.md) · [视频 Release](https://github.com/RedRoman/streaming-sr-h3/releases/tag/v4g-full226-20261108) · [脱敏 manifest](results/2026-11-08/v4g-full226/manifest.public.json)

本轮补上了 v2 缺的真实生产长度：不再用 37 帧真实 + 反转 30 帧的合成 67-latent，而是从真实 345 帧源取连续 226 帧，编码为 67 latent，六门因果审计在真实全长序列上重跑全部通过（`all_passed = true`，阈值 0.01）；五路信号各整段 VAE 解码 226 帧不裁剪；六面板视频 226 帧、24 fps、9.4167 秒，服务器 MPEG-4 与本地 H.264 副本各自全帧解码 226 帧、exit 0。原始 teacher 权重指纹恢复已核验。

结论分两层：**单就 upscaler 模块本身**，它已是不偷看未来、状态有界、可逐 chunk 流式调用且与整段前向一致的因果上采样算子，机制这一层已确定；**但「正式管线中已确定的可交付组件」仍不是**，因为还没接入自强迫 DMD 生成链做在线逐段联调、没有留出效用评测证明相对原始双向 teacher 的收益、也没有接口与时间 / 延迟 / 显存契约。见上面的分析文档。

边界：整段 VAE 编解码是离线批处理，不是在线流式 VAE 链路，也不代表实时性能；LR 是 HR 合成下采样退化，没有真实 GT，不给 PSNR / SSIM；teacher 不是 GT；未做人工实际检视，不声称看过视频。A/B 训练双臂未启动，冻结审阅包未改动，v2 的 120 帧视频已发布在 `1f513c6`，本轮不重复生成或发布该内容。

以下保留 2026-11-07 的 v2 可视化阶段记录，以及 2026-11-05 的 Method-2 阶段记录；其中「进行中」表示当时状态。

## 2026-11-07 阶段：v4g 因果可视化（已发布 v2）

[修正后的 v2 对比视频、指标与验收说明](results/2026-11-07/v4g-causal-viz-v2/README.md)：120 帧、24 fps、5 秒，teacher 权重已恢复核验，视频完整解码通过。37 latent 可视化与历史合成 67 latent 审计分别标注，尚无真实 226 帧端到端视频验收。本轮演示从 HR 合成 LR，与下方旧方案的天然 LR 配对设定不同。

[视频 Release](https://github.com/RedRoman/streaming-sr-h3/releases/tag/v4g-causal-viz-20261107-v2) · [复现代码与环境](code/v4g_causal_viz/README.md)

**日期**：2026-11-05 ｜ **硬件**：昇腾 910B3 ｜ **状态**：v1 完成并评测（负结果已保留），v2 训练中

独立于对齐-过去权重改造（用户的 method-1）的平行方案：从头训一个因果 student，
蒸馏冻结的 Open-H3 双向 latent upsampler（teacher）的输出，不需要 GT 配对。

## 方案
- Teacher：`minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`（fp32 冻结），输入 draft latent (24,37,12,21) → 输出 (24,37,24,42)。
- Student：CausalStudentUpscaler ~15M 参数。bilinear 残差基线 + head conv→PixelShuffle(2) + 6×CausalMemBlock（拼接过去帧记忆，逐帧因果）+ zero-init 输出。支持 parallel / sequential(带缓存) 双前向，一致性校验。
- 训练对：真实管线配对（draft = 低清生成视频的 H3 latent；reference = 同内容高清生成视频的 H3 latent，仅用于评测）。**无 bilinear downsample 合成**——draft 天然就是真实 LR。
- UltraFlash 的 bilinear-downsample 造对方法不适用/不需要：我们的 LR 是生成器真输出，非合成退化；差异在于真实配对稀缺 vs 合成对可无限生成。

## v1（纯 latent MSE + 0.1 temporal）——负结果
- 3000 步，~0.45s/步，一致性 4.8e-7 通过（因果性无泄漏）。
- 训练 clip 蒸馏保真极好（latent_mse 0.0042），但 holdout 上畸变明显：PSNR 22.24 dB vs teacher 30.11 dB、用户 method-1 25.31 dB；val latent_mse 训练中 0.251→0.280 恶化 → 2-clip 过拟合 + 无感知约束。
- 结论：畸变主因怀疑是 loss 缺像素/感知约束。

## v2（进行中）——单变量改动：仅换 loss
- 引入 songhua 跑法（h3_frozen_mapper672_x2_flashvaed_20261021）的 decode 后像素合同，adapted 到 H3 latent 空间（他的权重不可直接复用：mapper 输出 96ch Wan 空间 → FlashVAED；我们是 24ch H3 空间 → H3 VAE）。
- Loss = latent MSE + 0.1 temporal + 1.0 pixel L1 + 1.0 LPIPS(alex) + 0.33 wavelet(LH/HL/HH)，像素项在冻结 H3 VAE decode 输出上计算、梯度穿过 decoder 只更新 student。每步随机 5-latent-帧窗口（解码 17 帧）控制成本，~4.5s/步。
- songhua 完整合同中的 eWarp(RAFT×0.1) 与 IQA reward 刻意推迟到下一轮，保持单变量归因。
- 进度日志：`logs/student_pixel_v2_20261105.log`（step 460 时 LPIPS 0.36→0.13，持续下降）。

## 目录
- `code/train_student_distill.py` — v1 训练（架构、双前向一致性、验证与 summary）
- `code/train_student_pixel_v2.py` — v2 训练（v1 + 像素合同）
- `code/eval_student_5s.py` — 5s 全量评测：teacher/student/reference latent、vs-reference 与 vs-teacher 指标、因果性校验（future_cut / prefix_cut / 逐帧流式一致性）、解码 mp4
- `results/train_summary.json` — v1 训练总结
- `results/eval/summary_v1_student_holdout.json` — v1 holdout 评测
- `results/videos/` — 3 个训练 clip 的 teacher/student/reference 对比 mp4（v1）
- `logs/` — 训练与评测原始日志

## 已知环境要点（历史运行）
- 需要将 `ASCEND_OPP_PATH` 设置为有效的 CANN 8.5.2 OPP 目录；原运行环境使用单独安装的 OPP。
- vendored diffusers：`<VENDORED_DIFFUSERS>/src`（AutoencoderKLMiniMaxH3），其 dependency_versions_table.py 的 peft 已补丁 0.17→0.14。
- H3 VAE decode 必须先 `vae.disable_tiling()`（T<5 块触发 torch.cat 空列表 bug）；T=5 窗口 decode+fwd ≈1s。
- LPIPS 需 per-image (N,3,H,W)；DWTForward 不接受 5D。
