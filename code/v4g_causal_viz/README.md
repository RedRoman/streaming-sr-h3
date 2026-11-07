# v4g 因果流式可视化：可复现源码包

本目录对应 2026-11-07 的 v2 可视化管线。发布副本只把执行源码中的机器私有路径改为环境变量或目录内相对路径，并要求启动时明确指定物理 NPU 和已有共享锁。模型计算、teacher 恢复、24 通道误差图、缓存校验和验收逻辑保持原样。`sources.json` 分别记录执行源码与公开副本的 SHA256；两者不同不能混称为同一文件。

输入固定为 `VIDGEN5S_0086441`，37 latent 帧解码为 124 帧后保留前 120 帧，24 fps、5 秒。本包不证明 226 帧真实视频端到端推理。结果与媒体验收见仓库的对应日期结果目录；CPU smoke 仅验证管线结构。

## 运行必需品

自行准备以下本地文件；本包不包含视频数据集、模型权重、optimizer/RNG 或推理缓存。用 `run_contract.json` 中实际运行记录的文件 SHA256/大小核对输入、原 teacher 和 v4g checkpoint。

- `inputs/VIDGEN5S_0086441.mp4`：实际输入视频。缺少同一视频不能声称复现同一结果；仅更改文件名不能建立样本身份。
- `assets/v4g_final.pt`：v4g checkpoint，须含 `model_state_dict`。
- `assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`：原始 teacher 权重。
- `assets/h3-base/vae/`：H3 VAE 配置、权重和索引，供 `AutoencoderKLMiniMaxH3.from_pretrained(..., subfolder="vae")` 本地加载。

`H3_VIZ_SOURCE`、`H3_VIZ_CHECKPOINT`、`H3_VIZ_UPSCALER`、`H3_VIZ_BASE_MODEL` 可指向已有文件；不会自动下载。VAE 身份只记录配置 SHA256、权重文件大小/mtime，不包含所有 VAE 权重内容的 SHA256。大小与 mtime 不是权重内容完整性证明。

## 环境

`environment.json` 记录实际环境中已核对的版本：Python 3.12.0/aarch64、PyTorch 2.7.1、torch-npu 2.7.1.post2 等。使用匹配的 Ascend 驱动、CANN runtime 与 OPP；FFmpeg 需提供 `fps` 过滤器统计和可用视频编码器。

实际导入的 diffusers 来自 `https://github.com/huggingface/diffusers.git` 的 `19b759a898647cc6de60a2093b2fc86ffebabe36`，由 `PYTHONPATH` 优先选中。环境的 distribution metadata 虽显示 `0.33.1`，不能据此使用 stock 0.33.1 替代。该 checkout 的唯一已报告修改是把依赖版本表的 `peft>=0.17.0` 放宽为 `peft>=0.14.0`。可在准备好的指定 revision 上应用记录补丁：

```bash
python3 patches/apply_diffusers_peft_metadata_patch.py /path/to/pinned-diffusers
```

补丁脚本先核对 git HEAD，且只替换这一处元数据。设置 `H3_VIZ_PYTHONPATH` 包含该 checkout 的 `src` 和所需 CANN Python 路径。依赖版本表放宽不等于验证任意 peft 版本；完整 Python 依赖闭包、驱动、FFmpeg build 未全部固定，故不承诺跨环境逐位一致。

## 启动与检查

在本目录执行：

```bash
cp config.example.env config.local.env
# 编辑 config.local.env：填入已有资源、runtime、获准使用的物理 NPU 和共享锁。
source config.local.env
bash runs/launch_viz_v2.sh v2_reproduction_01 "$H3_VIZ_PHYSICAL_NPU" --check
bash runs/launch_viz_v2.sh v2_reproduction_01 "$H3_VIZ_PHYSICAL_NPU"
```

`config.example.env` 是路径配置模板；`run_contract.json` 仅描述本次固定实验配置，脚本不会把它作为运行参数读取。调用 launcher 时物理 NPU 参数必填，无默认卡。`--check` 不启动模型；正式启动仍重新检查。

三个 `H3_VIZ_*_LOCK` 必须是该主机、该物理 NPU 与已有任务共同遵守的设备锁、任务锁、VAE 锁的真实路径。不要填写独立的新锁来绕开协作协议。launcher 同时持有三把非阻塞 flock，要求设备查询成功且无任何已有进程；锁被占用、设备有进程或查询不能确认时以 75 退出。持锁延续至后台 worker 结束。运行目录/tag 已存在则拒绝覆盖；重新尝试使用新 tag。

输出写入 `$H3_VIZ_ROOT/runs/`：独立 log、exit JSON、包含 tag 的产物目录。`H3_VIZ_ROOT` 默认是本包目录；启动脚本始终使用本包的 Python 代码。示例会把本地 assets、inputs、缓存、输出及 `config.local.env` 排除在 Git 之外。

必须检查 exit JSON 的真实退出码，并完整解码产出视频，核对帧数、尺寸、fps、时长及 SHA256。再核对 `stats.json`、teacher 指纹和 manifest 的五路解码/assembly 状态。仅存在 MP4 文件或 CPU smoke 成功不能作为真实 NPU 验收。

## 管线与证据边界

- 源视频经 Lanczos 放大/裁切至 1920×1088，LR 是这些帧的 area 缩小版本（1280×704）。这是本次可视化的数据处理；不能沿用旧方法文档的“天然生成 LR”描述。reference 是 HR 经 VAE 编码解码后的参考，不是超分辨率真值；不据此编造 GT 指标。
- 五路分别为 v4g streaming、v4g full、恢复原权重的 teacher、未来 latent 加噪、reference。teacher 在加载 v4g 前保存原状态，恢复后进行参数指纹核验。
- 24 通道误差图先变为显示图；37→120 的时间轴重采样仅供显示。没有验证 VAE 的逐点像素时间映射，因此没有像素级因果切点线。
- `--from-cache` 校验输入/权重身份、形状和校验和，并写入新 attempt。完整 RGB 缓存可不初始化模型重绘；部分缓存仍需 VAE。launcher 尚未暴露该选项，需在同样完整的设备锁协议下使用，不要绕开 launcher 抢占设备。
- 缓存 manifest 包含本地绝对路径和 mtime，迁移路径会导致身份不匹配。公开脱敏 manifest 供核验来源，不能直接替代私有缓存 manifest 做 resume。新生成日志/manifest 也应在公开前脱敏。
- 历史 `audit_v4g_full.py` 读取的是 `upscaler_prod_pairs/holdout.jsonl` 中 `VIDGEN5S_0380588`，不是本视频的 0086441。其 67 latent 输入由 37 帧 draft 拼接帧 1:31 的反转尾部构成，用于更长的 latent 等价性检查；不是 226 帧真实视频验证。历史 audit 的随机扰动没有完整固定 RNG 状态，不能承诺精确数值重现。
- 历史审计 E1 虽把指标命名为 `prefix_37_of_67_vs_37_max_abs`，代码对 `long_draft` 和短 `draft` 都只循环前 37 帧；没有执行完整 67 帧流式过程后再截取前 37 帧。该项实际比较的是相同前缀的两次 37 帧处理，不能据其名称宣称完成了完整 67 帧后的前缀检验。D 项的 67 帧 streaming/full 比较和 E2 的两次完整 67 帧重跑应单独解读。
- 历史审计 F 的 `passed` 硬编码为 `True`，而顶层 `all_passed` 只聚合各项 `passed`。本次原始记录中 `reset_clears_cache`、`signature_mismatch_raises`、`frames_zero_after_reset` 三个观察值确实均为 `True`；但 `all_passed` 本身不足以独立证明这些状态机条件，必须逐项查看实际观察值。本包保留原执行源码，没有把修正后的 gate 伪装成历史执行结果。
- 这些因果性与流式等价性检查针对 latent upscaler。它们不等于整个 VAE 编码—上采样—解码流程具备在线因果性，也不证明端到端实时性、延迟或吞吐；相应结论需要独立的 VAE 时间依赖与性能实测。
- `audit_v4g_full.py` 与 pair helper 为历史证据/复现依赖保留；它们不包含此机器的训练数据或自动分配设备能力，不应直接在未协调的 NPU 上运行。

## 本地结构检查

```bash
python3 experiment/supervisor_smoke_viz_v2.py experiment/viz_v4g_causal_stream_v2.py
bash -n runs/launch_viz_v2.sh
```

smoke 使用小尺寸合成数据、假的 decoder/video writer，不加载真实模型、不证明画质或真实 NPU 性能。实际执行版本已经通过该 smoke；公开副本的验证记录见 `validation.json`，不要混同两个阶段。

## 上游来源

`upstream/open_h3/` 是此次执行的实际文件副本，文件 SHA 见 `sources.json`。node 自述来源为 `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`；实际执行 node 对应的 upstream commit 未核实，不能把当前 main 当作执行版本。`upstream/LICENSE` 原样保留从该项目当前 main 获取的 MIT 许可证（包括原文件字面格式），来源：

<https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/LICENSE>
