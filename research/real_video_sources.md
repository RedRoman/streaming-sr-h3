# 真实自然运动视频小集

2026-10-15。**8条完整自然运动源视频、695帧、4×LR已获取并同步H100。** 本地逐帧像素/hash核验通过；远端3404个payload SHA256及原JPEG CRC核验通过，未使用GPU。权威入口为 `data/real_video/manifest.json`、`verification.json`、`remote_verification.json`。

## 已核验的官方来源

- [DAVIS官方下载页](https://davischallenge.org/davis2017/code.html)链接至[TrainVal 480p ZIP](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip)。本次仅HTTP Range取所选完整序列，不下载整套Full-HD/4K数据。
- 所选序列全属于DAVIS2016。[原始CVPR2016论文§3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.pdf)说明原序列为24fps、短片；实际帧数以ZIP中的原始JPEG成员逐一统计。
- 数据许可读取ZIP内 `DAVIS/README.md` 与 `DAVIS/SOURCES.md`：Credits措辞有Attribution4.0，明确Terms of Use为[CC BY-NC4.0](https://creativecommons.org/licenses/by-nc/4.0/)，本项目按后者记录。BSD是helper代码仓库的声明，不能替代数据档案中的许可。8条所选视频均不在SOURCES.md列出的外部素材例外清单。

## 源视频级划分

| 项目split | 源视频 | 完整帧数 | fps | 时长秒 | 官方split |
|---|---|---:|---:|---:|---|
| train | bear | 82 | 24 | 3.42 | 2016/2017 train |
| train | bus | 80 | 24 | 3.33 | 2016/2017 train |
| train | car-turn | 80 | 24 | 3.33 | 2016/2017 train |
| train | flamingo | 80 | 24 | 3.33 | 2016/2017 train |
| val | cows | 104 | 24 | 4.33 | 2016/2017 val |
| val | dance-twirl | 90 | 24 | 3.75 | 2016/2017 val |
| test | bmx-trees | 80 | 24 | 3.33 | 2016/2017 val |
| test | soapbox | 99 | 24 | 4.12 | 2016/2017 val |

共695帧。项目test是官方val的保留子集，不冒充DAVIS官方隐藏test；全部同源帧始终留在一个split。官方ImageSets/2016与2017 train/val列表已从同一ZIP提取，脚本逐个验证成员归属。DAVIS source-video ID作为源分组；不声称跨演员/拍摄场景绝对独立。

## 数据与操作记录

`source_jpeg`逐字节保留官方JPEG，ZIP CRC32及SHA256校验。HR是JPEG解码成RGB、仅右/下modcrop到4的倍数后无损PNG；不是超分或生成模型输出。4×LR使用当前Pillow的`Image.Resampling.BICUBIC`反混叠resize，精确版本写manifest；没有额外噪声，也不声称等同MATLAB bicubic。

原始帧全尺寸保留。额外 `h3_windows_39.json` 仅指定每源第一段39个真实连续帧、固定256×256整数对齐ROI；不会循环、ping-pong或拼接制造长回访。H3分块22帧、重叠5帧意味着最多21输入帧的块内等待，须与零等待协议区分。39帧长度是模型适配，不改变真实帧率。

DAVIS segmentation放在 `segmentation_gt_only`，仅作为评价/显式oracle条件，不能悄悄作为普通SR输入；若用到mask，需应用与HR相同的modcrop。当前小集没有真值光流/深度/几何。

获取器支持512KiB有界range、最多16个HTTP请求，并保留已下载前缀。每次range记录Content-Range和字节数。初次整序列range在实测每连接8–10KB/s下会超时，已停止仅本任务的父进程/4个curl子进程，保留部分字节转入分块；没有因观察工具超时而盲目重启或重下整包。

## 完成后的权威产物

- `data/real_video/manifest.json`：每源/每帧来源、split、尺寸、fps、时间戳、算子、许可证与hash。
- `data/real_video/frames.csv`：全部帧source/HR/LR路径及hash。
- `data/real_video/source_metadata/`：官方README、SOURCES和split原文。
- `data/real_video/verification.json`：独立逐帧重建与hash核验，源级split检查、磁盘预算。
- `data/real_video/contact_sheet.png`：每源首/中/末真实帧目视检查。
- `data/real_video/LIVE_STATE.json`、`network_ranges.json`：实际运行句柄和下载记录。

全量有效range约98.79MB，辅助探测约0.44MB；按每请求最大重试次数计算的保守payload上界188.85MB，低于3GB。含原JPEG、全部HR/LR、GT分割、原始索引和312帧H3裁剪后，本地约609MB，低于含临时文件2GB上限。恢复后的分块获取进程运行575.8秒；该数字不含前期探测/首次慢连接及最终同步，不能冒称总耗时。

H100统一根目录：`/home/saiadmin/streaming_sr_20261015/data/real_video/`。正式模型窗口入口`h3_windows_39.json`，8源各39帧、256×256GT与64×64LR，先crop再下采样以免边界halo算子不匹配。全量数据就绪标志为`remote_verification.json`的`passed=true`，并核对其manifest SHA256。当前manifest SHA256：`1b138247798c9965c897f22f19b5414f12c18d334699dcf7548e133f838ce3d7`。

复现：`python3 experiments/real_video_data/prepare_davis.py --workers 4`；完成后 `python3 experiments/real_video_data/verify_and_preview.py`。获取脚本依赖本次保存的官方ZIP索引尾部/前缀；它们同样保留在data目录，避免反复下载整包。

局限：真实自然运动成立，但4×退化为合成；480p JPEG并非RAW相机GT；8条短片不构成长回访benchmark，也不足以支撑大规模训练或泛化结论。REDS/Vid4尚未下载，本轮不将检索清单计为已获得数据。
