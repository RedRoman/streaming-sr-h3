# 真实运动的 LR-only 对应接口与控制实验

2026-10-15。8条独立DAVIS源视频，每条前39帧固定256×256 crop、64×64 LR；当前只评价经典运动接口与非生成式控制，未执行H3或学习SR。

使用OpenCV 5.0 DIS medium，两个方向均只从已到达LR计算；forward/backward一致性与颜色残差合成置信。每17帧直接匹配初始anchor尝试恢复chain漂移。该分数不是对象身份概率。参数在运行heldout前固定，没有以test选择参数。source代码hash与版本在config.json。

| 源视频 | bicubic | 时序融合 | 固定参考 | 相邻可信覆盖 | canonical可信覆盖 | HR warp RMSE可信 / 不可信 |
|---|---:|---:|---:|---:|---:|---:|
| test/bmx-trees | 27.425 | 27.251 | 27.408 | 52.5% | 5.1% | 0.0400 / 0.2683 |
| test/soapbox | 28.465 | 28.378 | 28.348 | 91.2% | 42.4% | 0.0259 / 0.3645 |
| train/bear | 23.934 | 23.864 | 23.863 | 95.6% | 22.9% | 0.0426 / 0.1943 |
| train/bus | 22.934 | 22.931 | 22.910 | 87.6% | 33.0% | 0.0357 / 0.2430 |
| train/car-turn | 26.549 | 26.533 | 26.411 | 94.6% | 64.5% | 0.0221 / 0.1918 |
| train/flamingo | 25.179 | 25.028 | 25.136 | 85.5% | 10.9% | 0.0503 / 0.1885 |
| val/cows | 22.860 | 22.828 | 22.791 | 91.0% | 25.7% | 0.0459 / 0.2504 |
| val/dance-twirl | 25.781 | 25.606 | 25.733 | 66.6% | 9.5% | 0.0469 / 0.2693 |

PSNR单位dB，按每源逐帧平均。时序融合为固定0.5×匹配置信的旧估计加权；固定参考为anchor图像warp加当前LR填充。它们全程只接收LR。HR warp诊断在输出完成后才读取HR，包含真实光照/遮挡变化，因此不是纯光流endpoint error。

结果：相邻匹配置信能筛选较低HR投影误差的区域；但单anchor链的覆盖明显衰减，特别是快速运动/新内容进入画面。简单颜色融合在全部8条视频均略低于bicubic，不能通过时间平滑替代细节恢复。正向用途是将匹配可信区域作为H3条件记忆的写入/读出边界，低可信区域回退当前LR；进一步比较多anchor、学习重定位与无对应控制。

独立正确性审计：已知3,-2像素平移的endpoint error=0.02484 LR pixel；identity warp误差0；独立重建相同前缀map差0；重复UID不变；只读warping不改变state。图像和逐帧结果已实际检查。

接口：lr_correspondence.npz中的current_to_previous/current_to_canonical为当前LR网格→参考LR绝对像素中心坐标，shape [39,H,W,2]；pair_confidence/canonical_confidence为[39,H,W]。估计器只读取每个索引之前的LR；H3后续分块等待需另计，不能借此声称H3零等待。

局限：8条2–4秒原clip、这里39帧，不覆盖长间隔身份回访；DIS是经典对照，不是原创核心；当前RGB融合不生成新细节。

官方API：[OpenCV DISOpticalFlow](https://docs.opencv.org/5.0/main_modules/classcv_1_1DISOpticalFlow.html)。
