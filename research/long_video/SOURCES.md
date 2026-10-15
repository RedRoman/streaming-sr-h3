# 长片段数据扩展来源（2026-10-15）

当前DAVIS8源仅2–4秒，不能支持长间隔回访主张。下一轮准备LaSOT候选，先固定原片来源与真实遮挡/再现事件，数据选择不依赖模型输出。

- [LaSOT官方CVPR2019论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fan_LaSOT_A_High-Quality_Benchmark_for_Large-Scale_Single_Object_Tracking_CVPR_2019_paper.pdf)说明视频来自YouTube Creative Commons素材，1400序列平均2512帧，最短1000帧；论文以30fps换算时长。后续需单独记录每源帧率缺失，不把名义换算当实测源fps。
- [官方使用与下载页](https://vision.cs.stonybrook.edu/~lasot/download.html)允许个人、学术与教育使用；工具代码许可证不代替源视频权利。
- [官方工具仓库](https://github.com/HengLan/LaSOT_Evaluation_Toolkit)链接官方分发；其annotations可用于核对镜像数据元信息。
- 实际候选镜像：[l-lt/LaSOT](https://huggingface.co/datasets/l-lt/LaSOT/tree/a97464600e8c1ab91eb91a47332a20732830585e)，固定revision a97464600e8c1ab91eb91a47332a20732830585e。第三方镜像而非作者官方账号，逐range SHA、ZIP CRC与成员来源记录；目前没有下载全category来核对完整官方archive checksum。

初步候选为book、hat两类，目标是具有纹理的刚体/衣饰及真实遮挡再现。当前仅下载ZIP中央目录、遮挡/出视野/框标注、首帧预览；未将标注作为模型输入，未构造人工循环或伪长视频。是否适合作SR纹理任务，须检查真实图片内容与原分辨率后决定。新方法最终确认协议应在训练与模型选择前固定新来源split。
