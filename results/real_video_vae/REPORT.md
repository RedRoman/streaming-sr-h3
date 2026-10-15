# 8条真实视频的H3 VAE先验与相同观测一致性对照

日期：2026-10-15。16个固定case全部实际执行：8个独立DAVIS源视频，各取前22帧，256中心ROI与原生832×480视野两套。主干是冻结H3原生FP16 VAE，没有DiT，没有新记忆训练。每种分辨率的统计单位为8条源视频；不能把两套视野或352帧当独立样本。

## 协议与可核查性

HR仅用于预先构造低清数据与评价。原生480p按固定(8,0,832,480)中心mod32 crop，Pillow bicubic得到208×120 LR；256使用原固定ROI。推理阶段只读取LR文件，浮点Torch antialias bicubic上采样后调用官方VAE wrapper。该路径未使用AddGuide的lanczos/uint8预处理，不能与另一guide设置混为完全相同条件。

输入manifest记录逐LR SHA；352张实际输出从H100传回后逐SHA核对通过。权重完整SHA为7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522。代码/manifest SHA、前台session和上下文见execution_provenance.json，各case inference.json保留实际时间、latent形状和观测截止帧21。

对所有方法施加同一经典25步Dykstra输出投影，算子为浮点Torch AA bicubic且RGB约束[0,1]。两种基础读出都先按相同8bit精度存储；DC后浮点评价。投影无HR拟合，也不被写作真实证据。实际Pillow降采样残差另行报告，不声称浮点与整型算子严格一致。矩形算子独立核对最大误差3.33e-16。

## 按源平均结果

|视野|方法|PSNR↑|SSIM↑|LPIPS-Alex↓|实际Pillow LR RMSE↓|
|---|---|---:|---:|---:|---:|
|256|bicubic|25.222|0.7009|0.4277|0.014242|
|256|h3_vae|24.681|0.6708|0.4080|0.023168|
|256|bicubic_dc|25.833|0.7292|0.3796|0.000197|
|256|h3_vae_dc|26.071|0.7361|0.3416|0.000177|
|native480|bicubic|26.343|0.6997|0.4607|0.012474|
|native480|h3_vae|25.854|0.6758|0.4327|0.019322|
|native480|bicubic_dc|26.930|0.7291|0.4167|0.000149|
|native480|h3_vae_dc|27.159|0.7356|0.3732|0.000136|

|视野|VAE+DC相较bicubic+DC PSNR变化|LPIPS变化|两项源胜率|
|---|---:|---:|---|
|256|+0.238dB|-0.0380|8/8、8/8|
|native480|+0.229dB|-0.0434|8/8、8/8|

8源包含4train/2val/2project-test。本实验没有学习或按test选参数；aggregate.json分别给出三个split，不把原始测试集名称改成训练集。两种视野内容不同，不能将均值差直接解释为分辨率因果效应。源级描述性bootstrap保存于aggregate.json；小样本且全为同一数据集，不作跨域普遍结论。

## 当前意义

一致性修复后，冻结H3 VAE的先验在8条视频、两套视野中均改善PSNR与LPIPS，相对相同DC的bicubic仍有收益。这给H3路线提供可重复的积极基础，同时也是新记忆方法必须超越的强控制。收益并未证明持续身份、长回访、DiT生成或新状态结构的贡献。原VAE裸输出的PSNR通常更低，说明读出必须尊重真实LR约束。

480p热身后encode约1.54秒、decode约0.35秒；256约0.13/0.04秒，首次有额外初始化。峰值allocated约7.08GB。这里只统计E/D GPU阶段，不含模型加载、I/O、输出显示或21帧块等待，不称端到端实时。

后续保持这些固定强控制，检验新的锚定残差状态是否能利用近景/重现信息产生额外增益。测试过的源应明确作为当前探索证据；稳定方法确认需要预先冻结协议和未用于结构选择的新来源。

![原生480p：bmx-trees相同DC对照](/Users/hezihao/Documents/New project/streaming_sr_20261015/results/real_video_vae/bmx-trees_native480/comparison_projected.png)
