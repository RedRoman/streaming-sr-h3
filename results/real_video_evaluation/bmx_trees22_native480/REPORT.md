# 原生832×480的H3真实运动基线

实际H3 BF16 50层基础主干，正常50-sigma/49 Euler forward，22帧、208×120真实观测LR。全部forward finite，采样88.54秒/该阶段含load112.31秒、峰值allocated8.23GB。阶段跨H100与5090，不能相加称实测在线延迟。输入/输出和guide control的逐文件SHA归档于h3_setup对应guide_input_manifest.json。

主控制VAE-only与DiT使用同一官方guide量化、同一5090 decoder后端；它与real_video_vae的float-upsample版本分开。固定text-only中性Qwen条件，不含GT语义；官方I2V首帧视觉条件属于后续单独协议诊断。

|方法|PSNR↑|SSIM↑|LPIPS↓|Pillow LR RMSE↓|时序误差增量↓|
|---|---:|---:|---:|---:|---:|
|bicubic|31.335|0.8711|0.2565|0.008341|0.01774|
|h3_vae_only|29.312|0.8156|0.2567|0.020473|0.02432|
|h3_bf16|27.796|0.7954|0.2710|0.029429|0.03091|
|bicubic_dc|32.123|0.8851|0.2118|0.000142|0.01934|
|h3_vae_only_dc|32.150|0.8780|0.1729|0.000138|0.01972|
|h3_bf16_dc|32.071|0.8755|0.1699|0.000135|0.02015|

相同经典DC后，DiT的LPIPS略好于VAE-only（0.16994 vs0.17292），PSNR略低（32.071 vs32.150）。单片段结果不足以证明独立主干增益，且49步代价高；保留用于后续同预算guide/adapter训练的强控制。两个H3版本均比同DC bicubic感知指标更好，不能由此声称新记忆有效。

新的208×120 LR-only双向DIS对应独立计算，并逐输入SHA核对；可信区域覆盖90.0%。时间量扣除HR运动/外观残差，不能与256不同视野的覆盖或误差直接排序。两套空间视野和内容不同，不作为纯resolution ablation。

![相同DC下的实际输出](comparison_projected.png)
