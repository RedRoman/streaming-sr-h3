# H3真实运动读出与观测一致性对照

日期：2026-10-15。单条 DAVIS/bmx-trees 的前22帧，256×256 HR / 64×64 LR。真实运动，合成4×退化；只用这一段不能推断跨视频稳定性。

## 固定执行协议

H3使用官方pruned BF16 FL2VA主干、FP16 VAE、固定中性prompt及已缓存的NVFP4-AWQ文本编码。正常50-sigma / 49 Euler forward全部finite，采样41.72秒，含加载卸载64.83秒、峰值allocated7.67GB。输入仅第0–21帧LR，逐文件SHA与实际编码记录吻合。块内等待上界21帧，未测真实实时采集或端到端吞吐。旧manual guide与官方AddGuide的uint8预处理存在最多1/255差异，另有官方接口诊断，不能声称两者逐像素等价。

DC为经典输出空间投影：floating antialiased bicubic线性等式与[0,1] RGB box的25步Dykstra。没有HR拟合，三个基础方法使用同一投影。每个22帧方法约0.27秒CPU；实际float算子最大逐帧LR RMSE约1.3e-4，有限步数不声称精确收敛。数据实际Pillow整型算子的误差另列。

## 结果

|方法|RGB PSNR↑|RGB SSIM↑|LPIPS-Alex↓|Pillow LR RMSE↓|时序误差增量↓|
|---|---:|---:|---:|---:|---:|
|bicubic|27.232|0.8363|0.2821|0.013109|0.03438|
|h3_vae_only|25.406|0.7639|0.2808|0.031301|0.04260|
|h3_bf16|22.011|0.6714|0.3195|0.063411|0.06436|
|bicubic_dc|28.104|0.8565|0.2304|0.000314|0.03464|
|h3_vae_only_dc|28.229|0.8511|0.1926|0.000311|0.03453|
|h3_bf16_dc|27.938|0.8414|0.1944|0.000332|0.03577|

LPIPS按官方[PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) v0.1 AlexNet，输入[-1,1]，网络与校准权重SHA存入summary.json。SSIM为11×11、sigma1.5、valid边界、RGB平均。时序量以LR-only DIS对应，在置信度≥0.5区域计算输出运动残差减HR运动残差，避免把正确运动本身算作闪烁；平均覆盖49.7%。此量仍依赖估计对应，不等同人眼闪烁分。

## 判断与后续

原始H3输出偏离LR观察；一致性投影可以大幅修复这一点。相同DC下，H3 BF16的LPIPS优于bicubic，但VAE-only也取得相当或略好的PSNR/LPIPS。当前没有证明49步DiT生成或持续记忆的额外优势，不能把经典DC收益计为新方法贡献。

积极方向是保持真实LR决定的结构，让H3先验及可学习历史状态负责欠观测细节。下一步扩大源视频和原生空间分辨率，并对所有候选保留VAE-only、bicubic及相同DC控制；以真实decoder、同前缀和同guide预算验证独立历史信息收益。

![相同DC下的公平对照](comparison_projected.png)

comparison.png另展示raw输出与单个H3修复，不适合当作所有方法等处理的比较；comparison_projected.png全部控制均使用DC。所有逐帧原始指标、量化PNG和算子审计已保存。
