# 冻结H3：采样内观测一致性强控制

本实验是DDNM/FlowDPS启发的工程强基线，不是原创记忆方法、逐行论文复现或精确后验采样。全部预声明协议保持：未训练H3基础主干、相同官方LR guide/真实中性text/seed101/正常50点49Euler；仅在index28/45/48（σ.9/.516129/.2）修正post-CFG clean video estimate，强度1。没有扫描sigma/强度或使用test挑结果。

修正为 z0 + E(P25(D(z0))) − E(D(z0))；原Euler、audio clean estimate和guide保持。内部用float RGB、canonical FP16 H3VAE原生22帧、scale1。P25是既有AA bicubic+RGB box的固定Dykstra25，不保证与数据的Pillow量化完全等价；E/D不是互逆，像素投影结果不能冒充最终真实D输出。完整raw与最终DC25同时报告。

train bear_256原版/identity-hook完整49步video与audio latent最大差均0，identity delta严格0；实际P25在可行RGB固定点误差<1e−12。4个val case全49NFE、guide逐检查max0、hook未改audio；530基础参数张量版本不变、basegrad0。非零video修正可能经之后联合DiT影响audio轨迹，不宣称最终audio与base相同。

| 空间上下文 | 控制 | PSNR↑ | SSIM↑ | LPIPS↓ | LR RMSE↓ |
|---|---|---:|---:|---:|---:|
| 256 | Bicubic raw | 24.426 | 0.7177 | 0.41080 | 0.0164902 |
| 256 | Bicubic + final DC25 | 25.097 | 0.7503 | 0.35850 | 2.20662e-05 |
| 256 | H3 VAE raw | 23.618 | 0.6795 | 0.38574 | 0.0289113 |
| 256 | H3 VAE + final DC25 | 25.388 | 0.7591 | 0.31226 | 1.583e-05 |
| 256 | H3 BF16 base raw | 17.406 | 0.4853 | 0.43879 | 0.123078 |
| 256 | H3 BF16 base + final DC25 | 24.948 | 0.7358 | 0.29908 | 2.50358e-05 |
| 256 | H3 内置投影 raw | 22.517 | 0.6461 | 0.34385 | 0.0405417 |
| 256 | H3 内置投影 + final DC25 | 25.062 | 0.7396 | 0.26843 | 1.86956e-05 |
| native480 | Bicubic raw | 25.044 | 0.6268 | 0.49151 | 0.0145487 |
| native480 | Bicubic + final DC25 | 25.641 | 0.6675 | 0.44550 | 1.11912e-05 |
| native480 | H3 VAE raw | 24.536 | 0.6004 | 0.45408 | 0.0230354 |
| native480 | H3 VAE + final DC25 | 25.922 | 0.6777 | 0.39854 | 8.99592e-06 |
| native480 | H3 BF16 base raw | 23.194 | 0.5892 | 0.46072 | 0.0397326 |
| native480 | H3 BF16 base + final DC25 | 25.892 | 0.6759 | 0.39208 | 9.46265e-06 |
| native480 | H3 内置投影 raw | 24.889 | 0.6415 | 0.38038 | 0.0195967 |
| native480 | H3 内置投影 + final DC25 | 26.019 | 0.6829 | 0.35236 | 7.07512e-06 |

新运行共6条轨迹/294次DiT、27次E、15次采样内D、4次末端D。2条train轨迹只作严格latent/audio parity；不额外重复末端D。其余4条是val方法控制。v2的8个VAE/base控制case共176PNG逐SHA复用，额外NFE没有隐藏。成本含CPU投影和传输，详见cost_audit.json及各sample_result逐hook记录。
6条采样wall time合计168.552s，其中hook D 5.672s、hook E 20.908s、CPU projection 65.824s；cold load/总进程时间另见LIVE_STATE.json，不能相加重复或当实时吞吐。

全部12个输出case/264PNG本地逐SHA通过；来源拆分在sourcewise_metrics.json与evaluation/*/metrics.json。仅2个独立val源×2空间上下文，单seed/22帧；两空间上下文FOV不同，不是纯分辨率消融。实际观测end21、名义块内等待上界21帧，没有实测采集/稳态流式延迟。

经典依据：[DDNM](https://arxiv.org/abs/2212.00490)、[作者实现](https://github.com/wyhuai/DDNM)、[FlowDPS](https://arxiv.org/abs/2503.08136)。采样内projection/residual re-encoding是当前H3上的工程移植；它不更新历史memory，也不将生成结果写成新证据。

两个独立source的native480均改善：

- cows_native480：23.475dB / LPIPS 0.42253；相对同DC base ΔPSNR+0.145、ΔLPIPS-0.04704，相对同DC VAE +0.124/-0.05461。
- dance-twirl_native480：28.563dB / LPIPS 0.28218；相对同DC base ΔPSNR+0.108、ΔLPIPS-0.03240，相对同DC VAE +0.071/-0.03775。

256的PSNR仍低于VAE，且只有2source/单seed val，不能外推test或记忆收益。native480采样约70.3–70.6s，相对v2 base约30.6s约2.3倍wall time；同49NFE不代表同算力，因为额外VAE/CPU投影成本已实际发生。
