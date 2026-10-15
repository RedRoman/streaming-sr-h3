# H3 采样内观测一致性：最小外部强控制建议

这是冻结H3的经典逆问题对照，不引入历史记忆、不新增学习模块；当前仅建议，未启动采样。原生BF16官方49步、同LR guide/text/seed101/prefix0..21全部保留，原始官方输出也保留。v2接口修复与三控制已经完成，独立于本建议。

[DDNM原文](https://arxiv.org/abs/2212.00490)及[作者代码](https://github.com/wyhuai/DDNM)通过修正clean estimate的观测分量进行零样本线性恢复。[DPS原文](https://arxiv.org/abs/2209.14687)使用测量似然梯度处理更一般有噪声问题。[FlowDPS原文](https://arxiv.org/abs/2503.08136)把clean/noise分解和测量指导扩展至flow及latent transformer。下述H3的稀疏像素投影/残差重编码是受这些工作的启发而作的工程移植，不是逐行复现DDNM/DPS/FlowDPS，也不声称是精确后验采样或新的原创机制。

固定协议只取一条：官方50点sigma表的0-based index **28、45、48**（σ **0.9、0.51612903、0.2**），强度固定1，不在val上扫描点位/步长。钩子接在官方wrapper计算clean estimate之后、Euler更新之前：

```
z0 = z_sigma - sigma * v_video
r = D(z0)
r_projected = P_current_LR_25(r)
delta = E(r_projected) - E(r)
z0_corrected = z0 + delta
# 将z0_corrected交回原Euler；audio clean estimate保持原值
```

使用当前块内已观测的22个LR和既有AA bicubic+RGB box的Dykstra25投影。内部RGB保持float，不经PNG/uint8；VAE用canonical FP16原生22帧、scale1。不得直接decode高噪声z_sigma；不得把新z0重新造噪声替换Euler轨迹。paired residual避免把E(D(z0))的codec漂移全加到z0，但**不保证编码器能实现理想像素投影**，必须记录修正前/后真实D的LR误差和delta范数，不能把投影RGB当新证据。

先在train bear_256运行原官方49与零修正49数值控制；P(r)=r时复用同一E(r)，delta严格0，latents/最终PNG应与原路径一致。这个零修正检查仅验证接入，不据图像挑参数。随后只做既有2个独立val源×2空间上下文的4个case；官方base raw/最终DC25直接复用v2已验证SHA的控制，新增内置投影的raw/最终DC25均完整报告。只用最终结果，不按val选sigma或强度；test留待协议冻结后。

最小新增6条49步轨迹，即294次DiT NFE；main每case另外6次E+3次中途D+1次最终D。现成H100实测49NFE约256=5.24s、native480=30.65s；末端D约.66/4.49s。E没有分分辨率单独计时（混合16次编码含load共16.76s），故预计值明确为估计：256内置投影约10–15s/case，native480约55–80s/case；总新增GPU约3–5min另加load/接入审计，CPU指标约2–4min。VAE与BF16 DiT共驻留预计约50–60GB，但需fresh admission、实测峰值和实际调用/传输耗时；若Comfy自动offload，必须按真实wall time报告，不沿用加法估计。

可微D的LR梯度控制可作为下一条既有DPS/FlowDPS风格对照：在stop-gradient的z0上只反传D和测量损失，不反传49步DiT。现有A/B已有真实D eager/原生前向一致和latent梯度非零证据；但梯度步长单位依赖H3 VAE，不应照搬SD3数值或根据val挑步长。最小第一轮先完成上述无学习率的投影残差控制，再决定是否另行预声明梯度控制；本文件不启动或暗含额外训练。
