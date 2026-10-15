# 官方BF16流IO v2前置审计

同bear_256原生22帧，normal50网格index46，σv=.43902439。零adapter与guide末层bias+0.01两条件均实测：训练video输入、physical audio输入、v预测与实际官方包装层maxabs0，训练x0对官方单步Euler最大误差2.38e−7。guide/text/layout和process_in/out匹配，v目标epsilon−zHR。HR latent仅在此诊断中构造监督噪声桥，不能作为SR输入使用。

v2 helper复现BaseModel顺序：video noisy stream转BF16；audio先把σv*noise转BF16，再乘BF16(σa/σv)；返回video velocity遵从native BF16输出。guide payload保持FP32，loss显式FP32累计。对原始FP32流入口的差异已经隔离，不修改legacy训练/math文件。

随后真实全部50层反传：guide输入梯度0.01823247、LoRA梯度0.00259535、guide projection梯度0.17352806；530基础张量版本不变、basegrad0。无optimizer更新。审计session71242 terminalexit0，20.39s，peakallocated41,365,472,256bytes。

完整源/输入/helper SHA和逐条件数值在result.json；修复训练硬校验此审计通过且helper SHA相同。新训练目录native_sr_pilot400_native_io_v2，从同初始化/seed/case/sigma/noise/lr重跑三控制各400；不接旧checkpoint。
