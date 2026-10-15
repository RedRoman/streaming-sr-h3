# 复用采样内DC的接口与边界

完成且源SHA固定的实现是 `experiments/h3_backbone/sample_inner_dc.py`：`sample(case,label)`在62行，实际`hook(args)`在80行，105行注册`ModelPatcher.set_model_sampler_post_cfg_function`。SHA为`8b2809a51b59009697551c785b2b1a7a372cc2dda0f5661f3f61cfaaf3029bd8`。它是闭包，依赖当前样例的low、固定shape、VAE、计数器；不是直接可跨case共享状态的全局函数。每条49步轨迹必须新建/重置计数器。

实际Comfy回调的`args['denoised']`是clean estimate，不是velocity；用已知AV shape unpack，video `[1,24,7,H/16,W/16]`，audio `[1,32,2,37]`。取video经standard FP16 `comfy.sd.VAE.decode`得到`[22,H,W,3]` RGB `[0,1]`；两支`VAE.encode`均输入独立clone的float RGB，wrapper负责2RGB−1归一化。内部不要新增PNG量化或再归一化一次。

测量接口独立于条件guide：闭包中的`low`是float32 `[22,3,H/4,W/4]`、`[0,1]`，只用于已有`BicubicConsistency`的AA bicubic/box Dykstra25。条件guide、text、audio clean estimate不由该hook修改。delta加到clean video，不能同号加raw v；若改v须是−delta/sigma。官方Euler更新保持源代码原样。

可接受只有当前far LR的static22：令`low = current_far_LR[None].expand(22,-1,-1,-1).clone()`；若H3目标latent/输出画布就是同一完整far视角，所有22帧的DC都只读这张far LR。条件tokens可以另含已到达near/复合near guide，hook不读取near作为测量。如果实际输出画布自身是near/far拼贴，则不能拿单张far测量对拼贴全图投影；需先把目标读出定义为far画布或显式测量对应区域，不能暗改几何。

当前motion runner的`input_prefix=[0,21]`/等待21帧元数据针对真实22张不同LR，**不能原样继承到static22**。static重复只提供1次独立观测，应记录其真实far时间、distinct index、latest evidence；复制22次不增加证据，若far当前帧已到达则等待0。近景参考若来自已知过去，另记录到达时刻；不用HR/未来视角作测量或caption。

当前协议固定point28/45/48、unit残差、seed101、同中性真实context和基础权重。0创新测试显式identity P，复用同E(r)使delta严格0；同时已测实际P25的可行RGB固定点。它保留原官方路径为控制。A如更换读出teacher，应另版本缓存并保持所有控制一致，不能混旧新teacher。本文只是可复用约束说明，没有改A代码/启动更多样例。
