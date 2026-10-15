# RelayVSR 外部强控制资源审计

主线仍为H3；Relay仅官方外部比较基线，目前**没有启动GPU推理**。H100现有official commit `e856773350f53036d825404c9ea9228e6dffadca`，源码clean，官方CLI真实import/--help exit0。独立`external/relay_venv`已有全部pyproject依赖（torch2.10、diffusers0.37.1、triton3.6、av18.1等）；无需改环境或额外text encoder/VAE，Wan checkpoint内置prompt与模型配置。

两份官方文件现已完整核验，HF固定revision `d43dc5545f9a3bd1c7730b68ebff4ab1f9199bd1`：

- `external/RelayVSR/weights/wan.safetensors`：10,066,399,024 bytes，SHA256 `638acb6a27c17ba87b2e66a1770f17e61a99ad87cfff6fa3d3e05c2d57f2eb98`。
- `data/flash.safetensors`：127,759,784 bytes，SHA256 `7b8b8c99275fd5c4501b0589cc7b4c4506456a5df83160ef4e207e147b30f714`。需显式传--flash-checkpoint；不为默认路径复制大文件。

旧Wan只有7,685,209,714 bytes，缺2,381,189,310。前台40657/PID1321838通过本机HTTP代理的单次SSH-R18797续传，真实HTTP206的Content-Range从原byte准确接续，350.27s结束exit0，全文件SHA通过后atomic rename。旧不完整canonical先改名.partial防止误加载。public HF元数据、final状态、源SHA与包版本见同目录JSON；当前没有未完成下载。

官方协议直接复用inference.main：K16窗包含起/末两endpoint，first output最多等待15个未来LR；22帧的Wan引用为0/15/21，Flash两窗0..15、15..21，重叠15只更新历史、不重复emit，因此3次Wan NFE、23次Flash decode、22个输出。24fps名义块内采集上界.625s，另加计算；不是已经测量的实时延迟。Wan固定one-step timestep1000、shift5、CFG1、seed42，4历史keyframe KV，Flash为S step60000 EMA、两帧video attention。

可运行集合为已有4train bear/bus/car-turn/flamingo与2val cows/dance-twirl，各256和native480两种上下文，所有case固定prefix0..21，输入只LR。先LR64x64/22frame单例实测JIT/显存，fresh idle H100（native480先要求>60GBfree），项目wrapper内设置.8显存cap，不卸载其他服务；模型文件约10.19GB不代表真实激活峰值。通过后按固定同协议顺序12case，保存lossless PNG、原始MP4、逐输出last_consumed input/lookahead、WAN/Flash成本与每LR SHA；统一raw/DC25/相同LPIPS评价，test不用于调参。

命令模板（仅计划，未运行）：

```bash
CUDA_VISIBLE_DEVICES=6 external/relay_venv/bin/python experiments/relay_baseline/measure_official.py \
 --repository external/RelayVSR --input data/real_video/h3_256/train/bear/lr_x4 \
 --fps 24 --max-frames 22 --window-size 16 \
 --wan-checkpoint external/RelayVSR/weights/wan.safetensors \
 --flash-checkpoint data/flash.safetensors \
 --output results/relay_baseline/official/bear_256/output_x4.mp4 \
 --png-dir results/relay_baseline/official/bear_256/png \
 --metrics results/relay_baseline/official/bear_256/measurement.json
```

`measure_official.py`仅观察官方main的调用，不改官方source/预测。observer wall time含同步/PNG写入，须与纯model stage时间分开。其`--protocol causal_repeat_start`是已准备好的严格因果条件变体：每15帧刷新Wan，当前reference重复为两endpoint并resetFlash；22帧仅2次Wan和22次Flash，未来等待0。因此必须明确改协议/不同计算量，不能宣称官方完整法或同预算。未训练reference adapter；后续H3 memory机制不由这个外部基线替代。


精确WAN权重header核查：5,033,156,288个tensor标量，约5B，BF16文件10,066,399,024字节；30层、24heads×128。早期1.3B描述作废；Relay仅外部比较基线。
