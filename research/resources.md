# 计算资源与依赖实测

核查时间：2026-10-15 14:16–14:20（Asia/Shanghai）。本次只读检查现有计算节点、现有环境和少量定向目录；未启动、停止或改动任何其他项目的训练/服务，未读取私钥或密码。以下空闲情况是检查时的快照。

## 可立即推进的资源

### H100 节点 `sai-gpu161`

- 连接：`ssh -o BatchMode=yes -o ConnectTimeout=10 sai-gpu161`。本机 `~/.ssh/config` 已配置经 `sai-bastion` 的 ProxyJump。
- 身份：`saiadmin`，UID/GID 1001；hostname `aigc27`。
- 硬件：8 × NVIDIA H100 80GB HBM3，每卡 `81559 MiB`。
- 14:17:58、14:18:21 两次快照：GPU **1、5、6、7** 均仅占 `1 MiB`、利用率 `0%`，`nvidia-smi --query-compute-apps` 中这些卡没有进程。
- GPU 0、2、3、4 分别有 `VLLM::EngineCore` 服务，显存约 67.9–69.9GB。必须保留。
- 当前 shell 找不到 `squeue`、`sinfo`、`sbatch`、`qstat`、`pbsnodes`；在核查的 `/`、`/home`、`/home/saiadmin`、`/data1`、`/data2` 下未发现 AGENTS 文件，在 home 深度 2 范围未发现 reserve/reservation 命名文件。不能由此推断不存在任何外部预约制度。
- 推荐隔离项目目录：`/home/saiadmin/streaming_sr_20261015`（本次没有创建）。home 所在根盘余约 168GB、总 877GB。
- `/data1` 余 2.9TB，`/data2`、`/data3`、`/data4` 各余约 3.3TB；挂载点为 root 所有，不能直接假定当前账号可在根目录创建项目。

现成 Python 环境（仅 import 检查，没有分配 CUDA 张量）：

| 解释器 | Python | PyTorch/CUDA | 已有依赖 |
|---|---|---|---|
| `/home/saiadmin/miniconda3/bin/python` | 3.13.12 | 2.10.0+cu128 / 12.8 | numpy、Pillow、cv2、torchvision、transformers；无 scipy/skimage/diffusers |
| `/home/saiadmin/miniconda3/envs/mdc-crab/bin/python` | 3.10.20 | 2.5.1 / 12.4 | numpy、Pillow、scipy、skimage、cv2、torchvision、transformers；无 diffusers |
| `/home/saiadmin/miniconda3/envs/qwen25omni/bin/python` | 3.10.20 | 2.6.0+cu124 / 12.4 | numpy、Pillow、scipy、skimage、cv2、torchvision、transformers；无 diffusers |

`/home/saiadmin/.codex/AGENTS.md` 明确要求：**“All commands must run in the foreground.”** 并禁止未被用户显式请求的 `nohup`、`&`、`disown`、`setsid`、`screen`、`tmux`、detached/background execution。训练应使用前台 SSH 和可轮询的 exec session。它还要求不改无关代码、不悄悄修改默认参数、只在必要时新增依赖、如实报告验证范围。

建议先使用 GPU 1 单卡、小模型和独立项目目录。开跑前重新执行 GPU 进程与显存查询；不要修改上述已有 Conda 环境，可只复用解释器，新增依赖应放项目自有环境。

### 本地 Apple MPS

- 系统：macOS 26.3.1 arm64，物理/逻辑 CPU 均 10 核，内存 32GiB。
- 已验证解释器：`/Users/hezihao/Documents/New project/route3_conditional_world_20261013/.venv/bin/python`。
- Python 3.14.5，PyTorch 2.14.1，numpy、Pillow、scipy 已有；skimage、cv2、transformers、diffusers 未安装。
- `torch.backends.mps.is_available()` 为 True；实际 `torch.ones(2, device='mps') + 1` 返回 `tensor([2., 2.], device='mps:0')`。
- 可以直接复用解释器进行轻量合成数据/门控模型训练，不应修改其所属研究项目、依赖或数据。
- 默认 `/opt/homebrew/bin/python3` 是 Python 3.14.5，有 numpy/Pillow/scipy，无 torch。Homebrew Python 3.12.13 无上述科学计算依赖。
- Codex bundled Python 3.12.14 有 numpy/Pillow，无 torch/scipy/skimage/cv2。路径由 `load_workspace_dependencies` 实测取得。

## 当前不应投入新竞争任务的节点

### RTX 5090 现有 H3 节点

- 可复用连接：`ssh -S /tmp/codex-qfvdn-native.sock -o BatchMode=yes -p 28610 root@connect.bjb2.seetacloud.com`。
- hostname：`autodl-container-c12c46bdd8-576aa334`。
- 单张 RTX 5090，32607MiB；核查时 12717MiB 已用，GPU 利用率 91%。
- 四个 CUDA Python 进程均为现有 ComfyUI/H3 服务：PID 160856（8188）、324962（8190）、330142（8189）、354738（8191）。显存分别约 880、10814、498、498MiB。
- `/root/autodl-tmp` 总 1.1TB，余 794GB。
- 解释器 `/root/autodl-tmp/h3-w4a8-20261011/env/bin/python`：Python 3.12.3、torch 2.14.1+cu130，有 numpy/Pillow/scipy/torchvision/transformers，无 skimage/cv2/diffusers。
- 当前已有任务活跃，本项目没有向该卡提交任务。

### 其他节点

| 节点 | 本次结果 |
|---|---|
| `sai-gpu160` / hostname `aigc26` | 可连接，8×H100。每张卡都有现有进程；GPU0约26.6GB，GPU4约19.2GB，其余约69.9–70.6GB。服务包括 VLLM、sglang、HunyuanVideo Foley。保留全部已有任务。 |
| `huawei-modelarts-30778` | 可通过 `server-audit-30778/connect-30778.sh` 连接；8×Ascend 910B3 64GB，当前所有卡 AICore 100%，HBM约44.8–45.3GB，各有 Python 训练进程。旧的10月8日“无进程”记录已过期。`/temp` 余5.9TB，`/cache` 余18TB。 |
| `seetacloud`（SSH config 指向的另一实例） | 可连接到 `autodl-container-9fe14b930d-958bc866`，但 `nvidia-smi` 报 `No devices were found`。 |
| `lyg0371` | BatchMode 公钥认证失败：`Permission denied (publickey,password)`。没有尝试交互密码或改认证。 |

## 数据与模型复用核查

- 在 RTX 5090 节点的 `/root/autodl-tmp`、`/root/.cache/huggingface` 深度3定向搜索，没有发现 FastVR、RelayVSR、FlashVSR、REDS、Vimeo 对应路径。已有内容主要是 H3 权重与产物，不是本项目的流式超分实验数据。
- 在 H100 `sai-gpu161` 的 `/data1`–`/data4`、`/home/saiadmin/model_cache`、`/home/saiadmin/.cache/huggingface` 深度3定向搜索，没有发现 FastVR、RelayVSR、FlashVSR、REDS、Vimeo、DIV2K、Flickr/DF2K、Set5/Set14 对应路径。
- 昇腾旧盘点包含 H3、Wan、Flux、Self-Forcing 等权重，但不是已验证可直接加载的 FastVR/RelayVSR。未完整遍历数据集或加载权重。
- “没有发现”仅限上述定向范围，不代表整台服务器没有资源；数据下载与任何训练基线仍需在本项目目录中明确记录来源、样本划分与实际运行证据。

## 对实验推进的具体结论

1. **存在可用训练算力，不需要先陷入本地硬件受限的代理实验。** H100 GPU1是首选；本地MPS适合机制验证和数据管线调试。
2. 第一个训练任务宜限制单张空闲H100，使用独立目录与前台命令，保存配置、随机种子、训练曲线、checkpoint与评估结果。其余空闲卡应待方法与数据管线跑通后再决定是否并行。
3. 当前没有证据表明 FastVR/RelayVSR 的代码、权重及标准数据集已经准备好；需要另行建立真实基线链路。合成遮挡/重访实验可先验证状态更新机制，但不能冒充真实视频或扩散底座结论。

## 追加：H100 网络出口核查（2026-10-15 14:20–14:22）

结论：`sai-gpu161` 同时存在默认 DNS 不可用、直接公共网络 TCP 连接不可达的问题。**仅给 curl 加 `--resolve` 不能解决当前下载。** 本地正常联网下载，再通过 SSH/SCP 上传，是已验证组成环节可行的路径；本次没有代替主任务执行下载或上传，没有改网络配置，也没有终止主任务已有的 curl/git。

证据：

- 远端环境没有 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 或大小写对应变量，未打印任何代理认证内容。
- `/etc/resolv.conf` 由 Tailscale 管理，仅配置 `100.100.100.100`、`fd7a:115c:a1e0::53` 和其 tailnet 搜索域；文件声明不应手工编辑。
- 默认 DNS 的 curl HEAD 对 `github.com`、`raw.githubusercontent.com`、`www2.eecs.berkeley.edu`、`download.pytorch.org` 全部 `Resolving timed out after 4000 milliseconds`，没有进入 TCP 阶段。
- 本地普通 DNS 返回 198.18/15 范围的代理假IP，不应直接复制到远端。本地通过 Google DoH 取得真实A记录：GitHub `20.205.243.166`、raw GitHub `185.199.108.133`（另有109/110/111）、Berkeley `128.32.139.28`、PyTorch `65.8.76.30`（另有123/80/66）。这些仅为当时解析结果，不应永久写死。
- 用真实IP执行远端 `curl --noproxy '*' --resolve domain:443:real_ip`：上述四站均 DNS阶段立即完成，但 TCP connect 5秒超时，`time_connect=0`、`time_appconnect=0`。
- 公共DNS `dig @1.1.1.1`、`dig @8.8.8.8` 各2秒超时；直接 `http://1.1.1.1/` 的TCP80也3秒超时。这支持“存在外部出口/路由问题”，但不能据此确定具体防火墙或网关责任。
- 只读查看现有本地监听发现 `127.0.0.1:8118`、16000、8310、8311。逐个测试现成HTTP代理：8118返回 CONNECT 503，即使用 `--connect-to` 指定GitHub真实IP仍503；16000中止CONNECT；8310/8311是已有Python应用，返回CONNECT501。没有找到当前可用的现成代理。

本地 HEAD 实测：

| 资源 | 状态 |
|---|---|
| `https://github.com/` | HTTP 200 |
| `https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz` | **HTTP 200**，可用官方 BSDS500/BSR 包地址 |
| `https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS500/data/images.tgz` | HTTP 404，不应继续重试旧路径 |

下一步建议：在本项目本地目录下载官方数据/源码并记录哈希，通过已连通的 `scp ... sai-gpu161:/home/saiadmin/streaming_sr_20261015/...` 上传；若源码用本地 `git clone` 获取，再上传工作副本或 `git archive`。需要新增 Python 依赖时可本地准备 wheel，但必须匹配远端 Linux/Python/CUDA平台，不能直接复制本机 macOS wheel。

## 追加：已验证本机代理与前台 SSH 反向转发（14:23）

**已找到可供远端命令联网的办法：本机 127.0.0.1:7897 HTTP 代理，经前台 SSH `-R` 转发。** 远端 PyPI 与 Hugging Face 的 HEAD 均已成功 HTTP 200。这更新了上一节“尚未找到可用代理”的状态；无需修改远端全局 DNS、代理或系统配置。

- `scutil --proxy` 显示 HTTP/HTTPS/SOCKS 都启用，地址均为 `127.0.0.1:7897`；端口实际可连接。未读取或打印认证信息。
- 本机显式 `--proxy http://127.0.0.1:7897`：PyPI `/simple/pip/` 与 HF 首页均 HTTP 200、CONNECT 200。
- 本机显式 `--proxy socks5h://127.0.0.1:7897`：HF 首页 HTTP 200，因此7897是可用的混合协议代理。
- 一次性前台反向转发命令：`ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -R 127.0.0.1:18797:127.0.0.1:7897 sai-gpu161 '…远端命令…'`。
- 该连接内远端 `curl --proxy http://127.0.0.1:18797 -I` 访问 `https://pypi.org/simple/pip/`、`https://huggingface.co/` 均 HTTP 200、CONNECT 200、TLS握手约0.67秒。
- 同一连接内 GitHub CONNECT成功200，但TLS握手5秒超时；因此不能把 PyPI/HF 成功泛化为所有外站成功。
- 测试命令已执行完毕，SSH隧道随前台命令退出，没有保留后台隧道。

实际下载应将 `HTTPS_PROXY=http://127.0.0.1:18797 HTTP_PROXY=http://127.0.0.1:18797` 仅设置在上述 SSH 连接的具体 Python/pip/HF 命令前。转发端口18797仅绑定远端127.0.0.1，且必须随本次前台 SSH 保持；单独启动一个未带 `-R` 的 SSH 命令不会拥有此代理。继续使用前先确保端口没有被其他程序占用。

## 追加：H3实资产与接入进展（2026-10-15 15:28）

后续实时状态见 `experiments/h3_backbone/LIVE_STATE.md` 与 `results/h3_setup/live_jobs.json`。5090在各次queues全0/0、GPU idle时用新项目低分辨率前台进程完成了H3VAE、原生NVFP4 DiT/guide接口，不曾卸载/停止已有服务；初始91%繁忙快照不再代表后续时段。

- H100完整VAE：`/home/saiadmin/streaming_sr_20261015/external/h3_assets/minimax_h3_video_vae_fp16.safetensors`，5207808496 bytes，全SHA256 `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` 已通过，session84910 exit0。A/B可真实编码与解码。
- H100项目环境：`external/h3_venv/bin/python`，Python3.13/torch2.10+cu128，项目自有依赖；`external/ComfyUI_H3`实际VAE/DiT imports通过。没有变更已有conda环境。
- 40.226GB BF16 DiT下载在session81647、PID1135745续传，16并发复用旧54个已验证64MiB ranges。15:21–15:24真实增长1.208GB/141秒=8.57MB/s；最终完整SHA仍待完成，不用sparse逻辑大小充数。
- 原机完整BF16 donor路径、大小、header和已有canonical fullSHA核查成立；可在原机分进程VAE→BF16 CPUoffload→VAE路径接通，不必等迁移。全局CPU内存需留余量，runner只可终止自己进程。
- 真实中性prompt已由现成NVFP4-AWQ Qwen编码：`external/h3_assets/h3_qwen_context_neutral_restore.pt`，29×5120 BF16、tag1、raw prompt/hidden50协议。文本编码器量化单独标注；所有控制复用相同文本，没有未来GT物体语义。
- 5帧DiT smoke是同一已观测LR重复5次的静态guide，不能支持真实motion流式吞吐/等待结论。旧羊prompt与三只猪源图不匹配，已标为跨语义接口证据。

## 追加：BF16主干真实运行与视频证据（15:38）

原机已完成无DiT量化/无LoRA的40.226GB pruned BF16 FL2VA接入，文本单独标为NVFP4-AWQ编码器/BF16真实hidden50。分进程VAE encode、BF16 DiT CPUoffload、VAE decode避免同时CPU驻留；.5GPU上限与20GBreserve、CPU只杀自身guard有效，未卸载常驻服务。首个10GBreserve运行只触自身allocatorcap而失败，保留失败JSON；后续成功运行峰值7.6GB。

- 静态单LR重复5帧：4步真实forward/PNG与50sigma49Euler正常步数均已归档 `results/h3_setup/dit_bf16_neutral_guide5_256*`，仅接口，不能推断流式质量/采集延迟。
- 真实DAVIS/bmx-trees：LR0..21、64²→256²，nativeT7/audioT37，49forward全finite。sample41.715s，含load64.829s，peak7.675GB；encode6.296s，decode8.584s（包含额外VAEguide-only对照）。全部22帧输出/对照/latents/输入SHA在 `results/h3_setup/dit_bf16_neutral_davis22_normal49/`。最大块内lookahead21，未用未来块/HR语义/geometryoracle，无实际采集等待测量。
- 简单单clip RGB表征meanframePSNR：bicubic27.229、VAEguide25.406、BF16normal49 22.011dB；是需进一步学习约束的负结果，不作泛化结论。根线程做统一真实视频评价，A/B训练后再做同条件memory注入。
- 原机当前无本resource子任务GPU进程，所有compute session终态见 `results/h3_setup/compute_live_jobs.json`。
- H100完整VAE供A/B使用；BF16下载15:38为11.945/40.226GB verified ranges，16workers近实测10.90MB/s，PID1135745与session81647仍live，最终canonicalSHA仍待完成。
