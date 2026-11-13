# 运行与接续手册（2026-11-13）

本文件面向接手执行的 agent。当前交接只做只读核实、整理和发布；没有启动新训练、迁移、安装或 NPU 作业。用户已有接管授权，后续在原定范围内完成必要准备和有限执行不需要再次询问用户；但必须遵守源码、数据、预算和资源门，不能把旧资源快照当作当前准入。

首先读仓库根 [HANDOFF.md](../../../HANDOFF.md)，再读本目录的 `runtime_local_snapshot.json` 和 `runtime_ops_snapshot.json` 及仓库根 `TAKEOVER.md` 顶部。TAKEOVER 后面的时间线保留了早期“pending/running/0步”等历史状态，不可覆盖顶部的最终结论。

## 1. 当前真实状态与接续优先级

1. H3 真实 TRAIN 单步 AdamW 的原任务 **超时失败**。更新与保存产物已被独立 CPU 审计接受；严格重载后的实际输出及 BF16 合并输出尚待独立 continuation。不要重做 backward 或 AdamW。
2. continuation **b** 的源码、10/10 真实 CPU fixtures、执行合同、数值 admission 和原主机默认输入 inspection 均已准备。已有通用有界 runner 支持此任务；还需要为选择的主机/卡写一份新的 plan 与 launch admission，并做新鲜资源双检。无需重写 launcher。
3. 31112→本地的唯一迁移下载已经 **FAILED，exit255**。本地仅 `105492480 / 718719172` 字节，不能解包或安装。源归档完整，不能重打包替代。先查失败日志、核对 partial 的前缀，再选择支持恢复的单次传输方案；不要并行开第二条完整下载。
4. **31112 原主机接续可能比完成迁移更快**：本次快照卡4–7无 PID，原始依赖也在该机；这是瞬时观察，不是保留卡位。选择一台执行后，另一台绝不能再跑同一 continuation。
5. 输入保护头支线已按预声明停止迭代：matched/shuffled 各1000步及完整 DEV 评估已完成，两头均100%选 K1，科学筛选失败。后续两项 TRAIN 线性诊断也已完成。不要继续加步、调阈值或重跑。
6. FaverNet/AverNet 双模型三 native 任务已 **6/6** 推理与评分完成，含最终 Aver/compression；无需补跑。新 guide-matched 正式 LoRA 仍为0步。
7. 本地 automation `omni-r` 当前 **PAUSED**，间隔20分钟；本次未修改。不可沿用旧“ACTIVE”表述，也不在交接时自动唤醒或创建新 NPU 任务。

### 本次只读主机快照

两台各执行一次有界只读 SSH。UTC 对应新加坡时间 +8小时：

| 主机工具别名 | 快照 UTC | 瞬时无 NPU PID 的卡 | 仍有他人 PID 的卡 | MemAvailable | 工作盘可用字节 |
|---|---|---|---|---:|---:|
| `huawei-modelarts-31984` | 2026-11-13 02:41:49 | 2、3、7 | 0、1、4、5、6 | 1,206,000,500 KiB | 14,438,663,376,896 |
| `huawei-modelarts-31112` | 2026-11-13 02:42:03 | 4、5、6、7 | 0、1、2、3 | 1,542,511,588 KiB | 16,138,327,240,704 |

两机这次均未发现命令行含本项目绝对 `.py/.sh` 路径的活进程；这是限定匹配范围，不是对全机所有程序的归属认定。tmux 中仍有其他任务会话，不要结束它们。`npu-smi` 的原始输出保存在快照内。此次没有尝试共享 reservation 锁；快照中的 `lock_files=[]` 是有限路径查找结果，**不证明共享卡锁空闲**。

## 2. 主机、目录、运行时与资产

远端统一逻辑根：

```sh
TASK_ROOT=/cache/zihaohe/omni_openh3_20261111
TASK_PYTHON="$TASK_ROOT/venv/bin/python"
cd "$TASK_ROOT"
```

| 位置 | 用途与边界 |
|---|---|
| 31984 的同名根 | 原数据、物化数据、P1/保护头 CPU 与短 NPU任务、native CPU baselines、备份；迁移目标尚缺 continuation 文件 |
| 31112 的同名根 | 当前 H3 全22帧、真实 AdamW 失败现场、已审计 checkpoint、continuation b完整准备、迁移源归档 |
| 30794 的同名根 | 历史 Stage2/Rphi/SR/joint任务；本次未登录核实，不把历史卡6/卡7可用当事实 |
| `/cache/zihaohe/streaming_sr0` | 原资产与原项目；不覆盖、不清理、不就地改运行文件 |
| `$TASK_ROOT/code/ComfyUI` | 固定 Comfy 源；原审计commit `5a665890ded5020c282f63c4ea2dd9e083e025ff`，仍以实际逐文件pins为准 |
| `/cache/zihaohe/envs/minimax-h3-npu` | 31112/当前源码使用的真实底层运行时；不要替换成历史30794的 `/cache/yunfeng/...` |
| `$TASK_ROOT/venv` | 项目Python及依赖；不在共享环境中 pip install/upgrade |

本次 SSH 快照用系统 `python3` 收集 JSON，显示3.12.0；它**不是**实验 Python。冻结 runtime 预期为 Python `3.12.14 | packaged by Anaconda, Inc. | (main, Aug 27 2026, 14:37:13) [GCC 14.3.0]`、Torch `2.10.0+cpu`、torch_npu `2.10.0.post2`。CPU轮次设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，不导入模型或伪造可见NPU。

NPU worker 通过已有 `code/run_takeover_command.sh` 加载 `env.sh`：CANN9.0.1、ATB、固定 Comfy、离线权重路径等。`env.sh` 要求显式 `ASCEND_RT_VISIBLE_DEVICES`，还会建立本项目临时目录，不能把 source 它称为纯只读检查。runner 选择一个物理卡并映射给 worker；producer 内部用逻辑 NPU0。不要修改整个终端/其他会话的全局卡环境。

### 实际必要大资产

父 manifest 为 `configs/openh3_20261111/refiner_turbo_30794.json`，SHA `1023bd6e1df1552240cc7ff4bfac960836d2eeda05e680ecd2eb4b752bf7d975`。名字含30794不表示只可在30794使用；以其中实际路径和SHA为准。

| 资产绝对路径 | 字节 | SHA-256 |
|---|---:|---|
| `/cache/zihaohe/streaming_sr0/assets/comfy_h3/minimax_h3_fl2va_pruned_bf16.safetensors` | 40,225,724,176 | `a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4` |
| `/cache/zihaohe/streaming_sr0/assets/comfy_h3/minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| `/cache/zihaohe/omni_openh3_20261111/assets/refiner/open_regen_2k_refiner_rank384_bf16.safetensors` | 5,868,409,512 | `4bf30291343eef5e51ee867ba37dd31bce3133d1da1ace1a245a7130b88818c2` |
| `/cache/zihaohe/omni_openh3_20261111/assets/openh3_split/openh3_turbo128_comfy.safetensors` | 1,956,171,936 | `0b5e411b519ea5041734d44fc102b0dfd6b6a29737048ad1a80a522fb633fa99` |
| `/cache/zihaohe/streaming_sr0/assets/comfy_h3/h3_qwen_context_neutral_restore.pt` | 299,557 | `fb03abcc1eb569d2fcd360fe512f2ae0d707298f4a7ad8b6a35b609e1d97d5d2` |

31984旧inventory发现4个大模型已存在，但当时只是stat，**其目标端完整SHA尚待核**。四文件共53,258,114,120字节，不在迁移包里。31112的Turbo已从已有表示精确重建到原全文件SHA，不再启动历史Turbo失败传输，也不要用稀疏文件或假权重占位。

## 3. H3 saved continuation 的准确接续

### 必需身份与结果边界

读取实际31112路径，31984只有迁移安装并核验后才可同路径复用：

- `runs/h3_real_adamw_update_20261112a/queue_status.json`：原任务 `failed`，SHA `28299a42e3e76ce00b5e8fe7dae9bd471ebfa7fc7950051199857280de9d9edb`。
- `runs/h3_real_adamw_update_20261112a/eval/partial.json`：SHA `ff278a84665a7cae78246dc017cbbfe0b0ddb2e93ac9f7d08c6fa31c0309b646`。
- `runs/h3_real_adamw_update_20261112a_finalize/finalize_status.json`：仍为 `failed`。旧CPU终态审计运行完成不等于原producer通过。
- `runs/h3_saved_adamw_partial_audit_20261112b/output/audit.json`：SHA `45455ffb415b853ccb1c5afc7ce7ca0d1d47dc5ac731184a39c34512d9d250d6`，要求 `complete=true` 且 `saved_subset_ready_for_continuation=true`；原失败、reload与merge未验收的字段仍保留。
- `runs/h3_real_adamw_update_20261112a/eval/real_train_AdamW_step1.safetensors`：149,313,664字节，SHA `b805168bc6d1af3506db5788b9c22a6eec1c30f804dba34268013001558932be`。
- 同目录 `real_train_AdamW_step1_optimizer.safetensors`：298,511,448字节，SHA `569eeaf6d470036cb75b216bfd1dcf0fcd824aadbfe283a0f779657e4055664e`。
- TRAIN cache：`runs/train_vae_cache_legacy_smoke_20261112e_31112/cache/manifest.json`，SHA `a2c5a5b7607aaa0bce74e0bd573a98f5fb36ecb0a8ee61d5726d6d72c134d62e`；下面 `arrays/rphi_guide_latent.npy` 和 `arrays/raw_warm_start_latent.npy` 是固定 `Z_R` 与 `Z_raw`。不要因为迁移到31984就改掉 `_31112` 路径名。

源仍为 TRAIN `2BdYaGuVAKo` 的 compression，22×480×832，latent `[1,24,7,30,52]`。该源含切镜，所以用于实际负载工程验收，不能当连续视频质量主实验。target只用于离线描述性评分，不进入模型/guide/warm start；续验0encode、0Rphi、0backward、0optimizer step、0新增训练checkpoint保存。

只做：加载400个独立FP32 LoRA tensor与1200个AdamW state → 3NFE+1decode → 对原 `updated` 的video/audio/实际FP16 decoder input/RGB逐字节相同 → 声明200矩阵FP32累加后一次BF16转换 → 3NFE+1decode。合并误差只描述，不虚设等价阈值；真实部署RGB不变时不得称有效训练。原 `original_run_accepted=false` 永远不改，不能把两次资产加载与分开的预算拼成一次成功12NFE任务。

### 冻结源码、合同和可用 runner

```sh
CONT_PREP="$TASK_ROOT/preparation/h3_adamw_saved_continuation_preparation_20261112b"
CONT_ENTRY="$CONT_PREP/code/check_h3_adamw_saved_continuation.py"
CONT_CONTRACT="$CONT_PREP/execution_contract.json"
CONT_RUNNER="$TASK_ROOT/preparation/omni_followup_runtime_20261112b/code/run_omni_followup_once.py"
```

| 文件 | SHA-256 |
|---|---|
| continuation b entry | `bebc1fa7ed902d521f99d9b22987e07fc91d722647d63d561844f04a51b023c8` |
| 最终 execution_contract.json | `be8e73ba1fd8d6d84af2115365164012b10ba907f0adc0db32b0ef61742c7c5e` |
| producer_admission.json | `a4741c3f0e6405fdeb566c5d9db5906c5cba938d384b2a6db33cefd576ec9916` |
| generic runner b | `8df017e4f1aa48bff3bd80a1f3361061f28f84b01c65991bc119eddeb023f30e` |
| generic runner旁 `owned_process.py` | `efabd330e5b49653fc01275bba18c9c408de43fd56e194685102bb313e097388` |

入口b的10项真实CPU测试已经通过；a的 `actual_lr→lr` 字段映射失败保留，不再重跑a。源码镜像及当时实际inspection在本地仓库 `results/takeover_20261111/h3_continuation_ready_20261112b/payload/`。不要把 template 中的占位内容当最终合同。

原主机已有输入inspection，不需无原因重复。换到31984并完成安装/全SHA后，需要新的目标端默认inspection。下列输出目录是**未来新输出**，不是已经存在的结果；选择未使用名字，禁止覆盖：

```sh
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=KILL 900s "$TASK_PYTHON" "$CONT_ENTRY" \
  --pretrained-manifest "$TASK_ROOT/configs/openh3_20261111/refiner_turbo_30794.json" \
  --execution-contract "$CONT_CONTRACT" \
  --out "$CONT_PREP/inspection_31984_handoff"
```

这条命令无 `--execute`，不加载H3。它会读实际checkpoint与必需数组，不只是stat。大模型在默认inspection中仍是sizes-only，不能代替完整依赖SHA验证。

### 缺的是一份新的 plan 与 launch admission，不是新模型实现

`run_omni_followup_once.py` 已支持 `kind=saved_adamw_continuation`。接手者在新的prep中写plan，不修改上述旧数值合同、旧队列或源码。所需字段：

- `schema=omni.followup_bounded_once.v1`、固定root、`kind=saved_adamw_continuation`、当场选择的 `physical_card`（0–7整数）。
- `resource={wall_seconds:3600, CPU_threads:4, rss_bytes:137438953472, grace_seconds:30}`。
- `supervision_source=$TASK_ROOT/preparation/omni_followup_runtime_20261112b/owned_process.py`。
- `execution_contract=$CONT_CONTRACT`、`pretrained_manifest=$TASK_ROOT/configs/openh3_20261111/refiner_turbo_30794.json`。
- 新 `run` 必须直接在 `$TASK_ROOT/runs/` 下；`out` 必须等于 `run/output`，不得复用原失败目录或已有inspection。
- `argv` 必须精确为下方worker数组，无额外采样/训练参数。
- `pins` 用绝对路径绑定runner、supervisor、execution、entry、manifest、`code/takeover_queue.py`、`code/run_takeover_command.sh`、`env.sh`、全部数值runtime pins及各前置验收文件。
- `required_json_values` 至少绑定独立subset audit的 `complete=true`、`saved_subset_ready_for_continuation=true`；跨主机增加目标完整依赖核验。
- `report_file=report.json`；`report_requires` 至少要求 `complete=true, arithmetic_persistence_passed=true, strict_reload_passed=true, budget_passed=true`。**不要要求functional_deployment_ready=true来掩盖负结果**；它是独立科学字段，需要如实报告。
- launch admission独立文件：`schema=omni.followup_once_admission.v1`、`approved=true`、`plan_sha256`、`runner_sha256`、同kind与physical_card。数值producer admission已存在，但不能替代这份新鲜资源启动绑定。

worker argv（仅放进plan，不在裸终端直接启动）：

```text
[
  "/cache/zihaohe/omni_openh3_20261111/venv/bin/python",
  "/cache/zihaohe/omni_openh3_20261111/preparation/h3_adamw_saved_continuation_preparation_20261112b/code/check_h3_adamw_saved_continuation.py",
  "--pretrained-manifest", "/cache/zihaohe/omni_openh3_20261111/configs/openh3_20261111/refiner_turbo_30794.json",
  "--execution-contract", "/cache/zihaohe/omni_openh3_20261111/preparation/h3_adamw_saved_continuation_preparation_20261112b/execution_contract.json",
  "--out", "新run绝对路径/output", "--execute"
]
```

新plan写好并审查其SHA后，设置 `CONTINUATION_PLAN` 为这份**实际存在**的绝对路径。先核plan，再单次启动；此处不给虚构“已存在queue”的路径：

```sh
"$TASK_PYTHON" "$CONT_RUNNER" "$CONTINUATION_PLAN" --inspect-only
```

只有上条通过、确认两机没有重复continuation、做好新鲜资源检查后，才执行已有runner：

```sh
"$TASK_PYTHON" "$CONT_RUNNER" "$CONTINUATION_PLAN"
```

可通过 `ssh_run_bg` 在新命名会话执行同一条命令，但不能前台/后台各启动一次。以新run的 `status.json`、`worker.log`、`output/report.json` 或 `output/partial.json` 为准，不以tmux存在当作模型运行/成功。

## 4. 资源、锁和作业管理

generic runner先核pins与admission，再检查固定卡health=OK、无任何PID、空闲至少60,000MB；取得 `/cache/zihaohe/.npu-reservations/card_N.lock` 的非阻塞exclusive flock，等待15秒，再核pins、卡、内存、磁盘。要求主机MemAvailable≥160GiB、项目盘free≥10GiB。shared flock只约束遵守协议的任务，不能替代集群调度器，所以第二次资源检查不能省。

资源不足时状态为 `not_launched`，不是等待/自动换卡。选择另一卡必须新plan/admission和新run，不能修改正在运行或已尝试的plan。**不能kill其他人的PID、删除其锁、结束其tmux或清理他人目录**。有锁文件不等于锁被持有；删除锁文件还会破坏合作锁语义。

runner建立自己的新进程组，采样该组同时RSS，128GiB上限；达到3600秒或RSS上限，先TERM自己组、最多30秒后KILL仍存活成员。这里只能清理recorded owned PG，不得把旧PID当活进程使用。终态还要核pins未变、remaining_owned_live_processes为空。它不是cgroup硬隔离，不要把采样RSS称作内核级限额。

| 阶段 | CPU/内存/时间 | 是否已完成 |
|---|---|---|
| 源missing archive打包 | 2CPU、4GiB AS、600s硬限 | 已完成；不要重打包 |
| 目标archive inspection/install（分别一次） | 2CPU、4GiB AS、900s；execute需≥3GiB额外盘空间 | 未部署/未执行 |
| 目标完整1406依赖SHA | 2CPU、4GiB AS、1800s | 未执行；读4大模型共53.26GB但不加载模型 |
| saved-subset CPU审计 | 2CPU、4GiB RSS、900+30s | b已完成；无需重复 |
| continuation b | 4CPU、128GiB owned-PG RSS、3600+30s；1NPU | 尚未启动；6NFE/2decode/1merge |
| continuation后独立CPU实际数组验收 | 可沿900s/2CPU/4GiB提出新有限合同 | 消费新report和数组；不重跑模型或训练 |

旧finalizer可作代码参考，但不能不改schema就声称它已审计continuation。producer可以在已批准的有界runner中先执行，完成后再独立核400adapter/1200optimizer状态、strict reload实际数组、200允许merge写入与输出差异。CPU审计不会增加NFE；不要因为没有专用自动finalizer就重新实现整个launcher或阻塞必要准备。

## 5. 唯一迁移：失败partial、安装与目标验证

源归档仍位于31112：

```text
/cache/zihaohe/omni_openh3_20261111/preparation/h3_saved_continuation_migration_31984_20261112a/output/missing_dependencies.tar.gz
```

完整大小718,719,172字节；SHA `b26949c1b9c8b9fbd98d94249146c4135a19dda16ec824a465c3987124fd3f2a`。346缺失路径→150去重blob、151tar普通成员；unique源字节1,084,653,343。源pack实际39.21秒，所有源与打包流SHA通过。`actual_manifest.json` SHA `b862d92543b1e994d8896d75936fb94c081ffd2cb2be4e8bd3de65e8e8e20d20`。

唯一失败下载会话 `h3-saved-continuation-missing-download-20261112a` 的最新确认在 `runtime_local_snapshot.json`。工具可先只读：

```text
transfer_status({"session":"h3-saved-continuation-missing-download-20261112a"})
```

`ssh_download_bg` 的接口只声明rsync、缺失时回退scp，没有承诺断点续传。不要直接再次调用并假定会续传。下一步先读该session诊断，确认原进程已退出，核本地105,492,480字节partial前缀确与固定源一致，再使用实际支持恢复/分块的实现只传剩余部分；保留原失败与prefix验证证据。如必须新session，明确它是同一传输的受控恢复且只有一个活传输，不删除partial、不重打包。

本地完整后才能一次上传31984；不上传partial。完整文件的尺寸与SHA可以这样核，使用实际本地仓库路径：

```sh
python3 - <<'PY'
from pathlib import Path
import hashlib
p=Path('/Users/hezihao/Documents/New project/omni-openh3-takeover-20261111/preparation/h3_saved_continuation_migration_31984_20261112a/missing_dependencies.tar.gz')
assert p.stat().st_size == 718719172, '下载尚未完整，停止'
h=hashlib.sha256()
with p.open('rb') as f:
    for b in iter(lambda:f.read(8<<20),b''):h.update(b)
assert h.hexdigest() == 'b26949c1b9c8b9fbd98d94249146c4135a19dda16ec824a465c3987124fd3f2a'
print('complete archive SHA verified; not yet installed')
PY
```

目标installer是本仓库 `preparation/h3_saved_continuation_target_install_20261112a`，freeze SHA `7e9121188f36775b223f5005e719f2444f15c73cd3460dca0771dfa82f7104f7`。12/12本地小fixture通过；未在目标执行。把这一新prep按SHA部署到31984后，用完整归档的实际目标位置设置变量：

```sh
INSTALL_PREP="$TASK_ROOT/preparation/h3_saved_continuation_target_install_20261112a"
MIGRATION_ARCHIVE="$TASK_ROOT/preparation/h3_saved_continuation_migration_31984_20261112a/output/missing_dependencies.tar.gz"
```

下列输出目录全部是未来新目录。先一次default inspection（不安装），看receipt后单独执行安装；不可一条命令盲链两步：

```sh
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=KILL 900s "$TASK_PYTHON" "$INSTALL_PREP/install_once.py" \
  --archive "$MIGRATION_ARCHIVE" --out "$TASK_ROOT/runs/h3_saved_continuation_target_install_20261112a_inspect"
```

```sh
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=KILL 900s "$TASK_PYTHON" "$INSTALL_PREP/install_once.py" \
  --archive "$MIGRATION_ARCHIVE" --out "$TASK_ROOT/runs/h3_saved_continuation_target_install_20261112a" --execute
```

installer验证完整archive和成员allowlist，不调用tar.extract。已有相同文件保留inode/mode；冲突即失败；每个新目标独立staged copy再exclusive link，不与同blob的其他快照共享inode。新文件0444。它是有journal、异常时仅回滚自有inode的操作，**不是多文件全局原子事务**；硬杀/崩溃可能留下partial安装，按journal核对后再决定新操作，不覆盖旧文件。

安装完成后另做全量只读SHA，不能拿原stat-only inventory提升状态：

```sh
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=KILL 1800s "$TASK_PYTHON" "$INSTALL_PREP/verify_target.py" \
  --out "$TASK_ROOT/runs/h3_saved_continuation_target_full_verify_20261112a" --full-sha
```

要求 `full_dependencies_ready=true`。它核1,406路径含1,364 frozen pins和4大模型全SHA，以及sys.version、安装分发包版本；不导入Torch/Comfy、不验证NPU数值。新主机实际runtime imports、相同noise、严格输出byte门仍由continuation检查。跨机舍入不同也不得事后放宽strict门；保留失败，0重试。

## 6. 失败恢复与不可重跑列表

| 情况 | 正确处理 |
|---|---|
| SSH超时、状态不可读 | 不推断worker死了；下一次有限只读status/owned PID核实。不要直接重启或kill |
| runner `not_launched` | 保留目录，定位资源/pins/锁原因；需要再尝试时新plan、新admission、新run，不覆盖旧状态 |
| producer超时/错误 | 保存partial、worker.log、status、owned cleanup；仅接受实际已核组件，不把完成counter视为严格门通过 |
| archive下载失败 | 先查日志和prefix，恢复同一partial；不能解包或上传未完成文件 |
| 安装冲突/回滚不完整 | 停止并核transaction journal及inode；不删除已有目标或试图“覆盖修复” |
| 跨机strict reload失败 | 保留实际video/audio/decoder输入/RGB与runtime证据；不调整seed、LR、容差或额外NFE重试 |
| arithmetic通过但部署RGB不变 | 明确工程持久化通过、功能门未过；不是质量收益，不能自动进入正式LoRA训练 |
| 科学screen failed但runner completed | 两者均保留。执行成功不等于方法成功 |

不要重跑：原H3 AdamW a及旧finalizer、已验saved-subset b、full22联合反传、VAE cache e、nonzero synthetic b、SR300步、Rphi Stage2三臂、全部6次native baseline、P1教师/已完成headroom与保护头两臂1000步、四臂完整DEV评估、两项TRAIN线性诊断、原始40源/修订25源材料化及其终态审计。具体已完成目录见TAKEOVER顶部与对应root_acceptance；不以历史文件名中的日期/“a”推断需要复跑。

P1旧3臂历史效用方案未准入，保护头支线已失败停止；未来新结构应另起预声明实验，不能借“接管”延长旧run。6/6 native只是单历史DEV源、offline22/maxfuture21，下一步多源比较应冻结同输入/target/window/metric矩阵并只补缺格。不存在新独立frozen test，不能把已调参DEV改名为test。

## 7. 下一agent的前30分钟

1. **0–5分钟**：读本文件、两份快照、TAKEOVER顶部；确认GitHub发布分支和本地HEAD由根handoff记录更新。明确当前只需补H3 reload/merge，正式guide-matched训练仍0步。
2. **5–10分钟**：检查唯一transfer session和失败日志；不要开启传输。各主机做一次新鲜资源/status读，确认没有已启动continuation。若31112已有空闲卡，优先考虑原机直接接续，避免被迁移等待绑住。
3. **10–20分钟**：读取冻结generic runner、最终continuation合同、subset acceptance。为当前选择的唯一主机/卡编写新plan和launch admission，核完整pins、1364runtime依赖、报告门与预算；跨到31984则先走完整恢复下载/上传/安装/1406SHA，未满足时不能启动NPU。
4. **20–30分钟**：通过runner `--inspect-only`；检查原/目标两机没有重复任务，确保新run目录不存在。资源与锁门由已有runner在实际启动时完成15秒双检。若准备已齐，可按用户接管授权单次执行这一有限continuation；否则交付明确缺口及可执行下一步，不重做既有模型工作。

之后等待status有意义变化，优先检查真实阶段/数组落盘，不高频轮询。完成后独立CPU读新实际数组与frozen状态，下载一次小JSON/log/必要小数组归档；大RGB/梯度/权重留远端并记录SHA。

## 8. 已保存的打包入口与Git边界

`preparation/h3_saved_continuation_migration_31984_20261112a/run_once.sh` 是515字节的**源missing文件打包helper**，不是continuation launcher，也不是训练脚本。它只调用 `pack_missing.py`，600s硬限、清空可见设备、2CPU/4GiB限制由pack实现；SHA `08e969b09e41c4087d24884cbc331f9876313a02161722a39c92b13ac51fffaa` 与当时preparation receipt一致。本次交接将这个小源码纳入版本控制，以保存真实pack入口；不要执行它重打包。

不应把718MB下载partial、模型权重、NPZ、编译缓存、重复source_bundle/backup/evidence大tar加入普通Git。本次不读SSH密钥或凭证；主机只用已配置的工具别名，交接不提供密码或私钥。恢复下载、上传与任何NPU运行都应写新的状态证据，不能回写旧 `local_actual_verification.json` 伪造当时完成状态。

本文件依据的原始只读快照：`runtime_ops_snapshot.json`；唯一transfer与automation状态依据：`runtime_local_snapshot.json`。这两份是观测证据，不是持续监控或资源预约。
