"""Produce a compact, evidence-linked report from completed actual runs."""
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[2]/'results/h3_factorized'
TEST=BASE/'test_three_seed'


def readcsv(path):return list(csv.DictReader(path.open()))
def mean(rows,key):return float(np.mean([float(r[key]) for r in rows]))
def selected(rows,method,scenario=None):return [r for r in rows if r['method']==method and (scenario is None or r['scenario']==scenario)]
def effect(payload,control):return next(x for x in payload['effects'] if x['scenario']=='ALL_five_stages' and x['comparison']=='factorized_swap minus '+control)


def main():
    raw=readcsv(TEST/'rgb_metrics.csv');dc=readcsv(TEST/'projected/rgb_metrics.csv');diag=json.loads((TEST/'diagnostics.json').read_text());swap=readcsv(TEST/'oracle_chart_swap_rgb.csv');oracles=readcsv(TEST/'error_source_oracles_rgb.csv')
    effects={k:json.loads((p/'paired_effects.json').read_text()) for k,p in [('raw',TEST),('DC',TEST/'projected')]};environment=json.loads((BASE/'environment.json').read_text());gradient=json.loads((BASE/'decoder_gradient_check.json').read_text());cache=json.loads((BASE/'cache_pilot/manifest.json').read_text())
    models=['input_bicubic','H3_current_LR','factorized_swap','factorized_no_swap','noninvertible_swap','mixed_memory','H3_HR_roundtrip']
    table='\n'.join(f"|{m}|{mean(selected(raw,m),'rgb_psnr'):.3f}|{mean(selected(dc,m),'rgb_psnr'):.3f}|" for m in models)
    eventtable='\n'.join(f"|{s}|{mean(selected(raw,'factorized_swap',s),'rgb_psnr'):.3f}|{mean(selected(dc,'factorized_swap',s),'rgb_psnr'):.3f}|{mean(selected(dc,'factorized_no_swap',s),'rgb_psnr'):.3f}|{mean(selected(dc,'H3_current_LR',s),'rgb_psnr'):.3f}|" for s in dict.fromkeys(r['scenario'] for r in raw))
    etable='\n'.join(f"|{family}: full−{c}|{effect(effects[family],c)['mean_dB']:+.3f}|[{effect(effects[family],c)['CI95_low']:+.3f},{effect(effects[family],c)['CI95_high']:+.3f}]|" for family in ['raw','DC'] for c in ['factorized_no_swap','noninvertible_swap','mixed_memory'])
    swapmeans={m:mean(selected(swap,m),'rgb_psnr') for m in models if m.startswith(('factorized','noninvertible','mixed'))}
    stable=[r for r in diag if r['method']=='factorized_swap'];state=stable[0];maxinverse=max(r['inverse_maxabs'] for r in stable);rmsinverse=mean(stable,'inverse_rms')
    ortable='\n'.join(f"|{s}|{mean(selected(raw,'factorized_swap',s),'rgb_psnr'):.3f}|{mean([r for r in oracles if r['mode']=='oracle_appearance' and r['scenario']==s],'rgb_psnr'):.3f}|{mean([r for r in oracles if r['mode']=='oracle_initial_HR_identity' and r['scenario']==s],'rgb_psnr'):.3f}|" for s in dict.fromkeys(r['scenario'] for r in raw))
    stage1secs=sum(json.loads(p.read_text())['seconds'] for p in (BASE/'latent_stage1').glob('*/complete.json'));stage2secs=sum(json.loads(p.read_text())['seconds'] for p in (BASE/'rgb_stage2').glob('*/complete.json'))
    final=json.loads((TEST/'complete.json').read_text());measure=readcsv(TEST/'projected/rgb_metrics.csv');rmsemean=mean(measure,'actual_measurement_rmse');rmsemax=max(float(r['actual_measurement_rmse']) for r in measure)
    source_results=json.loads((TEST/'source_paired_outcomes.json').read_text())['results'];return_effect=next(r for r in source_results if r['family']=='DC' and r['group']=='return' and r['comparison'].endswith('factorized_no_swap'))
    source_table='\n'.join(f"|{r['group']}：{r['comparison'].removeprefix('factorized_swap minus ')}|{r['mean_difference_dB']:+.3f}|{r['source_wins']}/16|[{r['source_bootstrap_CI95'][0]:+.3f},{r['source_bootstrap_CI95'][1]:+.3f}]|" for r in source_results if r['family']=='DC')
    report=f'''# B：真实 MiniMax-H3 可逆内容/外观状态实验

已实际完成三seed训练、独立来源heldout真H3解码评估及统一经典DC对照。当前仍是**候选表示架构**；不能仅凭可逆结构或counterfactual训练把它称为原创且有效的streaming SR核心。这里冻结实际H3 E/D，尚未使用H3 DiT生成先验，因此也不代表完整H3生成式视频SR效果。

可量化的积极信号是：相同DC的中性回访，full比no_swap高{return_effect['mean_difference_dB']:.3f}dB，{return_effect['source_wins']}/16来源获益。但full与非可逆控制相当，整体仍输当前LR→H3 VAE+DC；可逆性的独立性能贡献及有效SR核心均未建立。

## 可复现数据与预算

- 官方BSDS500来源划分32train/8val/16test；同split donor构造局部内容替换，来源/哈希/物理扰动seed见 `cache_pilot/manifest.json`。56源、每源6个HR监督状态/6个LR观测，共672次真实native256×22 E；22帧是同一已到达LR锚点复制的VAE上下文，不是未来帧或22份证据。
- 八步：中性远景A→外观1远景A→外观1近景揭示A→无可见证据query→中性回访A→近景局部内容换B→外观2远景B→同packet副本。退化为Gaussianσ.6/5tap反射+Torch bicubic antialias；远4×，中心128近2×，独立LR噪声σ.003。它是自然照片上的受控成像实验，**不是真实运动/长间隔回访数据**。
- 训练只加载train；val选择checkpoint；16test在架构、预算和分析协议固定后统一读取。HR/真实appearance只进入监督或单独标明的oracle诊断，正式observe/query仅用已到达LR、sensor footprint与packet ID。原始生成结果从不追加为真实观测。
- 四模型参数：full144748、no_swap144748、noninvertible144614、mixed143748。每seed每模型600步latent训练+200步微调、batch4；同seed完全相同source-batch调度。第二阶段每4步一次真D RGB backward，共每模型50次、全套600次；D权重全冻结。seed17/29/43，共9600 optimizer steps。第一阶段所有模型累计{stage1secs:.1f}s，第二阶段{stage2secs:.1f}s；原生缓存{cache['runtime_seconds']:.1f}s，最终评估{final['seconds']:.1f}s。
- 命令见 `../../experiments/h3_factorized/run_remote.sh`、参数与精确代码哈希见两阶段config/checkpoint；所有任务在H100 GPU7前台运行，session和状态见 `SESSION_REGISTRY.json`/`LIVE_STATE.json`。

## 实现与必要审计

24channel内容状态+12维LR推断appearance；四个conditional affine coupling提供F(c,a)/逆，且F(c,0)=c。LR-only update在canonical chart中写入，query只读；已见packet ID幂等。full以同内容跨外观、跨局部内容反事实交换监督；no_swap删此项；noninvertible采用独立encode/decode；mixed直接混合状态。共享更新head可更改受观测footprint，未受观测区域不直接写入。

修复了双零初始化死分支：chart末层小随机，仅appearance head为零。合成两步及真实train batch均确认appearance/chart/update有非零梯度和实际更新，记录在 `protocol_checks.json`/`real_batch_gradient_step*.json`。真D可微路径采用官方eager参考RoPE的局部context，原生/eager RGB maxabs={gradient['native_eager_rgb_maxabs']}，输入梯度RMS={gradient['input_gradient_rms']:.3g}，VAE权重梯度数={gradient['weight_gradients']}，peak CUDA={gradient['peak_cuda_bytes']/1e9:.2f}GB，见 `decoder_gradient_check.json`。

最终chart推断禁TF32，D仍真实FP16。最初val默认TF32下逆max达.0056；同checkpoint CPU严格FP32降至1.62e-5/RMS约1e-6，故精度行为必须显式记录。最终full heldout逆max={maxinverse:.3g}、均方根平均={rmsinverse:.3g}。所有heldout无可见证据query/重复packet均逐元素保持状态；这些协议也由控制模型满足，不能归作full独有性能贡献。

持久tensor成本每对象/256patch：content={state['content_state_bytes']}bytes，当前实现reference RGB={state['reference_state_bytes']}bytes，总{state['total_tensor_state_bytes']}bytes，另有packet ID元数据；不能只报内容168KiB而隐去参考图。

## Heldout RGB：原D与相同DC分别报告

表中是每source、seed、五个新证据事件的RGB PSNR平均(dB)，基线不重复计作三个独立seed。所有RGB误差对原始自然照片的监督目标计算，不以H3解码图充当GT；H3_HR_roundtrip仅是使用HR的表示上限参考，不是有效LR-only方法。DC对所有模型/基线相同：64步最小改变量affine/box Dykstra，远景全图4×、近景只实际中心128的2×观测；无观测query与反事实查询无伪LR约束。原始物理噪声测量重新采集后，56源缓存LR RGB全部bitwise一致；矩阵算子对原退化误差<3e-7。最终DC实际测量RMSE平均{rmsemean:.3g}、最大{rmsemax:.3g}。**DC是经典控制，不是本方法原创点。**

|方法|原生D|同预算DC|
|---|---:|---:|
{table}

|事件|full原D|full+DC|no_swap+DC|当前LR→H3 VAE+DC|
|---|---:|---:|---:|---:|
{eventtable}

|配对比较|平均差(dB)|95%探索区间|
|---|---:|---:|
{etable}

区间是4000次嵌套seed+循环source block(length2)配对bootstrap；只3seed/16来源且相邻episode复用donor，区间仍受相关性/小样本限制。不得把每帧或每个photometric变体当独立样本。逐seed均值和全部逐source数据在 `test_three_seed/paired_effects.json` 与CSV。

主要事件以来源为抽样单位再列：同一source/seed内部先平均两个lighting事件，再把三个训练seed在同一source内平均；胜率分母为16照片来源，**不是48个seed或独立视频**。下表是同预算DC后full减对应控制。区间为5000次source block bootstrap，条件于这三组已训练checkpoint；逐seed均差/胜率与16源配对值在 `source_paired_outcomes.json`/`source_paired_differences.csv`。

|事件：控制|配对差dB|来源胜数|来源95%探索区间|
|---|---:|---:|---:|
{source_table}

![原D比较]({TEST/'actual_H3_RGB_comparison.png'})

![相同DC比较]({TEST/'projected/actual_H3_RGB_comparison.png'})

## 机制诊断：训练到什么，缺口在哪里

单独oracle-HR-latent chart交换RGB：full={swapmeans['factorized_swap']:.3f}，no_swap={swapmeans['factorized_no_swap']:.3f}，noninvertible={swapmeans['noninvertible_swap']:.3f}，mixed={swapmeans['mixed_memory']:.3f}dB。这些使用HR latent与已知appearance作为**机制oracle**，不是有效LR-only方法。它能判断swap确实学习内容/外观的交换关系，但非可逆控制足以很强；原始架构的可逆性独立贡献须由heldout主表决定。

两种额外oracle严格分开：真appearance测试推断appearance的影响；仅初始HR身份后完全正常LR写入测试正确初始细节能否存活。它们都使用额外HR监督信息，只用于定位缺口，不进入有效方法排名，也没有基于它们重新选择模型。

|事件|正常full|真appearance oracle|仅初始HR身份oracle|
|---|---:|---:|---:|
{ortable}

完整记录 `error_source_oracles_rgb.csv`、`oracle_chart_swap_rgb.csv`、`LR_memory_neutral_query_rgb.csv`；未变外部区域单列outside PSNR，partial DC没有替它强制整图观测。报告应同时检查身份保持与真实内容更新，不能以“稳定不变”掩盖对新内容的迟滞。

真physical appearance的oracle反而没有改善，不能把瓶颈简单归因于appearance encoder；HR-chart的交换收益也不证明LR/HR共享了正确的canonical域。仅初始正确HR身份在第一次lighting有收益，但到return只剩少量收益，提示后续LR mixing会消耗已正确存在的细节。下一步只用val固定checkpoint逐步检验near写入及return覆盖，不用test选择新结构。

## 下一步最有价值的结构切口

当前核心把完整H3 code同时承担可观测粗结构、光照和持久细节，又以gated mixing写入新编码；VAE跨ROI的非局部响应与旧内容污染是需要解决的机制问题。优先测试**固定当前真实LR锚点+可逆appearance作用于持久细节残差**：z_read=z_LR+F(r_identity,a)。初始化残差来自冻结H3 DiT的一次真实生成proposal，后续只有新真实LR能改变内容；不把proposal当观测精度。共享相同经典DC，才能判断新架构是否同时保留可用旧细节并吸收新细节。这个修改是待实现候选，不能在本表中假装已经验证。

还需用decoder footprint/可信对应来约束内容写入，避免把像素ROI mask直接当作H3 latent可支持子空间。局部内容真的改变时应允许显式新episode并保留旧已证实身份；外观变化只读appearance。最有判别力的后续消融是同参数mixed残差、非可逆残差/no_swap、同一DiT调用预算、相同DC以及实际远→近→远视频。不要通过调大稳定性权重代替这个结构问题。

conditional coupling来自标准[Real NVP](https://arxiv.org/abs/1605.08803)；可逆递归本身也已有[Reversible RNN](https://arxiv.org/abs/1810.10999)，内容/外观分解已见[NeRF-W](https://arxiv.org/abs/2008.02268)。可发表贡献必须落在冻结视频生成latent里的可学习证据写入、持久细节身份与读出共同收益，而不是上述部件改名。

环境：Python{environment['python']}、Torch{environment['torch']}、CUDA{environment['cuda']}、Pillow{environment['Pillow']}、{environment['nvidia_smi']}。Comfy源码无.git，commit无法恢复；实际vae.py SHA256={environment['Comfy_VAE_code_sha256']}，完整预训练VAE SHA256={cache['vae_sha256']}。未改官方源码或权重。
'''
    (BASE/'factorized_report.md').write_text(report)
    print(BASE/'factorized_report.md')


if __name__=='__main__':main()
