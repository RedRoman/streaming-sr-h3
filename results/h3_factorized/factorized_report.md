# B：真实 MiniMax-H3 可逆内容/外观状态实验

已实际完成三seed训练、独立来源heldout真H3解码评估及统一经典DC对照。当前仍是**候选表示架构**；不能仅凭可逆结构或counterfactual训练把它称为原创且有效的streaming SR核心。这里冻结实际H3 E/D，尚未使用H3 DiT生成先验，因此也不代表完整H3生成式视频SR效果。

可量化的积极信号是：相同DC的中性回访，full比no_swap高0.093dB，13/16来源获益。但full与非可逆控制相当，整体仍输当前LR→H3 VAE+DC；可逆性的独立性能贡献及有效SR核心均未建立。

## 可复现数据与预算

- 官方BSDS500来源划分32train/8val/16test；同split donor构造局部内容替换，来源/哈希/物理扰动seed见 `cache_pilot/manifest.json`。56源、每源6个HR监督状态/6个LR观测，共672次真实native256×22 E；22帧是同一已到达LR锚点复制的VAE上下文，不是未来帧或22份证据。
- 八步：中性远景A→外观1远景A→外观1近景揭示A→无可见证据query→中性回访A→近景局部内容换B→外观2远景B→同packet副本。退化为Gaussianσ.6/5tap反射+Torch bicubic antialias；远4×，中心128近2×，独立LR噪声σ.003。它是自然照片上的受控成像实验，**不是真实运动/长间隔回访数据**。
- 训练只加载train；val选择checkpoint；16test在架构、预算和分析协议固定后统一读取。HR/真实appearance只进入监督或单独标明的oracle诊断，正式observe/query仅用已到达LR、sensor footprint与packet ID。原始生成结果从不追加为真实观测。
- 四模型参数：full144748、no_swap144748、noninvertible144614、mixed143748。每seed每模型600步latent训练+200步微调、batch4；同seed完全相同source-batch调度。第二阶段每4步一次真D RGB backward，共每模型50次、全套600次；D权重全冻结。seed17/29/43，共9600 optimizer steps。第一阶段所有模型累计634.8s，第二阶段343.8s；原生缓存95.5s，最终评估101.1s。
- 命令见 `../../experiments/h3_factorized/run_remote.sh`、参数与精确代码哈希见两阶段config/checkpoint；所有任务在H100 GPU7前台运行，session和状态见 `SESSION_REGISTRY.json`/`LIVE_STATE.json`。

## 实现与必要审计

24channel内容状态+12维LR推断appearance；四个conditional affine coupling提供F(c,a)/逆，且F(c,0)=c。LR-only update在canonical chart中写入，query只读；已见packet ID幂等。full以同内容跨外观、跨局部内容反事实交换监督；no_swap删此项；noninvertible采用独立encode/decode；mixed直接混合状态。共享更新head可更改受观测footprint，未受观测区域不直接写入。

修复了双零初始化死分支：chart末层小随机，仅appearance head为零。合成两步及真实train batch均确认appearance/chart/update有非零梯度和实际更新，记录在 `protocol_checks.json`/`real_batch_gradient_step*.json`。真D可微路径采用官方eager参考RoPE的局部context，原生/eager RGB maxabs=0.0，输入梯度RMS=2.47e-06，VAE权重梯度数=0，peak CUDA=10.71GB，见 `decoder_gradient_check.json`。

最终chart推断禁TF32，D仍真实FP16。最初val默认TF32下逆max达.0056；同checkpoint CPU严格FP32降至1.62e-5/RMS约1e-6，故精度行为必须显式记录。最终full heldout逆max=4.39e-05、均方根平均=2.2e-06。所有heldout无可见证据query/重复packet均逐元素保持状态；这些协议也由控制模型满足，不能归作full独有性能贡献。

持久tensor成本每对象/256patch：content=172032bytes，当前实现reference RGB=786432bytes，总958464bytes，另有packet ID元数据；不能只报内容168KiB而隐去参考图。

## Heldout RGB：原D与相同DC分别报告

表中是每source、seed、五个新证据事件的RGB PSNR平均(dB)，基线不重复计作三个独立seed。所有RGB误差对原始自然照片的监督目标计算，不以H3解码图充当GT；H3_HR_roundtrip仅是使用HR的表示上限参考，不是有效LR-only方法。DC对所有模型/基线相同：64步最小改变量affine/box Dykstra，远景全图4×、近景只实际中心128的2×观测；无观测query与反事实查询无伪LR约束。原始物理噪声测量重新采集后，56源缓存LR RGB全部bitwise一致；矩阵算子对原退化误差<3e-7。最终DC实际测量RMSE平均3.47e-05、最大0.000263。**DC是经典控制，不是本方法原创点。**

|方法|原生D|同预算DC|
|---|---:|---:|
|input_bicubic|25.268|25.721|
|H3_current_LR|25.032|25.639|
|factorized_swap|23.957|24.761|
|factorized_no_swap|23.979|24.785|
|noninvertible_swap|23.988|24.815|
|mixed_memory|23.445|24.557|
|H3_HR_roundtrip|29.069|29.460|

|事件|full原D|full+DC|no_swap+DC|当前LR→H3 VAE+DC|
|---|---:|---:|---:|---:|
|appearance_A1|23.892|24.765|24.828|25.651|
|reveal_A1|24.423|24.902|25.098|25.923|
|return_A0|24.401|25.083|24.990|25.642|
|content_replace_B0|23.095|24.328|24.356|25.699|
|appearance_B2|23.974|24.724|24.653|25.279|

|配对比较|平均差(dB)|95%探索区间|
|---|---:|---:|
|raw: full−factorized_no_swap|-0.022|[-0.150,+0.112]|
|raw: full−noninvertible_swap|-0.031|[-0.133,+0.075]|
|raw: full−mixed_memory|+0.512|[+0.236,+0.893]|
|DC: full−factorized_no_swap|-0.025|[-0.103,+0.060]|
|DC: full−noninvertible_swap|-0.055|[-0.146,+0.028]|
|DC: full−mixed_memory|+0.203|[+0.086,+0.363]|

区间是4000次嵌套seed+循环source block(length2)配对bootstrap；只3seed/16来源且相邻episode复用donor，区间仍受相关性/小样本限制。不得把每帧或每个photometric变体当独立样本。逐seed均值和全部逐source数据在 `test_three_seed/paired_effects.json` 与CSV。

主要事件以来源为抽样单位再列：同一source/seed内部先平均两个lighting事件，再把三个训练seed在同一source内平均；胜率分母为16照片来源，**不是48个seed或独立视频**。下表是同预算DC后full减对应控制。区间为5000次source block bootstrap，条件于这三组已训练checkpoint；逐seed均差/胜率与16源配对值在 `source_paired_outcomes.json`/`source_paired_differences.csv`。

|事件：控制|配对差dB|来源胜数|来源95%探索区间|
|---|---:|---:|---:|
|return：factorized_no_swap|+0.093|13/16|[+0.056,+0.131]|
|return：noninvertible_swap|-0.001|6/16|[-0.025,+0.028]|
|return：mixed_memory|+0.158|16/16|[+0.120,+0.198]|
|return：H3_current_LR|-0.559|0/16|[-0.743,-0.398]|
|lighting：factorized_no_swap|+0.004|8/16|[-0.015,+0.024]|
|lighting：noninvertible_swap|-0.062|3/16|[-0.105,-0.025]|
|lighting：mixed_memory|+0.063|11/16|[-0.003,+0.122]|
|lighting：H3_current_LR|-0.721|0/16|[-0.942,-0.532]|
|true_replacement：factorized_no_swap|-0.028|9/16|[-0.117,+0.040]|
|true_replacement：noninvertible_swap|+0.009|11/16|[-0.110,+0.090]|
|true_replacement：mixed_memory|+0.249|16/16|[+0.184,+0.324]|
|true_replacement：H3_current_LR|-1.372|0/16|[-1.671,-1.031]|

![原D比较](/Users/hezihao/Documents/New project/streaming_sr_20261015/results/h3_factorized/test_three_seed/actual_H3_RGB_comparison.png)

![相同DC比较](/Users/hezihao/Documents/New project/streaming_sr_20261015/results/h3_factorized/test_three_seed/projected/actual_H3_RGB_comparison.png)

## 机制诊断：训练到什么，缺口在哪里

单独oracle-HR-latent chart交换RGB：full=23.674，no_swap=19.134，noninvertible=23.724，mixed=21.512dB。这些使用HR latent与已知appearance作为**机制oracle**，不是有效LR-only方法。它能判断swap确实学习内容/外观的交换关系，但非可逆控制足以很强；原始架构的可逆性独立贡献须由heldout主表决定。

两种额外oracle严格分开：真appearance测试推断appearance的影响；仅初始HR身份后完全正常LR写入测试正确初始细节能否存活。它们都使用额外HR监督信息，只用于定位缺口，不进入有效方法排名，也没有基于它们重新选择模型。

|事件|正常full|真appearance oracle|仅初始HR身份oracle|
|---|---:|---:|---:|
|appearance_A1|23.892|23.488|24.687|
|reveal_A1|24.423|23.965|25.064|
|return_A0|24.401|24.157|24.583|
|content_replace_B0|23.095|22.969|23.258|
|appearance_B2|23.974|23.628|24.024|

完整记录 `error_source_oracles_rgb.csv`、`oracle_chart_swap_rgb.csv`、`LR_memory_neutral_query_rgb.csv`；未变外部区域单列outside PSNR，partial DC没有替它强制整图观测。报告应同时检查身份保持与真实内容更新，不能以“稳定不变”掩盖对新内容的迟滞。

真physical appearance的oracle反而没有改善，不能把瓶颈简单归因于appearance encoder；HR-chart的交换收益也不证明LR/HR共享了正确的canonical域。仅初始正确HR身份在第一次lighting有收益，但到return只剩少量收益，提示后续LR mixing会消耗已正确存在的细节。下一步只用val固定checkpoint逐步检验near写入及return覆盖，不用test选择新结构。

## 下一步最有价值的结构切口

当前核心把完整H3 code同时承担可观测粗结构、光照和持久细节，又以gated mixing写入新编码；VAE跨ROI的非局部响应与旧内容污染是需要解决的机制问题。优先测试**固定当前真实LR锚点+可逆appearance作用于持久细节残差**：z_read=z_LR+F(r_identity,a)。初始化残差来自冻结H3 DiT的一次真实生成proposal，后续只有新真实LR能改变内容；不把proposal当观测精度。共享相同经典DC，才能判断新架构是否同时保留可用旧细节并吸收新细节。这个修改是待实现候选，不能在本表中假装已经验证。

还需用decoder footprint/可信对应来约束内容写入，避免把像素ROI mask直接当作H3 latent可支持子空间。局部内容真的改变时应允许显式新episode并保留旧已证实身份；外观变化只读appearance。最有判别力的后续消融是同参数mixed残差、非可逆残差/no_swap、同一DiT调用预算、相同DC以及实际远→近→远视频。不要通过调大稳定性权重代替这个结构问题。

conditional coupling来自标准[Real NVP](https://arxiv.org/abs/1605.08803)；可逆递归本身也已有[Reversible RNN](https://arxiv.org/abs/1810.10999)，内容/外观分解已见[NeRF-W](https://arxiv.org/abs/2008.02268)。可发表贡献必须落在冻结视频生成latent里的可学习证据写入、持久细节身份与读出共同收益，而不是上述部件改名。

环境：Python3.13.12、Torch2.10.0+cu128、CUDA12.8、Pillow12.1.1、NVIDIA H100 80GB HBM3, 560.35.03, 81559 MiB。Comfy源码无.git，commit无法恢复；实际vae.py SHA256=93c0639bcb8d40524db0594db72ec3318291b0c6c76ff4492645129ea68a0884，完整预训练VAE SHA256=7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522。未改官方源码或权重。
