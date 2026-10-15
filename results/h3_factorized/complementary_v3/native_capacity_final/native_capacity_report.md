# B3 互补观测的真实 H3 容量审计

最终公平性结论：对每个near单独施加同样的LR物理预处理，再执行原生E/D与相同支持读出，最后decoded atlas达到26.9930dB，比同样加强的last-near 25.1789dB高1.8141dB；batch atlas→E/D matched为26.9900dB，仅差-0.0030dB。最初约1.1dB差几乎全部由经典预处理解释，neutral已知几何下尚无明确必须学习的更新缺口。正面结果是强last-near之上的互补收益仍真实存在并被native表示保留。

互补观测增益能穿过真实原生 H3 E/D。8 个自然验证源、两种固定相反顺序、4 个重叠 near2× 足迹，neutral appearance、无遮挡、已知相机；没有训练或test访问，没有H3 DiT。当前far固定同一真实LR packet，隔离表示容量，不声称已经实现18步光照/遮挡/改变协议或真实运动视频。HR只用于模拟传感读数、评估和单列oracle，所有有效状态只由实际LR构造。

| 第4次near后的相同far DC | 全图PSNR | union | 旧覆盖独占区 | 新覆盖独占区 | 重叠区 |
|---|---:|---:|---:|---:|---:|
| 当前 LR→H3 VAE | 24.2112 | 23.9296 | 24.6461 | 26.4960 | 23.7113 |
| pixel last-near | 25.0794 | 25.4548 | 24.6930 | 30.1153 | 27.6927 |
| 经典 pixel atlas | 26.1395 | 27.7925 | 28.6358 | 30.1167 | 27.7691 |
| 完整历史线性 inverse | 26.1569 | 27.7676 | 28.6165 | 30.1239 | 27.7154 |
| H3 last-near full-code | 24.7065 | 24.7735 | 24.6485 | 28.4342 | 25.6986 |
| H3 last-near delta-code | 24.6997 | 24.7610 | 24.6476 | 28.3517 | 25.6453 |
| 逐packet H3 decoded atlas full | 25.2792 | 25.8636 | 26.6915 | 28.4301 | 25.7145 |
| 逐packet H3 decoded atlas delta | 25.2579 | 25.8191 | 26.6181 | 28.3449 | 25.6744 |
| 经典 atlas→H3 E/D | 25.8432 | 27.0744 | 27.9080 | 29.4302 | 26.9891 |
| 经典 inverse→H3 E/D | 25.8698 | 27.0708 | 27.9061 | 29.4269 | 26.9875 |
| atlas→H3 E/D，同历史支持读出 | 25.7996 | 26.9900 | 27.8310 | 29.4181 | 26.8961 |
| inverse→H3 E/D，同历史支持读出 | 25.7813 | 26.9414 | 27.7698 | 29.3619 | 26.8715 |
| HR→H3 E/D，仅oracle容量 | 28.1781 | 28.0249 | 28.8233 | 30.2595 | 28.0536 |
| 同物理预处理 H3 last-near full-code | 24.9271 | 25.1789 | 24.6493 | 29.4067 | 26.8533 |
| 同物理预处理 H3 last-near，完整D读出 | 24.9394 | 25.2044 | 24.6511 | 29.3919 | 26.9071 |
| 同物理预处理逐packet H3 decoded atlas | 25.8010 | 26.9930 | 27.8250 | 29.4066 | 26.9201 |

相同支持算子的匹配很关键：逐packet路径先分别E/D，再把各自历史near row-band残差合并；匹配控制先从全部真实LR融合，再E/D，最后施加完全相同的逐足迹平均row-band/current-far-null读出。两者都用同一当前LR锚点和DC，不把decoder输出加入真实观测ledger。另保留未加历史支持过滤的全E/D强控制，原始输出与DC后都在CSV。

| source内先平均两顺序，union配对差 | ΔdB | 源胜数 | 95% source bootstrap |
|---|---:|---:|---|
| H3_decoded_atlas_full − H3_last_fullcode | +1.0901 | 8/8 | [+0.9754, +1.2149] |
| H3_ED_pixel_inverse − H3_last_fullcode | +2.2973 | 8/8 | [+1.9985, +2.6794] |
| H3_ED_pixel_inverse_matched_support − H3_last_fullcode | +2.1679 | 8/8 | [+1.9158, +2.4815] |
| H3_ED_pixel_inverse_matched_support − H3_decoded_atlas_full | +1.0777 | 8/8 | [+0.9180, +1.2779] |
| H3_ED_pixel_atlas_matched_support − H3_decoded_atlas_full | +1.1264 | 8/8 | [+0.9671, +1.3307] |
| H3_decoded_atlas_full − H3_decoded_atlas_delta | +0.0445 | 7/8 | [+0.0192, +0.0753] |
| H3_ED_pixel_inverse − pixel_inverse | -0.6968 | 0/8 | [-0.8509, -0.5260] |
| H3_preconditioned_decoded_atlas_full − H3_preconditioned_last_fullcode | +1.8141 | 8/8 | [+1.6090, +2.0410] |
| H3_ED_pixel_atlas_matched_support − H3_preconditioned_decoded_atlas_full | -0.0030 | 1/8 | [-0.0148, +0.0149] |
| H3_ED_pixel_inverse_matched_support − H3_preconditioned_decoded_atlas_full | -0.0516 | 0/8 | [-0.0635, -0.0393] |
| H3_ED_pixel_inverse − H3_preconditioned_last_ED | +1.8664 | 8/8 | [+1.6384, +2.1357] |

经典 inverse 经H3 E/D相对其pixel结果丢失 0.6968dB union；但相对未经物理预处理的H3 last-near full仍有 2.2973dB互补收益。相同历史支持过滤后，先融合再E/D比未经预处理逐packet decoded atlas高 1.0777dB；但单near prefix1已存在约1.14dB，必须先排除bicubic near与物理反演预处理的混杂，不能据此声称非线性多观测学习缺口。

主匹配与强预处理控制实际256次native22编码+256次native22解码，71.36秒，GPU peak 7.277GB，native E均值128.03ms、D均值35.42ms；48步matrix-free inverse累计1.430秒/64prefix查询。初轮另52.15秒保留原样；全部物理审计合计123.51秒，没有优化步数。脚本按source/context逐次记录runtime_operations.csv，包含同步计时；评估复用确定性native输出缓存，不能把这些均值冒充整套流延迟。

物理核验：所有锚定路径prebox当前far观测差RMS最大3.26e-09。精度nativeE/D FP16，camera float64，DC float32，TF32关闭；同一观测4种footprint，与前轮CPU传感/噪声seed一致。22次重复图像只提供native时域context，不是22条观测。源级CI分母8，不把顺序/prefix当独立视频。

状态成本需区分payload、精度与临时缓存：one native code float32为172032B（原始FP16 encoder code可无损以86016B存储）；RGB residual sum+单通道count的float32 atlas为1048576B；4个64×64×3实际LR packet的float32 ledger只需196608B，或固定footprint的sum/count统计，不能刻意把经典baseline成本膨胀成无限ledger。所有方法另有当前LR49152B、4个packet/geometry的64B payload；实际float64审计数组为相应非native状态的两倍，详见state_cost.csv。共享camera算子655360B、当前native latent172032B、当前decoded RGB及native执行缓存另计，完整真实GPU峰值已报告。Python容器元数据不伪装成精确payload字节。decoder前融合的强控制仍需保留原始LR统计以继续更新，不能只报它查询时输出的一个code。

下一步先测冻结“当前D上下文→可见LR物理修订→E重编码”递归控制的累积遗忘；经典投影+重编码不命名原创。只有跨appearance/遮挡/实际内容变化出现旧区保留与新证据修订的可测缺口，才训练同预算anchor条件写入算子：旧detail、当前anchor latent、当前packet detail/geometry/confidence→canonical revision；真实D过去LR保留+当前可见LR纠错loss与无冲突顺序组合训练。与同输入GRU、非可逆/无swap、直接融合与完整LR inverse→E/D比较。此neutral容量审计本身不能证明可逆解耦，也不支持把经典预处理的摊销称原创。

完整源hash、输入packet hash、命令、模型SHA、版本、native次数/状态/时间/峰值在complete.json；rgb_metrics.csv含raw/DC每prefix的union/old/new/overlap及past-LR residual，state_cost.csv同时列float32部署payload与float64审计payload。输入packet.pt只在远端保留，明确不含HR。
