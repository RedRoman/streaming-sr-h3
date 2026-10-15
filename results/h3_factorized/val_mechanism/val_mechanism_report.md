# 固定B1 checkpoint的val机制诊断

仅8个val来源，固定seed17/29/43；source内部先平均seed，未用test选新结构。真实native256×22 FP16 H3 D，严格FP32 chart/观测算子。

近景确实提供可用新细节：8/8 val source的近景支持/far-null contrast误差有改善。固定同一已观测near appearance query时，真实near测量RMSE为.04522→.02870→.05028（pre-near/post-near/return），并非只测latent L1。定义C=Nfar Pnear_emb Nfar为线性contrast（不是正交projector或新算法）；实测Afar C和C Pfar近零。HR仅用于评估contrast误差，不进入写入或下面的解法probe。

contrast MSE为0.000863445→0.000367419→0.000678112；return保留近景MSE增益37.4%，丢失62.6%。这定位了真实细节被后续far update覆盖的机制。当前更新gate均值.692；observation注入项本身将contrast RMSE由.01894恶化到.02544，校正项未救回，完整逐项干预在CSV。

保持相同当前return DC：

|读出|PSNR dB|
|---|---:|
|normal_return|23.8507|
|current_LR_VAE_return|24.2119|
|classical_anchor_plus_supported_memory_probe|24.6689|
|classical_anchor_plus_direct_near_code_probe|24.7550|

current-LR anchor+支持残差probe在所有模型/算子冻结、无额外训练下改善，far测量RMSE仍与当前LR-VAE逐数一致(.01739924)。direct-near-code probe复用B1已训练的appearance/chart，只去掉递归写入；“零训练”是零额外训练，不是声称全流程没有学习。两种probe都是B2必须保留的强控制，不能把经典残差注入/商空间控制收益归给可逆chart。

B2计划：当前LR latent及其nativeD为锚点，历史状态只存真实near LR与其LR-only far proxy的编码差；初始proposal为零。读出为D(zLR)+C_t[D(zLR+T(r,a))-D(zLR)]，同footprint far不能重写near细节。actual sensor footprint只作为latent更新条件，读出支持由真实pixel观测算子限定，避免pixel mask等同latent支持的错误。四相同预算控制及上述零额外训练强控制共同评价；appearance权重仍10。可逆/内容外观分解/经典投影都不是独立原创claim。
