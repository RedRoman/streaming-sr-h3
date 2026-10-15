# Frozen H3 sampling-inner DC: 8 DAVIS sources

Complete: 16 cases from 8 independent existing DAVIS sources, each in 2 spatial contexts; 1056 output PNG hashes verified. Four train sources (bear, bus, car-turn, flamingo), two val sources (cows, dance-twirl), and two previously used project-test sources (bmx-trees, soapbox). Project-test is not a sealed holdout. Sealed DIV2K was never accessed. No training or result-dependent parameter selection.

Frozen BF16 MiniMax-H3 base, real neutral context from the NVFP4-AWQ text encoder, canonical FP16 VAE, seed 101, CFG 1, and the official 50 sigma points / 49 Euler steps. At the three preregistered indices 28, 45, and 48, apply the unit residual to post-CFG clean z0: E(P25(D(z0))) − E(D(z0)). Euler, audio carry, and guide remain unchanged, and the base has no gradients. This is a DDNM/FlowDPS-inspired engineering control, not an original memory method or exact paper reproduction.

The four val cases reuse SHA-identical prior outputs (264 PNGs), with exactly equal original numeric metrics. CPU metrics were recomputed because absolute/relative TORCH_HOME strings changed the cache key; val GPU inference was not repeated. The new 12 base and 12 inner-DC trajectories and 4 LR-only project-test guides ran after protocol, source, and input hashes were frozen. Matched VAE-only uses the same official guide latents.

The 256 crop [296,112,256,256] and native 832×480 crop [8,0,832,480] cover different fields of view. This is not a pure resolution ablation. Frame means are averaged per case, then equally across the 8 sources within each context; 16 cases do not count as 16 independent observations. Split-specific means and the shared 6-source train+val subset for external-baseline comparison are in sourcewise_metrics.json. Raw precedes final DC25; the _dc25 suffix applies the same final projection to each control. sampling_dc_dc25 means inner DC plus final DC25.

|Context|Method|Sources|PSNR dB|LPIPS Alex|SSIM|
|---|---|---:|---:|---:|---:|
|256|bicubic|8|25.2242|0.42988|0.7013|
|256|bicubic_dc25|8|25.8351|0.38117|0.7296|
|256|sampling_dc|8|24.2285|0.35988|0.6601|
|256|sampling_dc_dc25|8|25.9361|0.30134|0.7257|
|256|unadapted|8|20.5222|0.42720|0.5541|
|256|unadapted_dc25|8|25.8071|0.32697|0.7223|
|256|vae_only|8|24.6654|0.40736|0.6705|
|256|vae_only_dc25|8|26.0714|0.34095|0.7361|
|native480|bicubic|8|26.3454|0.46428|0.7002|
|native480|bicubic_dc25|8|26.9334|0.41929|0.7295|
|native480|sampling_dc|8|26.2287|0.35889|0.7024|
|native480|sampling_dc_dc25|8|27.2457|0.32654|0.7374|
|native480|unadapted|8|24.0790|0.43339|0.6488|
|native480|unadapted_dc25|8|27.0948|0.36273|0.7317|
|native480|vae_only|8|25.8331|0.43171|0.6758|
|native480|vae_only_dc25|8|27.1608|0.37225|0.7356|

256: vs base DC25, PSNR improves 8/8, LPIPS improves 8/8; vs VAE DC25, PSNR improves 2/8, LPIPS improves 8/8.
native480: vs base DC25, PSNR improves 7/8, LPIPS improves 8/8; vs VAE DC25, PSNR improves 7/8, LPIPS improves 8/8.

|Source|Split|Context|ΔPSNR vs base DC25|ΔLPIPS vs base DC25|ΔPSNR vs VAE DC25|ΔLPIPS vs VAE DC25|
|---|---|---|---:|---:|---:|---:|
|cows|val|256|+0.2048|-0.02987|-0.2045|-0.05654|
|cows|val|native480|+0.1455|-0.04704|+0.1237|-0.05461|
|dance-twirl|val|256|+0.0234|-0.03144|-0.4473|-0.03113|
|dance-twirl|val|native480|+0.1080|-0.03240|+0.0706|-0.03775|
|bear|train|256|+0.1378|-0.03423|-0.0684|-0.07276|
|bear|train|native480|+0.2884|-0.03273|+0.0184|-0.06622|
|bus|train|256|+0.2983|-0.01782|+0.1140|-0.03087|
|bus|train|native480|+0.1947|-0.03223|+0.2212|-0.03959|
|car-turn|train|256|+0.0187|-0.03118|+0.0495|-0.05651|
|car-turn|train|native480|+0.0342|-0.04739|+0.0594|-0.05281|
|flamingo|train|256|+0.1274|-0.01924|-0.2899|-0.02067|
|flamingo|train|native480|+0.3012|-0.03637|+0.1198|-0.04587|
|bmx-trees|project-test previously used|256|+0.0955|-0.02025|-0.1909|-0.01885|
|bmx-trees|project-test previously used|native480|-0.0119|-0.01750|-0.0903|-0.02053|
|soapbox|project-test previously used|256|+0.1267|-0.02109|-0.0448|-0.02961|
|soapbox|project-test previously used|native480|+0.1468|-0.04382|+0.1564|-0.04830|

New extension: 1176 DiT NFE + 76 E (4 guide / 72 inner) + 72 D (36 inner / 36 final). The complete 16-case dataset costs 1568 DiT NFE + 112 E (16 guide / 96 inner) + 96 D (48 inner / 48 final), excluding the prior 98-NFE parity guard. Extension wall time 824.14s, summed sampling 700.11s, CPU metric process 854.09s overlaps the GPU and includes waiting. Peak allocated memory 47.483GB. Exact per-case, projection, E, and D costs are in cost_audit.json. Equal 49 DiT calls does not mean equal compute. A nominal 22-frame block needs a 21-frame wait (.875s at 24fps) plus compute; these are offline timings, not measured live ingestion.

Protocol SHA `1bd8992498b6cd1cfe4852e290032048bef25bff59a118cfa2b2ece7cd4f22b7`. Input SHA `73fdaf9c77c4a89b9fa6e4ae98fd7a98db320ee975e3709a24045b7a92a47247`. Source manifest and metric-loop equivalence are archived. The first preflight failed before model loading or GPU work because an import shadowed a Path variable. The variable-name fix and failed attempt are retained in preflight_attempt0; parameters and math never changed.
