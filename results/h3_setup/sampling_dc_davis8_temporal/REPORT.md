# Fixed LR-flow temporal residual audit

All 8 existing DAVIS sources ×2 FOV contexts, 22 frames each. Base, matched VAE-only and sampling-inner DC are evaluated at raw output and after identical final DC25. No DiT, decoder, training, parameter search, or GPU work. The original 8-source archive stays unchanged.

For every pair t=1..21, use the root evaluate.py warp and confidence upsampling verbatim: bilinear LR flow scaled to output size, zero padding, align_corners=False; confidence ≥0.5. The score is RMSE of (output_t−warp(output_t−1)) − (HR_t−warp(HR_t−1)) within that mask. All inputs and outputs are RGB float32 in [0,1]. DC25 internally uses float64 and returns float32, with no extra PNG rounding after projection.

Flow and confidence use observed LR only. HR appears only in evaluation. The 15 existing root caches are reused by SHA; bmx native480 was copied from the existing local cache, and only missing soapbox native480 was computed using unchanged cache_motion.py and fixed DIS parameters (direct interval17, photo sigma0.05, cycle sigma1). Every flow provenance LR prefix is SHA-matched to the exact case. Pair RMSE is averaged within each source, then equally across the 8 sources per context. Source/split/common-trainval6 aggregates and all 2016 pair rows are retained. Empty-mask handling and valid pair counts are explicit.

This is flow-relative error after subtracting true motion/appearance residual. It is not a direct flicker score, and flow confidence restricts its coverage. Two spatial contexts are not two independent sources. Project-test was used earlier and is not a sealed holdout; sealed DIV2K remains untouched.

|Context|Method|Sources|Temporal excess RMSE ↓|Mask coverage|Valid pairs|
|---|---|---:|---:|---:|---:|
|256|sampling_dc|8|0.0445824|84.2503%|168/168|
|256|sampling_dc_dc25|8|0.0368137|84.2503%|168/168|
|256|unadapted|8|0.0629522|84.2503%|168/168|
|256|unadapted_dc25|8|0.0372862|84.2503%|168/168|
|256|vae_only|8|0.0366739|84.2503%|168/168|
|256|vae_only_dc25|8|0.0350271|84.2503%|168/168|
|native480|sampling_dc|8|0.0311804|94.4905%|168/168|
|native480|sampling_dc_dc25|8|0.0285458|94.4905%|168/168|
|native480|unadapted|8|0.0368106|94.4905%|168/168|
|native480|unadapted_dc25|8|0.0292499|94.4905%|168/168|
|native480|vae_only|8|0.0282608|94.4905%|168/168|
|native480|vae_only_dc25|8|0.0284098|94.4905%|168/168|

256, raw, innerDC versus unadapted: mean temporal-RMSE change -0.0183698, decreases on 8/8 sources.

256, raw, innerDC versus vae_only: mean temporal-RMSE change +0.0079086, decreases on 0/8 sources.

256, final DC25, innerDC versus unadapted: mean temporal-RMSE change -0.0004726, decreases on 5/8 sources.

256, final DC25, innerDC versus vae_only: mean temporal-RMSE change +0.0017866, decreases on 0/8 sources.

native480, raw, innerDC versus unadapted: mean temporal-RMSE change -0.0056301, decreases on 7/8 sources.

native480, raw, innerDC versus vae_only: mean temporal-RMSE change +0.0029196, decreases on 0/8 sources.

native480, final DC25, innerDC versus unadapted: mean temporal-RMSE change -0.0007041, decreases on 6/8 sources.

native480, final DC25, innerDC versus vae_only: mean temporal-RMSE change +0.0001361, decreases on 4/8 sources.

CPU process wall time 271.537s, including initial flow setup. The metric does not use LPIPS; the source table joins previously frozen LPIPS values only for comparison. Full metric source and flow hashes are in protocol.json and flow_manifest.json.

Initial preflight attempts found one existing flow only local and the canonical cache helper absent remotely; these files were copied unchanged and the failed logs retained. No metric evaluation ran before the successful protocol freeze.
