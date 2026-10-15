# B5 fixed2×2 result: observed-detail bypass works; learned historical correction does not

All four predeclared672-update arms completed on32TRAIN sources. The primary weights are last672 for all arms, not val-selected. VAL8 development sources were evaluated only after all training. Queries are averaged within source, then10000 paired source bootstraps; this is one trained seed17, not56 independent videos. These BSDS photographs were used previously elsewhere in the project; they are not a newly sealed confirmation set. The camera, appearance changes, occlusions and replacements are simulated. No sealed DIV2K/old test, real-motion evaluation or DiT is claimed.

The physical bridge recovers actual observed detail that the learned4-channel pathway was losing. Removing current-native-code conditioning from the neural readout does **not** activate learned history use. Neither learned skip1 arm improves its zero-training H3 decoded-atlas base. This is an information-pathway diagnosis, with no demonstrated new memory core or compression advantage.

## Fixed causal graph and successful acceptance

`x = B_S + H_common[ D(z_current + delta_theta) - D(z_current) ]`.

B0 is current LR-VAE; B1 is the common H3 decoded-photo atlas. B/H both use the same external causal LR IRLS nuisance/reliability controller and no arm's learned appearance. Learned appearance remains only in the neural residual/core. The bridge uses saved actual-LR-derived full24 native codes, not delta+proxy recomposition. The addition is in pixels, with no reencoding afterward. Every arm retains full24/proxy24 codes and computes identical phi; the skip-off ablation's extra phi work is explicit. A0 removes current native code only from neural readout; the physical frozen D still depends on the current code, and q/confidence metadata remains available.

ActualD acceptance passed: raw and common **single** finalDC zero-correction pixel maxabs0 for both bases, native-phi route maxabs0, A0 neural-anchor poison exact0 while A1 responds, future first-prefix/empty-evidence maxabs0. The initial zero final-head gives nonzero head gradient but exactly zero RGB-only encoder/core gradient; after one fixed Adam update both branches are nonzero, and frozen VAE weight gradients stay0. Eight discarded audit updates are separate. Active counts146192 vs146578 (A0 width74),0.264% difference; no dummy padding. Every VAL current-native route also has zero-base pixel maxabs0.

## Primary source-averaged DC union quality

| Current native code in neural readout | No bridge S0 | NativeD pixel bridge S1 |
|---|---:|---:|
| A1 present |24.12894|26.79023|
| A0 absent |24.07015|26.76572|

Zero-training current LR-VAE23.84270; zero-training H3 decoded atlas26.79536; stronger raw LR-photo atlas27.51398; fixed LR-only joint inverse25.75004. B4's separately trained current-image neural control is24.10618, factor-cache24.15909 and replay24.08248 under the same evaluation data/camera/DC. B4 weights have their original warm-start decoder and rounded delta input; those are external reference controls, not extra cells in the internally matched B5 factorial.

The average bridge effect is+2.67844dB (8/8 positive source differences,CI[2.46350,3.00023]). The average current-code conditioning effect is+0.04165 (8/8,CI[0.01963,0.07661]); the interaction is−0.03429 (CI[−0.05665,−0.01180]). These source effects are conditional on one trained seed and this development protocol, not architecture-wide claims.

Relative to each own zero-training base, A1S0 improves current LR by+0.28624dB (8/8,CI[0.14858,0.49158]); A0S0 by+0.22745 (8/8,CI[0.11696,0.39063]). A1S1 minus its H3 atlas is−0.00513 (1/8,CI[−0.00890,0.00054]); A0S1 is−0.02964 (0/8,CI[−0.05198,−0.01394]). A1S1 remains0.72375dB below the stronger raw-LR atlas on every source. The+2.68dB bypass gain must not be attributed to learning or the affine chart.

## Neural versus physical evidence dependence

For neural interventions, the physical B/H remains correct and fixed. Actual minus no-measured-delta quality is0.000124/0.000053/−0.000016/0.000003dB for A1S0/A0S0/A1S1/A0S1. Actual minus shuffled neural delta+proxy is−0.000280/−0.000012/0.000009/0.000011dB. Delayed neural q/weight freezes are likewise negligible. Removing readout z therefore reduces the shortcut's quality somewhat, but does not force dependence on measured history; the frozen nonlinear decoder can still adapt a nearly universal correction to current z. Do not call A0 completely current-content blind.

Conversely, hold each trained NN prediction fixed and change only the physical bridge: removing it reduces A1S1 quality2.93057dB (8/8,CI[2.62968,3.30859]); shuffling its observed phi reduces4.97050 (8/8,CI[4.15561,5.90472]). A0S1 gives the same pattern2.92587/4.96389dB. The pipeline now uses true past evidence through the established atlas, while the learned residual/action still contributes no independent evidence benefit. Numerical1e-5dB neural effects are not meaningful improvements despite an occasional narrow bootstrap interval.

The preceding B4 staged diagnosis is archived separately: packet h RMS.1570 versus across-source std.00221; delta removal changes h.00269, but swapping neural anchor changes readout much more. The fixed single-TRAIN success28.2840dB used a different diagnostic known-q/one-packet setting and is not pooled here. The new factorial confirms that removing only the neural anchor condition is insufficient to repair multi-source evidence learning.

## Conditions, unchanged areas and no-observation query


### union_psnr

| Condition | A1S0 | A0S0 | A1S1 | A0S1 | H3 atlas | LR atlas |
|---|---:|---:|---:|---:|---:|---:|
|neutral_return|24.19754|24.14012|26.84008|26.80656|26.84374|27.56836|
|appearance_cue|24.35889|24.29640|27.04378|27.01485|27.04813|27.77390|
|true_content_change|23.71541|23.66078|26.36006|26.35120|26.36782|27.06971|

### old_unmodified_psnr

| Condition | A1S0 | A0S0 | A1S1 | A0S1 | H3 atlas | LR atlas |
|---|---:|---:|---:|---:|---:|---:|
|neutral_return|24.19754|24.14012|26.84008|26.80656|26.84374|27.56836|
|appearance_cue|24.35889|24.29640|27.04378|27.01485|27.04813|27.77390|
|true_content_change|24.32817|24.26212|27.05540|27.03959|27.06877|27.77271|

### new_detail_psnr

| Condition | A1S0 | A0S0 | A1S1 | A0S1 | H3 atlas | LR atlas |
|---|---:|---:|---:|---:|---:|---:|
|neutral_return|24.85072|24.81684|27.68484|27.64805|27.69371|28.40522|
|appearance_cue|25.20915|25.17096|28.12713|28.09315|28.13975|28.86003|
|true_content_change|22.33492|22.26815|24.42015|24.45800|24.44904|25.04821|

### overlap_psnr

| Condition | A1S0 | A0S0 | A1S1 | A0S1 | H3 atlas | LR atlas |
|---|---:|---:|---:|---:|---:|---:|
|neutral_return|23.50832|23.42127|26.18469|26.15452|26.18821|26.97330|
|appearance_cue|23.88793|23.79759|26.59061|26.56200|26.59488|27.36521|
|true_content_change|23.63755|23.57147|26.26599|26.25891|26.27329|26.96487|

Complete raw/DC source-level tables, confidence intervals and comparisons are in summary_seed17/*.csv. All condition/region rows were retained; union is the predeclared primary. Raw union quality shows the same pattern: H3 atlas25.64364, A1S1 25.63586, A0S1 25.61779, A1S0 23.45109, A0S0 23.39650, current23.16904.

No-observation query15 uses actual retained far13 as anchor, admits near14, applies **no** invented current-LR DC, and remains read-only. Union PSNR current22.71678, A1S0 22.94729, A0S0 22.89842, A1S1 25.05710, A0S1 25.05363, H3 atlas25.06722 and LR atlas25.57653. This again attributes the useful history to the physical bridge.

## Full state, timing and revision accounting

Actual incremental prefix inference was run on one fixed TRAIN episode with the final weights, not new samples. Owned FP16 native/proxy buffers store only arrived packets; duplicate ID insertion is idempotent. Vectorized versus incremental neural outputs are exactly0 maxabs for all four. Explicit factor withdrawal agrees with fresh-prefix reconstruction within2.27e-6 and restore within1.44e-6; repeat reads do not write. These algebraic checks establish implementation correctness, not visual benefits of a learned rebase.

| At8 packets /2 known groups | Bytes |
|---|---:|
| Full24 native FP16 archive |688128|
| Proxy24 native FP16 archive |688128|
|4ch factors/weights/group moments/cached gains/canonical/IDs |372000|
| Reference LR / packet LR descriptors / group metadata controller |55440|
| Current LR / native anchor / decoded RGB buffer |1007616|
| Persistent state subtotal, excluding model/cameras |2811312|
| Common camera matrices |524288|
| A1/A0 adapter FP32 weights |584896/586440|
| Frozen VAE weights |5207737968|
| Per-query decoded phi scratch at8 |6291456|
| One bridge RGB buffer |786432|
| Actual unoptimized helper's all7 bridge stack scratch |5505024|

Python allocator overhead is excluded. Some derivable uniform runtime buffers are retained and counted. The actual helper computes all7 bridge scratch rows at each query; only the current prefix-valid row is read. The per-query full/proxy source archive is real, not a four-channel compressed-state claim. Raw LR archive393216B plus the same controller is much smaller than this bridge-plus-neural evidence payload. No storage advantage is shown.

Holding group count2 fixed, the logical persistent subtotal grows as1191152+202520N bytes: N8/32/128/512 gives2811312/7671792/27113712/104881392B, before model/cameras. Values above8 are complexity arithmetic, not additional observations or actual dataset sizes. Decoded-phi scratch grows786432N B. If retaining decoded phi between queries to avoid repeatedD, its6.29MB at8 is additional persistent storage; do not charge zero compute and also omit this memory.

Actual prefix timing medians: whole neural state update0.745ms, neural query0.663ms, one-factor withdrawal0.825ms, fixed two-group restore0.781ms; one phi nativeD33.386ms, correctionD+H33.990ms, finalDC64 5.083ms. The unoptimized all7 fusion helper is16.385ms/query; withdrawal fusion with phi already cached16.188ms, without E/D. Native-only ledger policy recomputes each arrived phi per query: the audit made124phiD+28correctionD=152D,0E. Skip-off arms computed the same phi for budget matching, but their minimal implementation could omit it. Overall pixel read is linear in active packet count; the earlier closed two-group chart action alone being constant in N does not make this renderer constant-cost.

Main physical training:2688 actualD backward updates,482.501s, peak10814982656B; discarded acceptance8 updates/2.904s separately. TRAIN phi256D/0E in10.256s; VAL phi64D/0E separately. Evaluation60.088s with1208D/0E,3136raw/DC metric rows, peak5645481984B; prefix audit6.373s/152D/0E. The native-context22copies are codec context only, not22independent observations. Every actualD is the frozen native H3 FP16 decoder with TF32 disabled; eager autograd only scopes the supervised backward path. There is no new DiT call.

Foreground session70083/PID1381677 completed exit0; evaluation+runtime76701 also exit0. Afterward a fresh check81711 confirmed GPU7 used1MiB/free81110MiB/util0, and it was released. There was no training restart, val-selected checkpoint, loss-weight sweep, or early stop. Local/remote weights0/224/448/672, hashes, logs and metrics are retained. Native phi .pt files remain remote; their manifests/hashes are local. Main training source snapshot/protocol was frozen before launch; later evaluation/runtime scripts are archived separately.

## Boundary and next use

This protocol deliberately keeps the near-row historical band fixed. Independent H_band diagnosis by A shows that it excludes a substantial component of clean target detail in the near-camera nullspace. That is a distinct representational restriction, not an explanation that rescues a new method claim from these results. B5 does not test unrestricted generative high-frequency state or a pre-cue conditional codec.

The useful handoff is the observed full24/native/proxy/RGBphi cache and verified common camera/bridge contract for a genuinely different causal representation. No further training of this architecture is warranted by this pilot. A new conditional codec hypothesis must preserve genuinely observed information, show a meaningful history intervention, and measure unseen high-frequency/private detail without relying on a hard band that prohibits it.

Classical projection, residual learning and model-based readout are established, including [Deep Null Space Learning](https://arxiv.org/abs/1806.06137), [RegNets](https://arxiv.org/abs/1812.00965) and [MoDL](https://arxiv.org/abs/1712.02862). The affine chart and factor-cache/replay mechanisms also overlap established equivariant/set/recursive inference. This report does not relabel them as original.
