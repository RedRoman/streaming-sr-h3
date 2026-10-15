# B2 current-LR anchored persistent supported detail

Frozen B1 val diagnostics established that near supplies real information, then far absolute-latent writes destroy62.6% of its operator-band MSE gain. Fixed current-LR anchor+history-support readout improves return by.457dB after the same DC without changing far observation RMSE. This is the reason for B2, not hyperparameter tuning.

B2 has the same four charts, parameter budget,32train/8val sources,600+200 Adam steps, batch4, LR.002/.0005, appearance weight10, counterfactual.5 and RGB5 every4 steps. No extra training on the16 heldout sources. Initial proposal is explicitly zero; future fixed H3-DiT proposal has a separate non-evidence interface, unused here. HR never initializes actual state.

The persistent state is only residual detail, initialized zero. Input detail is E(actual near LR)-E(its LR-only far proxy), not HR residual. Finer near packets can update a global latent residual, with sensor footprint as a condition; far updates appearance/current anchor but never overwrites this known fine band. A revisited near can overwrite changed content; reveal is a sensor scale, not packet novelty/counts.

Centered conditional residual chart T(r,a)=F(r,a)-F(0,a) ensures every model has zero readout at r=0. The inverse is correspondingly centered. Decoder readout is D(z_currentLR)+N_current P_historyNear N_current[D(z_currentLR+T(r,a))-D(z_currentLR)]. Current near supports all the historical band, so its added residual is mathematically0; far can read it. All four models share this classical observation quotient and final DC. Neither projection, content/appearance split, nor conditional coupling alone is claimed novel.

Stage1 supervision uses measured-detail latent labels E(noisefree near2x view)-E(noisefree far4x view), generated from the same training photo labels; these labels never enter state initialization or inference. RGB stage2 uses actual frozen H3 D. Cached baseline D(currentLR) is stored float32 bitwise to ensure zero-residual equality. Strict FP32 charts/projectors disable TF32; native D is FP16.

Required audits: actual train-cache zero residual equals native LR-VAE RGB; near read cannot double-add historical band; far measurement derivative w.r.t detail is zero while old near measurement derivative is nonzero; far/query/duplicate packets leave residual state unchanged; both chart/appearance get nonzero gradients after two steps.

Strong controls include current LR-VAE and both fixed B1 readout probes: retained post-near memory and direct last-near-code transport, each current-LR anchored with the same support/null operator and same DC. These use fixed B1 learned charts, require zero ADDITIONAL training, and are not falsely described as entirely unlearned pipelines. B2 must not attribute the quotient's gain to learned reversibility.
