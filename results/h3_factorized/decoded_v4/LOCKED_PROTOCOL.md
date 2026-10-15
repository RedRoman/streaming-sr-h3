# Locked B3R decoded-RGB v4: seven controls,672 updates each

Authorized after the completed v3 pilot/endpoint and the fixed single-TRAIN-source objective diagnosis. This is a new training-objective experiment, not a repair/overwriting of valid prior results. No source/hyperparameter/weight sweep; no sealed data access.

## Budget and sources

Reuse exactly the existing32TRAIN/8VAL sources and measured cache. Main training loads TRAIN only. Each of224(source,far-query) pairs is presented once in a seeded permutation, repeated3balanced passes,672updates/control,batch1,Adam.0005,clip5,seed17. Every single update differentiates the actual frozen native H3 FP16 D. Original latent L1 main loss is removed completely. Loss is fixed to

`5*(actualD RGB L1 +0.1 causal confidence-weighted actual past-LR L1)+10 physical appearance MSE+0.5 equivariance L1`.

no-equivariance has its defined zero equivariance term. All other relative coefficients remain as v3. HR is training/evaluation supervision only. No qGT/neutral code is an inference input. Same causal LR-only IRLS controller and source geometry, same far-null/history-row support and64stepDC. Native VAE frozen; TF32 off; 22 temporal copies only supply native codec context, not extra measurements.

All six original controls initialize from their own stage1 **last600**. Seventh `anchor_only` copies the complete full closed-action stage1-last600 state, without an additional pretraining run or chosen initialization. Save224/448/672 and optimizer checkpoints; evaluate only **last672**. No val selection or val-based early stop. All7×672 updates run even when interim train behavior is weak. Physical total4704actualD backward calls, in addition to all v3/diagnosis compute recorded separately.

## Current-image learned strong control

Same146224 architecture parameters as full: existing51channel encoder takes only current query code highpass24+local-lowpass24+ones/coords;4channel state is recomputed at every query, existing action is used as query-conditioned style, same decoder/appearance head. No historical packet image/code, group features or coarse descriptor is read by its neural `rollout`. Geometric arrived-footprint coverage is query metadata; this supplies the neural confidence/mask, without content-derived history reliability.

The shared post-decoding row-band reliability filter still uses the same causal historical controller as other methods. Therefore this is **current-image-only neural SR with shared historical controller/support**, not a claim that the entire baseline uses no historical statistics. Evaluate an additional geometry-only filter variant at the same fixed checkpoint to expose this dependency. Poison all historical images/codes/group features/descriptors while holding query metadata fixed: neural predictions must be exactly unchanged. The common training appearance/past-LR supervision can inspect controller metadata and observed-LR labels; those are loss branches, not neural query inputs.

Its equivariance auxiliary uses TRAIN-only current-query actual LR neutral-counterpart codes: upsample the actual current LR, inverse-photo with supervised training nuisance q, native E. This uses no HR texture and no past near image; the known q/neutral code is strictly an auxiliary label. Do not build/read val/test counterfactual labels. Main neural query path never reads them. All other controls retain their original near-pair equivariance labels. Both means use the same0.5 coefficient; appearance query+valid-group supervision and physical losses have the same coefficients.

The inherited encoder's input semantics differ for the current-image arm; this potential warm-start distribution disadvantage is explicit. No compensating schedule/weight/init tuning. Algebraically cancelled final biases are mechanically frozen in all methods; architecture/active counts and actual gradient audits are recorded. No dummy parameter padding.

## Pretraining acceptance checks

Use one actual TRAIN cache sample and actual nativeD loss/backward for each control. Require nonzero encoder/readout/core/appearance branch gradients; zero VAE gradients; finite gradients. Test future-packet poison does not affect first prefix query, no-evidence latent and real RGB readout equal currentLR-VAE, anchor-only historical image/code/group/descriptor poison exact invariance. Check active parameter counts differ below1.5%. Use existing independently audited E/D/physical camera; no large Jacobian cache.

## Fixed final evaluation

Eight VAL sources, per-source means before10000source bootstrap, raw and identicalDC. Keep currentLR-VAE, raw/photo/reliability atlas, native decoded atlas, retained-LR jointinverse/nativeED, lastnear, six learned controls and current-image learned control. Report source wins, return/appearance/change and old/new/overlap regions.

Fixed checkpoint interventions: remove measured delta, shuffle history while keeping controller metadata fixed, freeze delayed appearance rebase, freeze reliability reweight, both frozen. Current-image neural path must remain unchanged under image/code history poison; shared filter effects are reported separately. Evaluate t15 no-visible query by reusing latest real far13 code/metadata as its past anchor, updating only the actually arrived near14 evidence; no fresh observation is invented, no current-LR DC applied, repeat query does not write state. Give explicit field-level state bytes (including reference/currentLR/native anchor/decoded buffer, h/weights/ID/group/descriptors/moments/action weights) and whole-revision versus isolated-group-action versus native-query-D costs.

No single-source fit or algebraic closure can establish a new memory method. A useful result requires newly observed detail benefit over the parameter-matched current-image model, along with meaningful action/revision gains compared with strong factor-cache/GRU/classical controls. Otherwise record the failure/equivalence honestly.
