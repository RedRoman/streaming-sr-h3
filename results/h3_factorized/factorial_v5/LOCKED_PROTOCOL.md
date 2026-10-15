# B5 locked structural 2×2 pilot: native-code conditioning × observed pixel bridge

This is a causal architecture diagnosis after B4, not a claim that a residual skip, affine action, classical atlas or null-space readout is new. B4's every-step native-D objective improved quality, but its full model matched the current-image neural control and ignored history. The fixed single-TRAIN fit remains separate. No sealed DIV2K/old test is accessed.

## Inputs and exact RGB graph

All four arms retain the same actual-LR-derived **full24 native code and proxy24 native code for every arrived packet**, FP16 storage, and compute `packet_delta = full.float() - proxy.float()` before the shared existing 4-channel lift. This is not the separately rounded old half-delta route. It uses the identical saved full code for both the bridge and neural input. Full codes were built from actual near LR and reference-far upsampling, with no HR input. All four also retain the same actual LR controller descriptors, IDs/group tags, weights, group moments, current LR/native anchor and action parameters. Known capture-group tags are simulator-supplied nuisance groups, common to all arms; no automatic discovery claim. This pilot makes **no persistent compression claim**.

Let `z_c` be the actual current LR native code, `x_c = D(z_c)` the route-aligned cached current RGB. Let `phi_i = D(full24_i)` for each arrived unique packet. The common, frozen causal LR IRLS photometric/reliability controller defines `B_1 = ClassicalRevision.decoded_atlas(x_c, phi, records)` using the existing relative pixel photometry, per-footprint near row projection and outer current-far null projection. `B_0=x_c`. The native decoded atlas is the exact B4 H3 decoded-atlas algorithm; neither it nor its support uses any arm's learned appearance. It is a strong empirical pixel-photo bridge, not native-latent photometric equivariance. `phi` and bridge are detached, actual observed evidence only; zero DiT proposal. For compute-matched ablation, all four compute the same phi/bridge; the skip-off minimal implementation could omit these D calls, and both costs must be stated.

For arm `(A,S)` with current-native-code neural conditioning `A∈{0,1}` and physical bridge skip `S∈{0,1}`:

`delta_theta = neural_readout(h, learned_query_appearance, confidence[, z_c if A=1])`

`x_raw = B_S + H_common[ D(z_c + delta_theta) - D(z_c) ]`.

This is **pixel-domain addition**, with no E reencoding after addition. The subtracted term is always the same current `D(z_c)`, never the bridge. `H_common` is the existing far-null / causal reliability-weighted historical near-row support / far-null operator, fixed by the same external LR controller for all arms. Final observed queries use the same one application of 64-step bounded DC; no-observation query15 uses retained far13 anchor and no invented current-LR/DC. Raw outputs receive the common clamp only when raw metrics are computed. Physical D is frozen and always sees `z_c`, including A=0. A=0 removes current code only from the neural readout, not from physical D or all metadata; it is **no-current-native-code conditioning**, not completely current-content blind.

With zero correction, S=0 must equal current LR-VAE, S=1 must equal the specified H3 decoded atlas, pixel for pixel before final DC and after **the same single** final DC application. This parity is mandatory before training. Shared H never filters B a second time.

## Mechanical parameter/init controls

All arms use the same existing closed-action 51→64→64→4 encoder, appearance18→48→48→12 and affine core. Copy all their weights from the same full stage1-last600 checkpoint. Reinitialize both neural readout types with the same seed and zero final weight/bias; the encoder/core remain their nonzero trained initial state. A=1 is 41→64→64→24 with inputs h4+z24+confidence1+appearance12; A=0 is 17→w→w→24 with h4+confidence1+appearance12, w mechanically chosen to minimize total parameter-count difference, never by val. No dummy input/padding parameters. Algebraically cancelling final biases are frozen as before. Active counts must agree within1.5%. S does not change any neural parameters.

The first actual-D-only RGB backward must give nonzero final-head gradient; zero encoder/core gradient at the zero head is the expected mathematical initialization. After one fixed Adam update, the second actual-D-only backward must demonstrate encoder and mixed-group core gradients. Do not change initialization to force first-step encoder gradients. Auxiliary gradients are recorded separately. Frozen VAE has zero weight gradients. Poison future packet full/proxy codes: first prefix prediction/base must remain unchanged. Repeated packet IDs are not extra writes; read queries are pure.

## Fixed budget, supervision and endpoints

Reuse B4's exact32TRAIN/8VAL source-level development split. Train loads only32TRAIN. Seed17, balanced224(source,query) pairs in the same seeded permutation for each of3 passes,672updates per arm; four arms all run. Batch1, Adam0.0005, gradient clip5. Each update differentiates one actual frozen native FP16 D through the existing audited eager-autograd context; TF32 off. Loss unchanged:

`5*(actualD RGB L1 + .1 shared confidence-weighted past actual-LR L1) + 10 appearance MSE + .5 equivariance L1`.

No latent reconstruction loss, no new shuffle penalty, no coefficient/init/epoch sweep. HR and known appearance/counterfactual labels are supervision only. Inference controller remains causal LR-only. Learned appearance is used only inside neural residual/core and the common supervised loss, never in B/H. Save0/224/448/672 plus optimizer; primary comparison is **last672**, no val selection/weak-signal early stop. Extra physical compute is separate from all earlier experiments.

## Fixed diagnosis and reporting

First finish fixed B4 staged interventions: across-source packet-h variation, actual vs zero measured delta vs shuffled delta/proxy vs swapped neural-anchor code (physical current anchor held fixed) vs a constant first-VAL-source packet h; report h/canonical/readout/nativeD/support/DC RMS and per-source GT effect. This is diagnosis only, no eligible method.

After all4×672, evaluate VAL8 raw/DC, per-source means then10000 source bootstrap, all7 original queries and query15. Keep B4 same-capacity current-image / factor-cache / replay and strong current LR-VAE / LR-photo atlas / joint inverse / H3 decoded atlas controls. Report each arm's gain over **its own zero-training base**, not the bridge's existing history gain as a learned gain. Intervene neural history removal/shuffle while leaving the correct physical bridge fixed; separately remove/shuffle bridge evidence to quantify its classical contribution. Freeze delayed q/weights separately. Report return/appearance/change and union/old/new/overlap, no-observation protection.

Count all full24/proxy24 factor archives, h/weights/ID/group/descriptors, group sums, controller and current LR/native/RGB buffers, shared cameras, trained weights, native VAE, fixed phi/bridge scratch or optional RGB cache. Record E/D/phi construction and query fusion/NN/DC times separately; N packet/group dependence and fixed two-group action costs are distinct. Offline precomputed common phi/bridge is amortized training input, not free deployment computation or compressed state. No claim of new memory from numerical closure, a positive bridge skip, or generic residual correction alone. Standard model-based residual/null-space overlap is explicit: [Deep Null Space Learning](https://arxiv.org/abs/1806.06137), [RegNets](https://arxiv.org/abs/1812.00965), [MoDL](https://arxiv.org/abs/1712.02862).

## Accepted implementation details before launch

Independent checks passed in2.904s on actual nativeD: both zero-base raw/DC parities and common cached-phi route maxabs0; A1 anchor-poison response>0, A0 exactly0; future/empty maxabs0. Active parameters A1=146192, A0=146578 (width74), difference0.264%. First zerohead RGB-only encoder/core gradients0 as expected, second backward nonzero for both, VAE weight gradients0. Eight discarded acceptance updates are separate from2688 main updates. Neutral equivariance auxiliary retains the old separately rounded half neutral-delta label; observed input now full.float−proxy.float. This small shared route difference is explicit, without a new weighting or cache re-encoding change. Existing near-row support limits learned correction to observed-near rowspace, excluding near-camera-nullspace generated detail; it stays fixed for this factorial diagnosis.
