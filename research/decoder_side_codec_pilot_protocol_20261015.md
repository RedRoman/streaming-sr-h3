# Decoder-side codec fixed pilot v1

Status: implementation/API audit, then root-authorized four-arm 600-update pilot.
This is a DSC-inspired baseline candidate and an information-path experiment.
It is not yet an original mechanism, a video SR result, or an H3 decoder result.

## Dataset and chronology

Use `results/h3_core/decoder_side_natural_v2`, 32 TRAIN and 8 VAL sources with
the existing source-disjoint development split. The natural primary and VAL
tensors are bitwise equal to v1. TRAIN additionally has the declared
`0.05+0.90X` auxiliary base and its mechanically range-safe nullspace worlds.
Preserve v1's negative ambiguity audit: future far4 gives weak new directions
at the declared noise/quantization scale. This task tests the value of coarse
decoder side information for storage, not strong joint-posterior ambiguity.

Writer input is only uint8 past RGB64x64, converted to float/255. It cannot see
future phase, future pixels, source ID, supervision or construction labels.
Commit one genuine uint8 packet before reading any future. Reader receives
that packet, one selected future RGB32x32, its arrived phase, and shared model
weights. It never receives all four possible future images at once, fine
indices, writer features, source identity or an instance dictionary. The same
packet is independently read for all four phases; its hash must stay fixed.

Every physical TRAIN world is supervised under all four future phases.
Pair supervision uses only the fixed interior auxiliary base versus the
future-null collision for its construction phase; their future uint8 tensors
must be equal. The original natural primary is not the collision base.
Past-null-reveal worlds receive ordinary supervised losses only: no demand for
different codes when their writer input is identical. VAL has natural worlds
only and is evaluated once at the last checkpoint, never used for selection.

## Frozen model layout and wire contract

Primary layout is H=W=16, G=4, D=8. A common small writer maps past64 to
features `[B,16,16,4,8]`. Shared frozen books are E256 `[4,256,8]` and E128
`[4,128,8]`. Fit both books once using only the common initial writer's TRAIN
features, deterministic seed17 and 20 Lloyd iterations; initialize centers by
equally spaced entries of a seed17 shuffled TRAIN feature pool. Empty centers
keep their previous value. No RGB loss, future cue, VAL, or auxiliary labels
enter this fit. Save features/initial writer/books SHA. This fit is common
preparation, with its actual CPU/GPU time reported separately; no learned
warmup or extra optimizer updates are hidden outside the 600 per arm.

All arms clone the same initial writer and future-conditioned reader, then
train them for 600 updates. Only books are frozen. Nearest assignment uses
hard-forward soft-distance-backward ST with temperature0.5, so the pair loss
can change the ordinary VQ128 writer's actual code decisions too.

A future-only CNN takes exactly one future32 plus its two phase coordinates,
producing fine logits `[B,16,16,4,256]`. All arms train this selector with the
same nearest-fine CE. The RGB reader receives reconstructed code features,
the same future-only expected-fine features, and the same future/phase inputs;
it predicts RGB128. The common CNN architecture and its real parameter counts
are to be saved in the driver protocol before launch. No arm gets extra RGB
channels or HR input. Do not pad with unused parameters to imply parity.

The wire contains 1024 symbols in H,W,G row-major order, seven bits each:
7168 bits = **896 bytes per instance**, no floating tensor substitute. Bit k
of symbol n occupies `(7*n+k)%8` in byte `floor((7*n+k)/8)`, LSB first. The
reader knows the globally fixed shape; no per-instance shape/header is hidden.
Shared codebooks, partition map and neural weights are separately reported
as model storage, not silently charged as zero total storage.

## Four arms and the exact changed assignment

1. `plain_cvq`: nearest E128 center, conditional RGB reader, no pair loss.
2. `cvq_pair`: identical conditional VQ128, same pair loss as3/4.
3. `categorical_pair`: fine256 nearest index j; learn Theta[G,256,128], row
   softmax/ST argmax assigns j to a seven-bit bin. A bin can have any occupancy.
4. `balanced_pair`: same fine256, same Theta and parameter count; the only
   assignment change is an exactly capacity-two fine-to-bin partition.

For balanced, soft P has row mass1 and column mass2, obtained with32 log
Sinkhorn iterations at temperature0.5. Hard P solves linear assignment of
256 fine indices onto256 slots (two slots for every bin) by Hungarian matching.
ST is `P_hard+(P_soft-P_soft.detach())`. Fine nearest assignment uses the same
form, with softmax of negative squared distance/0.5. Thus forward is exactly
discrete, while both writer and partition get gradients. This is a surrogate
gradient and must not be described as the gradient of the hard codec.

During training, e_j is nearest-fine onehot ST, u=eP is bin onehot ST, and
membership m_j=sum_b u_b P_jb. The reader computes
`w_j=m_j*softmax(future_logits)_j / sum_k m_k*softmax(future_logits)_k`
and reconstructs `sum_j w_j E256_j`. At wire inference m is obtained from the
received hard bin and the frozen hard partition only. Categorical considers
**all** members of a bin, not a top-two truncation. Balanced has exactly two.
Direct VQ128 reconstructs E128[received_index]; its future-conditioned reader
and common selector remain trainable. The two VQ arms have fewer quantizer
parameters: report this. Categorical versus balanced is the assignment-only,
exact-parameter control; CVQ128 is the same-rate standard conditional-codec
control. Neither comparison alone establishes novelty.

Numerics: forward probabilities use a masked softmax **within the hard bin**,
so even extremely unlikely received bins normalize to one. Membership's ST
differential is added as a zero-valued analytic conditional-probability
derivative. Only the absent-member likelihood ratio in this backward
surrogate is capped at1e6; actual forward and cue-logit derivatives retain the
exact conditional distribution. The API test includes logits of+/-1000.

## Fixed losses and update budget

Pixel values are float0..1; each MSE is a mean over its complete tensor.
Let Xhat be RGB128 and X its supervision-only target. Let Zhat be reflection
padding Xhat by4 pixels into136. N and Fq are the saved actual separable
operators. The common loss is:

`Lcommon=MSE(Xhat,X)+.1*MSE(N(Zhat),past/255)`
`        +.1*MSE(Fq(Zhat),future_q/255)`
`        +.01*MSE(writer_feature,nearest_E256.detach())`
`        +.01*CE(future_fine_logits,nearest_E256_index.detach())`.

The136 reflection padding is a declared readout boundary convention, not the
actual source halo. Its observation residual is measured and equally applied
to every arm. HR/full136 fields are used only as labels and pair-target
construction; no HR tensor reaches writer or reader.

For each collision pair define its **past-observed-band** target directly in
near-measurement coordinates, not an unobservable HR difference:
`Delta_obs=N(full136_collision-full136_auxbase)`.
With the same future cue, use
`Lpair=MSE(N(reflectpad4(Xhat_collision-Xhat_auxbase)),Delta_obs)`
`      /(mean(Delta_obs^2)+.003^2)`.
This keeps the target exactly observed in the past. A near-only HR row-space
projection could fail to remain in the shifted future nullspace; therefore
it is deliberately not imposed as an RGB pair target. The common future loss
constrains both predicted worlds to their identical actual future observation.
Total is `Lcommon+.01*Lpair` in arms2/3/4 and Lcommon in arm1. The denominator
and weights are fixed here, not selected using VAL. Pair labels/loss are
identical across all pair arms. No special conflict loss is reserved for the
candidate; auxiliary selector and commitment are common across all four.

All arms use Adam, lr1e-3, betas(.9,.999), eps1e-8, weight_decay0, seed17,
600 updates, last600 checkpoint only. Same deterministic source/world/phase
and collision-pair schedule, same batch count, and same initialization for
all shared layers. Each update uses the next ordinary world/query item plus
one next collision pair (same source schedule and all four construction
phases in fixed cycles); arm1 performs the same pair readout and logging but
its pair loss multiplier is zero. Driver saves the exact item schedule SHA.
No VAL sweep, early stopping or best-checkpoint selection. Report ordinary
forward/read and pair extra compute, optimizer steps, model parameters,
packet bytes, fitting/training/evaluation seconds and peak memory separately.

## Required preflight, outputs and direct baselines

Before GPU admission: bitpack/unpack; exact hard-forward vs independent
fresh-reader feature equality; nonzero writer/selector/Theta gradients where
applicable; frozen books; balanced exact2 occupancy; categorical arbitrary
occupancy; identical past gives identical bytes; changing future/phase cannot
change written bytes; inference only the selected future. Save checks and
source SHA. `decoder_side_codec.py --selftest-out ...` covers the quantizer
API; A's driver must add image-level causality and four-arm schedule checks.

Evaluate all8 natural VAL sources and all four phases, averaging phases
within source before equal-weight source mean. Save raw RGB/PSNR, actual near
and future reprojection residual, actual896-byte packets, packet/weight SHA,
and writer-only byte audit. Intervene future drop/shuffle and use same bytes;
this measures side-information dependence, not a causal video result.
Compare root's real896B DCT/WebP + common-future physical reader, uncompressed
past, current-future-only, plus the four neural arms. Report model overhead.

Only if this pilot shows a reproducible partition benefit against both
categorical and CVQ+samepair should it be repeated at fixed448/1792B and on
new held sources. H3 connection is a subsequent separate test: actual LR
H3 E or cached actualD response features replace pixel features and a frozen
actual H3 D supplies common RGB/LR losses. Count all E/D calls and retain
full-code/atlas/conditional-codec controls. This pixel pilot has **zero H3
calls** and cannot establish decoder-aware H3 memory or a new paper core.

Implementation: `experiments/h3_backbone/decoder_side_codec.py`.
Driver/data ownership: A, `src/h3_core/codec_data.py` and its new pilot driver.
B independently reviews the API/physical pairing without editing these files.
