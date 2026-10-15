# B3R fixed seed17 pilot: closed appearance reinterpretation of evidence moments

Status: preregistered before cache/training.32 official BSDS train /8 val sources, same hashes/donors as B1; 本B3R不访问旧16test；BSDS这些源曾在早期项目实验出现，不称项目全新封存test. Seed17 first,600 latent+200 actual frozen-native-D steps per control,batch4,Adam .002/.0005,appearance10,equivariance.5,RGB5 every4. No loss/step tuning. Three seeds only after structural checks and one-seed mechanism interpretation.

## Candidate hypothesis and direct overlap

Learn a4-channel lift h=phi(delta_native_code,LR_proxy_native_code,footprint) whose weighted homogeneous moments admit an invertible spatial affine appearance action H(a). For capture group g retain canonical sum M_g=sum w H(a_g)^-1[h;1]. When its appearance posterior changes a→b, update M_g←H(b)^-1 H(a) M_g once. This action-only substep is independent of the number of packets in that group; total controller reliability updates may still examine descriptors and are not claimed O(1). Individual withdrawal/reweight subtracts/adds the stored compressed factor under current H, without native E/D replay. Query normalizes summed moments, applies query appearance action, and a small current-LR-anchor-conditioned decoder adapter feeds actual frozen H3 D. Zero evidence gives exactly current LR-VAE.

Affine actions, sufficient statistics and sum aggregation are established ideas. Deep Sets (https://arxiv.org/abs/1703.06114) and Homomorphism Autoencoder (https://proceedings.mlr.press/v202/keurti23a/keurti23a.pdf) overlap directly; appearance adaptation/memory also overlaps RSDN,RECAP-Forcing,Multi-Memory Streams. This is a candidate about compressed H3 evidence reinterpretation and physical correction, not a novelty claim for those components or amortized projection/reencoding.

## Frozen controls

Six learned controls: closed_action; closed_action_no_equiv; nonlinear factor-cache MLP with the SAME4ch payload and original factor storage allowed; nonlinear group-map without closure; ordinary ConvGRU; ConvGRU allowed to replay all stored factors after posterior/reliability revision. All receive the same LR-only photometric estimates, nuisance-group tags, descriptors,confidence,geometries,anchor and query. Match used trainable parameters within1.5%, report all counts. No unused parameter padding.

Classical controls: robust LR-only photometric IRLS, signed confidence-weighted residual atlas with retrospective weight/reliability revision, full retained-LR inverse and native E/D of its result; same physical DC. Keep preconditioned last-near and ordinary non-retrospective atlas. Initial no-proposal interface explicitly zero. No HR or generated output is inserted as actual evidence.

##18-step mechanism episode

Canonical256 crop;4 near128 footprints at(32,32),(32,96),(96,32),(96,96),2× camera;far4×. Gaussian.6/kernel5/reflect+bicubic,noise.003. A is a real natural crop,B replaces[64:128,64:128]with same-split natural donor. q0 neutral,q1/q2 RGBgain±.18,bias±.025,smoothshade6coeff±.09. Foreground corruption uses a donor region inside previously seen overlap, but its true mask is NEVER an input.

|t|packet|
|---:|---|
|0|far A0 reference|
|1|near0 A1|
|2|near1 A1|
|3|far A1: delayed same-capture photometric cue|
|4|far A0 return|
|5|near2 A2 with transient foreground in overlap|
|6|near3 A2|
|7|far A2: delayed cue/reliability evidence|
|8|far A0 return|
|9|exact duplicate packet t6, no write|
|10|independent noisy near3 A2, newID|
|11|far A1|
|12|near0 B1 actual local replacement|
|13|far B2, change/reliability confirmation|
|14|near1 B2 with transient foreground|
|15|no visible observation, read-only query|
|16|near0 B1 independent confirmation|
|17|far B0 final|

Capture-group tags are simulator-provided known nuisance grouping: packets1/2/3 share locked exposure group1,5/6/7 group2; later uses reuse those controlled settings. They supply no numerical q,visibility,content epoch or HR identity. ALL controls share these tags. This protocol does not claim real LR automatically discovers appearance groups. Current group estimates come only from causal robust photometric regression against the reference far LR. Full far cues refine the group, and coarse8×8 RGB descriptors of retained packets support causal retrospective reliability from actual LR disagreement. True foreground/change masks remain supervision/evaluation-only; pre-confirmation ambiguity is not treated as magically distinguishable.

Main evaluation far query steps3/4/7/8/11/13/17. Report old-unmodified/newly-replaced/overlap RGB, current/past LR residuals, appearance revision gain, removal gain, source wins,correction latency. HR targets are training/evaluation only; no test selection. All controls use physical current-far quotient and identical64-step DC. Train native latent targets are HR-derived clean2×-supported detail labels, never runtime state; equivariance pairs are training-only counterpart measurements.

## Mandatory pretraining checks and falsification

Check actual cache,realD zero equality,nonzero gradients in lift/action/readout/appearance; group rebase equals explicit per-factor recomputation; duplicate idempotence; local withdrawal equals recomputation; query purity; no h=0 escape. Log feature variance,detail RGB energy/new-region recovery,not only consistency. h=0/currentLR cannot count as a successful method even if some region averages look safe.

Claim supported only if correction preserves acquired fine detail and improves confirmed content/appearance recovery over GRU,while quality/cost compares credibly to SAME-compression factor-cache and strong LR-only atlas/inverse. Closed-action algebra matching replay alone is a structural check,not empirical novelty. If factor-cache matches all quality/compute tradeoffs,report equivalence/limited inductive-bias result.

Cost includes reference LR,currentLR/anchor native code/decoded buffer,all factor h/weight/id/group/coarse descriptor,group sums and appearance state,shared action weights/camera matrices,and any raw factors needed by replay.8 actual near packets/source are not8 independent sources. Packet-count32/128/512 compute stress may reuse features solely to measure complexity; it never enlarges sample count. Separate group correction,controller work,group-count dependence,native encoding,and query D calls.

Implementation clarification before first training: RGB5 multiplies actualD RGB L1 plus0.1 confidence-weighted actual past-LR reprojection loss; all six controls share it. Shared used-parameter count is146028–146576 (within0.4%); no padding. Readout/appearance/cache controller payloads and per-query H3 decoder buffers are counted separately.

Evaluation clarification before first val result: all six full models and fixed-checkpoint frozen-photo/frozen-weight/frozen-both interventions use all7far queries. Add diagnostic delta=0 and cross-source-shuffled factor inputs to distinguish acquired detail from an anchor-conditioned prior trigger; these diagnostic interventions are not eligible methods. Strong LR inverse applies a relative spatial photometric gain/bias before the exact camera (48CG,ridge.01 from earlier train-only neutral calibration), plus common64stepDC. Retain currentLR-VAE, last-near pixel/native code/nativeED, nonretro/retro atlas, raw-LR inverse, nativeED atlas/inverse with common historical support, and atlas with full model learned photometric estimates to isolate appearance-head benefit. D-after-photo need not be exactly latent equivariant; these are empirical strong controls. Saturated RGB means the inverse unclipped affine photo model is approximate. Same causal confidence is resized to camera/feature grids, which have different numerical resolution.

Endpoint amendment after original pilot training, before separate endpoint evaluation: parent requested final600/200 rather than original val-selected best. Preserve original rgb_stage2/val_seed17 selection pilot. Formal endpoint run rgb_endpoint_stage2 will initialize ALL six controls from latent_stage1/last.pt(step600), repeat the same200-step RGB schedule/lr/loss, and evaluate only last.pt(step200), with no validation-score choice. No architecture/data/weight tuning. Physical extra1200 updates/300trueDbackward are explicitly additional compute, not hidden. Original full latent best500 and RGB selected-step variability are not presented as endpoint results.
