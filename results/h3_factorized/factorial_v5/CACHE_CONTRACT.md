# Observed cache handoff for a different pre-cue codec

Remote project root `/home/saiadmin/streaming_sr_20261015` on `sai-gpu161`; native caches remain remote. Source development split and provenance: `results/h3_factorized/revision_v3/cache/manifest.json` (32TRAIN/8VAL). These BSDS photos were used previously in the project; not a fresh confirmation set. Sealed DIV2K and old test are untouched. Reuse TRAIN for learning and VAL only after a fixed protocol. Camera/appearance/foreground/content replacement are simulated static-source episodes, not real natural motion.

For source ID, load `revision_v3/cache/ID.pt` with torch weights_only=False. Actual-observation fields:

- `classical_inputs.packet_full_latents`: FP16 `[8,24,7,16,16]`, native E of actual noisy near-LR physical preconditioned full view. No HR enters this code. Do not treat it as appearance-neutral or a clean texture latent.
- `inputs.packet_proxy`: FP16 same shape, actual-LR-derived proxy native code. B5 neural input uses `full.float()-proxy.float()`. Legacy `inputs.packet_delta` is separately rounded FP16; it is a slightly different route, do not silently mix.
- `classical_inputs.near_LR`: actual `[8,3,64,64]` observations. `classical_inputs.query_LR`: `[7,3,64,64]` actual far observations. `inputs.reference_LR` is actual first-far reference.
- `inputs.query_latents`: FP16 `[7,24,7,16,16]`; `query_base_rgb`: FP32 `[7,3,256,256]` route-aligned nativeD of current LR code.
- `packet_steps=[1,2,5,6,10,12,14,16]`, `packet_fp=[0,1,2,3,3,0,1,0]`, `packet_groups=[0,0,1,1,1,0,1,0]`; crop top-lefts `(32,32),(32,96),(96,32),(96,96)`, crop128 at2× camera. Step9 duplicates packet at6 and is not a new factor. Query steps3/4/7/8/11/13/17. Group tags are supplied nuisance groups, not inferred identities.
- `inputs.group_features`: `[18,2,18]` causal LR IRLS q12+quality6 as each cue arrives; `weights`: `[18,8,1,16,16]` causal reliability. These are time-indexed controller metadata; use only arrival-prefix values. `query_features/groups`, `coarse_descriptors` are observed metadata.

`supervision` is a separate branch: RGB/clean detail/true nuisance/neutral counterfactual labels may supervise TRAIN only; they must never initialize or update runtime state. Existing cache has future packets physically present on disk, so a new pre-cue write path must explicitly filter `packet_steps<=arrival`; never use a later group_features/weights row to write an earlier state. Pixel/latent interventions in B4/B5 prove first-prefix output independence but do not license an unfiltered new encoder.

Common actualD phi is in `factorial_v5/phi/ID.pt`: `decoded` FP32 `[8,3,256,256]` from frozen nativeFP16D of the saved **full code**, not delta+proxy recomposition; `bridge` FP32 `[7,3,256,256]` from fixed external causal LR q/weights and H3 decoded-photo atlas. `metadata` records source cache SHA, VAE SHA, actualD time/calls and byte sizes. Train/val complete JSONs list every source. Native context repeats22frames for codec context only; they are not22 evidence samples. H3 E/D asset SHA is `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`.

A pre-cue codec can reuse full/proxy codes or full actualD phi without opening DiT. Keep any query-dependent future-cue teacher strictly on the loss/evaluation side. The frozen B5 pixel bridge is standard fusion and serves as a direct-information strong control. `B5_final_report.md` states its quality and full state/computation cost; neither four-channel compression nor independent learned-memory value was established.
