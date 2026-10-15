# B4 decoded-RGB fixed endpoint: objective repair, no demonstrated memory benefit

All seven controls completed672 balanced updates on32TRAIN sources, every update differentiating the actual frozen native H3 FP16 decoder. The formal checkpoint is last672 for every arm; no VAL loading/selection during training. Eight development VAL sources were evaluated after training; queries are averaged within source before10000 source bootstraps. This is one trained seed17 on natural photos with simulated cameras/appearance/change, not fresh project-wide heldout data, real video, or DiT evaluation. Sealed DIV2K and old16test were not accessed.

| Method | DC union PSNR, source mean |
|---|---:|
| Current LR-VAE |23.84270|
| Closed-action full |24.11443|
| No equivariance |24.07241|
| Factor-cache MLP |24.15909|
| Nonlinear group |24.12149|
| Online GRU |24.05545|
| Replay GRU |24.08248|
| Current-image neural, shared controller |24.10618|
| Same current-image neural, geometry-only support |24.10616|
| H3 decoded-photo atlas |26.79536|
| H3 E/D LR joint inverse |26.51293|
| LR-photo revision atlas |27.51398|

Full minus current LR-VAE is +0.27173dB (8/8 source wins;95% source-bootstrap CI[0.12581,0.48068]); full minus current-image neural is +0.00825 (3/8,CI[−0.01824,0.04658]); full minus factor-cache is −0.04466 (3/8,CI[−0.13578,0.00899]); full minus replay is +0.03194 (7/8,CI[0.01202,0.05678]). Full minus strong LR atlas remains −3.39955 (0/8). Relative to the valid earlier selection-pilot and fixed-endpoint supplements, decoder-aware supervision fixes a major quality problem, but the resulting gain is also obtained by the current-image learned control. It does not establish a new memory/action benefit.

## Causal dependence and staged diagnosis

Holding controller/metadata fixed, removing full's measured packet delta changes quality by only0.001123dB (CI[−0.002137,0.004876]); shuffling delta+proxy changes it by0.000535 (CI[−0.000837,0.001829]). All seven heads are audited in history_dependence/all_head_history_interventions.csv. MLP/nonlinear/GRU changes are smaller still. No-equivariance shows a modest shuffle drop0.02510 (CI[0.00295,0.04753]) but is worse than full/current-image quality; this does not establish successful evidence retention. Freezing photometric revision changes full by−0.000032dB; freezing weights by0.000427. Nonzero mixed-group RGB-only gradients verify that encoder/core/appearance paths are differentiable, with zero VAE weight gradients; weak causal use is not a dead-branch claim.

Staged fixed-weight interventions (pathway_diagnosis/stages.csv) place the loss of detail before readout: packet h RMS0.157019, across-source standard-deviation RMS0.002206. Removing actual measured delta changes h by0.002690, canonical h by0.003323, predicted24-code by0.009918, nativeD RGB by0.001106, supported RGB by0.000366 and finalDC RGB by0.000359 RMS. Only swapping the neural readout's current native-code condition, while keeping physical decoder/current-source anchor fixed, changes predicted code by0.23799 and final RGB by0.007930, and loses0.08834dB. A nearly constant history can therefore act as a gate for current-image SR. Shared support reduces both interventions, but does not alone explain why history was weak upstream.

The constant-h diagnostic uses the **first VAL source's existing packet features** for every source. Its first source is self, not an independent intervention; summary.json separately reports the other seven actual interventions. It is a diagnostic counterfactual, not a valid reconstruction method.

The completed single-TRAIN RGB300 fit is a distinct, known-q, one-packet capacity/objective diagnosis: removing its actual delta changes h by0.19710 and finalRGB by0.013741 RMS, losing1.74816dB (28.28402→26.53586). The multi-source full last672 under that same diagnostic input changes h only0.002632/finalRGB0.000171 and loses0.000692dB (26.54812→26.54743). Thus local trainability does not transfer into learned historical evidence in the formal source-development experiment. Do not pool that single-source fit with source-heldout averages.

## No-observation query and cost

Query15 reuses actual far13 code/metadata, includes arrived near14, invents no current LR, applies no DC, and leaves state unchanged on repeated read. Union PSNR is22.94224 full,22.94015 current-image neural,22.94005 cache and25.57653 LR atlas. The same near-evidence failure persists without a new visible observation.

Physical training was4704 updates/nativeD backwards plus the separate acceptance and posthoc gradient probes; main training844.942s, VAL85.602s. SSH29841 closed255 after all seven complete files/checkpoints were written; read-only verification confirmed completion and no process, and no training was restarted. Formal VAL native calls were168E/1016D, peak allocated7,213,607,424bytes. Main-training peak was not recorded and is not inferred from the smaller audit peak.

Actual six-state runtime audit is in runtime_seed17/. At eight packets, h/weights/moments/metadata runtime payload372000B, shared LR controller55440B, current LR/native/RGB buffer1007616B, camera operators524288B, full adapter584896B and frozen VAE5207737968B. The raw LR archive alone393216B plus controller is only about29.6KB larger than the learned runtime+controller; trained weights eliminate any general total-storage saving. This actual implementation counts some unused uniform buffers. Current-image neural recomputes its4-channel current feature and does not retain historical image/code/h; it still shares historical LR controller/support, explicitly not a wholly memoryless pipeline. Its parameter count is mechanically matched, not its unnecessary history storage.

For full, isolated two-group appearance rebase is about0.50ms independent of packet count; whole revision scanning/allocation grows1.41ms at8 to39.71ms at512. The512 stress duplicates existing features for timing only and is not512 independent observations. Native E/D/photometric controller and queries are separately charged, not included in isolated group-action timing. Four-channel state alone is not a complete cost estimate.

## Structural follow-up boundary

B5's frozen2×2 separates removing neural current-native-code conditioning from a direct full24nativeD pixel-photo bridge. Exact zero correction must restore each common physical base. This is an information-pathway diagnosis, not novelty: atlas fusion, null-space correction and residual learning are established. Existing near-row support also excludes generative detail in the near-camera nullspace; B5 intentionally holds it fixed to isolate the two factors. A future generative-detail hypothesis needs a separate protocol rather than interpreting this restricted band as an unrestricted SR model.

Original B3R val-best selection protocol was legitimate; the separate endpoint and B4 experiments do not retrospectively relabel it as an error. All old results/checkpoints remain archived. Detailed reproducibility comes from B4_DECODED_PROTOCOL.md, main_seed17/config.json, val_seed17/complete.json with checkpoint SHA256, source manifests, logs and SESSION_REGISTRY.json.
