# Run log

## 2026-10-15 — project start
- Read source brief and preserved it as source_brief.md.
- Current goal is active; no prior experiment in this new project has been verified.
- Local hardware: Apple M5, 32 GB unified memory; 38 GiB local disk available at first audit.
- Source verification, resource audit and controlled mechanism implementation dispatched in parallel.
- A reachable 32 GB RTX 5090 was found but has existing jobs; no jobs were interrupted and no GPU load was launched there.

## 2026-10-15 — first completed evidence
- Mechanism main32 seeds and two noise stress suites finished; results establish evidence accumulation value and identity/change ambiguity. No novelty claim for MAP.
- Global Fourier blur/shift ledger implemented and checked against independently constructed dense spatial operators (max error <4e-14).
- Official BSDS500 downloaded and SHA256 verified locally and remotely; extracted image split counts200/100/200.
- Six ~60k-parameter models are in three-seed H100 training; public RelayVSR weights are downloading via a foreground command-scoped proxy.
- One discarded Fourier run had an incorrect finite-window timestamp treatment of repeated packets; preserved with explicit SUPERSEDED note and corrected before result synthesis.

## User steering — prefer H3 foundation model
The user explicitly requested using the H3 base model as much as possible. Existing workspace manifests identify this as MiniMax-H3, not Hunyuan. Primary model development is now H3-based. FastVR/RelayVSR are comparison baselines and interface references. Completed numerical and small-model studies remain diagnostics, not substitutes for H3 model results. Prioritize the existing BF16 FL2VA base, FP16 VAE and cached conditioning; quantized variants must be labelled separately.

## Fixed-budget learning and H3 preparation
- Six small models ×3 seeds finished1000 steps each (18,000 optimizer steps;351.37s H100 training); main+extended results saved locally.
- Learned non-Gaussian residual improves under one/four phase observations, but eight-phase fidelity is below the strong fitted Gaussian MAP; persistence-specific accuracy benefit is not yet stable.
- Branch-memory32seed suite completed; exactMHT equivalence explicitly retained. Old-version retrieval is beneficial but is also achieved by standard versionMHT.
- H3 primary model identified as MiniMax-H3; complete40.23GB BF16 donor and5.21GB VAE located. Direct copy throughput was poor; resource worker is preparing fixed-revision parallel download on H100.
- h3_data.py creates known-camera natural-image39frame sequences and separate LR-only packets;7 previews generated locally and source sent to H100.

## Fast H3 route verified
- At approximately14:52 Shanghai, direct5090 read-only check found GPUutil0%,11,613/32,607MiB resident and all four Comfyqueues8188–8191 empty. No resident service was stopped.
- Resource/core workers instructed to run low-resolution batch1 H3VAE cache/JVP directly where the verified FP16VAE already resides, then transfer small caches toH100 for adapter training. This avoids treating slow asset copying as a reason to stop all H3 work.
- H100 BF16/VAE downloads remain foreground tracked; sparse partial logical sizes must not be read as downloaded bytes. Their actual process/complete-range status is stored in results/h3_setup/live_jobs.json.

## Actual H3 evidence
- Existing5090 FP16VAE SHA verified; actual128single E/D worked but21.46dB and grid artefacts, so no claim of restorationquality.
- Standard-wrapper controllednativecontext comparison completed:256single25.67,256x5frames26.61,256x22frames28.53;512single27.21,512x5frames28.61;128x5frames22.04dB. ContinueH3 usingappropriatevideo context.
- 3source/32atom singleframe teacher cache is retained asinterfacepilot; no broadtraining onit. FDhalfstep39–54% requiresLRspace/numericalaudit.
- H3chart/transport model, stochasticcontrols andsamecapacitybaselines implemented; actualDiT guidegeneration remainsnextwork.

## 15:16 — actual DiT and real-motion interface
- Native H3 NVFP4-all 50-block model completed 4 genuine forward calls and 5-frame256² guide sampling; real Qwen layer50 condition was used. Main BF16 quality remains unverified.
- Visual review caught prompt/source mismatch (single sheep text versus three pigs image). Keep this run as integration evidence only; neutral shared prompt required for quality comparisons.
- Native22 FP32 teacher (same FP16-stored weights) reduced LR epsilon sensitivity to5.16%; actual local LR prediction RMSE7.95e-6. Proceed to actual native-context training with precision tracked.
- DAVIS8 sources/695 frames and8x39 H3 crops are completely prepared locally/remotely; independent3404payload hash and sourceCRC verification passed.
- Root implemented and executed LR-only DIS correspondence on all8 windows; prefix, duplicate and read-only checks passed. Confidence predicts lower HR warp discrepancy but anchor coverage decays; RGB temporal blending is slightly worse than bicubic in all8clips. These are classical controls, not H3 method results.
- CandidateB content/appearance counterfactual training assigned to mechanism_suite, distinct from candidateA chart/transport. Both require actual H3 decoder outputs and source-disjoint evaluation.

## 16:00 — true H3 training and real-video readout
- B completed three seeds/four controls:9600 optimizer steps,600 true-D RGB backward;16 independent-photo test episodes evaluated. Return swap gain does not establish overall advantage: sharedDC full24.761 vs LR-VAE25.639. Residual-state structural revision follows evidence.
- A strict FP32/noTF32 teacher cache56 sources complete; fixed three-seed nonlinear transport suite started. Old Bures absolute-floor bug corrected in v2 reevaluation; no phantom advantage retained.
- Actual BF16 H3 bmx22 normal49 pipeline complete. SameDC yields LPIPS.1944 versusbicubic.2304, but VAE-only.1926; one-clip diagnostic only.
- Root ran frozen H3 VAE on all8 real videos at256 and true832x480,22frames each;16cases/352output frames verified by hashes. CPU matchedDC/LPIPS evaluation running.
- Official AddGuide cached parity and full sampler direct-vs-official video/audio latent parity passed; actual pretrained block0 autograd/LoRA2step audit passed. Full50layer gradient waits verified BF16 transfer, not claimed complete.

## 16:25 — larger spatial context and long real source clips
- Native832x480 H3 BF16 normal49 forward complete and evaluated with matched official guide quantization/5090 decode controls. DC LPIPS DiT.16994 vs VAE.17292, PSNR32.071 vs32.150; oneclip only.
- FrozenVAE8realclips×twofields all finished and evaluated; sharedDC VAE beats bicubicPSNR/LPIPS on8/8 each, meanPSNR+.238/.229dB andLPIPS−.0380/.0434. No persistent-memory claim.
- LaSOT pinnedmirror two-categorymetadata+previews inspected; four predeclared contiguous600frame clips fetched and CRC/SHA verified (2400frames, nominal20s each at30fps; actualper-sourcefps unknown). Officialsplit recorded, no loop/pingpong. RawJPEG retained; HR/LR preparation and methods notyetexecuted. data/long_video/manifest.json.
- A/B original3seed studies stable negative for novelmodule; positiveval mechanism now supports current-observation anchor + historical-supported residual, both continuing structural revisions.

## 16:50 — full backbone adaptation and long-stream controls
- BF16 H100 complete40.23GB file SHA verified; full50layer native LoRA/guide2step backward finite, frozenbasegrad0,41.394GB peak. Direct/native forwardparity0.
- Three fixed nativeSR adapter controls launched GPU6 session68260,400updates each on4trainvideos×2res. LoRA-only400 complete, fixedvalvelocity .208553→.193380; RGBquality unproved pending49stepreadout.
- A2 projected3seed6000trueRGB steps complete: valtransport24.81474 vsMAP24.82167, no significant独立增量. Incorrect current-only rebasingcontrol excluded and isolated samebudgetrepair; nonlinear actualD LRinverse next.
- B2 seed17 retains evidencedetail, return+.473dB vsLR-VAE but loses zero-training directnear by.072dB. Seeds29/43 fixedbudget running session45407.
- LaSOT2400source/HR/LR allremoteSHAchecked. H3longVAE session68251 GPU5 uses140native22blocks,5overlap, emit2400frames once; no historic rewrite. New fixed adjacentLRonlyDIS maps2400complete; long quality/return/boundary evaluation next.
- ProjectlocalLPIPS package+AlexNet/metricweights copied from alreadyvalidated localenvironment to external/eval_deps and eval_torch_cache; sameSHA verified, identicalimage distance0. No sharedPythonenvironment mutation.
