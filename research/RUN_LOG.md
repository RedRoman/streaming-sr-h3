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

## 17:30 — long-stream evidence and native condition-memory interface
- LongVAE2400frames/140blocks and all4source metrics complete; rawPNG2400localSHAallpass. Sourceavg sameDC PSNR31.659→31.978(+.3184),LPIPS.18192→.15225,4/4wins. Temporal3/4better, longgapROI2/3better; no persistentstateclaim.
- Real targetannotation visualaudit clarifiesbook20targetbookis onbookshelfbehindperson, notlargehandheldcover. target_annotation_preview retains evaluation-onlytargetlabel/boxes.
- SharedSSHinterruption closedmultipleforegroundjobs; root44364 resumed evaluation afterverifyingGPU5empty/PIDgone, retained2completedcases. Outputrsync83144resumedthenall2400verified. No detach/serviceinterruptions.
- NativeSR3controls400each +49stepreadouts complete; velocityLoRA.193380,guide.204384,joint.186553 vsinit.208553; finalRGBpendingresource13078.
- A2faircurrentrepaircomplete24.60394. TrueD40stepLRinverse numericallyfixedall8valfinite,ROI+.27875vsMAP; additional40backward/42Dpercase explicit.
- A trueH3initializationthree seeds147full50forwardscomplete, native22sameLR; softlowdiversity, noqualitywinvsVAE. 5step48,280parammemoryconditionadapter backwardworks butunboundedtarget1.03447 caught; preservedlegacy, boundedtargetrerunpending.
- B2three-seedcompleted: return+.47003vsLRVAE but-.07307vsfrozenB1directnear; noindependentreversiblegain. B3realnativeE/Dcomplementarycapacityauditpositivebutcontrols beingmatched.
- RelatedworkcheckedprimarySequentialFlowMatching2026-02andSequentialPosteriorSampling2024; genericposteriortransport/warmstartalreadycovered. Newclaimmustbeconcreteidentity/cuecorrectionability, notrenaming.

## 2026-10-15T17:46:51.324123+08:00 本轮完成与下一固定实验

四条600帧长流评价/2400PNG逐SHA归档完成。H3原生SR三控制1200更新与440帧评价/最终adapter归档完成，未稳定超过VAE；原生480 LPIPS/PSNR取舍分列。A bounded5step nativeguide未胜经典near，unbounded target首次尝试保留legacy。B3配平每packet右逆后融合路径差仅−0.003dB，原非线性缺口为预处理混杂。根报告已更新完整数字。

已授权A固定4train/2val conditional-response路径一致实验，B3R固定32train/8val一seed learned compressed group evidence修订训练；保留set/ensemble/classic/factorcache/replayGRU强控制。resource另做一次固定case官方sampler/训练目标数学核验，不增训练。root GPU5释放，无后台任务。论文核心仍未确立，goal保持active。

## 2026-10-15T17:51:38.236150+08:00 原生训练IO一致性修复

resource固定case40557审计发现v1 FP32 stream IO与官方BF16 wrapper的数值差（v RMSE .02575，x0 .01130）；对齐后逐元素差0，其余目标/时间/guide路由正确。root已授权保留v1，先梯度/一致性审计，再从相同初始化和相同400预算重跑三控制v2，不追加旧checkpoint训练。GPU6继续由resource协调。

## 2026-10-15T19:47:31.050660+08:00 已修复数值路径与确认来源

v2官方BF16 IO parity/真反传71242通过，14233新三控制训练已完成LoRA400并进入49readout。A56actual H3样本缓存已完成，准备四头各300固定训练。发现BSDS全部test在早期已评价；99597以前台下载官方DIV2K filename-hash选32源，只字节校验并封存，不看像素和指标。

## 2026-10-15T20:09:08.501066+08:00 原生训练与真实视频历史反演

Root19599和43441均exit0：六源×22帧四控制及GT观测模型诊断完成。H3历史修订train平均+.12076dB，但val−.37376；flamingo/dance条纹伪影及GT更不符合估计观测模型已目视/数值确认，无参数调优。A四头1200实际D更新完成，loop未胜普通set；full24诊断没有揭示DiT独立增益。B3R真实检查通过后55254正式六控制600+200训练。原生IOv2全部1200更新及固定49采样完成，88349解码/评价中。Goal继续active，核心尚未确立。

## 2026-10-15T20:26:28.512224+08:00 完整结果、基线投影配平与新固定实验

v2 H3全训练/采样/评价/440PNG归档完成，native480 joint PSNR25.37960/LPIPS.374858，仍取舍。B3R六控制600+200及原预声明best-selection评价完成，full23.67001低于LRVAE23.84270和强LRphotoAtlas27.51398，delta0反而改善；61593另按统一端点完整补充，不把原预声明选择称错误。A统一严格精度LPIPS完成，无独立优势。Root历史反演归档发现有限投影非幂等，基线补齐最终25投影后全部t0逐位不变；逆问题输出不变，匹配val平均−.374052dB。新固定A条件函数压缩48H3样本+3×300与H3采样内三sigma DC强控已授权，不扫val/不触碰封存确认集。

## 2026-10-15T20:31:44.587531+08:00 本轮归档完成与实际信息损失定位

Root matched-final-projection六个float outputs已全部本地SHA验证，torch.equal核对逆问题输出与旧版逐位不变、全部t0最终对齐；95879/21174完成，根无GPU任务。B3R endpoint61593与单packet结构诊断16869均完成，prelift26.2379dB→learned23.5511，实际D监督稀疏（每模型50次）限制结论。现授权同一TRAIN源latent-L1/每步actualD-RGB各300明确诊断，不扫val。Relay官方Wan完整10,066,399,024bytes SHA通过40657；无GPU推理。37根文件42867已同步，确认集仍封存，核心目标保持active。


## 2026-10-15T20:58:59.096918+08:00 采样内观测约束与真实解码训练

H3固定innerDC pilot native480两val源同时提高PSNR/LPIPS，完整8源扩展98912/18691前台执行。B单TRAIN actualD-RGB目标提高至28.2840，latent-only26.0128；七控制各672真D训练29841已启动。A raw48轨迹及三头900更新完成仍未胜CNP/强逆问题，独立DC72版81881运行。ROOT Relay24case/528PNG96909完整；全12case原生空间LR-onlyflow补齐87809，统一评价17464启动。首次12674在LPIPS包metadata读取失败，尚无指标，已保留源码并按实际缺失记录而非安装依赖。Goal仍active，核心与真实长期记忆尚缺。


## 2026-10-15T21:24:10.990827+08:00 全量结果与结构性下一步

B4全部4704actualD训练与8源终评完成，full24.11443但anchor24.10618，未学到有用历史。A DC72+900训练也完成；完整prior/A/AB核验发现早期q已独立onehot且压缩postarrival，不能当因果jointmemory检验。H3固定innerDC8源结果native4807/8PSNR和8/8LPIPS胜VAE，完整1344文件SHA本地/远端一致。Relay24case/2112评价完成，SSH255经独立PID/完整输出核验非任务失败，不重跑。下一步检查有用观测detail从encoder到readout的丢失与anchorshortcut，构造直通信息路径；不扫超参。根64806统一A感知评价，resource1248纯CPU时序评价中。


## 2026-10-15T21:30:28.079334+08:00 根评价归档与下一架构预算

A统一96读出感知62541完成：DC mixed LPIPS两源胜CNP但PSNR/SSIM损失且仍弱于强反演，完整取舍保留。首次64806因少数control是float64不能直接进float32LPIPS失败，显式统一LPIPS输入精度后完成，不改模型输出。R全部时序完成，native480相对VAE时序均值略差，256全8源更差；不把LPIPS收益说成所有视频指标提高。新B2×2获授权4×672完整预算，独立分anchor神经捷径与已观测细节直通；先零点/成本核验和协议冻结，A独立审阅。ROOTGPU5无活跃任务，未使用封存数据。


## 2026-10-15T21:59:54.291118+08:00 — B5 completed; delayed future-LR validation

B5 70083/76701 exit0: all four672 budgets,2688actualD updates; physical bridge+2.67844dB but learned residual does not beat its zero-training atlas and neural-history interventions negligible. A pureCPU actual-H audit shows near-row band excludes useful GT directions; no B5 changes. R/A now design a genuinely pre-cue finite-byte near/far codec with conditional-DSC controls.

Root wrote and froze delayed_lr_validation.py protocol1f75680d1fd99931a19797543b8db12060858b389589a269d8e0720564051d07 before metrics. Score46475exit0,HR-evaluation57184exit0. Freeze reference21 withH3context0..21; observeonlynewLR22..37, primaryfirst4fixed. Six-source PSNRbase25.521713/always25.548109/gate25.702369,5/6signcorrect. VAL2still−.346624;dance falseaccept and longerhorizon nonmonotone. Fullprotocol/scoreslock/SHA/GTrows/report/plot saved; plot visually checked. 0newE/D/GPU/training. No sealed source read.
