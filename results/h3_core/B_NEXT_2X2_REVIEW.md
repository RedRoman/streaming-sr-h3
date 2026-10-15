# Independent review of the next B2×2 protocol

Status: concrete implementation read-only review completed. Read experiments/h3_factorized/B5_FACTORIAL_PROTOCOL.md, revision_bridge.py, train_bridge_revision.py, cache_bridge_phi.py and check_bridge_revision.py. No blocking issue found in the source; actual acceptance JSON was independently read and all required structural checks pass. No GPU or new training was started by this reviewer. Reviewed current `revision_model.py`, `revision_dense.py`, `revision_utils.py`, `cache_revision.py`, `train_decoded_revision.py`, and decoded_v4's fixed-last672 report.

## Required common forward and zero point

Let C=D(z_current) be the authoritative native-route current-anchor RGB. Let B_b be the shared base for bridge mode b∈{off,on}, before the common final DC. Let H_t be the common historical-support readout. A valid additive form is

x_raw = B_b + H_t[D(z_current + δ_phi) − C],
x_final = DC_current(x_raw).

The subtraction is C, not B_b and not a separately decoded bridge code. Otherwise δ=0 can subtract/duplicate the measured bridge. B_b must itself use the same declared support/photometric convention in all bridge-enabled arms. A zero correction must reproduce the *same base through the same finalDC route*; running a second64-step bounded projection on a previously projected base can produce a nonzero numerical difference and must not be confused with model failure. Route/dtype-matched C or an audited runtime C is required.

The current `Readout.historical_band` is linear in its RGB argument when weights are held fixed. Thus this additive residual form is coherent, including its two current-null operations and per-packet near-support weighting. It is a controlled readout, not an exact nonlinear native-coordinate nullspace.

## What no-anchor must mean

Remove current z from neural readout features, not merely its gradient. Detach(z) still exposes all current content. Replacing it with zero while retaining unused weights creates inactive parameters and requires an active-parameter audit. Mechanically match usable width/parameters if input channels differ. The frozen physical D(z_current+δ) necessarily retains current z in every arm; this is correct. Appearance/reliability metadata can remain shared if reported. This condition is therefore no neural current-native-code conditioning, not blindness to every current signal.

Even a no-z neural head can make an input-independent δ that a nonlinear D applies differently to different anchors. Therefore history use remains an empirical intervention question. Zero evidence, nonzero parameter gradients and a subtraction f(h,z)−f(0,z) do not guarantee useful historical dependence: a nearly constant nonzero h can enable an anchor-conditioned SR shortcut. Keep drop/shuffle interventions on the neural history while holding a correct bridge fixed, and compare each learned arm against its own0-correction base.

## Base attribution, provenance and costs

If the photometric bridge uses each arm's learned appearance estimate, its base changes with both architecture and training. Zero correction can still be correct, but gains are no longer attributable only to the decoder residual. Prefer a common external causal LR appearance/controller for phi; otherwise separately report each arm's B-only output and gradient path as well as a common fixed bridge.

`cache_revision.py` builds `packet_full_latents` from `view(refup,nearLR)` and its proxy from that same LR composite. This is a valid observed carrier, not GT. Bridge decoding should use one authoritative saved-full-code route across arms: D(fullcode) and D(delta+proxy reconstructed in another precision) have already been measured to differ slightly. Do not silently switch them between conditions.

All arms should retain the same native/proxy/archive fields and actually execute the same phi calls, as required by the frozen design. Count retained full-code/proxy/weights/IDs/controller/query scratch, adapter parameters, actualD calls and runtime separately. A4ch h state cannot be presented as the whole state once the full native observations are also retained. Bridge-off phi can be a deliberate compute control; it does not justify counting unused neural weights as active parameters.

The evidence-bearing bridge itself is a strong established baseline. The factorial main effect of bridge tests information transmission, while the anchor-conditioning interaction tests the current-image shortcut. Neither establishes a new memory mechanism by itself. Improvement attributable to learned history requires beating the shared fixed bridge and remaining sourcewise sensitive to correct rather than shuffled measurements.

Concrete calls match the declared batch1 shape: near_LR[N,3,64,64], phi list entries[1,3,256,256]. Native full/proxy recomputation, detached external-q base, single currentD subtraction, no neural z in A0, mechanically chosen decoder width and fresh zero final head are implemented. The672 training schedule uses only TRAIN and backward stays inside the audited decoder context. All arms share one explicitly amortized actualD phi/base cache, rather than recomputing phi at every training update.

Two nonblocking checks were sent to the owner: assert a nonzero anchor-poison effect in A1 as well as zero effect in A0 after the initialization update; and record the small auxiliary route difference between legacy half-rounded neutral_delta and the new full.float−proxy.float observed delta. The primary RGB graph is unchanged.

Scope for the next core: historical_band includes P_near r P_near^T and therefore mechanically removes near-measurement null directions, even if a generative prior predicts useful correlated detail there. This remains fixed in B5 for diagnosis. A separate CPU-only GT-direction audit compares the current hard filter with the same far-null/spatial support without P_near; it is an evaluation diagnostic and does not authorize or modify a training branch.

The actual `results/h3_factorized/factorial_v5/acceptance/acceptance.json` was independently inspected. All four raw/one-DC zero-correction differences are exactly0; cached phi route maxabs0; active parameters146192/146578 differ0.264%. A1 neural-anchor poisoning changes prediction1.80e−4/2.06e−4, while A0 is exactly0. First RGB-only backward has nonzero head and zero encoder/core gradients as required; after one fixed update both encoder/core gradients are positive, though small. Frozen VAE gradients, future neural/base poisoning and empty-evidence output differences are0. These are finite interface/causality checks, not evidence of a learned historical-quality contribution. The owner has begun the independent four×672 foreground run; this reviewer did not start or alter it.
