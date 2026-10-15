# Fixed historical-band CPU diagnostic

Existing8val sources,7original queries/source. Actual B far/near camera matrices match their simulator to2.98e−7/2.38e−7. This is one application of an evaluation-only GT correction direction, with no E/D/DiT/GPU/optimization. It does not change B5 and is not an optimized oracle upper bound.

| GT-direction readout | Full PSNR after sameDC | Active-union PSNR after sameDC |
|---|---:|---:|
|GT_direction_far_null_spatial_support|27.807383|46.092119|
|GT_direction_full_far_null|46.926683|46.758655|
|GT_direction_hard_near_row|25.915396|27.713089|
|base|24.163627|23.842698|

The pooled active-crop near-null error-energy ratio is40.6368% on average. This concerns the far-null GT residual of the current LR-VAE base, not the accuracy of an actual generated detail proposal. Crops overlap, so this is an operator diagnostic rather than a count of independent pixels. H includes weighted blending and outer far-null projection and is not itself an orthogonal projector; the ratio is not a theorem that40.6% of all final-image error is irrecoverable.

Removing only the near row projector P_near, with the same actual confidence/count/spatial support and far-null operations, changes the GT-direction active-union result by+13.8153 to+21.7659dB across all8sources. Both branches preserve the far measurement to RMS below5.1e−9 for their correction directions. The spatial branch has no access to that GT direction at inference: its much higher value only demonstrates which useful directions the hard-band version mechanically excludes. It is not a valid SR baseline or evidence of learnable H3 headroom at this accuracy.

For the next causal core, keep the fixed B5 hard-band result as an information-transfer diagnosis. A separate generative branch may use far-null plus operational spatial support or soft observation likelihood to allow prior-correlated high frequencies. It must still pass real LR residuals, true HF error, unrelated/private query stability and same-source shared-prior/strong-inverse controls. Do not substitute a random PCA freeze for this measurement audit.

Runtime4.837s CPU;224direction/readout cases. All GT use is evaluation-only and no sealed source is touched.
