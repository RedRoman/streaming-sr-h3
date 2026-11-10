streaming-sr-h3 — delivery videos, 2026-11-03

Clips are chosen by a rule, never by eye. Where a judge file exists the worst clips by OUR OWN metric
come first, so the cases that argue against us lead. The 256x256 zoom crops are placed by the window
holding the most detail the x4 operator discards, not by hand.


1_synthetic_x4/           the headline comparison
  Tiles: LR (bicubic x4) | InfVSR | FlashVSR+ | ours 2-step | ground truth
  On 77 clips we lead all four reference metrics: LPIPS 0.1929 vs 0.2072, PSNR 25.78 vs 24.23,
  and both GT-range variants. Noise floor is 0.0009 LPIPS.

2_audio_joint/            the one result you can hear
  _audio_degraded.mp4     same video, audio NOT restored      (mel 82.20 dB on this clip)
  _audio_restored.mp4     same video, audio restored          (mel  5.33 dB)
  _video_identical.mp4    |video_A - video_B| amplified 32x
  The third file is the claim, not an illustration: restoring the audio must not disturb the video.
  Over 77 clips the video stream is bit-identical between the two arms on 77 of 77. If that file is
  not pure black, the claim is false. It is 0.03 MB because an all-black video compresses to nothing.
  Audio over 77 clips: mel 28.247 -> 8.718 dB, closing 81.3% of the reachable gap (codec floor 3.725).

3_x6_unseen_operator/     an operator no arm ever trained on
  Tiles: LR (bicubic x6) | ours | ground truth
  x6 fills 52.0% of the bicubic LPIPS gap where a trained x4 fills 53.0%. The one-hot conditioned
  arm cannot be evaluated here at all -- it raises an error rather than scoring badly, because its
  vocabulary has no slot for x6.

4_real_optics_mvsr4x/     real degradation WITH a reference
  Tiles: LQ | InfVSR | FlashVSR+ | ours | ground truth
  Phone wide/tele paired capture, so the degradation is a real lens. Every generative method falls
  below a near-no-op bicubic in PSNR here -- ours by 0.84 dB, InfVSR by 1.18, FlashVSR+ by 1.90 --
  because parallax and registration error punish synthesised detail. We hold the fidelity side of the
  trade: highest PSNR of the three, null energy nearest the truth. FlashVSR+ holds the sharpness side
  and leads LPIPS.

5_real_degradation_videolq/   real degradation, NO reference
  Tiles: LR | ours r5_k3 | ours omni_m5
  No ground truth exists, and we measured that no-reference metrics cannot be trusted on this task:
  a NIQE-family score fitted on our own ground truth cannot separate FlashVSR+ from the ground truth
  (median 3.01 vs 2.98). So the frames are the deliverable, not a score.


Two things to look at that the numbers do not show well:

  Block handovers. The video is generated 22 frames at a time. Until today the overlap was cut rather
  than blended, so the picture jumped every 17 frames. These renders predate that fix; the fix reduces
  the jump 63% at zero cost to image quality and is now part of the delivery path. If you see periodic
  jumps, that is what they are, and they are already fixed for the next render.

  Flat regions. Our generated detail has the highest directional accuracy of any arm measured
  (null_cos 0.4876 against FlashVSR+ 0.3970) but only 0.544 of the truth's energy, so we are the least
  hallucinatory and the least sharp. FlashVSR+ puts back 0.845 of the energy with less of it pointing
  the right way. Whether that trade is set correctly is a judgement call, and it is the main thing
  worth your eyes.
