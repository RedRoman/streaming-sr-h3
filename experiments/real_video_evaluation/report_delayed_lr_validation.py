"""Summarize the fixed delayed-LR diagnostic without selecting a horizon."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/real_video_history_inverse/delayed_lr_validation'
v = json.loads((OUT/'evaluation.json').read_text())
s = json.loads((OUT/'scores.json').read_text())
p = json.loads((OUT/'LOCKED_PROTOCOL.json').read_text())
rows = v['sources']
means = v['source_equal_means']
source_gain = [r['candidate_psnr_gain'] for r in rows]
gate_gain = [r['candidate_psnr_gain'] if r['retain_at25'] else 0 for r in rows]

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.spines.top':False, 'axes.spines.right':False})
fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.5), gridspec_kw={'width_ratios':[1.2,1]}, layout='constrained')
x = np.arange(6)
ax[0].bar(x-.18, source_gain, .36, label='Always retain inverse update', color='#98aec5')
ax[0].bar(x+.18, gate_gain, .36, label='Fixed 4-arrival LR gate', color='#177e89')
ax[0].axhline(0,color='#657084',linewidth=.8)
ax[0].set_xticks(x,[r['source'] for r in rows],rotation=25,ha='right')
ax[0].set_ylabel('Reference-frame PSNR change vs H3 VAE (dB)')
ax[0].set_title('Freeze at frame 21; decide at arrival 25')
ax[0].legend(fontsize=8,loc='lower left')
h = v['diagnostic_horizons_not_selected']
ax[1].plot([r['horizon'] for r in h],[r['source_equal_psnr'] for r in h],'-o',markersize=3,color='#177e89',label='LR gate')
ax[1].axhline(means['never_apply_base']['psnr'],color='#667085',linestyle='--',label='Never retain update')
ax[1].axhline(means['always_apply_candidate']['psnr'],color='#98aec5',linestyle=':',label='Always retain update')
ax[1].axvline(4,color='#b44355',linestyle='--',linewidth=1,label='Predeclared primary horizon')
ax[1].scatter([4],[h[3]['source_equal_psnr']],color='#b44355',zorder=4)
ax[1].set_xticks([1,2,4,8,12,16]); ax[1].set_xlabel('Number of genuinely later LR arrivals')
ax[1].set_ylabel('Six-source mean reference-frame PSNR (dB)')
ax[1].set_title('All horizons shown; none selected by GT')
ax[1].legend(fontsize=8,loc='lower right')
fig.suptitle('Delayed physical evidence can retract some harmful details',fontweight='bold')
fig.savefig(OUT/'source_and_horizon.png',dpi=180)
fig.savefig(OUT/'source_and_horizon.pdf')
plt.close(fig)

lines = [
    '# Delayed LR validation of a frozen H3-based history update', '',
    'This is a fixed classical diagnostic, not a learned core or a new compression method. '
    'All six previously used DAVIS train/val sources are included; clip-average failures were known beforehand. '
    'The primary four-arrival decision and source order were frozen before these scores or frame-21 GT metrics were read.', '',
    'The existing H3 VAE context and historical inverse candidates use only frames 0–21. '
    'We freeze their frame-21 outputs, then use actual unseen LR frames 22–37. '
    'Using earlier frames inside the 22-frame VAE block as “future validation” would leak future evidence; this protocol avoids that.', '',
    'For each later LR frame, the same LR-only DIS map warps the baseline and candidate. '
    'The fixed floating AA-bicubic camera predicts its observed LR. '
    'With w = confidence × 1(confidence≥0.5), score = sum w[(y−A base)^2−(y−A candidate)^2]. '
    'At arrival 25, retain the inverse candidate iff the cumulative first-four score is positive and '
    'its observable difference energy exceeds 1e−12; otherwise retain the baseline. '
    'No score threshold, source exclusion, or horizon is fitted. Scores and their SHA lock are written by '
    'a separate HR-free command before GT evaluation.', '',
    '## Primary result: frozen frame 21, read at arrival 25', '',
    '| State retained | Six-source PSNR | Six-source SSIM |',
    '|---|---:|---:|',
]
for name, label in [('never_apply_base','Never apply: H3 VAE + same final DC'),('always_apply_candidate','Always apply: historical inverse + same final DC'),('delayed_lr_gate','Fixed first-four delayed LR gate')]:
    m=means[name];lines.append(f"| {label} | {m['psnr']:.6f} | {m['ssim']:.6f} |")
lines += ['',f"The gate improves mean PSNR by {means['delayed_lr_gate']['psnr']-means['never_apply_base']['psnr']:.6f} dB over never applying and "
          f"{means['delayed_lr_gate']['psnr']-means['always_apply_candidate']['psnr']:.6f} dB over always applying. "
          f"It gets the HR PSNR sign right on {v['primary_sign_correct']}/6 sources. "
          'This evaluates a retrospectively retained frame-21 state after four new observations; '
          'it is not a frame-25 reconstruction or a zero-delay video output.', '',
          'Splitting the pre-existing source groups, TRAIN4 base/always/gate are '
          '25.513883/25.726789/25.958180 dB; VAL2 are 25.537373/25.190748/25.190748 dB. '
          'Thus the two validation sources still average −0.346624 dB relative to base: '
          'the positive six-source mean does not establish validation-set robustness.', '',
          '| Source | Inverse PSNR gain at t21 | Retain at25? | First4 LR MSE gain | First4 observable energy | Mean valid coverage |',
          '|---|---:|---|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['source']} | {r['candidate_psnr_gain']:+.6f} | {'yes' if r['retain_at25'] else 'no'} | {r['lr_gain_mse_first4']:+.4e} | {r['lr_observable_energy_mse_first4']:.4e} | {100*r['first4_mean_coverage']:.2f}% |")
lines += ['', 'The rule retracts the harmful bear and flamingo updates and retains the useful bus, car-turn and cows updates. '
          'It incorrectly retains dance-twirl: reference PSNR falls 0.857346 dB, coverage averages only 30.53%, '
          'and the normalized score is close to zero (6.93e−7). Confidence/coverage are diagnostics, not calibrated probabilities. '
          'The actual uint8 Pillow camera, estimated motion, deformation and visibility differ from the scoring model; '
          'low reprojection error is still not a guarantee of correct detail.', '',
          '## All horizons and scope', '',
          'At two and three future arrivals the sign happens to be correct on 6/6 sources; those are secondary descriptive results, '
          'not substitutes for the predeclared four-arrival result. Longer horizons sometimes re-admit flamingo and '
          'reject dance-twirl. More evidence under a misspecified correspondence model is not monotonically more reliable. '
          'Frames and horizons are repeated observations of six sources, not independent samples. '
          'No statistical generalization claim or sealed-set confirmation is made.', '',
          '![Source changes and full horizon curve](source_and_horizon.png)', '',
          'Future-warp masked HR MSE is also retained per source/frame in evaluation.json as a secondary diagnostic. '
          'Its first-four mean gain has the same sign as the reference-frame gain for each of the six sources, '
          'including a negative gain for dance-twirl. It evaluates a different target and is not mixed with the primary reference-frame result.', '',
          '## Constructive consequence', '',
          'Delayed LR observations provide an actionable, imperfect signal for withholding or retracting previously proposed details. '
          'A future learned memory should preserve enough alternative evidence to reconsider an interpretation, '
          'and should distinguish insufficient evidence from verified detail. The simple score alone does not resolve model bias, '
          'nor does it establish a novel architecture. Its positive result justifies testing such revisions on actual future observations.', '',
          '## Reproduction and cost', '',
          f"LR scoring: {s['cpu_seconds']:.6f}s; GT evaluation: {v['cpu_seconds']:.6f}s inside measured functions, excluding interpreter/import/hash setup where outside those timers. "
          'Four Torch CPU threads, one OpenCV thread; zero GPU, E, D, DiT or optimizer updates. '
          'Torch '+s['torch']+', OpenCV '+s['cv2']+'. Existing model inference and inverse-solving costs are not included in this incremental diagnostic.', '',
          '`experiments/real_video_evaluation/delayed_lr_validation.py` runs `freeze`, `score`, then `evaluate` in separate commands. '
          'Protocol/source SHA, all prefix input SHA, candidate SHA, future LR SHA and LR geometry SHA are checked. '
          'No HR image is opened by the scoring phase; no sealed DIV2K source is accessed.', '',
          'Protocol SHA256: `'+v['protocol_sha256']+'`.',
          'Scores SHA256: `'+v['scores_sha256']+'`.', '']
(OUT/'REPORT.md').write_text('\n'.join(lines))
manifest={str(path.relative_to(ROOT)):{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
          for path in sorted(OUT.iterdir()) if path.is_file() and path.name!='local_manifest.json'}
(OUT/'local_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'files':len(manifest),'report':str(OUT/'REPORT.md')}))
