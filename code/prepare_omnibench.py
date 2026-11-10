"""OmniBench as an eval set, and OmniVR's own released outputs as the first joint audio-video baseline.

Until now every baseline in this project restores video only (FlashVSR+, InfVSR, RealBasicVSR,
Real-ESRGAN), so the joint audio-video claim had no peer. OmniVR published its full benchmark AND its own
prediction videos, which makes the comparison possible without running their model: 71 with_gt clips
(LQ synthesised from HQ sources, GT aligned frame by frame, 48 kHz audio with legacy old-film degradation)
and 129 no_gt real old-film clips, all 121 frames at 24 fps, plus predictions for both tracks.

Comparing against their released outputs is better than running their code: it removes any chance that we
misconfigured their model, and it is the configuration they chose to publish.

Three preprocessing decisions, each forced and each recorded in every meta.json:

  their 1088 rows against the benchmark's 1080.  Their README says dimensions are padded to a multiple of
  32 and cropped back, which would mean finding the right 8 rows. Their code says otherwise: infer.py
  defaults --height to 1088 and line 196 is
      video = F.interpolate(video, size=(target_h, target_w), mode="bilinear", align_corners=False)
  so the input is RESIZED 1080 -> 1088 and the output stays at 1088. We resize back, bilinear, mirroring
  them. This was not decidable from the pixels: a Laplacian cross-correlation registration scored 0.3064
  for the best of nine crops against 0.3024 for the resize (results/deploy/omnibench_align2.json), i.e.
  indistinguishable, and the raw-MAE argmin wandered over 0-6 rows across clips. The source settled it.

  geometry.  Frames are 1920x1080 and every number in this project is 832x480. An 832x480 window is cut
  by the same most-null-energy rule as mvsr4x / realvsr / videolq, so only the degradation source changes.

  the input handicap, which must be stated wherever these numbers appear.  OmniVR restores at the source
  resolution; this project is an x4 chain, so our observation is the window downsampled by four, 208x120.
  Our arm therefore sees 1/16 of the pixels theirs sees. That makes the comparison conservative for us,
  not favourable, and a win under it means more than the raw numbers do -- but a loss is partly explained
  by it, so the handicap is recorded in the manifest rather than left for a reader to notice.

121 frames does not fit the 17/22 block plan, so the last block is [99,121] emitting [107,121]; seven
blocks and six handovers per clip, the same as the 124-frame sets.
"""
import argparse, hashlib, json, sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
from prepare_mvsr4x import null_energy_window, radial_mtf

OB = Path('/cache/zihaohe/baselines/omnibench')
DST = ROOT / 'data/real_eval/omnibench'
PRED = ROOT / 'results/baseline_omnivr_omnibench'
CROP_W, CROP_H = 832, 480
SCALE = 4
FRAMES = 121
SR = 48000
BLOCKS = [{'k': k, 'range': [17 * k, 17 * k + 22], 'new': [17 * k + 5 if k else 0, 17 * k + 22]}
          for k in range(6)] + [{'k': 6, 'range': [99, 121], 'new': [107, 121]}]


def read_video(p, n=FRAMES):
    out = []
    with av.open(str(p)) as c:
        for f in c.decode(video=0):
            out.append(f.to_ndarray(format='rgb24'))
            if len(out) >= n:
                break
    return np.stack(out) if out else None


def read_audio(p, sr=SR):
    with av.open(str(p)) as c:
        if not c.streams.audio:
            return None
        rs = av.audio.resampler.AudioResampler(format='s16p', layout='stereo', rate=sr)
        acc = []
        for fr in c.decode(audio=0):
            for g in rs.resample(fr):
                acc.append(g.to_ndarray().copy())
    return (np.concatenate(acc, axis=-1).astype(np.float32) / 32768.0) if acc else None


def to_1080(v, h=1080):
    """their 1088 rows back to the benchmark's 1080, the inverse of their own input resize"""
    if v.shape[1] == h:
        return v
    t = torch.from_numpy(v).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(h, v.shape[2]), mode='bilinear', align_corners=False)
    return t.permute(0, 2, 3, 1).clamp(0, 255).numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='with_gt', choices=['with_gt', 'no_gt'])
    ap.add_argument('--clips', type=int, default=0)
    a = ap.parse_args()
    from operator_desc import operator_mtf
    assumed = operator_mtf(SCALE)
    wg = a.group == 'with_gt'
    lqd = OB / ('omnibench/with_gt/lq' if wg else 'omnibench/no_gt')
    gtd = OB / 'omnibench/with_gt/gt'
    prd = OB / ('predictions/withgt_step02400_neg_cfg' if wg else 'predictions/nogt_step01800_neg_cfg')
    dst = DST if wg else DST.with_name('omnibench_nogt')
    dst.mkdir(parents=True, exist_ok=True)
    pred_out = PRED if wg else Path(str(PRED) + '_nogt')

    names = sorted(x.name for x in lqd.glob('*.mp4'))
    if a.clips:
        names = names[:a.clips]
    kept, gaps = [], []
    for nm in names:
        sid = nm[:-4]
        lq = read_video(lqd / nm)
        gt = read_video(gtd / nm) if wg else None
        pr = read_video(prd / nm)
        if lq is None or pr is None or (wg and gt is None):
            print(json.dumps({'skip': sid[:28], 'reason': 'decode'}), flush=True)
            continue
        T = min(FRAMES, lq.shape[0], pr.shape[0], gt.shape[0] if wg else FRAMES)
        if T < FRAMES:
            print(json.dumps({'skip': sid[:28], 'reason': 'frames', 'T': int(T)}), flush=True)
            continue
        pr = to_1080(pr[:T], lq.shape[1])
        ref = gt if wg else lq
        cy, cx = null_energy_window(ref[0])
        out = dst / sid
        (out / 'lr').mkdir(parents=True, exist_ok=True)
        if wg:
            (out / 'hr').mkdir(parents=True, exist_ok=True)
        (pred_out / sid / 'stream').mkdir(parents=True, exist_ok=True)
        mtf_acc, ctl_acc = [], []
        for t in range(FRAMES):
            w_lq = lq[t][cy:cy + CROP_H, cx:cx + CROP_W]
            lr = cv2.resize(w_lq, (CROP_W // SCALE, CROP_H // SCALE), interpolation=cv2.INTER_CUBIC)
            Image.fromarray(lr).save(out / 'lr' / f'{t:06d}.png')
            Image.fromarray(pr[t][cy:cy + CROP_H, cx:cx + CROP_W]).save(
                pred_out / sid / 'stream' / f'{t:06d}.png')
            if wg:
                w_gt = gt[t][cy:cy + CROP_H, cx:cx + CROP_W]
                Image.fromarray(w_gt).save(out / 'hr' / f'{t:06d}.png')
                if t % 40 == 0:
                    up = cv2.resize(lr, (CROP_W, CROP_H), interpolation=cv2.INTER_CUBIC)
                    mtf_acc.append(radial_mtf(up, w_gt).numpy())
                    syn = cv2.resize(cv2.resize(w_gt, (CROP_W // SCALE, CROP_H // SCALE),
                                                interpolation=cv2.INTER_CUBIC),
                                     (CROP_W, CROP_H), interpolation=cv2.INTER_CUBIC)
                    ctl_acc.append(radial_mtf(syn, w_gt).numpy())
        au_lq, au_gt = read_audio(lqd / nm), (read_audio(gtd / nm) if wg else None)
        au_pr = read_audio(prd / nm)
        if au_lq is not None:
            np.save(out / 'audio_lr.npy', au_lq)
        if au_gt is not None:
            np.save(out / 'audio_gt.npy', au_gt)
        if au_pr is not None:
            np.save(pred_out / sid / 'audio.npy', au_pr)
        meta = {
            'id': sid, 'source': 'omnibench', 'group': a.group,
            'category': ('synthetic_legacy_degradation_with_reference' if wg
                         else 'real_old_film_no_reference'),
            'frames': FRAMES, 'hr_wh': [CROP_W, CROP_H], 'lr_wh': [CROP_W // SCALE, CROP_H // SCALE],
            'orig_wh': [int(lq.shape[2]), int(lq.shape[1])], 'crop_yx': [int(cy), int(cx)],
            'degradation': {'kind': 'realistic', 'origin': 'omnivr_legacy_pipeline' if wg else 'real_old_film'},
            'has_hr': bool(wg), 'audio_sr': SR,
            'audio_present': {'lq': au_lq is not None, 'gt': au_gt is not None, 'omnivr': au_pr is not None},
            'baseline': {'name': 'OmniVR', 'checkpoint': 'step_02400' if wg else 'step_01800',
                         'strategy': 'neg_cfg', 'source': 'authors released predictions, not re-run here',
                         'height_handling': 'their output is 1088 rows; resized back to %d bilinear, '
                                            'mirroring infer.py line 196 which resizes the input '
                                            '1080 -> 1088 the same way' % lq.shape[1]},
            'input_handicap': 'OmniVR restores at the source resolution (1920x1088). Our arm is an x4 '
                              'chain and receives this window downsampled by four (208x120), i.e. 1/16 '
                              'of the pixels. The comparison is conservative for us.',
        }
        if wg and mtf_acc:
            m, c = np.mean(mtf_acc, 0), np.mean(ctl_acc, 0)
            g = float(np.linalg.norm(m - c)); gaps.append(g)
            meta['assumed_operator'] = {
                'D': 'bicubic x4',
                'measured_radial_mtf': [round(float(x), 4) for x in m],
                'synthetic_x4_control_mtf': [round(float(x), 4) for x in c],
                'assumed_radial_mtf_white_noise_probe': [round(float(x), 4) for x in assumed],
                'l2_gap_vs_same_measurement_x4': g,
                'note': 'same convention as mvsr4x and realvsr'}
        (out / 'meta.json').write_text(json.dumps(meta, indent=1))
        kept.append(sid)
        print(json.dumps({'clip': sid[:28], 'crop_yx': [int(cy), int(cx)],
                          'gap': round(gaps[-1], 4) if gaps else None,
                          'audio': meta['audio_present']}), flush=True)

    man = {'clips': kept, 'blocks': BLOCKS, 'frames': FRAMES,
           'lr_wh': [CROP_W // SCALE, CROP_H // SCALE], 'hr_wh': [CROP_W, CROP_H],
           'has_hr': bool(wg), 'source': 'omnibench/' + a.group, 'audio_sr': SR,
           'baseline_arm': str(pred_out),
           'note': 'OmniBench (OmniVR). The first set here with a JOINT audio-video baseline. Read '
                   'input_handicap in any meta.json before quoting a comparison.'}
    (dst / 'manifest.json').write_text(json.dumps(man, indent=1))
    print()
    print(json.dumps({'written': len(kept), 'group': a.group, 'dst': str(dst),
                      'baseline_arm': str(pred_out), 'blocks': len(BLOCKS),
                      'mtf_gap_mean': round(float(np.mean(gaps)), 4) if gaps else None}, indent=1))


if __name__ == '__main__':
    main()
