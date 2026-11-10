"""Where is OmniVR's 1088-row output aligned against the 1080-row ground truth?

Their README says dimensions are padded to multiples of 32 internally and cropped back, but the released
with_gt predictions are 1920x1088 while the LQ and GT are 1920x1080. Eight rows of padding sit somewhere,
and cropping the wrong eight rows would shift their frames by up to 8 pixels. On 480-row crops that is a
1.7% vertical shift, which destroys PSNR and LPIPS -- it would make their arm look far worse than it is
and the conclusion would be an artefact of our own preprocessing.

So measure the offset instead of assuming it: for each candidate offset dy in 0..8, score
mean |pred[dy:dy+1080] - LQ| and report the argmin. The LQ is the reference rather than the GT because
a restoration stays geometrically aligned to its input even where it changes appearance.
"""
import json, sys
from pathlib import Path

import av
import numpy as np

ROOT = Path('/cache/zihaohe/streaming_sr0')
OB = Path('/cache/zihaohe/baselines/omnibench')
NCLIP = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def frames(p, n=3, stride=40):
    out = []
    with av.open(str(p)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i % stride == 0:
                out.append(f.to_ndarray(format='rgb24'))
            if len(out) >= n:
                break
    return out


def main():
    names = sorted(x.name for x in (OB / 'omnibench/with_gt/lq').glob('*.mp4'))[:NCLIP]
    per, best = {}, []
    for n in names:
        lq = frames(OB / 'omnibench/with_gt/lq' / n)
        pr = frames(OB / 'predictions/withgt_step02400_neg_cfg' / n)
        if not lq or not pr or pr[0].shape[0] <= lq[0].shape[0]:
            per[n] = {'skip': True, 'lq': list(lq[0].shape) if lq else None,
                      'pred': list(pr[0].shape) if pr else None}
            continue
        H = lq[0].shape[0]
        pad = pr[0].shape[0] - H
        sc = []
        for dy in range(pad + 1):
            e = np.mean([np.abs(p[dy:dy + H].astype(np.float32) - l.astype(np.float32)).mean()
                         for p, l in zip(pr, lq)])
            sc.append(round(float(e), 3))
        per[n] = {'pad_rows': pad, 'mae_by_offset': sc, 'argmin': int(np.argmin(sc)),
                  'spread': round(float(max(sc) - min(sc)), 3)}
        best.append(int(np.argmin(sc)))
        print(json.dumps({'clip': n[:28], **per[n]}), flush=True)
    res = {'clips': len(best), 'offsets': best,
           'consensus': int(np.bincount(best).argmax()) if best else None,
           'unanimous': len(set(best)) == 1 if best else None, 'per_clip': per}
    (ROOT / 'results/deploy/omnibench_align.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'per_clip'}, indent=1))


main()
