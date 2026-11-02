"""Build the VIDGEN eval set: HR = 832x480 centre crop of the native 1344x768 frames (no rescaling, so the
ground truth keeps its full detail), LR = the same clip-consistent degradation used in training.
One 22-frame block per clip, taken from the middle of the clip."""
import hashlib, json
from pathlib import Path
import av, numpy as np
from PIL import Image
from degrade import degrade
R = Path('/cache/zihaohe/streaming_sr0')
OUT = R / 'data/vidgen_eval'
T, H, W = 22, 480, 832
sel = json.loads((OUT / 'selection.json').read_text())
recs = []
for c in sel['clips']:
    name = c['id']
    d = OUT / name
    if (d / 'meta.json').exists():
        recs.append(json.loads((d / 'meta.json').read_text())); continue
    (d / 'hr').mkdir(parents=True, exist_ok=True); (d / 'lr').mkdir(parents=True, exist_ok=True)
    cont = av.open(c['video']); vs = cont.streams.video[0]
    n = vs.frames or 124
    start = max(0, n // 2 - T // 2)
    frames, i = [], 0
    for fr in cont.decode(video=0):
        if i >= start and len(frames) < T: frames.append(fr.to_ndarray(format='rgb24'))
        i += 1
        if len(frames) >= T: break
    cont.close()
    assert len(frames) == T, (name, len(frames))
    arr = np.stack(frames)
    h0, w0 = arr.shape[1:3]
    y0, x0 = (h0 - H) // 2, (w0 - W) // 2
    hr = arr[:, y0:y0 + H, x0:x0 + W]
    rng = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    lr, dp = degrade(hr, rng)
    files = {}
    for j, f in enumerate(range(start, start + T)):
        ph, pl = d / 'hr' / f'{f:06d}.png', d / 'lr' / f'{f:06d}.png'
        Image.fromarray(hr[j]).save(ph); Image.fromarray(lr[j]).save(pl)
        files[str(f)] = {'hr_sha256': hashlib.sha256(ph.read_bytes()).hexdigest(),
                         'lr_sha256': hashlib.sha256(pl.read_bytes()).hexdigest()}
    rec = {'id': name, 'video': c['video'], 'category': c['category'], 'kw': c['kw'],
           'source_wh': [w0, h0], 'crop_xy': [x0, y0], 'hr_wh': [W, H], 'frames': list(range(start, start + T)),
           'degradation': dp, 'frame_records': files}
    (d / 'meta.json').write_text(json.dumps(rec, indent=1))
    recs.append(rec)
    print(json.dumps({'clip': name, 'frames': [rec['frames'][0], rec['frames'][-1]], 'deg': dp.get('kind'),
                      'crf': dp.get('h264_crf'), 'jpeg': dp.get('jpeg_q')}), flush=True)
(OUT / 'manifest.json').write_text(json.dumps({'n': len(recs), 'hr_wh': [W, H], 'frames_per_clip': T,
    'hr_rule': 'centre crop of native 1344x768, no rescaling', 'clips': [r['id'] for r in recs]}, indent=1))
print('prepared', len(recs))
