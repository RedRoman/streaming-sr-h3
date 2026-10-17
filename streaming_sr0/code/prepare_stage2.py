"""Stage-2 data: 12 fresh VIDGEN clips, full 124 frames each (not a single block).
Same selection rule as Stage 1 (metadata only), excluding the 16 clips already used."""
import hashlib, json, re
from pathlib import Path
import av, numpy as np
from PIL import Image
from degrade import degrade
SRC = Path('/temp/liwei/vidgen1m-h3-5s-selected-493-20261017')
R = Path('/cache/zihaohe/streaming_sr0')
OUT = R / 'data/stage2_eval'
T, H, W, N = 124, 480, 832, 12
KW = r'(fabric|textile|cloth|knit|wool|sweater|lace|embroider|weav|fur|hair|feather|scale|leaf|leaves|grass|moss|bark|wood grain|carv|engrav|brick|stone|rust|mesh|net|pattern|texture|crochet|basket|straw|sand|gravel|crystal|bead)'
used = set(json.loads((R/'data/vidgen_eval/manifest.json').read_text())['clips'])
rows = [json.loads(l) for l in open(SRC/'dataset.jsonl')]
cand = []
for r in rows:
    if r['id'] in used: continue
    s = r.get('source', {})
    cap = (s.get('caption') or '') + ' ' + (r.get('enhanced_prompt') or '')
    hits = sorted(set(m.group(0).lower() for m in re.finditer(KW, cap, re.I)))
    cand.append({'id': r['id'], 'video': str(SRC/r['relative_video_path']), 'category': s.get('category'),
                 'quality': s.get('quality_score') or 0, 'vd': (s.get('h3_5s_scores') or {}).get('visible_detail', 0),
                 'kw': hits, 'n_kw': len(hits)})
cand.sort(key=lambda c: (-c['n_kw'], -c['vd'], -c['quality'], c['id']))
seen, picked = {}, []
for c in cand:
    k = c['category'] or 'none'
    if seen.get(k, 0) >= 2 or c['n_kw'] == 0: continue
    seen[k] = seen.get(k, 0)+1; picked.append(c)
    if len(picked) >= N: break
OUT.mkdir(parents=True, exist_ok=True)
recs = []
for c in picked:
    name = c['id']; d = OUT/name
    if (d/'meta.json').exists(): recs.append(json.loads((d/'meta.json').read_text())); continue
    (d/'hr').mkdir(parents=True, exist_ok=True); (d/'lr').mkdir(parents=True, exist_ok=True)
    cont = av.open(c['video']); frames = []
    for fr in cont.decode(video=0):
        frames.append(fr.to_ndarray(format='rgb24'))
        if len(frames) >= T: break
    cont.close()
    assert len(frames) == T, (name, len(frames))
    arr = np.stack(frames); h0, w0 = arr.shape[1:3]
    y0, x0 = (h0-H)//2, (w0-W)//2
    hr = arr[:, y0:y0+H, x0:x0+W]
    rng = np.random.default_rng(int(hashlib.sha256(('s2_'+name).encode()).hexdigest()[:8], 16))
    lr, dp = degrade(hr, rng)   # one parameter draw for the whole 124-frame clip
    files = {}
    for f in range(T):
        ph, pl = d/'hr'/f'{f:06d}.png', d/'lr'/f'{f:06d}.png'
        Image.fromarray(hr[f]).save(ph); Image.fromarray(lr[f]).save(pl)
        files[str(f)] = {'hr_sha256': hashlib.sha256(ph.read_bytes()).hexdigest(),
                         'lr_sha256': hashlib.sha256(pl.read_bytes()).hexdigest()}
    rec = {'id': name, 'video': c['video'], 'category': c['category'], 'kw': c['kw'], 'frames': T,
           'hr_wh': [W, H], 'crop_xy': [x0, y0], 'degradation': dp, 'frame_records': files}
    (d/'meta.json').write_text(json.dumps(rec, indent=1)); recs.append(rec)
    print(json.dumps({'clip': name, 'cat': c['category'], 'deg': dp.get('kind'), 'crf': dp.get('h264_crf'), 'jpeg': dp.get('jpeg_q')}), flush=True)
(OUT/'manifest.json').write_text(json.dumps({'n': len(recs), 'frames': T, 'hr_wh': [W, H],
    'blocks': [{'k': k, 'range': [17*k, 17*k+22], 'new': [17*k+5 if k else 0, 17*k+22]} for k in range((T-22)//17+1)],
    'excluded_stage1': sorted(used), 'clips': [r['id'] for r in recs]}, indent=1))
print('prepared', len(recs), 'categories', seen)
