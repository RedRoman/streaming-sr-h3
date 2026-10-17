"""Select revisit events (annotation-only rule in PROTOCOL_step0.md) and materialize HR/LR frames for their blocks."""
import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import scan_lasot as S  # noqa: E402

ROOT = Path('/cache/zihaohe/streaming_sr0')
MAX_EVENTS, PER_CATEGORY = 32, 3


def natkey(seq):
    return int(re.search(r'-(\d+)$', seq).group(1))


def select(scan_dir):
    per_cat = {}
    for f in sorted(scan_dir.glob('*.json')):
        if f.name in ('protocol.json', 'summary.json'):
            continue
        d = json.loads(f.read_text())
        picks = []
        for r in sorted(d['sequences'], key=lambda r: natkey(r['sequence'])):
            ev = [e for e in r['events'] if e['pass_primary']]
            if ev:
                picks.append((r, max(ev, key=lambda e: (e['absent_frames'], -e['absent_start']))))
        per_cat[d['category']] = picks[:PER_CATEGORY]
    chosen, rnd = [], 0
    while len(chosen) < MAX_EVENTS and any(len(v) > rnd for v in per_cat.values()):
        for cat in sorted(per_cat):
            if rnd < len(per_cat[cat]) and len(chosen) < MAX_EVENTS:
                chosen.append((cat, *per_cat[cat][rnd]))
        rnd += 1
    return chosen


def outputs(k):
    return range(0, 22) if k == 0 else range(17 * k + 5, 17 * k + 22)


def best_block(lo, hi, nframes):
    best = None
    for k in range(0, (nframes - 22) // 17 + 1):
        inter = len(set(outputs(k)) & set(range(lo, hi)))
        if inter and (best is None or inter > best[1]):
            best = (k, inter)
    return best[0] if best else None


def pick(frames, n=3):
    frames = list(frames)
    if len(frames) <= n:
        return frames
    return [frames[round(i * (len(frames) - 1) / (n - 1))] for i in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scan', type=Path, default=ROOT / 'data/lasot_scan')
    p.add_argument('--out', type=Path, default=ROOT / 'data/events')
    p.add_argument('--cache', type=Path, default=ROOT / 'data/lasot_ranges')
    p.add_argument('--select-only', action='store_true')
    a = p.parse_args()
    tree = {x['path'][:-4]: x['size'] for x in json.loads((ROOT / 'data/lasot_mirror_tree.json').read_text()) if str(x.get('path', '')).endswith('.zip')}
    chosen = select(a.scan)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cat, row, ev in chosen:
        seq, n = row['sequence'], row['frames']
        s, r = ev['absent_start'], ev['return']
        kpre, kpost = best_block(s - 22, s - 3, n), best_block(r + 3, r + 22, n)
        kctl = kpre - 1 if kpre and kpre > 0 else None
        blocks = {'kpre': kpre, 'kpost': kpost, **({'kctl': kctl} if kctl is not None else {})}
        rec = {'category': cat, 'sequence': seq, 'frames': n, 'source_wh': row['source_wh'], 'hr_scale': row['hr_scale'],
               'hr_crop_xywh': row['hr_crop_xywh'], 'event': ev, 'blocks': blocks,
               'pairs_revisit_a': pick(f for f in outputs(kpre) if s - 22 <= f < s - 3),
               'pairs_revisit_b': pick(f for f in outputs(kpost) if r + 3 <= f < r + 22),
               'pair_control': [max(outputs(kctl)), min(outputs(kpre))] if kctl is not None else None}
        manifest.append(rec)
    (a.out / 'selection.json').write_text(json.dumps({'protocol': 'PROTOCOL_step0.md', 'events': manifest}, indent=1))
    print(json.dumps({'selected': len(manifest), 'categories': sorted({m['category'] for m in manifest})}), flush=True)
    if a.select_only:
        return
    for rec in manifest:
        cat, seq = rec['category'], rec['sequence']
        size = tree[cat]
        entries, _ = S.central(cat, size, a.cache)
        names = {e['name']: e for e in entries}
        imgs = sorted(nm for nm in names if nm.startswith(f'{seq}/img/') and nm.endswith('.jpg'))
        ann = S.prefetch(cat, size, [names[f'{seq}/{f}'] for f in ('full_occlusion.txt', 'out_of_view.txt', 'groundtruth.txt')], a.cache)
        flag = lambda f: [int(x) for x in ann[f'{seq}/{f}'].decode().replace('\n', ',').split(',') if x.strip()]
        absent = [bool(x or y) for x, y in zip(flag('full_occlusion.txt'), flag('out_of_view.txt'))]
        boxes = [[float(v) for v in line.split(',')] for line in ann[f'{seq}/groundtruth.txt'].decode().strip().splitlines()]
        need = sorted({f for k in rec['blocks'].values() for f in range(17 * k, 17 * k + 22)})
        raw = S.prefetch(cat, size, [names[imgs[f]] for f in need], a.cache)
        sc, (cx, cy, cw, ch) = rec['hr_scale'], rec['hr_crop_xywh']
        folder = a.out / seq
        (folder / 'hr').mkdir(parents=True, exist_ok=True)
        (folder / 'lr').mkdir(parents=True, exist_ok=True)
        frames = {}
        for f in need:
            im = Image.open(io.BytesIO(raw[imgs[f]])).convert('RGB')
            if sc != 1.0:
                im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
            hr = im.crop((cx, cy, cx + cw, cy + ch))
            lr = hr.resize((cw // 4, ch // 4), Image.BICUBIC)
            hp, lp = folder / 'hr' / f'{f:06d}.png', folder / 'lr' / f'{f:06d}.png'
            hr.save(hp); lr.save(lp)
            x, y, w, h = boxes[f]
            frames[f] = {'absent': absent[f], 'bbox_hr_xywh': [x * sc - cx, y * sc - cy, w * sc, h * sc],
                         'source_sha256': hashlib.sha256(raw[imgs[f]]).hexdigest(),
                         'hr_sha256': hashlib.sha256(hp.read_bytes()).hexdigest(), 'lr_sha256': hashlib.sha256(lp.read_bytes()).hexdigest()}
        rec['hr_wh'], rec['lr_wh'] = [cw, ch], [cw // 4, ch // 4]
        (folder / 'event.json').write_text(json.dumps({**rec, 'frame_records': frames}, indent=1))
        print(json.dumps({'event': seq, 'blocks': rec['blocks'], 'frames': len(need)}), flush=True)


if __name__ == '__main__':
    main()
