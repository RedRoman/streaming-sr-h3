"""Revisit identity metrics for PROTOCOL_step0.md (CPU). GT is used only for registration and evaluation."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
os.environ.setdefault('TORCH_HOME', str(ROOT / 'torch_home'))


def load(p):
    return np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255


def unobserved(x):
    """N(x) = x - U(D(x)): the component a x4 antialiased-bicubic observation does not determine."""
    t = torch.from_numpy(x).permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // 4, w // 4), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    return (t - u)[0].permute(1, 2, 0).numpy().astype(np.float32)


def region(box, shape, margin=.1):
    x, y, w, h = box
    x0, y0 = int(np.floor(x - margin * w)), int(np.floor(y - margin * h))
    x1, y1 = int(np.ceil(x + w + margin * w)), int(np.ceil(y + h + margin * h))
    H, W = shape[:2]
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def gray(x):
    return cv2.cvtColor(np.uint8(np.clip(x, 0, 1) * 255 + .5), cv2.COLOR_RGB2GRAY)


class Pair:
    """Register frame b onto frame a using GT only; apply the identical warp to any field of frame b."""

    def __init__(self, ga, gb, box_a, box_b):
        self.ra, rb = region(box_a, ga.shape), region(box_b, gb.shape)
        ax0, ay0, ax1, ay1 = self.ra
        bx0, by0, bx1, by1 = rb
        self.size = (ax1 - ax0, ay1 - ay0)
        self.scale = float(np.sqrt((box_b[2] * box_b[3]) / max(box_a[2] * box_a[3], 1e-6)))
        self.rb = rb
        A = gray(ga[ay0:ay1, ax0:ax1]); Bc = gray(self.crop_b(gb))
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        f_ab = dis.calc(A, Bc, None)  # a(x) ~ b(x + f_ab(x))
        f_ba = dis.calc(Bc, A, None)
        h, w = A.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.mx, self.my = gx + f_ab[..., 0], gy + f_ab[..., 1]
        back = cv2.remap(f_ba, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e3)
        fb = np.linalg.norm(f_ab + back, axis=-1)
        inside = (self.mx >= 0) & (self.mx <= w - 1) & (self.my >= 0) & (self.my <= h - 1)
        box_mask = np.zeros((h, w), bool)
        x, y, bw, bh = box_a
        box_mask[max(0, int(y) - ay0):max(0, int(y + bh) - ay0), max(0, int(x) - ax0):max(0, int(x + bw) - ax0)] = True
        self.mask = (fb < 1.0) & inside & box_mask
        self.coverage = float(self.mask.sum() / max(box_mask.sum(), 1))

    def crop_b(self, x):
        bx0, by0, bx1, by1 = self.rb
        c = x[by0:by1, bx0:bx1]
        interp = cv2.INTER_AREA if (bx1 - bx0) > self.size[0] else cv2.INTER_CUBIC
        return cv2.resize(c, self.size, interpolation=interp)

    def crop_a(self, x):
        ax0, ay0, ax1, ay1 = self.ra
        return x[ay0:ay1, ax0:ax1]

    def warp_b(self, x):
        return cv2.remap(self.crop_b(x), self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


_LPIPS = None


def lpips_dist(x, y):
    global _LPIPS
    import lpips
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval()
    def t(z):
        z = torch.from_numpy(np.ascontiguousarray(z)).permute(2, 0, 1)[None].float() * 2 - 1
        h, w = z.shape[-2:]
        if min(h, w) < 64:
            s = 64 / min(h, w); z = F.interpolate(z, size=(round(h * s), round(w * s)), mode='bicubic', align_corners=False)
        return z
    with torch.no_grad():
        return float(_LPIPS(t(x), t(y)))


def psnr(x, y, m=None):
    d = (x - y) ** 2
    mse = float(d[m].mean()) if m is not None else float(d.mean())
    return 10 * np.log10(1 / max(mse, 1e-12))


def pair_metrics(fa, fb, ev, ydir, gdir):
    rec = ev['frame_records']
    ga, gb = load(gdir / f'{fa:06d}.png'), load(gdir / f'{fb:06d}.png')
    pr = Pair(ga, gb, rec[str(fa)]['bbox_hr_xywh'], rec[str(fb)]['bbox_hr_xywh'])
    base = {'a': fa, 'b': fb, 'scale': pr.scale, 'coverage': pr.coverage}
    if not (2 / 3 <= pr.scale <= 1.5):
        return {**base, 'dropped': 'scale'}
    if pr.coverage < .3:
        return {**base, 'dropped': 'coverage'}
    out = dict(base, dropped=None)
    m = pr.mask
    nga, ngb = unobserved(ga), unobserved(gb)
    gA, gBw = pr.crop_a(ga), pr.warp_b(gb)
    out['gt'] = {'change_unobs': float(((pr.crop_a(nga) - pr.warp_b(ngb))[m] ** 2).sum()),
                 'lpips_change': lpips_dist(gA, gBw), 'registration_psnr': psnr(gA, gBw, m)}
    for mode, folder in ydir.items():
        ya, yb = load(folder[fa] / f'{fa:06d}.png'), load(folder[fb] / f'{fb:06d}.png')
        nya, nyb = unobserved(ya), unobserved(yb)
        ea = pr.crop_a(nya - nga)[m]; eb = pr.warp_b(nyb - ngb)[m]
        den = float((ea ** 2).sum() + (eb ** 2).sum())
        change = float(((pr.crop_a(nya) - pr.warp_b(nyb))[m] ** 2).sum())
        yA, yBw = pr.crop_a(ya), pr.warp_b(yb)
        out[mode] = {'ICR': float(((ea - eb) ** 2).sum() / max(den, 1e-12)),
                     'unobs_energy_a': float((pr.crop_a(nya)[m] ** 2).mean()), 'gt_unobs_energy_a': float((pr.crop_a(nga)[m] ** 2).mean()),
                     'change_ratio': change / max(out['gt']['change_unobs'], 1e-12),
                     'lpips_change': lpips_dist(yA, yBw),
                     'psnr_a': psnr(yA, gA, m), 'psnr_b_warped': psnr(yBw, gBw, m),
                     'lpips_a': lpips_dist(yA, gA), 'lpips_b_warped': lpips_dist(yBw, gBw)}
    return out


def seed_metrics(f, ev, folders, gdir):
    """Same frame, seeds 101 vs 202 (no registration): how much of the unobserved band is noise-driven."""
    x, y, w, h = ev['frame_records'][str(f)]['bbox_hr_xywh']
    g = load(gdir / f'{f:06d}.png')
    H, W = g.shape[:2]
    m = np.zeros((H, W), bool)
    m[max(0, int(y)):min(H, int(y + h)), max(0, int(x)):min(W, int(x + w))] = True
    ng = unobserved(g)
    out = {'frame': f, 'gt_unobs_energy': float((ng[m] ** 2).mean())}
    fields = {}
    for mode, folder in folders.items():
        ny = unobserved(load(folder / f'{f:06d}.png'))
        fields[mode] = (ny - ng)[m]
        out[f'{mode}_energy_ratio'] = float((ny[m] ** 2).mean() / max(out['gt_unobs_energy'], 1e-12))
    e1, e2 = fields['innerdc'], fields['innerdc_s202']
    out['ICR_seed'] = float(((e1 - e2) ** 2).sum() / max((e1 ** 2).sum() + (e2 ** 2).sum(), 1e-12))
    return out


def panel(fa, fb, ev, ydir, gdir, path):
    rec = ev['frame_records']
    ga, gb = load(gdir / f'{fa:06d}.png'), load(gdir / f'{fb:06d}.png')
    pr = Pair(ga, gb, rec[str(fa)]['bbox_hr_xywh'], rec[str(fb)]['bbox_hr_xywh'])
    rows = []
    for mode, folder in ydir.items():
        ya, yb = load(folder[fa] / f'{fa:06d}.png'), load(folder[fb] / f'{fb:06d}.png')
        rows.append(np.concatenate([pr.crop_a(ga), pr.crop_a(ya), pr.warp_b(yb), pr.warp_b(gb)], axis=1))
    img = np.concatenate(rows, axis=0)
    s = 256 / img.shape[0] * len(rows) if img.shape[0] < 256 * len(rows) else 1
    im = Image.fromarray(np.uint8(np.clip(img, 0, 1) * 255 + .5))
    if s != 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.NEAREST)
    im.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', type=Path, default=ROOT / 'data/events')
    p.add_argument('--blocks', type=Path, default=ROOT / 'results/step0_blocks')
    p.add_argument('--out', type=Path, default=ROOT / 'results/step0_metrics')
    p.add_argument('--modes', nargs='+', default=['vae_only', 'innerdc'])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((a.events / 'selection.json').read_text())['events']
    results = []
    for e in sel:
        seq = e['sequence']
        ev = json.loads((a.events / seq / 'event.json').read_text())
        gdir = a.events / seq / 'hr'
        frame_dir = {}
        ok = True
        for mode in a.modes:
            frame_dir[mode] = {}
            for name, k in ev['blocks'].items():
                folder = a.blocks / mode / seq / f'block{k:03d}'
                if not (folder / 'result.json').exists():
                    ok = False; break
                for f in range(17 * k, 17 * k + 22):
                    frame_dir[mode].setdefault(f, folder)
            # new outputs override overlap frames from the previous block
            for name, k in sorted(ev['blocks'].items(), key=lambda kv: kv[1]):
                folder = a.blocks / mode / seq / f'block{k:03d}'
                for f in (range(0, 22) if k == 0 else range(17 * k + 5, 17 * k + 22)):
                    frame_dir[mode][f] = folder
        if not ok:
            print(json.dumps({'skip_incomplete': seq}), flush=True); continue
        revisit = [pair_metrics(fa, fb, ev, frame_dir, gdir) for fa in ev['pairs_revisit_a'] for fb in ev['pairs_revisit_b']]
        control = []
        if ev.get('pair_control'):
            fa, fb = ev['pair_control']
            if not ev['frame_records'][str(fa)]['absent'] and not ev['frame_records'][str(fb)]['absent']:
                control = [pair_metrics(fa, fb, ev, frame_dir, gdir)]
        mid = lambda v: v[len(v) // 2]
        panel(mid(ev['pairs_revisit_a']), mid(ev['pairs_revisit_b']), ev, frame_dir, gdir, a.out / f'panel_revisit_{seq}.png')
        if control and control[0]['dropped'] is None:
            panel(*ev['pair_control'], ev, frame_dir, gdir, a.out / f'panel_control_{seq}.png')
        seed = []
        kpre = ev['blocks']['kpre']
        sfold = {m: a.blocks / m / seq / f'block{kpre:03d}' for m in ('vae_only', 'innerdc', 'innerdc_s202')}
        if all((v / 'result.json').exists() for v in sfold.values()):
            seed = [seed_metrics(f, ev, sfold, gdir) for f in ev['pairs_revisit_a']]
        r = {'sequence': seq, 'category': ev['category'], 'absent_frames': ev['event']['absent_frames'], 'revisit_pairs': revisit,
             'control_pairs': control, 'seed_sensitivity': seed}
        (a.out / f'{seq}.json').write_text(json.dumps(r, indent=1))
        results.append(r)
        print(json.dumps({'event': seq, 'revisit_kept': sum(x['dropped'] is None for x in revisit), 'control_kept': sum(x['dropped'] is None for x in control)}), flush=True)

    def ev_mean(r, kind, mode, key):
        v = [x[mode][key] for x in r[kind] if x['dropped'] is None]
        return float(np.mean(v)) if v else None
    summary = {'events_total': len(results), 'modes': {}}
    for mode in a.modes:
        rows = []
        for r in results:
            rv, cv = ev_mean(r, 'revisit_pairs', mode, 'ICR'), ev_mean(r, 'control_pairs', mode, 'ICR')
            rows.append({'sequence': r['sequence'], 'ICR_revisit': rv, 'ICR_control': cv,
                         'change_ratio_revisit': ev_mean(r, 'revisit_pairs', mode, 'change_ratio'),
                         'lpips_change_revisit': ev_mean(r, 'revisit_pairs', mode, 'lpips_change')})
        rv = [x['ICR_revisit'] for x in rows if x['ICR_revisit'] is not None]
        paired = [(x['ICR_revisit'], x['ICR_control']) for x in rows if x['ICR_revisit'] is not None and x['ICR_control'] is not None]
        summary['modes'][mode] = {'per_event': rows, 'ICR_revisit_mean': float(np.mean(rv)) if rv else None, 'n_revisit_events': len(rv),
                                  'ICR_control_mean_paired': float(np.mean([c for _, c in paired])) if paired else None,
                                  'ICR_revisit_mean_paired': float(np.mean([r for r, _ in paired])) if paired else None,
                                  'n_paired_events': len(paired)}
    seed_rows = [{k: float(np.mean([x[k] for x in r['seed_sensitivity']])) for k in ('ICR_seed', 'innerdc_energy_ratio', 'vae_only_energy_ratio')}
                 for r in results if r['seed_sensitivity']]
    summary['seed_sensitivity'] = {'n_events': len(seed_rows), **({k: float(np.mean([x[k] for x in seed_rows])) for k in seed_rows[0]} if seed_rows else {})}
    gt_rows = [ev_mean(r, 'revisit_pairs', 'gt', 'lpips_change') for r in results]
    summary['gt_lpips_change_revisit_mean'] = float(np.mean([g for g in gt_rows if g is not None])) if any(g is not None for g in gt_rows) else None
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({m: {k: v for k, v in s.items() if k != 'per_event'} for m, s in summary['modes'].items()}, indent=1))


if __name__ == '__main__':
    main()
