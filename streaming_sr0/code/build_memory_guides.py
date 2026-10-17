"""Step 2 (PROTOCOL_step2.md): build memory-augmented guide pixels for each kpost block, plus the classical
(no-H3) baselines. CPU only. GT is used for: oracle localization (boxes, all variants), oracle flow (oflow variants),
and oracle content (gt variants) -- exactly as declared in the protocol."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pixel_consistency import BicubicConsistency
from revisit_metrics import ROOT, load, region

E = ROOT / 'data/events'
GATE_TAU, FB_TAU, SOFT_SIGMA = 0.04, 1.0, 2.0
VARIANTS = ['gt_oflow', 'gt_lrflow', 'self_lrflow', 'wrong_oflow', 'gt_oflow_nogate', 'wrong_oflow_nogate']


def up(lr, hw):
    t = torch.from_numpy(lr).permute(2, 0, 1)[None].double()
    return F.interpolate(t, size=hw, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)[0].permute(1, 2, 0).numpy().astype(np.float32)


def down_up(x):
    t = torch.from_numpy(x).permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // 4, w // 4), mode='bicubic', align_corners=False, antialias=True)
    return F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)[0].permute(1, 2, 0).numpy().astype(np.float32)


def gray(x):
    return cv2.cvtColor(np.uint8(np.clip(x, 0, 1) * 255 + .5), cv2.COLOR_RGB2GRAY)


def resize_to(x, size):
    interp = cv2.INTER_AREA if x.shape[1] > size[0] else cv2.INTER_CUBIC
    return cv2.resize(x, size, interpolation=interp)


class Reg:
    """Register reference region onto target region of frame t (target <- ref)."""

    def __init__(self, tgt_img, ref_img, box_t, box_r):
        self.rt, self.rr = region(box_t, tgt_img.shape), region(box_r, ref_img.shape)
        x0, y0, x1, y1 = self.rt
        self.size = (x1 - x0, y1 - y0)
        A, B = gray(tgt_img[y0:y1, x0:x1]), gray(self.crop_ref(ref_img))
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        f_tr, f_rt = dis.calc(A, B, None), dis.calc(B, A, None)
        h, w = A.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.mx, self.my = gx + f_tr[..., 0], gy + f_tr[..., 1]
        back = cv2.remap(f_rt, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e3)
        inside = (self.mx >= 0) & (self.mx <= w - 1) & (self.my >= 0) & (self.my <= h - 1)
        self.fb_ok = (np.linalg.norm(f_tr + back, axis=-1) < FB_TAU) & inside
        self.scale = float(np.sqrt(box_r[2] * box_r[3] / max(box_t[2] * box_t[3], 1e-6)))

    def crop_ref(self, x):
        x0, y0, x1, y1 = self.rr
        return resize_to(x[y0:y1, x0:x1], self.size)

    def warp(self, ref_content_full):
        return cv2.remap(self.crop_ref(ref_content_full), self.mx, self.my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def build_frame(base, warped_region, rt, extra_mask, gate=True, lr=None):
    """Composite warped region into base, gate (LR only), soft-mask, add N(W) to base. Returns guide, hard mask."""
    x0, y0, x1, y1 = rt
    W = base.copy(); W[y0:y1, x0:x1] = np.clip(warped_region, 0, 1)
    m = np.zeros(base.shape[:2], bool); m[y0:y1, x0:x1] = extra_mask
    if gate:
        e = np.abs(down_up(W) - base).mean(-1)  # base = U(LR_t), so this compares U(D(W)) with U(LR_t)
        e = cv2.GaussianBlur(e, (0, 0), 2.0)
        m &= e < GATE_TAU
    s = cv2.GaussianBlur(m.astype(np.float32), (0, 0), SOFT_SIGMA)[..., None]
    return np.clip(base + s * (W - down_up(W)), 0, 1).astype(np.float32), m


def box_mask(box, shape, rt):
    x0, y0, x1, y1 = rt
    x, y, w, h = box
    m = np.zeros(shape[:2], bool)
    m[max(0, int(y)):min(shape[0], int(y + h)), max(0, int(x)):min(shape[1], int(x + w))] = True
    return m[y0:y1, x0:x1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'data/step2_guides')
    p.add_argument('--only', nargs='*')
    a = p.parse_args()
    sel = json.loads((E / 'selection.json').read_text())['events']
    for i, e in enumerate(sel):
        seq = e['sequence']
        if a.only and seq not in a.only:
            continue
        out = a.out / seq
        if (out / 'record.json').exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        ev = json.loads((E / seq / 'event.json').read_text())
        fr = ev['frame_records']
        k, kpre = ev['blocks']['kpost'], ev['blocks']['kpre']
        ar = e['pairs_revisit_a']; fref = ar[len(ar) // 2]
        nxt = sel[(i + 1) % len(sel)]; nar = nxt['pairs_revisit_a']; nref = nar[len(nar) // 2]
        nev = json.loads((E / nxt['sequence'] / 'event.json').read_text())
        g_ref = load(E / seq / 'hr' / f'{fref:06d}.png')
        hw = g_ref.shape[:2]
        lr_ref_up = up(load(E / seq / 'lr' / f'{fref:06d}.png'), hw)
        self_ref = load(ROOT / 'results/step0_blocks/innerdc' / seq / f'block{kpre:03d}' / f'{fref:06d}.png')
        wrong_full = load(E / nxt['sequence'] / 'hr' / f'{nref:06d}.png')
        box_r, box_wrong = fr[str(fref)]['bbox_hr_xywh'], nev['frame_records'][str(nref)]['bbox_hr_xywh']
        frames = list(range(17 * k, 17 * k + 22))
        guides = {v: [] for v in VARIANTS}; masks = {v: [] for v in VARIANTS}
        rec = {'sequence': seq, 'f_ref': fref, 'wrong_source': [nxt['sequence'], nref], 'frames': frames, 'per_frame': {}}
        for f in frames:
            lr = load(E / seq / 'lr' / f'{f:06d}.png'); base = up(lr, hw)
            row = {'visible': not fr[str(f)]['absent']}
            if fr[str(f)]['absent'] or min(fr[str(f)]['bbox_hr_xywh'][2:]) < 4:
                for v in VARIANTS:
                    guides[v].append(base); masks[v].append(np.zeros(hw, bool))
                rec['per_frame'][f] = row; continue
            box_t = fr[str(f)]['bbox_hr_xywh']
            g_t = load(E / seq / 'hr' / f'{f:06d}.png')
            ro = Reg(g_t, g_ref, box_t, box_r)              # oracle flow (GT)
            rl = Reg(base, lr_ref_up, box_t, box_r)         # deployable flow (LR only)
            bm_o, bm_l = box_mask(box_t, hw, ro.rt), box_mask(box_t, hw, rl.rt)
            gv, mv = build_frame(base, ro.warp(g_ref), ro.rt, ro.fb_ok & bm_o)
            guides['gt_oflow'].append(gv); masks['gt_oflow'].append(mv)
            gv, mv = build_frame(base, rl.warp(g_ref), rl.rt, rl.fb_ok & bm_l)
            guides['gt_lrflow'].append(gv); masks['gt_lrflow'].append(mv)
            gv, mv = build_frame(base, rl.warp(self_ref), rl.rt, rl.fb_ok & bm_l)
            guides['self_lrflow'].append(gv); masks['self_lrflow'].append(mv)
            # wrong content: next event's GT target region, resized into this event's ref region frame, then the SAME oflow warp and mask
            wx0, wy0, wx1, wy1 = region(box_wrong, wrong_full.shape)
            rx0, ry0, rx1, ry1 = ro.rr
            wrong_as_ref = g_ref.copy()
            wrong_as_ref[ry0:ry1, rx0:rx1] = resize_to(wrong_full[wy0:wy1, wx0:wx1], (rx1 - rx0, ry1 - ry0))
            m_gt = masks['gt_oflow'][-1]
            x0, y0, x1, y1 = ro.rt
            gv, mv = build_frame(base, ro.warp(wrong_as_ref), ro.rt, m_gt[y0:y1, x0:x1], gate=False)
            guides['wrong_oflow'].append(gv); masks['wrong_oflow'].append(mv)
            # secondary (amendment 20:55): no LR gate; wrong uses the identical flow and mask
            gv, mv = build_frame(base, ro.warp(g_ref), ro.rt, ro.fb_ok & bm_o, gate=False)
            guides['gt_oflow_nogate'].append(gv); masks['gt_oflow_nogate'].append(mv)
            gv, mv = build_frame(base, ro.warp(wrong_as_ref), ro.rt, ro.fb_ok & bm_o, gate=False)
            guides['wrong_oflow_nogate'].append(gv); masks['wrong_oflow_nogate'].append(mv)
            bx = box_mask(box_t, hw, (0, 0, hw[1], hw[0])).sum()
            row.update({'scale': ro.scale, 'box_px': int(bx), **{f'coverage_{v}': float(masks[v][-1].sum() / max(bx, 1)) for v in VARIANTS},
                        'fb_ok_oflow': float((ro.fb_ok & bm_o).sum() / max(bm_o.sum(), 1)), 'fb_ok_lrflow': float((rl.fb_ok & bm_l).sum() / max(bm_l.sum(), 1))})
            rec['per_frame'][f] = row
        # classic (no H3): Dykstra-25 onto the LR observation, same operator as inner-DC
        low = torch.stack([torch.from_numpy(load(E / seq / 'lr' / f'{f:06d}.png')).permute(2, 0, 1) for f in frames])
        op = BicubicConsistency(tuple(hw), (hw[0] // 4, hw[1] // 4))
        base_all = np.stack([up(load(E / seq / 'lr' / f'{f:06d}.png'), hw) for f in frames])
        for name, arr in [('none', base_all)] + [(v, np.stack(guides[v])) for v in VARIANTS]:
            proj, _ = op.project(torch.from_numpy(arr).permute(0, 3, 1, 2).double(), low.double(), iterations=25)
            cdir = out / f'classic_{name}'; cdir.mkdir(exist_ok=True)
            for f, fr_img in zip(frames, proj.permute(0, 2, 3, 1).numpy()):
                cv2.imwrite(str(cdir / f'{f:06d}.png'), cv2.cvtColor(np.uint8(np.clip(fr_img, 0, 1) * 255 + .5), cv2.COLOR_RGB2BGR))
        for v in VARIANTS:
            np.save(out / f'guide_{v}.npy', np.stack(guides[v]).astype(np.float32))
            np.save(out / f'mask_{v}.npy', np.stack(masks[v]))
        vis = [r for r in rec['per_frame'].values() if 'scale' in r]
        rec['mean_coverage'] = {v: float(np.mean([r[f'coverage_{v}'] for r in vis])) if vis else 0.0 for v in VARIANTS}
        (out / 'record.json').write_text(json.dumps(rec, indent=1))
        print(json.dumps({'event': seq, 'visible_frames': len(vis), 'coverage': {kk: round(vv, 3) for kk, vv in rec['mean_coverage'].items()}}), flush=True)


if __name__ == '__main__':
    main()
