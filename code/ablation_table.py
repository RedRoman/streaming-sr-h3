"""The ablation table: innerDC target x quota gate, as a 2x2 on the held-out 96 clips.

Every held-out number reported so far came from one sampling run (deploy_runs/t4_r2_ho, innerDC projecting
onto R_phi(y)) with the quota gate applied afterwards. That confounds the two contributions: the
projection target lives inside sampling, the gate is post-processing, and a single column cannot say which
of them the gain comes from. The four cells below cross them, so each factor is read off twice -- once
with the other factor off and once with it on -- and an interaction is visible rather than assumed absent.

  dcy      innerDC projects onto the observation y,  no gate        deploy_runs/t4_dcy_ho
  dcy_q    innerDC onto y,                           gate on        deploy/ho_dcy_q
  dcr      innerDC onto R_phi(y),                    no gate        deploy_runs/t4_r2_ho
  dcr_q    innerDC onto R_phi(y),                    gate on        deploy/t4r2_ho_final

All four share the 4-step turbo schedule, the merged Stage-1 LoRA, R_phi = r2_hs, and gate parameters
s0=0.02, p=1.0, sharp=0.0, gain=1.15. The two sampled arms differ in the single `--dc` flag.

Baselines (InfVSR, FlashVSR) come from judge_ho_3way.json, scored by the same judge on the same clips.
FlashVSR is marked approximate: its CUDA block-sparse kernel is unavailable on Ascend, and the masked-SDPA
substitute drops 4.08% of the mask.

Metrics are reported on the realistic subset, which is where the method is aimed and where it trailed the
baselines; `--group all` includes the 19 clean-bicubic clips. Both temporal columns are shown on purpose:
warp_range rewards blur -- bicubic scores "best" on it -- while terr_null penalises it, so neither alone
is a temporal verdict.
"""
import argparse
import json
from pathlib import Path

ROOT = Path('/cache/zihaohe/streaming_sr0')
R = ROOT / 'results/deploy'

# (column label, metric key, lower is better)
COLS = [('LPIPS', 'lpips', True),
        ('PSNR', 'psnr', False),
        ('LPIPS|GTrange', 'lpips_gtrange', True),
        ('null_cos', 'null_cos', False),
        ('p8.terr_null', 'p8.terr_null', True),
        ('p8.warp_range', 'p8.warp_range', True)]

ABL_ARMS = [('dcy', 'innerDC->y,      no gate'),
            ('dcy_q', 'innerDC->y,      gate on'),
            ('dcr', 'innerDC->R_phi,  no gate'),
            ('dcr_q', 'innerDC->R_phi,  gate on  (final)')]

BASE_ARMS = [('infvsr', 'InfVSR'),
             ('flashvsr', 'FlashVSR (masked-SDPA, approx)'),
             ('flashvsr_plus', 'FlashVSR+ (masked-SDPA, approx)')]


def cells(summary, arm):
    v = summary.get(arm)
    if not v:
        return None
    return {k: (v[k]['mean'] if isinstance(v.get(k), dict) else None) for _, k, _ in COLS}


def fmt(x, key):
    if x is None:
        return '-'
    return ('%.4f' if key in ('lpips', 'lpips_gtrange', 'null_cos') else '%.3f') % x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', default=str(R / 'judge_abl.json'))
    ap.add_argument('--baselines', default=str(R / 'judge_ho_3way.json'))
    ap.add_argument('--group', default='realistic', choices=['all', 'realistic', 'bicubic', 'clean16'])
    a = ap.parse_args()

    rows, notes = [], []
    for path, arms, src in ((a.baselines, BASE_ARMS, 'baseline judge'),
                            (a.judge, ABL_ARMS, 'ablation judge')):
        p = Path(path)
        if not p.exists():
            notes.append(f'{src} missing: {p}')
            continue
        d = json.loads(p.read_text())
        g = d['summary'].get(a.group)
        if g is None:
            notes.append(f'{src}: no group {a.group!r}')
            continue
        notes.append(f'{src}: {p.name}, n={g.get("n")} clips in group {a.group!r}')
        for arm, label in arms:
            c = cells(g, arm)
            if c:
                rows.append((label, c))
            elif arm != 'flashvsr_plus':          # expected to be absent until that arm is produced
                notes.append(f'{src}: arm {arm!r} not in this judge')

    width = max(len(l) for l, _ in rows) if rows else 10
    head = ('%-*s' % (width, 'arm')) + ''.join('%15s' % c for c, _, _ in COLS)
    print(head)
    print('-' * len(head))
    for label, c in rows:
        print(('%-*s' % (width, label)) + ''.join('%15s' % fmt(c[k], k) for _, k, _ in COLS))

    # read each factor off twice, which is the point of crossing them
    by = {l: c for l, c in rows}
    pairs = [('innerDC y -> R_phi, gate off', 'innerDC->y,      no gate', 'innerDC->R_phi,  no gate'),
             ('innerDC y -> R_phi, gate on', 'innerDC->y,      gate on', 'innerDC->R_phi,  gate on  (final)'),
             ('gate off -> on, innerDC=y', 'innerDC->y,      no gate', 'innerDC->y,      gate on'),
             ('gate off -> on, innerDC=R_phi', 'innerDC->R_phi,  no gate', 'innerDC->R_phi,  gate on  (final)')]
    shown = [(n, x, y) for n, x, y in pairs if x in by and y in by]
    if shown:
        print()
        print(('%-*s' % (width, 'effect (delta, sign = direction of change)'))
              + ''.join('%15s' % c for c, _, _ in COLS))
        print('-' * len(head))
        for name, x, y in shown:
            d = [None if by[x][k] is None or by[y][k] is None else by[y][k] - by[x][k]
                 for _, k, _ in COLS]
            print(('%-*s' % (width, name))
                  + ''.join('%15s' % (('%+.4f' % v) if v is not None else '-') for v in d))
        print()
        print('lower is better: ' + ', '.join(c for c, _, lo in COLS if lo)
              + ' | higher is better: ' + ', '.join(c for c, _, lo in COLS if not lo))
    print()
    for n in notes:
        print('note:', n)


if __name__ == '__main__':
    main()
