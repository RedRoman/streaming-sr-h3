"""Score the multi-operator model and the matching single-operator models on identical data.

omni_stage1 reported its gains against the single-operator numbers as a hardcoded constant, but those were
measured differently: the single models were evaluated over the whole validation split at stride 4, while
the omni run uses 24 clips per operator at stride 8 to fit a CPU-only schedule. Subtracting one from the
other compares two different measurements, so the +0.09 / +0.06 / +0.51 deltas it printed do not mean
what they appear to mean. The same trap is waiting on the heterogeneous set: sgl_deblur's own log says
+8.378 dB over 426 frames while het_scale says +7.966 over 72, and those two numbers are not comparable.

Here every model sees the same clips, the same frames and the same code path. The crop difference during
TRAINING remains and is reported rather than hidden.

Two families of operator can be compared:

  --ops 2,4,8                     scale family: the same bicubic-chain operator at three rates
  --ops 4,deblur,denoise          heterogeneous family: downsampling, blur-only, and an operator whose
                                  null space is empty (denoise), so the three differ in kind, not degree

Arms whose tag contains `sev` are given the ground-truth severity of the clip. That is an oracle: severity
is not recoverable from a single LR frame (the supervised upper bound sits at AUC 0.61-0.68), so those
rows are an upper bound on conditioning, not a deployable configuration. They are labelled as such in the
output instead of being mixed into the same column as the blind arms.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))

# tag of the specialist trained on that operator alone, and the crop it was trained at
DEFAULT_SINGLE = {'2': 'r2_s2', '4': 'r2_hs', '8': 'r2_s8'}
HET_SINGLE = {'4': 'sgl_sr4', 'deblur': 'sgl_deblur', 'denoise': 'sgl_denoise'}
def train_meta(tag):
    """Read each specialist's own training record instead of keeping a hand-maintained table.

    The hardcoded crop table went stale the moment new specialists appeared (sp_s2 / sp_deblur /
    sp_denoise / r5_big), and printed `null` for every one of them -- which is exactly the provenance
    line that has to be right, because a crop or window-count difference between the omni model and its
    control is the confound this script exists to rule out.
    """
    for d in ('results/range_branch', 'results/omni'):
        f = ROOT / d / f'{tag}.json'
        if f.exists():
            j = json.loads(f.read_text())
            return {'crop': j.get('crop'), 'windows': j.get('train_windows'),
                    'iters': j.get('iters'), 'k': j.get('k'), 'params': j.get('params')}
    return None


def as_op(s):
    return int(s) if s.strip().isdigit() else s.strip()


@torch.no_grad()
def score(fn, data, dev, stride=8, chunk=4):
    """fn(x, op, sev) -> estimate, or None when this model has nothing for that operator.

    Returning None leaves the operator out of the row entirely rather than scoring a placeholder, so a
    missing specialist shows as an empty cell instead of a fabricated gain.
    """
    acc = {}
    for it in data.item:
        idx = list(range(0, it['y'].shape[0], stride))
        for a in range(0, len(idx), chunk):
            sel = idx[a:a + chunk]
            x = torch.from_numpy(it['y'][sel]).float().permute(0, 3, 1, 2).to(dev) / 255
            g = torch.from_numpy(it['t'][sel]).float().permute(0, 3, 1, 2).to(dev) / 255
            o = fn(x, it['scale'], it['sev'])
            if o is None:
                break
            d = acc.setdefault(it['scale'], {'m': [], 'b': []})
            for i in range(len(sel)):
                d['m'].append(float(10 * torch.log10(1 / (o[i] - g[i]).square().mean().clamp_min(1e-12))))
                d['b'].append(float(10 * torch.log10(1 / (x[i] - g[i]).square().mean().clamp_min(1e-12))))
    return {s: {'gain_dB': float(np.mean(d['m']) - np.mean(d['b'])),
                'psnr': float(np.mean(d['m'])), 'base': float(np.mean(d['b'])),
                'n_frames': len(d['m'])}
            for s, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-limit', type=int, default=None, help='default: the whole split')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ops', default='2,4,8')
    ap.add_argument('--single', default='', help='op=tag,... ; default picks by --ops')
    ap.add_argument('--omni', default='omni_scale,omni_scalesev')
    ap.add_argument('--out', default='compare.json')
    a = ap.parse_args()
    dev = torch.device(a.device)
    if a.device.startswith('npu'):
        import torch_npu  # noqa: F401
        from h3_comfy import patch_module_to, set_precision
        patch_module_to(); set_precision()
    from omni_stage1 import Multi, OmniRange
    from range_branch import RangeBranch

    ops = [as_op(s) for s in a.ops.split(',')]
    if a.single:
        single_tag = {as_op(k): v for k, v in (kv.split('=') for kv in a.single.split(','))}
    else:
        table = HET_SINGLE if any(isinstance(o, str) for o in ops) else DEFAULT_SINGLE
        single_tag = {as_op(k): v for k, v in table.items() if as_op(k) in ops}
    missing = [o for o in ops if o not in single_tag]
    if missing:
        print(json.dumps({'warn': 'no specialist for these operators, "single" row will omit them',
                          'operators': [str(m) for m in missing]}), flush=True)

    va = Multi('val', ops, limit=a.val_limit)
    head = {'val_items': {str(s): len(v) for s, v in va.by_scale.items()},
            'single_tag': {str(k): v for k, v in single_tag.items()},
            'train_meta': {v: train_meta(v) for v in single_tag.values()}}
    print(json.dumps(head), flush=True)
    rows, meta = {}, {}

    # the specialists, each only valid on the operator it was trained on
    sing = {}
    for op, tag in single_tag.items():
        st = torch.load(ROOT / f'results/range_branch/{tag}.pt', map_location='cpu')
        m = RangeBranch(**st['cfg']).eval().to(dev)
        m.load_state_dict(st['state'])
        sing[op] = m
    if sing:
        rows['single'] = score(lambda x, s, sev: sing[s](x[:, None]) if s in sing else None, va, dev)
        meta['single'] = {'oracle_severity': False, 'models': len(sing)}

    for tag in a.omni.split(','):
        p = ROOT / f'results/omni/{tag}.pt'
        if not p.exists():
            print(json.dumps({'skip': tag, 'reason': 'checkpoint missing'}), flush=True)
            continue
        st = torch.load(p, map_location='cpu')
        m = OmniRange(**st['cfg']).eval().to(dev)
        m.load_state_dict(st['state'])
        use_sev = 'sev' in tag
        rows[tag] = score(
            lambda x, s, sev, m=m, u=use_sev: m(
                x, torch.tensor([va.cond(s, sev, u)], dtype=torch.float32,
                                device=dev).expand(x.shape[0], -1)), va, dev)
        meta[tag] = {'oracle_severity': use_sev, 'models': 1, 'train': train_meta(tag)}

    out = {'ops': [str(o) for o in ops], 'rows': rows, 'meta': meta, **head}
    (ROOT / 'results/omni' / a.out).write_text(json.dumps(out, indent=1))

    cols = [o for o in ops]
    w = max(10, *(len(str(c)) + 2 for c in cols))
    print()
    print(('%-16s' + '%*s' * len(cols)) % tuple(['model'] + [x for c in cols for x in (w, str(c))]))
    for k, v in rows.items():
        cells = ['%.3f' % v[c]['gain_dB'] if c in v else '-' for c in cols]
        print(('%-16s' + '%*s' * len(cols)) % tuple([k] + [x for c in cells for x in (w, c)]))
    if 'single' in rows:
        base = rows['single']
        print()
        for k, v in rows.items():
            if k == 'single':
                continue
            cells = ['%+.3f' % (v[c]['gain_dB'] - base[c]['gain_dB'])
                     if c in v and c in base else '-' for c in cols]
            note = '  ORACLE severity' if meta[k]['oracle_severity'] else '  blind'
            print(('%-16s' + '%*s' * len(cols) + '%s') %
                  tuple([k] + [x for c in cells for x in (w, c)] + [note + ', vs specialists']))
    print()
    print('frames per operator:', json.dumps({str(c): rows[list(rows)[0]][c]['n_frames']
                                              for c in cols if c in rows[list(rows)[0]]}))


if __name__ == '__main__':
    main()
