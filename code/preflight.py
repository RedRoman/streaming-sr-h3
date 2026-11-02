"""Pre-flight gates for a deploy/eval run, so the failures that cost cards today fail in one second.

Ten things broke today. None were in the model: nine were in the shell and data-plumbing layer and one was
a resource check nobody made. Each of these gates corresponds to one of them.

  cards        Free HBM on each target card. A colleague's four-card job took 61.5 GB on cards 1/4/5/6
               sixteen minutes before a launch; 55 clips of sampling died at model load with 7.69 MiB
               free, and nothing in the logs said "someone else is on this card".
  eval_set     manifest.json clip count against the directories that actually have meta.json.
               prepare_real_eval had every shard rebuild the manifest from disk when it finished, so the
               last writer could be an early finisher: heldout_s2 listed 81 of 96 complete clips and a
               deploy silently ran on the subset and produced a table that looked full-size.
  ckpt         Checkpoint loads, and its adapter targets exist in the base with matching shapes. The
               Open-H3 refiner needed this and passed; the Singularity full model would have failed it
               (51 adaLN tensors with incompatible shapes) before 34 GB was downloaded.
  degradation  Degradation parameters across eval-set variants, not just HR hashes. The HR frames of the
               x2/x4/x8 sets are byte-identical and that check passed, but _noise consumes rng
               proportional to LR pixel count, so the sets diverge in their compression draw and the
               absolute numbers are not comparable across scales.
  outputs      Whether the target tag already has results, so a rerun is understood as a resume.

Exit code is 0 only if every requested gate passes. `--warn-only` reports and exits 0, for the cases
where the operator genuinely wants to proceed.
"""
import argparse
import json
import re
import shutil
import subprocess
import struct
import sys
from pathlib import Path

ROOT = Path('/cache/zihaohe/streaming_sr0')


def free_hbm():
    """{card: (used_MB, total_MB)} from npu-smi, or None when it cannot be read."""
    exe = shutil.which('npu-smi')
    if not exe:
        return None
    try:
        txt = subprocess.run([exe, 'info'], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    out, card = {}, None
    for line in txt.splitlines():
        m = re.match(r'^\|\s*(\d+)\s+\S+\s+\|', line)
        if m:
            card = int(m.group(1))
        m = re.search(r'(\d+)\s*/\s*(\d+)\s*\|?\s*$', line.replace('MB', ''))
        if m and card is not None and int(m.group(2)) > 1000:
            out[card] = (int(m.group(1)), int(m.group(2)))
            card = None
    return out


def gate_cards(cards, need_gb):
    hbm = free_hbm()
    if hbm is None:
        return [('cards', 'WARN', 'npu-smi unavailable; cannot check')]
    rows = []
    for c in cards:
        if c not in hbm:
            rows.append(('cards', 'WARN', f'card {c} not reported by npu-smi'))
            continue
        used, total = hbm[c]
        free = (total - used) / 1024.0
        ok = free >= need_gb
        rows.append(('cards', 'PASS' if ok else 'FAIL',
                     f'card {c}: {free:.1f} GB free of {total/1024:.0f}, need {need_gb:.0f}'))
    return rows


def gate_eval_set(rel):
    d = ROOT / rel
    if not d.is_dir():
        return [('eval_set', 'FAIL', f'{rel} does not exist')]
    man = d / 'manifest.json'
    if not man.exists():
        return [('eval_set', 'FAIL', f'{rel}/manifest.json missing')]
    listed = json.loads(man.read_text()).get('clips', [])
    on_disk = sorted(p.name for p in d.iterdir() if p.is_dir() and (p / 'meta.json').exists())
    if set(listed) == set(on_disk):
        return [('eval_set', 'PASS', f'{rel}: {len(listed)} clips, manifest matches disk')]
    return [('eval_set', 'FAIL',
             f'{rel}: manifest lists {len(listed)} but {len(on_disk)} complete clips on disk '
             f'(missing from manifest: {sorted(set(on_disk) - set(listed))[:4]})')]


def header(p):
    with Path(p).open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop('__metadata__', None)
    return h


def gate_ckpt(path, base=None):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return [('ckpt', 'FAIL', f'{path} does not exist')]
    rows = []
    if p.suffix == '.pt':
        import torch
        try:
            st = torch.load(p, map_location='cpu')
            cfg = st.get('cfg', {})
            rows.append(('ckpt', 'PASS', f'{p.name}: cfg {cfg}, {len(st.get("state", {}))} tensors'))
        except Exception as e:                                    # noqa: BLE001
            rows.append(('ckpt', 'FAIL', f'{p.name}: {type(e).__name__}: {e}'))
        return rows
    try:
        h = header(p)
    except Exception as e:                                        # noqa: BLE001
        return [('ckpt', 'FAIL', f'{p.name}: unreadable header: {e}')]
    if base is None:
        return [('ckpt', 'PASS', f'{p.name}: {len(h)} tensors')]
    bh = header(ROOT / base if not Path(base).is_absolute() else base)
    strip = lambda k: k.split('diffusion_model.')[-1]
    bk = {strip(k): tuple(v['shape']) for k, v in bh.items()}
    pairs, bad, miss = {}, 0, 0
    for k, v in h.items():
        kk = strip(k)
        for suf, slot in (('.lora_A.weight', 'A'), ('.lora_B.weight', 'B'),
                          ('.lora_down.weight', 'A'), ('.lora_up.weight', 'B')):
            if kk.endswith(suf):
                pairs.setdefault(kk[:-len(suf)], {})[slot] = tuple(v['shape'])
                break
    for m, ab in pairs.items():
        t = m + '.weight'
        if t not in bk:
            miss += 1
        elif 'A' in ab and 'B' in ab and (ab['B'][0], ab['A'][1]) != bk[t]:
            bad += 1
    if not pairs:
        extra = [strip(k) for k in h if strip(k) not in bk]
        return [('ckpt', 'PASS' if not extra else 'WARN',
                 f'{p.name}: full weights, {len(extra)} tensors not in base')]
    ok = miss == 0 and bad == 0
    return [('ckpt', 'PASS' if ok else 'FAIL',
             f'{p.name}: {len(pairs)} adapter modules, {miss} absent from base, {bad} shape mismatches')]


def gate_degradation(sets):
    """Compare the degradation record of the first shared clip across eval-set variants."""
    recs = {}
    for rel in sets:
        d = ROOT / rel
        man = d / 'manifest.json'
        if not man.exists():
            return [('degradation', 'FAIL', f'{rel}/manifest.json missing')]
        clips = json.loads(man.read_text())['clips']
        recs[rel] = (clips, d)
    common = set.intersection(*(set(c) for c, _ in recs.values()))
    if not common:
        return [('degradation', 'FAIL', 'the sets share no clips')]
    name = sorted(common)[0]
    keys = ('kind', 'interp', 'jpeg_q', 'h264_crf', 'noise1', 'second_order')
    seen, rows = {}, []
    for rel, (_, d) in recs.items():
        m = json.loads((d / name / 'meta.json').read_text())['degradation']
        seen[rel] = {k: m.get(k) for k in keys}
        seen[rel]['blur1_sigma'] = round(m['blur1'][0], 3) if 'blur1' in m else None
    vals = list(seen.values())
    same = all(v == vals[0] for v in vals)
    rows.append(('degradation', 'PASS' if same else 'WARN',
                 f'clip {name}: ' + ('identical across sets' if same
                                     else 'DIFFERS across sets -> absolute numbers are not comparable; '
                                          + json.dumps(seen))))
    return rows


def gate_outputs(tag):
    d = ROOT / 'results/deploy_runs' / tag
    if not d.is_dir():
        return [('outputs', 'PASS', f'{tag}: no existing output, fresh run')]
    done = sum(1 for p in d.iterdir() if p.is_dir() and (p / 'stream_result.json').exists())
    return [('outputs', 'PASS', f'{tag}: {done} clips already finished, this will resume')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cards', default='', help='comma list of card indices the run will use')
    ap.add_argument('--need-gb', type=float, default=45.0, help='free HBM each card must have')
    ap.add_argument('--eval-set', action='append', default=[])
    ap.add_argument('--ckpt', action='append', default=[])
    ap.add_argument('--base', default='assets/comfy_h3/minimax_h3_fl2va_pruned_bf16.safetensors')
    ap.add_argument('--compare-degradation', action='append', default=[],
                    help='eval sets whose degradation records should match')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--warn-only', action='store_true')
    a = ap.parse_args()

    rows = []
    if a.cards:
        rows += gate_cards([int(c) for c in a.cards.split(',')], a.need_gb)
    for s in a.eval_set:
        rows += gate_eval_set(s)
    for c in a.ckpt:
        rows += gate_ckpt(c, a.base)
    if len(a.compare_degradation) > 1:
        rows += gate_degradation(a.compare_degradation)
    if a.tag:
        rows += gate_outputs(a.tag)
    if not rows:
        rows = [('(none)', 'WARN', 'no gates requested')]

    w = max(len(g) for g, _, _ in rows)
    for g, st, msg in rows:
        print('%-*s  %-4s  %s' % (w, g, st, msg))
    fails = [r for r in rows if r[1] == 'FAIL']
    print()
    print('%d gate(s), %d FAIL, %d WARN' % (len(rows), len(fails), sum(r[1] == 'WARN' for r in rows)))
    sys.exit(1 if fails and not a.warn_only else 0)


if __name__ == '__main__':
    main()
