"""Touch the accelerator before loading the dataset, so a card that was won is not lost while reading it.

Both trainers construct their Data/Multi object -- which pulls every npz into RAM, 31 GB and several
minutes for range_pairs_x4big -- before the model is moved to the device. The card therefore reports no
process for the whole load, and a colleague's reservation daemon on this host spawns a holder for exactly
that condition: its loop skips any device where `m.query(d)` already shows a process. A one-element
allocation up front is enough to make the device non-idle, so the race disappears instead of being run.

Also the reason three runs have been lost at 60-70% of their iterations: a card taken mid-run by a holder
that sees an idle window. 1 GB is below any batch this project uses and is freed before training starts.
"""
from pathlib import Path

ROOT = Path('/cache/zihaohe/streaming_sr0/code')
CLAIM = """        # Hold the device before the dataset is read: see patches/claim_card_early.py
        _claim = torch.zeros(1 << 28, device=dev)
        del _claim
"""

for name in ('train_range2.py', 'omni_stage1.py'):
    p = ROOT / name
    s = p.read_text()
    if 'claim_card_early' in s:
        print('%s: already patched, skipped' % name); continue
    old = """        from h3_comfy import patch_module_to, set_precision
        patch_module_to(); set_precision()
"""
    if s.count(old) != 1:
        raise SystemExit('%s: anchor matched %d times' % (name, s.count(old)))
    p.write_text(s.replace(old, old + CLAIM))
    print('%s: patched' % name)

import ast
for name in ('train_range2.py', 'omni_stage1.py'):
    ast.parse((ROOT / name).read_text())
print('syntax OK')
