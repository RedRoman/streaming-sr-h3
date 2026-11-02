"""Allow prepare_real_eval to build a second, disjoint evaluation set.

The 24-clip set has done its job but it is now also where s0, the unsharp alpha and the detail gain were
tuned, and its bootstrap CI (+/-0.018) is wider than the margin we are claiming over InfVSR (0.0124). A
held-out set fixes both at once: the knobs stay fixed at the values chosen on the original 24, and every
number is reported on clips that played no part in choosing them.

Three additions, nothing else touched:
  --cand        pick from a different candidate list (candidates_big.json has 290, of which 282 are not
                in the original 24)
  --out-name    write under data/real_eval/<name> instead of overwriting the validated set
  --exclude     skip clip ids listed in a file, one per line: the original 24, and the clips that entered
                the Stage-7 LoRA training set, so the new set is clean of both
"""
import sys
from pathlib import Path

p = Path('/cache/zihaohe/streaming_sr0/code/prepare_real_eval.py')
s = p.read_text()
if '--cand' in s:
    print('already patched'); sys.exit(0)

s = s.replace("""    ap.add_argument('--keep', type=int, default=24)""",
              """    ap.add_argument('--keep', type=int, default=24)
    ap.add_argument('--cand', default=None, help='candidate json; default data/real_av/candidates.json')
    ap.add_argument('--out-name', default=None, help='subdirectory of data/real_eval; default the variant')
    ap.add_argument('--exclude', default=None, help='file of clip ids to skip, one per line')""")

s = s.replace("""    sel = json.loads(CAND.read_text())['selected'][:a.keep]
    root = OUT / a.variant""",
              """    cand = Path(a.cand) if a.cand else CAND
    if not cand.is_absolute():
        cand = ROOT / 'data/real_av' / cand
    skip = set()
    if a.exclude:
        skip = {l.strip() for l in Path(a.exclude).read_text().splitlines() if l.strip()}
    sel = [c for c in json.loads(cand.read_text())['selected'] if Path(c['file']).stem not in skip][:a.keep]
    root = OUT / (a.out_name or a.variant)""")

s = s.replace("""        man = {'n': len(done), 'frames': T, 'hr_wh': [W, H], 'blocks': blocks,
               'source': 'audioset_real', 'variant': a.variant, 'seed': SEED, 'clips': sorted(done)}""",
              """        # with nshards > 1 each worker only sees its own clips, so the manifest is rebuilt from disk
        done = sorted(d.name for d in root.iterdir() if (d / 'meta.json').exists())
        man = {'n': len(done), 'frames': T, 'hr_wh': [W, H], 'blocks': blocks,
               'source': 'audioset_real', 'variant': a.variant, 'seed': SEED,
               'candidates': str(cand), 'excluded': len(skip), 'clips': done}""")
p.write_text(s)
print('patched')
