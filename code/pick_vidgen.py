"""Pick texture-rich VIDGEN5S clips for the Stage-1 visual/quality readout.
Selection uses only the dataset's own metadata (category, scores, caption keywords) -- no model outputs.
Frozen before any evaluation runs."""
import json, re
from pathlib import Path
D = Path('/temp/liwei/vidgen1m-h3-5s-selected-493-20261017')
N = 20
# texture-bearing content words; the aim is fabric / fur / foliage / carving / small print
KW = r'(fabric|textile|cloth|knit|wool|sweater|lace|embroider|weav|fur|hair|feather|scale|leaf|leaves|grass|moss|bark|wood grain|carv|engrav|brick|stone|rust|mesh|net|pattern|texture|crochet|basket|straw|sand|gravel|crystal|bead)'
rows = [json.loads(l) for l in open(D / 'dataset.jsonl')]
cand = []
for r in rows:
    src = r.get('source', {})
    cap = (src.get('caption') or '') + ' ' + (r.get('enhanced_prompt') or '')
    hits = sorted(set(m.group(0).lower() for m in re.finditer(KW, cap, re.I)))
    q = src.get('quality_score') or 0
    h = src.get('h3_5s_score') or 0
    vd = (src.get('h3_5s_scores') or {}).get('visible_detail', 0)
    cand.append({'id': r['id'], 'video': str(D / r['relative_video_path']), 'category': src.get('category'),
                 'quality': q, 'h3_score': h, 'visible_detail': vd, 'kw': hits, 'n_kw': len(hits),
                 'caption': (src.get('caption') or '')[:150]})
cand.sort(key=lambda c: (-c['n_kw'], -c['visible_detail'], -c['quality'], c['id']))
# cap two per category so the set is not all food or all nature
seen, picked = {}, []
for c in cand:
    k = c['category'] or 'none'
    if seen.get(k, 0) >= 2: continue
    if c['n_kw'] == 0: continue
    seen[k] = seen.get(k, 0) + 1; picked.append(c)
    if len(picked) >= N: break
out = Path('/cache/zihaohe/streaming_sr0/data/vidgen_eval')
out.mkdir(parents=True, exist_ok=True)
(out / 'selection.json').write_text(json.dumps({'source': str(D), 'n': len(picked),
    'rule': 'texture keyword count desc, then visible_detail, then quality; max 2 per category; metadata only',
    'keywords': KW, 'clips': picked}, indent=1))
for c in picked: print(json.dumps({k: c[k] for k in ('id','category','n_kw','visible_detail','quality','kw')}), flush=True)
print('picked', len(picked), 'categories', seen)
