"""Clip order for the sampling runs: realistic first, so a partial run still covers the regime that decides
the comparison (19 of the 24 clips are realistic, and that is where we trail the baselines)."""
import json
from pathlib import Path
D = Path("/cache/zihaohe/streaming_sr0/data/real_eval/const")
m = json.loads((D / 'manifest.json').read_text())['clips']
k = {c: json.loads((D / c / 'meta.json').read_text())['degradation']['kind'] for c in m}
print(','.join(sorted(m, key=lambda c: (k[c] != 'realistic', c))))
