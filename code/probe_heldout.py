"""Find real-footage clips that are clean of every model in this project, for a held-out test set.

The 24-clip set is now carrying two jobs it cannot do at once: it is where s0, the unsharp alpha and the
detail gain were chosen, and it is where the margin over InfVSR (0.0124) is claimed against a bootstrap CI
of +/-0.018. Both are fixed by reporting on clips that chose nothing.

The obvious source turned out to be unusable: candidates_big.json's 290 clips are, id for id, exactly the
Stage-7 LoRA training set, so every clip that passed the original quality filter is contaminated. What is
left is the 694 files sitting in data/real_av/_cand that appear in neither candidate list. They are
already downloaded, so this re-applies build_real_av's own filter to them locally -- same band_ratio bar
(0.00504, the median of the VIDGEN clips the project accepted), same audio gates, same minimum geometry --
and writes the passers as a new candidate list.

Nothing is deleted: build_real_av unlinked files that failed its gates, and whether these 694 were ever
probed is unrecorded, so a file failing here is just not selected.
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
SRC = ROOT / 'data/real_av/_cand'
BAR = 0.00504            # bar_band_ratio recorded in candidates.json
NEED_FRAMES, FPS = 124, 24.0
MIN_W, MIN_H = 1280, 720


def one(name):
    import build_real_av as B
    f = SRC / name
    try:
        v = B.probe(f, 'v:0', ['width', 'height'])
        au = B.probe(f, 'a:0', ['codec_name', 'sample_rate'])
        if not v or not au:
            return {'file': name, 'why': 'no stream'}
        W, H = (int(z) for z in v.split(',')[:2])
        dur = B.duration(f)
        if W < MIN_W or H < MIN_H or dur < NEED_FRAMES / FPS + 0.5:
            return {'file': name, 'why': f'geometry {W}x{H} {dur:.1f}s'}
        fr = B.frames_at(f, 6, 832, 480, start=max(0.0, (dur - NEED_FRAMES / FPS) / 2))
        if fr is None:
            return {'file': name, 'why': 'decode'}
        st = B.audio_stats(f)
        if st is None:
            return {'file': name, 'why': 'audio'}
        br = B.band_ratio(fr)
        return {'file': name, 'rel': name, 'wh': [W, H], 'duration': dur, 'band_ratio': br,
                'sharp_ok': br >= BAR, **st}
    except Exception as e:                           # noqa: BLE001 - a bad file is a skip, not a crash
        return {'file': name, 'why': f'{type(e).__name__}: {e}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=32)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default=str(ROOT / 'data/real_av/candidates_heldout.json'))
    a = ap.parse_args()
    known = set()
    for f in ('candidates.json', 'candidates_big.json'):
        d = json.loads((ROOT / 'data/real_av' / f).read_text())
        for k in ('all', 'selected'):
            known |= {Path(x['file']).stem for x in d.get(k, [])}
    excl = set((ROOT / 'data/real_eval/exclude_ids.txt').read_text().split())
    todo = sorted(p.name for p in SRC.glob('*.mp4')
                  if p.stem not in known and p.stem not in excl)[:a.limit]
    print(json.dumps({'to_probe': len(todo), 'already_known': len(known)}), flush=True)
    with Pool(a.procs) as pool:
        rows = pool.map(one, todo)
    good = [r for r in rows if 'band_ratio' in r]
    ok = [r for r in good if r['sharp_ok'] and r['rms'] > 0.005 and r['above_8k_ratio'] > 1e-4]
    ok.sort(key=lambda r: -r['band_ratio'])
    Path(a.out).write_text(json.dumps({'bar_band_ratio': BAR, 'n_probed': len(rows),
                                       'n_decoded': len(good), 'n_pass': len(ok),
                                       'selected': ok, 'all': good}, indent=1))
    import collections
    why = collections.Counter(r.get('why', 'passed_decode') for r in rows)
    print(json.dumps({'probed': len(rows), 'decoded': len(good), 'passed': len(ok),
                      'reasons': dict(why.most_common(8))}, indent=1))


if __name__ == '__main__':
    main()
