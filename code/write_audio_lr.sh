#!/usr/bin/env bash
# Freeze the degraded audio into the eval set, so the audio side is reproducible the way the video side is.
#
# Right now every clip has clean audio_gt.npy and nothing else, and the audio evaluations generate the
# degraded signal on the fly from a seed. That is fine for one script and wrong for a joint claim: the
# video side ships its LR frames on disk with a per-frame sha256, and the audio side should too. Without
# it, "our audio restoration scores X" cannot be checked by anyone who does not rerun my rng.
#
# Seeded per clip by index exactly as degrade_audio's gate was (20261101 + i), so the numbers already
# measured -- passthrough mel 28.22, VAE-only 17.68, R_phi^audio 7.33, ceiling 2.99 -- refer to these
# files and not to a different draw.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export PATH=/cache/zihaohe/streaming_sr0/venv/bin:$PATH
python - "$@" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'code')
from degrade_audio import degrade_audio

R = Path('/cache/zihaohe/streaming_sr0')
sets = sys.argv[1:] or ['data/real_eval/heldout']
for rel in sets:
    D = R / rel
    clips = json.loads((D / 'manifest.json').read_text())['clips']
    n_w = n_skip = 0
    for i, c in enumerate(sorted(clips)):
        m = json.loads((D / c / 'meta.json').read_text())
        if not m.get('audio_present'):
            n_skip += 1; continue
        f = D / c / 'audio_lr.npy'
        if f.exists():
            n_w += 1; continue
        gt = np.load(D / c / 'audio_gt.npy')
        import torch
        dg, p = degrade_audio(torch.from_numpy(gt), np.random.default_rng(20261101 + i))
        arr = dg.numpy().astype(np.float32)
        np.save(f, arr)
        m['audio_degradation'] = p
        m['audio_lr_sha256'] = hashlib.sha256(f.read_bytes()).hexdigest()
        m['audio_gt_sha256'] = hashlib.sha256((D / c / 'audio_gt.npy').read_bytes()).hexdigest()
        (D / c / 'meta.json').write_text(json.dumps(m, indent=1))
        n_w += 1
        if n_w % 20 == 0:
            print(json.dumps({'set': rel, 'written': n_w, 'skipped_no_audio': n_skip}), flush=True)
    print(json.dumps({'set': rel, 'audio_lr_written': n_w, 'skipped_no_audio': n_skip,
                      'clips': len(clips)}), flush=True)
PY
echo "AUDIO LR DONE $(date '+%F %T')"
