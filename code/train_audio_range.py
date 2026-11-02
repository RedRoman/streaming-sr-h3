"""R_phi^audio: restore the audio latent, scored against the three bounds that make the number readable.

Same shape as the video side's contribution, deliberately. R_phi is a small network trained standalone on
(observation, target) pairs with the DiT frozen; this is the audio analogue on (degraded latent, clean
latent). Nothing about the sampler or the 20.1B DiT changes, which matters because train_stage1 gives the
audio stream "noise input, no loss" and altering that is a full retrain.

The bounds, measured in degrade_audio.py on the held-out set's own audio, in mel dB (the metric a 25x
perceptual codec can express -- LSD weights every linear bin equally and is dominated by near-silent
high-frequency bins the codec legitimately reconstructs):

    26.54   passthrough, the degraded signal itself              the floor to beat
    16.34   VAE round trip of the degraded signal                the no-training baseline, the real target
     2.77   VAE round trip of the clean signal                   the ceiling nothing can pass

Beating 26.54 proves nothing: the codec alone does that by pulling toward its own prior. The claim is
only interesting above the 16.34 line.

Content caveat, learned the hard way on the bd operator where a specialist scored -12.7 dB on Vid4
because "base dB" is content x operator and not an operator invariant: training pairs come from the
h3_unified pool and the bounds were measured on AudioSet recordings. So the eval below reports BOTH -- the
h3_unified val split, and the held-out AudioSet audio the bounds live on.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
PAIRS = Path('/cache/zihaohe/audio_pairs')
OUT = ROOT / 'results/audio_range'
BOUNDS = {'passthrough_mel_dB': 26.543, 'vae_only_mel_dB': 16.34, 'ceiling_mel_dB': 2.768}


class AudioRange(nn.Module):
    """Latent-domain residual net over [B,32,2,T]: 32 latent channels, stereo, T frames at 40 Hz.

    Convolves along time only. The stereo axis is kept as a spatial dim of size 2 rather than folded into
    channels so a 3x3 kernel still sees both channels at once, and the time axis is where all the
    structure is. The tail is zero-initialised, so an untrained model is exactly the identity and any
    gain is attributable.
    """

    def __init__(self, ch=128, blocks=8, c_in=32):
        super().__init__()
        self.head = nn.Conv2d(c_in, ch, (3, 3), padding=(1, 1))
        self.body = nn.ModuleList([nn.Conv2d(ch, ch, (3, 3), padding=(1, 1)) for _ in range(2 * blocks)])
        self.tail = nn.Conv2d(ch, c_in, (3, 3), padding=(1, 1))
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)

    def forward(self, z):
        h = self.head(z)
        for i in range(0, len(self.body), 2):
            r = F.relu(self.body[i](h), inplace=True)
            h = h + self.body[i + 1](r)
        return z + self.tail(h)


class Pairs:
    def __init__(self, split, limit=None):
        fs = sorted(PAIRS.joinpath(split).glob('*.npz'))
        if limit:
            fs = fs[:limit]
        self.c, self.d, self.meta = [], [], []
        T = None
        for p in fs:
            z = np.load(p)
            a, b = z['clean'][0], z['degraded'][0]          # [32,2,T]
            if T is None:
                T = a.shape[-1]
            if a.shape[-1] != T or b.shape[-1] != T:        # windows at a clip boundary can be short
                continue
            self.c.append(a); self.d.append(b)
            self.meta.append(json.loads(str(z['meta'])))
        assert self.c, f'no usable pairs in {PAIRS/split}'
        self.c = torch.from_numpy(np.stack(self.c)); self.d = torch.from_numpy(np.stack(self.d))
        self.T = T

    def batch(self, rng, n):
        i = rng.integers(len(self.c), size=n)
        return self.d[i], self.c[i]


@torch.no_grad()
def latent_scores(model, data, dev, chunk=64):
    """L1 in latent space, plus the identity's, so the gain is not confused with the scale of the data."""
    m, b = [], []
    for a in range(0, len(data.c), chunk):
        d = data.d[a:a + chunk].to(dev); c = data.c[a:a + chunk].to(dev)
        o = model(d)
        m.append(float((o - c).abs().mean()) * len(d)); b.append(float((d - c).abs().mean()) * len(d))
    n = len(data.c)
    return {'n': n, 'l1_model': sum(m) / n, 'l1_identity': sum(b) / n,
            'l1_reduction': 1 - (sum(m) / n) / max(sum(b) / n, 1e-12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='ar1')
    ap.add_argument('--ch', type=int, default=128)
    ap.add_argument('--blocks', type=int, default=8)
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--log-every', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--eval-audio', type=int, default=8,
                    help='held-out AudioSet clips to decode and score in mel dB against the bounds')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    dev = torch.device(a.device)
    import torch_npu  # noqa: F401
    import h3_comfy  # noqa: F401
    from h3_comfy import patch_module_to, set_precision
    patch_module_to(); set_precision()
    _claim = torch.zeros(1 << 28, device=dev); del _claim

    tr, va = Pairs('train'), Pairs('val')
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    model = AudioRange(a.ch, a.blocks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rec = {'tag': a.tag, 'seed': a.seed, 'ch': a.ch, 'blocks': a.blocks, 'iters': a.iters,
           'params': sum(p.numel() for p in model.parameters()),
           'train_pairs': len(tr.c), 'val_pairs': len(va.c), 'latent_T': tr.T,
           'bounds_mel_dB': BOUNDS, 'curve': []}
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)

    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        x, y = tr.batch(rng, a.batch)
        loss = F.l1_loss(model(x.to(dev)), y.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % a.log_every == 0 or it == a.iters:
            model.eval(); ev = latent_scores(model, va, dev); model.train()
            e = {'iter': it, 'loss': float(loss), 'val': ev, 'sec': round(time.perf_counter() - t0, 1)}
            rec['curve'].append(e); print(json.dumps(e), flush=True)
            torch.save({'state': model.state_dict(),
                        'cfg': {'ch': a.ch, 'blocks': a.blocks, 'c_in': 32}},
                       OUT / f'{a.tag}.pt')

    # ---- decode and score in mel dB on the held-out AudioSet audio the bounds were measured on
    import comfy.sd
    import comfy.utils
    from audio_line1 import align
    from degrade_audio import degrade_audio, lsd, mel_dist
    sd_, meta_ = comfy.utils.load_torch_file(
        str(ROOT / 'assets/comfy_h3/minimax_h3_audio_vae_fp32.safetensors'), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd_, device=dev, dtype=torch.float32, metadata=meta_)
    vae.first_stage_model.eval().requires_grad_(False)
    enc = lambda w: vae.encode(w[None].movedim(1, -1).to(dev))

    def dec(z):
        y = vae.decode(z).float().cpu()
        y = y[0] if y.ndim == 3 else y
        return (y.T if y.shape[0] != 2 else y).reshape(2, -1)

    D_ = ROOT / 'data/real_eval/heldout'
    clips = [c for c in json.loads((D_ / 'manifest.json').read_text())['clips']
             if json.loads((D_ / c / 'meta.json').read_text()).get('audio_present')][:a.eval_audio]
    rows = []
    model.eval()
    with torch.no_grad():
        for i, c in enumerate(clips):
            gt = torch.from_numpy(np.load(D_ / c / 'audio_gt.npy'))
            dg, p = degrade_audio(gt, np.random.default_rng(20261101 + i))
            # comfy's VAE.encode returns on CPU; the model lives on the device. The first run died with
            # "weight is on npu:0, different from other tensors on cpu".
            zd = enc(dg).to(dev)
            zr = model(zd)                                     # [1,32,2,T] in, same out
            w_vae, w_mod, w_ceil = dec(zd), dec(zr), dec(enc(gt).to(dev))
            r = {'clip': c, 'params': p,
                 'passthrough_mel_dB': round(mel_dist(*align(dg, gt)[:2]), 3),
                 'vae_only_mel_dB': round(mel_dist(*align(w_vae, gt)[:2]), 3),
                 'model_mel_dB': round(mel_dist(*align(w_mod, gt)[:2]), 3),
                 'ceiling_mel_dB': round(mel_dist(*align(w_ceil, gt)[:2]), 3),
                 'model_lsd_dB': round(lsd(*align(w_mod, gt)[:2]), 3),
                 'vae_only_lsd_dB': round(lsd(*align(w_vae, gt)[:2]), 3)}
            rows.append(r); print(json.dumps(r), flush=True)

    mean = lambda k: float(np.mean([r[k] for r in rows]))
    pt, vo, mo, ce = mean('passthrough_mel_dB'), mean('vae_only_mel_dB'), mean('model_mel_dB'), mean('ceiling_mel_dB')
    rec['final'] = rec['curve'][-1]
    # LSD is reported as a JOINT criterion, not as decoration. The 6000-iter run improved mel on 12/12
    # clips while LSD got WORSE on two of them (24cmo2fEQo8 12.82 -> 16.01, 43dQMHGLDHU 17.74 -> 27.84),
    # and those were also the two with the smallest mel gain. That is the same energy-up/correctness-down
    # split that killed the video sharpness arm, so a run that improves mel while regressing LSD on
    # several clips is not reported as an unqualified win.
    lsd_reg = [r['clip'] for r in rows if r['model_lsd_dB'] > r['vae_only_lsd_dB']]
    ml = float(np.mean([r['model_lsd_dB'] for r in rows]))
    vl = float(np.mean([r['vae_only_lsd_dB'] for r in rows]))
    rec['audio_eval'] = {'clips': len(rows), 'per_clip': rows,
                         'passthrough_mel_dB': pt, 'vae_only_mel_dB': vo,
                         'model_mel_dB': mo, 'ceiling_mel_dB': ce,
                         'model_lsd_dB': ml, 'vae_only_lsd_dB': vl,
                         'lsd_regressed_clips': lsd_reg,
                         'lsd_regressed_n': len(lsd_reg),
                         'mel_improved_n': int(sum(1 for r in rows
                                                   if r['model_mel_dB'] < r['vae_only_mel_dB'])),
                         'gap_closed_vs_vae_only': round((vo - mo) / max(vo - ce, 1e-9), 4),
                         'beats_vae_only': bool(mo < vo),
                         'verdict': ('restoration' if mo < vo - 0.3 and not lsd_reg else
                                     'restoration with LSD regressions' if mo < vo - 0.3 else
                                     'no better than the codec alone')}
    (OUT / f'{a.tag}.json').write_text(json.dumps(rec, indent=1))
    print()
    print('latent L1: identity %.5f -> model %.5f  (%.1f%% reduction)'
          % (rec['final']['val']['l1_identity'], rec['final']['val']['l1_model'],
             100 * rec['final']['val']['l1_reduction']))
    print()
    print('mel dB on %d held-out AudioSet clips, lower is better:' % len(rows))
    print('  %-30s %8.3f' % ('passthrough (degraded)', pt))
    print('  %-30s %8.3f   <- the no-training baseline' % ('VAE round trip only', vo))
    print('  %-30s %8.3f   <- R_phi^audio' % ('model', mo))
    print('  %-30s %8.3f   <- ceiling' % ('clean through VAE', ce))
    print()
    print('LSD mean: %.3f -> %.3f   mel improved on %d/%d clips, LSD REGRESSED on %d: %s'
          % (vl, ml, rec['audio_eval']['mel_improved_n'], len(rows),
             len(lsd_reg), lsd_reg or 'none'))
    print('gap closed against the VAE-only line: %.1f%%   verdict: %s'
          % (100 * rec['audio_eval']['gap_closed_vs_vae_only'], rec['audio_eval']['verdict']))


if __name__ == '__main__':
    main()
