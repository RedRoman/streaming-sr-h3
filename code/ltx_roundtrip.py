"""Does LTX-2's decoder carry our detail band better than H3's own? (the LTX mapper branch, cheapest test)

The LTX route is: our DiT latent -> mapper -> LTX-2 latent -> frozen LTX-2 decoder -> pixels. Before any
mapper is trained the frozen downstream module has to be shown capable of the job: a previous investigation
in this group chased a cross-model mapper and the bottleneck turned out to be the frozen downstream module
itself -- fed a perfect input it still scored 5.2 dB below bicubic.

So feed LTX-2's VAE a perfect input. Round-trip the true null component n = x - U(Dx) through its encoder
and decoder and compare against the same measurement on H3's VAE, which returned n at cosine 0.9155 and
31.93 dB (results/deploy/null_roundtrip.json).

There is a prior from the architecture, read off the checkpoint's own config rather than assumed. LTX-2:
patch_size 4 x compress_space 2 x compress_all 2 x compress_all 2 = 32x spatial, compress_time 2 x
compress_all 2 x compress_all 2 = 8x temporal, 128 latent channels -> 128/(32*32*8) = 0.0156 values per
pixel-frame. H3, from the measured shapes: 22 frames of 480x832 -> [24,7,30,52], i.e. 16x16 pixels and
3.14 frames per latent cell at 24 channels -> 0.0299. LTX-2's latent is about 1.9x less dense, and detail
is exactly what we are short of, so the expectation is that it does worse. This measures it.

ltx-pipelines is not importable in this checkout (ltx_pipelines.multigpu is absent from the package), so
the two modules are built straight from ltx-core's configurators and the checkpoint's embedded config.
"""
import json, os, struct, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
LTX = Path('/cache/zihaohe/baselines/omnivr/packages')
CKPT = Path('/cache/zihaohe/baselines/weights/ltx-2.3-22b-dev.safetensors')
sys.path.insert(0, str(LTX / 'ltx-core/src'))

D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/heldout')
NCLIP = int(os.environ.get('NCLIP', 4))
T = int(os.environ.get('T', 17))        # the encoder needs 1 + 8k frames; 480x832 is already a 32 multiple


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x):
    t = x.permute(0, 3, 1, 2).double()
    d = F.interpolate(t, scale_factor=.25, mode='bicubic', align_corners=False, antialias=True)
    r = F.interpolate(d, size=t.shape[-2:], mode='bicubic', align_corners=False).permute(0, 2, 3, 1).float()
    return r, x - r


def stats(nh, n):
    a, b = nh.reshape(-1).double(), n.reshape(-1).double()
    return {'energy_ratio': float((a @ a) / (b @ b)),
            'cos': float((a @ b) / (a.norm() * b.norm() + 1e-12)),
            'psnr': float(-10 * torch.log10(((a - b) ** 2).mean() + 1e-12))}


def ckpt_config():
    with open(CKPT, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    return json.loads(hdr['__metadata__']['config'])


def sub_sd(prefix):
    """vae.<prefix>.* and vae.per_channel_statistics.* with the prefixes stripped, matching ltx-core's
    VAE_{ENCODER,DECODER}_COMFY_KEYS_FILTER."""
    from safetensors.torch import safe_open
    out = {}
    with safe_open(str(CKPT), framework='pt') as f:
        for k in f.keys():
            if k.startswith(f'vae.{prefix}.'):
                out[k[len(f'vae.{prefix}.'):]] = f.get_tensor(k)
            elif k.startswith('vae.per_channel_statistics.'):
                out['per_channel_statistics.' + k[len('vae.per_channel_statistics.'):]] = f.get_tensor(k)
    return out


def main():
    import torch_npu  # noqa: F401
    from ltx_core.model.video_vae.model_configurator import (VideoDecoderConfigurator,
                                                             VideoEncoderConfigurator)
    cfg = ckpt_config()
    dev, dt = torch.device('npu:0'), torch.bfloat16
    enc = VideoEncoderConfigurator.from_config(cfg)
    dec = VideoDecoderConfigurator.from_config(cfg)
    rep = {}
    for name, mod, pre in (('encoder', enc, 'encoder'), ('decoder', dec, 'decoder')):
        miss, unexp = mod.load_state_dict(sub_sd(pre), strict=False)
        rep[name] = {'missing': len(miss), 'unexpected': len(unexp),
                     'missing_sample': [str(x) for x in list(miss)[:4]],
                     'unexpected_sample': [str(x) for x in list(unexp)[:4]]}
        # strict=False silently drops a mismatched tree, so the counts are reported rather than trusted
        assert not miss and not unexp, json.dumps(rep[name], indent=1)
    enc = enc.to(device=dev, dtype=dt).eval().requires_grad_(False)
    dec = dec.to(device=dev, dtype=dt).eval().requires_grad_(False)
    print(json.dumps({'state_dict': rep}), flush=True)

    clips = sorted(p.name for p in D_.iterdir() if (p / 'hr').is_dir())[:NCLIP]
    acc, per = {}, {}
    with torch.no_grad():
        for c in clips:
            f = sorted((D_ / c / 'hr').glob('*.png'))[:T]
            x = torch.stack([load(p) for p in f])
            r, n = split(x)
            out = {}
            for tag, img in (('gt', x), ('n05', (n + 0.5).clamp(0, 1))):
                v = img.permute(3, 0, 1, 2)[None].to(device=dev, dtype=dt) * 2 - 1
                z = enc(v)
                q = next(iter(dec.decode_video(z))).float().cpu()        # [f,h,w,c] in [0,1]
                m = min(q.shape[0], T)
                if tag == 'gt':
                    out['gt'] = stats(split(q[:m])[1], n[:m])
                    out['gt']['frame_psnr'] = float(-10 * torch.log10(((q[:m] - x[:m]) ** 2).mean() + 1e-12))
                else:
                    out['n05'] = stats(q[:m] - 0.5, n[:m])
                out[tag]['latent_shape'] = list(z.shape)
                out[tag]['frames_out'] = int(q.shape[0])
            per[c] = out
            for kk, vv in out.items():
                acc.setdefault(kk, []).append(vv)
            print(json.dumps({'clip': c, **{kk: {m: (round(y, 4) if isinstance(y, float) else y)
                                                 for m, y in vv.items()} for kk, vv in out.items()}}), flush=True)
    agg = {k: {m: round(float(np.mean([d[m] for d in v])), 4) for m in v[0] if isinstance(v[0][m], float)}
           for k, v in acc.items()}
    h3 = json.loads((ROOT / 'results/deploy/null_roundtrip.json').read_text())['mean']
    res = {'clips': clips, 'frames_per_clip': T, 'ltx2': agg, 'h3_same_measurement': h3,
           'state_dict': rep, 'per_clip': per,
           'latent_density_values_per_pixel_frame': {'ltx2': round(128 / (32 * 32 * 8), 5),
                                                     'h3': round(24 / (16 * 16 * 3.14), 5)}}
    (ROOT / 'results/deploy/ltx_roundtrip.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({'ltx2': agg, 'h3': h3}, indent=1))


main()
