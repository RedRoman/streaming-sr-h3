"""LTX-2 VAE encode/decode wall time against H3's, so the latency side of the LTX branch is a number.

Our block is 22 frames at 480x832 and H3's decode measured 1.05 s of a 17.3 s block. The LTX branch would
replace that decode with LTX-2's, so the question is what LTX-2's decode costs on the nearest geometry its
grid allows (25 frames, 1+8k). Both are warmed up first: an unwarmed Ascend call includes operator
compilation and has been off by two orders of magnitude in this project before.
"""
import json, os, sys, time
from pathlib import Path

import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
import ltx_roundtrip as LR


def sync():
    torch.npu.synchronize()


def bench(fn, n=3):
    fn(); sync()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); sync(); t.append(time.perf_counter() - t0)
    return round(min(t), 3), round(sum(t) / len(t), 3)


def main():
    import torch_npu  # noqa: F401
    from ltx_core.model.video_vae.model_configurator import (VideoDecoderConfigurator,
                                                             VideoEncoderConfigurator)
    cfg = LR.ckpt_config()
    dev, dt = torch.device('npu:0'), torch.bfloat16
    enc = VideoEncoderConfigurator.from_config(cfg); enc.load_state_dict(LR.sub_sd('encoder'))
    dec = VideoDecoderConfigurator.from_config(cfg); dec.load_state_dict(LR.sub_sd('decoder'))
    enc = enc.to(device=dev, dtype=dt).eval().requires_grad_(False)
    dec = dec.to(device=dev, dtype=dt).eval().requires_grad_(False)
    out = {}
    with torch.no_grad():
        v = torch.randn(1, 3, 25, 480, 832, device=dev, dtype=dt).clamp(-1, 1)
        out['ltx_encode_25f_min_mean_s'] = bench(lambda: enc(v))
        z = enc(v)
        out['ltx_latent_shape'] = list(z.shape)
        out['ltx_decode_25f_min_mean_s'] = bench(lambda: next(iter(dec.decode_video(z))))
        out['ltx_peak_GB'] = round(torch.npu.max_memory_allocated() / 1e9, 2)
    del enc, dec, z, v
    torch.npu.empty_cache()
    from h3_comfy import H3
    h3 = H3(load_dit=False)
    with torch.no_grad():
        x = torch.rand(22, 480, 832, 3)
        zz = h3.encode(x)
        out['h3_encode_22f_min_mean_s'] = bench(lambda: h3.encode(x))
        out['h3_latent_shape'] = list(zz.shape)
        out['h3_decode_22f_min_mean_s'] = bench(lambda: h3.decode(zz))
    out['context'] = ('H3 decode measured 1.05 s inside a 17.3 s / 22-frame block in the shipped run, of '
                      'which sampling is 8.69 s. Any added per-frame cost here lands on top of that.')
    (ROOT / 'results/deploy/ltx_latency.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


main()
