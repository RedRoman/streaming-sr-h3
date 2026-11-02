"""What does the colleague's one-step streaming SR stack cost per frame on a 910B?

Their pipeline (/temp/liusonghua/h3_wan_full_pipeline_20261022) replaces our generator entirely: H3 is kept
only as an encoder, the latent is mapped into Wan space, upsampled x3 in latent, and a 1.419B SR-DiT
produces the HR latent in ONE evaluation (fixed timestep 1000, z = noise - velocity) under a causal
block-sparse attention with a 3-block KV history, 2 latents per chunk; a 23 MB Flash-VAED decoder then
emits 8 frames per chunk with continuous state.

Against ours -- 20.11B, 4 steps, 22-frame blocks, 1.79 s/frame single-stream -- the parameter-passes alone
differ by 56x, so the interesting question is not whether it is faster but by how much, measured, on the
same hardware. Their own report gives 109-120 s per 85-frame sample, but that figure covers three full
decodes, LPIPS over 85 frames, and four mp4 encodes, so it is not a latency.

Timed here, warm, one process, one card, on random tensors of the right shape:
  srdit   one 2-latent chunk at their grid (96x168 Wan latent = 1344x768 RGB)
  srdit   the same at ours (64x104 = 832x512, the multiple-of-4 patch grid nearest our 832x480)
  decode  Flash-VAED on one 2-latent chunk, both grids
Each is divided by the 8 frames a chunk emits to give s/frame directly comparable to our 1.79.
"""
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace

PKG = Path('/temp/liusonghua/h3_wan_full_pipeline_20261022/ascend_eval')
WEIGHTS = Path('/temp/liusonghua/h3_wan_full_pipeline_20261022/weights')
sys.path.insert(0, str(PKG))

import torch
import torch_npu  # noqa: F401


def sync():
    torch.npu.synchronize()


def timeit(fn, warm=2, runs=5):
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    sync()
    return (time.perf_counter() - t0) / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load-weights', action='store_true',
                    help='5.3 GB from the shared mount; shapes and dtypes decide the cost, not the values, '
                         'so the default is a random init')
    ap.add_argument('--grids', default='96x168,64x104')
    ap.add_argument('--no-decoder', action='store_true')
    ap.add_argument('--out', default='/cache/zihaohe/streaming_sr0/results/deploy/bench_wan_chain.json')
    a = ap.parse_args()

    import common
    common.MODELS = WEIGHTS
    dev = torch.device('npu:0')
    torch.npu.set_device(dev)
    out = {'device': '910B3', 'chunk_latents': 2, 'frames_per_chunk': 8, 'grids': {}}

    import model as M
    sr = M.StreamingSR()
    if a.load_weights:
        ck = torch.load(WEIGHTS / 'srdit_h3_step2000.pt', map_location='cpu', weights_only=False, mmap=True)
        sr.load_state_dict(ck['model'], strict=True)
        del ck
    sr = sr.to(dev).eval().requires_grad_(False)
    sr.release_eval_cache = True
    out['srdit_params'] = sum(p.numel() for p in sr.parameters())

    prompt = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16)
    for g in a.grids.split(','):
        h, w = (int(x) for x in g.split('x'))
        rec = {'latent_hw': [h, w], 'rgb_hw': [h * 8, w * 8]}
        z = torch.randn(1, 16, 2, h, w, device=dev, dtype=torch.bfloat16)
        n = torch.randn_like(z)

        def run_sr(cache=[None]):
            with torch.no_grad(), torch.autocast('npu', dtype=torch.bfloat16):
                y, c = sr(n, z, prompt, 0, None)
            return y

        try:
            rec['srdit_s_per_chunk'] = timeit(run_sr)
            rec['srdit_s_per_frame'] = rec['srdit_s_per_chunk'] / 8
        except Exception as e:                      # noqa: BLE001 - report, do not hide
            rec['srdit_error'] = f'{type(e).__name__}: {e}'
        out['grids'][g] = rec
        print(json.dumps({g: rec}, default=float), flush=True)

    if not a.no_decoder:
        try:
            from model import StreamingDecoder
            fake = SimpleNamespace(scale=[torch.zeros(16, device=dev), torch.ones(16, device=dev)],
                                   dtype=torch.bfloat16)
            dec = StreamingDecoder.__new__(StreamingDecoder)
            from flash_decoder import FlashDecoder
            FlashDecoder.__init__(dec, fake, segment=2,
                                  weights=WEIGHTS / 'Flash-VAED/Flash_VAED_Wan.pth')
            dec = dec.to(dev).eval().requires_grad_(False)
            for g in a.grids.split(','):
                h, w = (int(x) for x in g.split('x'))
                zz = torch.randn(1, 16, 2, h, w, device=dev)

                def run_dec():
                    with torch.no_grad(), torch.autocast('npu', dtype=torch.bfloat16):
                        return dec(zz, None, first=True)
                s = timeit(run_dec, warm=1, runs=3)
                out['grids'][g]['decode_s_per_chunk'] = s
                out['grids'][g]['decode_s_per_frame'] = s / 8
                print(json.dumps({g: {'decode_s_per_chunk': s}}, default=float), flush=True)
        except Exception as e:                      # noqa: BLE001
            out['decoder_error'] = f'{type(e).__name__}: {e}'
            print(json.dumps({'decoder_error': out['decoder_error']}), flush=True)

    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print()
    print('%-10s %12s %12s %12s %12s' % ('grid', 'srdit/chunk', 'srdit/frame', 'dec/chunk', 'dec/frame'))
    for g, r in out['grids'].items():
        print('%-10s %12s %12s %12s %12s'
              % (g, '%.3f' % r['srdit_s_per_chunk'] if 'srdit_s_per_chunk' in r else 'ERR',
                 '%.4f' % r['srdit_s_per_frame'] if 'srdit_s_per_frame' in r else '-',
                 '%.3f' % r['decode_s_per_chunk'] if 'decode_s_per_chunk' in r else '-',
                 '%.4f' % r['decode_s_per_frame'] if 'decode_s_per_frame' in r else '-'))
    print()
    print('ours, single stream, all phases: 1.789 s/frame')


if __name__ == '__main__':
    main()
