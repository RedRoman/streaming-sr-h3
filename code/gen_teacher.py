"""Teacher targets for one-step distillation: run the locked 8-step pipeline over the training windows.

The deployed configuration is 8 Turbo steps with innerDC projecting onto R_phi(y) at two of them. One-step
distillation cannot keep those projections -- there is no step 5 or 7 to hook -- so the student has to learn
the composite map `guide -> final latent`, innerDC included. That makes the teacher output, not the HR
latent, the right regression target: the student should reproduce what the pipeline actually emits, which is
already consistent with the observation, rather than chase the ground truth and lose the consistency the
projections were buying.

Saved per window: the teacher's final video latent, plus the guide it was produced from, so training needs
nothing else. The HR target is kept alongside for diagnostics (how far the teacher itself sits from truth).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file, load_file

from h3_comfy import H3, ROOT, sync

DATA = ROOT / 'data/stage7_train'
LAT = ROOT / 'data/stage7_latents'
OUT = ROOT / 'data/stage7_teacher'
H, W = 480, 832


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--lora', type=Path, default=ROOT / 'results/stage1_lora/lora_step002000.safetensors')
    ap.add_argument('--turbo-steps', type=int, default=8)
    ap.add_argument('--rphi', type=Path, default=ROOT / 'results/range_branch/k1.pt')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    out_dir = OUT / a.split
    out_dir.mkdir(parents=True, exist_ok=True)

    import h3_comfy
    import lora as L
    import turbo as TB
    from safetensors.torch import load_file as lf
    TURBO = ROOT / 'assets/turbo/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors'
    h3 = H3(load_dit=True, dit='fl2va', lora_path=None)
    model = h3.patcher.model.diffusion_model
    rec_turbo = TB.merge(model, TURBO)
    L.inject(model, 32)
    n_lora = L.load(model, lf(str(a.lora)))
    h3.frozen = [(n, v, v._version) for n, v in model.named_parameters()
                 if not (n.endswith('.lora_a') or n.endswith('.lora_b'))]
    h3_comfy.SIGMAS = TB.turbo_sigmas(a.turbo_steps, 12.0)

    from range_branch import RangeBranch
    st = torch.load(a.rphi, map_location='cpu')
    rb = RangeBranch(**st['cfg']).eval(); rb.load_state_dict(st['state'])
    assert st['cfg']['k'] == 1

    files = sorted((DATA / a.split).glob('*.npz'))
    if a.limit:
        files = files[:a.limit]
    files = [f for i, f in enumerate(files) if i % a.nshards == a.shard]
    (out_dir / f'runtime_shard{a.shard}.json').write_text(json.dumps(
        {'turbo': rec_turbo, 'lora_tensors': n_lora, 'steps': a.turbo_steps,
         'sigmas': [round(float(x), 4) for x in h3_comfy.SIGMAS], 'files': len(files)}, indent=1, default=str))

    done = 0
    t_all = time.perf_counter()
    for f in files:
        q = out_dir / (f.stem + '.safetensors')
        if q.exists():
            done += 1
            continue
        d = np.load(f, allow_pickle=True)
        lr = torch.from_numpy(d['lr']).float().div(255).permute(0, 3, 1, 2)      # [T,3,h,w]
        with torch.no_grad():
            est = torch.cat([rb(lr[t:t + 1][None]) for t in range(lr.shape[0])]).clamp(0, 1)
        guide = h3.encode_guide(lr, [H, W])                                      # guide from raw y
        t0 = time.perf_counter()
        sampled, rec = h3.sample(guide, 'innerdc', low=est, hw=[H, W], seed=101)  # dc target = R_phi(y)
        sync()
        lat = LAT / a.split / (f.stem + '.safetensors')
        payload = {'teacher': sampled['video'].to(torch.float16).cpu(),
                   'guide': guide.to(torch.float16).cpu()}
        if lat.exists():
            payload['target'] = load_file(str(lat))['target'].to(torch.float16)
        save_file(payload, str(q))
        done += 1
        if done % 10 == 0:
            el = time.perf_counter() - t_all
            print(json.dumps({'done': done, 'of': len(files), 'sec_last': round(time.perf_counter() - t0, 1),
                              'eta_min': round((len(files) - done) * el / max(done, 1) / 60, 1)}), flush=True)
    print(json.dumps({'shard': a.shard, 'split': a.split, 'written': done, 'finished': True}), flush=True)


if __name__ == '__main__':
    main()
