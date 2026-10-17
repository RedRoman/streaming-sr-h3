"""Step 1 bug check (post-hoc): does the reference image reach the DiT, and does its content change the prediction?
One DiT forward at the first sampling step per variant (same noise, sigma, LR guide); sampling is aborted afterwards.
Variants: none (plain prompt) | gt | wrong | gt_text_wrong_rows (text sees gt, DiT rows from wrong) | gt_text_no_rows."""
import json

import numpy as np
import torch
from PIL import Image

from h3_comfy import H3, ROOT

E, C = ROOT / 'data/events', ROOT / 'results/step1_contexts'


class Stop(Exception):
    pass


def first_tensor(o):
    while isinstance(o, (tuple, list)):
        o = o[0]
    if hasattr(o, 'unbind') and getattr(o, 'is_nested', False):
        o = o.unbind()[0]
    return o.detach().float().cpu()


def main():
    h3 = H3(load_dit=True, dit='ref2va')
    model = h3.patcher.model.diffusion_model
    out = {}
    for seq in ['bottle-14', 'crab-12', 'tiger-7']:
        ev = json.loads((E / seq / 'event.json').read_text())
        k = ev['blocks']['kpost']
        low = torch.stack([torch.from_numpy(np.array(Image.open(E / seq / 'lr' / f'{f:06d}.png').convert('RGB')).copy()).permute(2, 0, 1)
                           for f in range(17 * k, 17 * k + 22)]).float() / 255
        hw = [ev['hr_wh'][1], ev['hr_wh'][0]]
        guide = h3.encode_guide(low, hw)
        neutral = torch.load(C / 'neutral.pt', weights_only=True)
        gt = torch.load(C / f'{seq}__gt.pt', weights_only=True)
        wrong = torch.load(C / f'{seq}__wrong.pt', weights_only=True)
        variants = {'none': (neutral, None), 'gt': (gt, gt), 'wrong': (wrong, wrong),
                    'gt_text_wrong_rows': (gt, wrong), 'gt_text_no_rows': (gt, None)}
        vel, info = {}, {}
        for name, (ctx, rows) in variants.items():
            cap = {}

            def hook(module, args, kwargs, output):
                p = kwargs['minimax_payload']
                cap.update(n_cond_video_latents=len(p.get('cond_video_latents', [])), n_refs=len(p.get('refs') or []),
                           segments=[s[2] for s in p['layout'].segments], seq_len=int(p['layout'].seq_len))
                cap['out'] = first_tensor(output)
                raise Stop()
            hd = model.register_forward_hook(hook, with_kwargs=True)
            try:
                h3.sample(guide, 'base', low=low, hw=hw, seed=101,
                          context={'prompt_embeds': ctx['prompt_embeds'], 'text_token_tags': ctx['text_token_tags']},
                          refs=None if rows is None else [h3.ref_block(rows['ref_image'])])
            except Stop:
                pass
            finally:
                hd.remove()
            vel[name] = cap.pop('out'); info[name] = cap
        ref = vel['gt']
        rel = lambda a, b: float((a - b).square().mean().sqrt() / b.square().mean().sqrt())
        info['relative_velocity_diff_vs_gt'] = {n: rel(v, ref) for n, v in vel.items() if n != 'gt' and v.shape == ref.shape}
        out[seq] = info
        print(json.dumps({seq: info}), flush=True)
    (ROOT / 'results/step1_diag_path.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
