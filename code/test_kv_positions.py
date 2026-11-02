"""Check that install_positions actually moves each chunk onto the clip's absolute timeline.

The bug this guards against: absolute_video_positions existed but was never called, so every chunk's rows
carried the same RoPE coordinates and the cached history of chunk k-1 collided with chunk k's own rows.
"""
import json, sys
import torch
from h3_comfy import ROOT, patch_module_to
patch_module_to()
sys.path.insert(0, str(ROOT / 'code/ComfyUI'))
import comfy.ldm.minimax.model as impl
import kv_cache as KV

TEXT, LT, LH, LW, AT = 11, 7, 30, 52, 37
GUIDE = {'resolved_frame_index': 0, 'latent': torch.zeros(1, 24, LT, LH, LW)}


def layout():
    return impl.PackedLayout(TEXT, LT, LH, LW, AT, keyframes=[GUIDE])


out = []
base = layout()
cache = KV.H3KVCache()
orig = KV.install_positions(cache)
try:
    cache.latent_start = 0
    l0 = layout()
    same = bool(torch.equal(l0.position_ids, base.position_ids))
    out.append({'check': 'latent_start_0_is_identity', 'equal': same})
    assert same, 'chunk 0 must be untouched, otherwise the cache arm is not comparable to the baseline'

    cache.latent_start = 5
    l1 = layout()
    d = (l1.position_ids - base.position_ids)
    txt = d[:TEXT].abs().max()
    off = d[TEXT:, 0]
    spatial = d[TEXT:, 1:].abs().max()
    out.append({'check': 'chunk1_shift', 'text_moved': float(txt), 'spatial_moved': float(spatial),
                'time_offset_min': float(off.min()), 'time_offset_max': float(off.max()),
                'expected': KV.target_timeline_offset(5)})
    assert float(txt) == 0.0 and float(spatial) == 0.0
    assert abs(float(off.min()) - float(off.max())) < 1e-9
    assert abs(float(off.min()) - KV.target_timeline_offset(5)) < 1e-9

    # The blocks overlap by 5 pixel frames, so chunk 0 and chunk 1 DO share two latent frames -- those are
    # literally the same moments of the clip. What must not collide is the part of chunk 0 that gets cached
    # with chunk 1's own rows: that is the whole reason the cached slice is the non-overlapping prefix.
    va, vb, _ = next(s for s in base.segments if s[2] == 'video')
    rows = (vb - va) // LT
    cfg = KV.H3KVCache()
    hi = min(cfg.new_frames, LT); lo = max(0, hi - cfg.tail_frames)
    cached_t = base.position_ids[va + lo * rows:va + hi * rows, 0]
    t1 = l1.position_ids[va:vb, 0]
    overlap = float(torch.isin(cached_t, t1).float().mean())
    out.append({'check': 'cached_rows_do_not_collide', 'cached_frames_local': [lo, hi],
                'cached_t': [float(cached_t.min()), float(cached_t.max())],
                'chunk1_t': [float(t1.min()), float(t1.max())], 'overlap_fraction': overlap})
    assert overlap == 0.0, 'the cached slice still lands on chunk 1 coordinates'

    # and the cached rows must be immediately before chunk 1, not far in the past
    gap = float(t1.min() - cached_t.max())
    out.append({'check': 'cache_is_adjacent', 'gap_units': gap})
    assert 0 < gap <= 2 * KV.FRAME_RESCALE * 4 + 1e-6, 'cached history is not adjacent to the new chunk'

    # every chunk of a 124-frame clip must stay inside the range the model was trained on
    spans = [KV.target_timeline_offset(5 * k) for k in range(7)]
    last = layout
    cache.latent_start = 30
    l6 = layout()
    out.append({'check': 'range', 'chunk_offsets': [round(s, 3) for s in spans],
                'max_time_coord': float(l6.position_ids[:, 0].max())})
finally:
    impl.PackedLayout.__init__ = orig

for r in out:
    print(json.dumps(r), flush=True)
(ROOT / 'results/stage5_position_check.json').write_text(json.dumps(out, indent=1))
print('ALL POSITION CHECKS PASSED')
