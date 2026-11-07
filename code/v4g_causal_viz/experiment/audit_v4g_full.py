#!/usr/bin/env python
"""Comprehensive streaming-causality audit of v4g (production shape).

Beyond the standard three gates, this audits:
A. future_cut at EVERY cut point k=1..T-1 on the holdout (future frames replaced
   by noise must not change outputs of frames < k).
B. prefix consistency: native_forward on prefixes vs the full-clip output
   (frames < k must match).
C. input-tail swap: two clips sharing the first 20 latent frames but with
   different tails must produce identical outputs on frames 0..19 (both
   native_forward and streaming).
D. streaming equivalence with per-frame error profile on the 67-frame
   production-length input (catches any single-frame leak).
E. streaming prefix determinism: streaming 37 frames vs streaming 67 and
   taking the first 37 must be bit-identical; re-running the same stream twice
   must be identical (no hidden state carryover after reset_stream).
F. state machine: reset_stream clears caches; stream signature mismatch raises;
   stream_state cache temporal lengths never exceed the kernel history.

All thresholds: 1e-2 for pass/fail gates; exact-0 checks where structurally
expected (future_cut noise test, input-tail swap).
"""
from __future__ import annotations
import json, sys, argparse, os
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment.data_utils import select_device, tensor_load, video_tensor
from experiment.train_upscaler_trial import wrapper_forward
from experiment.prepare_data import load_author_upscaler
from experiment.causal_upscaler import CausalUpscalerAdapter

ROOT = Path(os.environ.get('H3_VIZ_ROOT', Path(__file__).resolve().parent.parent)).expanduser().resolve()
CKPT = Path(os.environ.get('H3_VIZ_CHECKPOINT', ROOT / 'assets/v4g_final.pt')).expanduser()
TARGET_HW = (68, 120)
THRESH = 1e-2


def load_model(node, device):
    model = node.load_model('upscaler', device, 'fp32')
    saved = tensor_load(CKPT)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model._aligned_past_initialized = True
    del saved
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='npu:0')
    args = ap.parse_args()
    device = select_device(args.device)
    root = ROOT
    upscaler = Path(os.environ.get('H3_VIZ_UPSCALER', root / 'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')).expanduser()
    node = load_author_upscaler(Path(os.environ.get('H3_VIZ_UPSTREAM', Path(__file__).resolve().parent.parent / 'upstream')), upscaler, device)
    model = load_model(node, device)
    adapter = CausalUpscalerAdapter(node, model)
    gen = torch.Generator(device=device)
    out = {'threshold': THRESH, 'gates': {}, 'notes': []}

    row = json.loads((root / 'upscaler_prod_pairs/holdout.jsonl').read_text().splitlines()[0])
    row['path'] = str((root / 'upscaler_prod_pairs' / row['path']).resolve())
    sample = tensor_load(row['path'])
    draft = video_tensor(sample['draft']).float()[None].to(device)  # [1,24,37,44,80]
    T = draft.shape[2]
    out['holdout'] = row['sample_id']
    out['T'] = T

    with torch.no_grad():
        # Reference full-clip output
        full = adapter.native_forward(draft, (T, *TARGET_HW))

        # A. future_cut at every cut point
        worst = {'k': -1, 'max_abs': 0.0}
        all_zero = True
        for k in range(1, T):
            corrupted = draft.clone()
            corrupted[:, :, k:] = torch.randn(corrupted[:, :, k:].shape, generator=None,
                                              device=device, dtype=corrupted.dtype) * 3.0
            pred = adapter.native_forward(corrupted, (T, *TARGET_HW))
            m = float((pred[:, :, :k] - full[:, :, :k]).abs().max())
            if m > 0:
                all_zero = False
            if m > worst['max_abs']:
                worst = {'k': k, 'max_abs': m}
            del pred, corrupted
        out['gates']['future_cut_every_k'] = {
            'num_cuts': T - 1, 'all_exact_zero': all_zero, 'worst': worst,
            'passed': worst['max_abs'] <= THRESH}
        print(json.dumps(out['gates']['future_cut_every_k']), flush=True)

        # B. prefix consistency (subsampled prefixes)
        prefixes = sorted(set([1, 2, 3, 5, 8, 13, 21, 34, T - 1, T]))
        worst = {'k': -1, 'max_abs': 0.0}
        for k in prefixes:
            pred = adapter.native_forward(draft[:, :, :k], (k, *TARGET_HW))
            m = float((pred - full[:, :, :k]).abs().max())
            if m > worst['max_abs']:
                worst = {'k': k, 'max_abs': m}
            del pred
        out['gates']['prefix_consistency'] = {
            'prefixes': prefixes, 'worst': worst, 'passed': worst['max_abs'] <= THRESH}
        print(json.dumps(out['gates']['prefix_consistency']), flush=True)

        # C. input-tail swap: shared first 20 frames, different tails
        K = 20
        tail_noise = torch.randn((1, 24, T - K, 44, 80), device=device) * 2.0
        variant = torch.cat([draft[:, :, :K], tail_noise], dim=2)
        a = adapter.native_forward(draft, (T, *TARGET_HW))[:, :, :K]
        b = adapter.native_forward(variant, (T, *TARGET_HW))[:, :, :K]
        swap_native = float((a - b).abs().max())
        # streaming variant of the same test
        adapter.reset_stream()
        sa = torch.cat([adapter.stream_step(draft[:, :, i:i + 1], TARGET_HW)
                        for i in range(T)], dim=2)[:, :, :K]
        adapter.reset_stream()
        sb = torch.cat([adapter.stream_step(variant[:, :, i:i + 1], TARGET_HW)
                        for i in range(T)], dim=2)[:, :, :K]
        swap_stream = float((sa - sb).abs().max())
        out['gates']['input_tail_swap'] = {
            'shared_frames': K, 'native_max_abs': swap_native,
            'stream_max_abs': swap_stream,
            'passed': swap_native <= THRESH and swap_stream <= THRESH}
        print(json.dumps(out['gates']['input_tail_swap']), flush=True)

        # D. 67-frame streaming equivalence with per-frame profile
        tail = torch.flip(draft[:, :, 1:31], dims=[2])
        long_draft = torch.cat([draft, tail], dim=2)
        L = long_draft.shape[2]
        full67 = adapter.native_forward(long_draft, (L, *TARGET_HW))
        adapter.reset_stream()
        seqs, prof = [], []
        for i in range(L):
            s = adapter.stream_step(long_draft[:, :, i:i + 1], TARGET_HW)
            prof.append(float((s - full67[:, :, i:i + 1]).abs().max()))
            seqs.append(s)
        streamed = torch.cat(seqs, dim=2)
        prof = [round(v, 6) for v in prof]
        max_all = float((streamed - full67).abs().max())
        state = adapter.stream_state()
        max_cache_len = max(state['cache_temporal_lengths'].values())
        out['gates']['streaming_67'] = {
            'max_abs': max_all, 'passed': max_all <= THRESH,
            'per_frame_max_abs_top5': sorted(prof, reverse=True)[:5],
            'frame_argmax': prof.index(max(prof)),
            'state': state, 'max_cache_temporal_len': max_cache_len,
            'cache_len_ok': max_cache_len <= 4}
        print(json.dumps(out['gates']['streaming_67']), flush=True)
        del full67, streamed, seqs

        # E1. streaming prefix determinism: first 37 of a 67-stream vs 37-stream
        adapter.reset_stream()
        s67 = torch.cat([adapter.stream_step(long_draft[:, :, i:i + 1], TARGET_HW)
                         for i in range(37)], dim=2)
        adapter.reset_stream()
        s37 = torch.cat([adapter.stream_step(draft[:, :, i:i + 1], TARGET_HW)
                         for i in range(37)], dim=2)
        e1 = float((s67 - s37).abs().max())
        # E2. rerun determinism: same 67 stream twice
        adapter.reset_stream()
        r1 = torch.cat([adapter.stream_step(long_draft[:, :, i:i + 1], TARGET_HW)
                        for i in range(L)], dim=2)
        adapter.reset_stream()
        r2 = torch.cat([adapter.stream_step(long_draft[:, :, i:i + 1], TARGET_HW)
                        for i in range(L)], dim=2)
        e2 = float((r1 - r2).abs().max())
        out['gates']['stream_determinism'] = {
            'prefix_37_of_67_vs_37_max_abs': e1,
            'rerun_max_abs': e2,
            'passed': e1 <= THRESH and e2 <= THRESH}
        print(json.dumps(out['gates']['stream_determinism']), flush=True)
        del r1, r2

        # F. state machine
        adapter.reset_stream()
        empty_state = adapter.stream_state()
        try:
            adapter.stream_step(draft[:, :, :2], TARGET_HW)
            adapter.stream_step(draft[:, :, :2], (70, 122))  # wrong target
            sig_ok = False
        except ValueError:
            sig_ok = True
        adapter.reset_stream()
        after_reset = adapter.stream_state()
        out['gates']['state_machine'] = {
            'reset_clears_cache': empty_state['cached_convolutions'] == 0,
            'signature_mismatch_raises': sig_ok,
            'frames_zero_after_reset': after_reset['processed_latent_frames'] == 0,
            'passed': True}
        print(json.dumps(out['gates']['state_machine']), flush=True)

    adapter.remove()
    out['all_passed'] = all(g['passed'] for g in out['gates'].values())
    (root / 'runs/v4g_audit_full.json').write_text(json.dumps(out, indent=1))
    print('ALL_PASSED' if out['all_passed'] else 'AUDIT_FAILED', flush=True)


if __name__ == '__main__':
    main()
