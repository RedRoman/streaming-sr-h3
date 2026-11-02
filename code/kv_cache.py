"""Cross-chunk KV cache for the ComfyUI MiniMax-H3 DiT (Stage 5, PROTOCOL_stage5.md).

Chunk k still computes its own 7 latent frames, but its attention additionally sees the video K/V of the
previous W chunks. Information therefore flows between chunks through attention rather than through the guide
pixels, which is where accumulation would appear if it appears at all.

H3's RoPE is absolute (position_ids -> rope_freqs -> applied to q,k), and our 124-frame clips stay inside the
model's trained 124-362 frame range, so post-RoPE K/V can be cached directly provided each chunk's video rows
carry their clip-absolute time positions. `install_positions` performs that correction by shifting every
target-timeline segment of the layout; without it each chunk restarts at the layout cursor and the cached
history occupies the same coordinates as the current chunk's own rows.

Scope: this gives cross-chunk context, not a compute reduction -- each chunk still runs a full 7-latent-frame
forward. Emitting only the 5 new latent frames needs surgery on H3's 2+5k temporal layout and is left for later.
"""
import torch

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _t_spans(n, start=0):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[(start + k) % 5] for k in range(n)]


def target_timeline_offset(latent_start):
    """Clip-absolute time origin of a chunk whose first latent frame is `latent_start`."""
    return sum(_t_spans(latent_start)) if latent_start else 0.0


def install_positions(cache):
    """Place each chunk on the clip's absolute timeline before RoPE is computed.

    Without this every chunk restarts at the layout's own origin (`cursor = text_len`), so the cached history
    of chunk k-1 lands on exactly the same RoPE coordinates as chunk k's own rows: two sets of tokens at
    identical positions. That is a degenerate configuration and it shows up as regular stripe artefacts that
    grow with the number of cached chunks.

    The text segment stays put. Everything on the target timeline -- cond, cond_audio, audio, video -- shifts
    by the same amount, which keeps the guide-to-video geometry the model was trained with while moving the
    whole chunk later in absolute time.
    """
    import comfy.ldm.minimax.model as impl
    original = impl.PackedLayout.__init__

    def __init__(self, text_len, *a, **kw):
        original(self, text_len, *a, **kw)
        off = target_timeline_offset(cache.latent_start)
        if off:
            self.position_ids = self.position_ids.clone()
            self.position_ids[text_len:, 0] += off
    impl.PackedLayout.__init__ = __init__
    return original


class H3KVCache:
    """Per-layer post-RoPE video K/V of the last `window` chunks.

    Memory: 50 layers x window x 2 tensors of [heads, video_rows, head_dim] in bf16. At 832x480 that is
    2730 video rows per chunk, so roughly 3.9 GB per cached chunk; window 5 alongside the 40 GB base overflows
    a 64 GB card. We therefore cache only `tail_frames` latent frames per chunk: 2 frames is 780 rows, about
    1.1 GB per chunk, so window 2 costs ~2.2 GB. SolarWM can afford full chunks because sequence parallelism
    shards the heads across 8 ranks; we run one card.

    Which frames are cached matters as much as how many -- see the comment in `_cached_attention`."""

    def __init__(self, window=2, tail_frames=2, new_frames=5):
        self.window = window
        self.tail_frames = tail_frames
        self.new_frames = new_frames   # latent frames chunk k owns that chunk k+1 does not recompute
        self.layers = {}
        self.chunk = 0
        self.committing = False
        self.latent_start = 0      # this chunk's first latent frame within the clip; drives the RoPE offset

    def clear(self):
        self.layers.clear()
        self.chunk = 0
        self.latent_start = 0

    def history(self, layer):
        return self.layers.get(layer, [])

    def commit(self, layer, k, v):
        # clone, not just detach: the tail slice is a view and would otherwise pin the whole [1,heads,S,dim]
        # K/V tensor of every layer (~160 MB each, 8 GB over 50 layers) and exhaust the card.
        entries = self.layers.get(layer, [])
        self.layers[layer] = [*entries, (self.chunk, k.detach().clone(), v.detach().clone())][-self.window:]

    def advance(self):
        self.chunk += 1

    def stats(self):
        n = len(self.layers.get(0, []))
        rows = sum(e[1].shape[2] for e in self.layers.get(0, [])) if n else 0   # [1, heads, rows, dim]
        return {'chunks_cached': n, 'rows_cached': rows, 'layers': len(self.layers)}


def install(model, cache, quota=None):
    """Wrap Attention.forward so each layer attends over [cached video K/V | current K/V].

    Sampling runs 49 denoise steps; committing on every step would make a chunk read its own earlier steps as
    "history" and would pin 49x the memory. So the wrapper is read-only by default and only commits when
    `cache.committing` is set, which `commit_pass` does for a single forward on the finished latent.
    quota, when given, is called as quota(k, v, layer) -> (k, v) just before committing."""
    import comfy.ldm.minimax.model as impl
    original = impl.Attention.forward
    state = {'layer': 0}

    def forward(self, x, rope_freqs=None, transformer_options={}):
        layout = transformer_options.get('minimax_h3_layout')
        if layout is None or not transformer_options.get('_h3_kv_enabled', True):
            return original(self, x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        layer = transformer_options.get('block_index', state['layer'])
        return _cached_attention(self, x, rope_freqs, transformer_options, layout, cache, layer, quota)
    impl.Attention.forward = forward
    original_layout_init = install_positions(cache)

    class Handle:
        def __init__(self, c):
            self.cache = c

        @staticmethod
        def remove():
            impl.Attention.forward = original
            impl.PackedLayout.__init__ = original_layout_init
    return Handle(cache)


def _cached_attention(self, x, rope_freqs, transformer_options, layout, cache, layer, quota):
    import comfy.model_management
    import comfy.quant_ops
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
    s = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    v = v.view(s, self.heads, self.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = comfy.model_management.cast_to(self.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(self.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
        q, k = q[0], k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))

    va, vb, _ = next(seg for seg in layout.segments if seg[2] == 'video')
    kq = k.transpose(0, 1).unsqueeze(0)          # [1, heads, S, dim]
    vq = v.transpose(0, 1).unsqueeze(0)
    # Cache the tail of the chunk's NON-OVERLAPPING prefix, not its last frames. Blocks overlap by 5 pixel
    # frames, so chunk k's last 2 latent frames are exactly the ones chunk k+1 recomputes for itself: caching
    # them would put clean history on the same clip-absolute coordinates as the current chunk's own noisy
    # rows, which is degenerate and shows up as stripe artefacts. Local frames [new_frames-tail, new_frames)
    # are the newest history chunk k+1 does not already own.
    latent_t = layout.signature[1]
    frame_rows = (vb - va) // latent_t
    hi = min(cache.new_frames, latent_t)
    lo = max(0, hi - cache.tail_frames)
    vid_k, vid_v = kq[:, :, va + lo * frame_rows:va + hi * frame_rows], vq[:, :, va + lo * frame_rows:va + hi * frame_rows]

    hist = cache.history(layer)
    if hist:
        kq = torch.cat([*[h[1] for h in hist], kq], dim=2)
        vq = torch.cat([*[h[2] for h in hist], vq], dim=2)

    if cache.committing:
        ck, cv = (quota(vid_k, vid_v, layer) if quota is not None else (vid_k, vid_v))
        cache.commit(layer, ck, cv)

    out = optimized_attention(AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0)),
                              AttentionTensorContainer(kq), AttentionTensorContainer(vq),
                              self.heads, preferred_attention=self.comfy_attention, mask=None,
                              skip_reshape=True, transformer_options=transformer_options)
    return self.out_proj(out.squeeze(0))


def commit_pass(h3, latent_video, guide, hw, cache, seed=101):
    """One forward on the finished chunk, with committing enabled, so the cache holds that chunk's K/V."""
    import comfy.nested_tensor
    import torch as _t
    from h3_comfy import SIGMAS, native_training_math
    T = latent_video.shape[2]
    frames = (T - 2) // 5 * 17 + 5
    audio_t = round(frames / 24 * 40)
    kf = [{'resolved_frame_index': 0, 'latent': guide}]
    ctx = h3.context
    positive = [[ctx['prompt_embeds'][None], {'minimax_token_tags': ctx['text_token_tags'], 'minimax_keyframes': kf}]]
    latent = comfy.nested_tensor.NestedTensor((latent_video, _t.zeros(1, 32, 2, audio_t)))
    cache.committing = True
    try:
        with _t.no_grad(), native_training_math():
            import comfy.sample
            # one model evaluation at the final sigma; only the K/V it writes matters, the output is discarded
            comfy.sample.sample(h3.patcher, comfy.sample.prepare_noise(latent, seed), 1, 1., 'euler', 'simple',
                                positive, [], latent, seed=seed, disable_pbar=True, sigmas=SIGMAS[-2:])
    finally:
        cache.committing = False
    return cache.stats()
