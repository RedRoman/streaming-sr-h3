"""Block-causal attention masks for the ComfyUI MiniMax-H3 DiT (Stage 4, PROTOCOL_stage4.md).

Only the target video segment's rows are restricted: text / keyframe condition / audio rows stay fully visible,
since they are given conditions rather than autoregressive state. Queries in video latent-frame t may attend to
video rows of latent-frame t' when floor(t'/C) <= floor(t/C); C=1 is strict frame causality.

The mask is installed by wrapping Attention.forward, which upstream calls with mask=None.
"""
import torch

_CACHE = {}


def build_mask(layout, chunk, device):
    """[S, S] bool, True = may attend. Cached per (signature, chunk)."""
    key = (layout.signature, chunk, str(device))
    if key in _CACHE:
        return _CACHE[key]
    S = layout.seq_len
    m = torch.ones(S, S, dtype=torch.bool, device=device)
    va, vb, _ = next(s for s in layout.segments if s[2] == 'video')
    n_video = vb - va
    latent_t = layout.signature[1]
    frame_rows = n_video // latent_t
    assert frame_rows * latent_t == n_video, (n_video, latent_t)
    t = torch.arange(n_video, device=device) // frame_rows      # latent-frame index per video row
    g = torch.div(t, chunk, rounding_mode='floor')              # chunk index
    m[va:vb, va:vb] = g[:, None] >= g[None, :]
    _CACHE[key] = m
    return m


def install(model, mode, chunk=1):
    """mode: 'bidir' (no-op) | 'causal' (chunk=1) | 'causal_cw{C}'. Returns a handle with .remove()."""
    import comfy.ldm.minimax.model as impl
    if mode == 'bidir':
        return type('H', (), {'remove': staticmethod(lambda: None), 'mode': 'bidir'})()
    original = impl.Attention.forward

    def forward(self, x, rope_freqs=None, transformer_options={}):
        layout = transformer_options.get('minimax_h3_layout')
        if layout is None:
            return original(self, x, rope_freqs=rope_freqs, transformer_options=transformer_options)
        mask = build_mask(layout, chunk, x.device)
        opts = dict(transformer_options)
        opts['_h3_causal_mask'] = mask
        return _masked_attention(self, x, rope_freqs, opts, mask)
    impl.Attention.forward = forward

    class Handle:
        def __init__(self, m):
            self.mode = m

        @staticmethod
        def remove():
            impl.Attention.forward = original
    return Handle(mode)


def _masked_attention(self, x, rope_freqs, transformer_options, mask):
    """Same math as the upstream Attention.forward, with an explicit attention mask."""
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
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
    qc = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    kc = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    vc = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(qc, kc, vc, self.heads, preferred_attention=self.comfy_attention,
                              mask=mask, skip_reshape=True, transformer_options=transformer_options)
    return self.out_proj(out.squeeze(0))
