"""MiniMax-H3 3D latent upscaler (LBH LatentResizer3D v1), reimplemented from the published architecture spec
(LBH-123-AI/Minimax_h3_latent_Upscaler, minimax_h3_latent_upscaler_3d_conv_v1/config.json, Apache-2.0) and verified
against the released state dict's tensor shapes. Operates on H3's 24-channel latents; only H and W are scaled.

forward: normalize -> conv_in -> in_blocks -> trilinear(H,W) -> out_blocks -> norm_out -> silu -> conv_out -> denormalize
"""
import json

import torch
from torch import nn
from torch.nn import functional as F

MEAN = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949,
        -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736,
        0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
        0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
        -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
STD = [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833,
       2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114,
       1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
       0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
       3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523]
LAYOUT = ['res', 'temporal', 'res', 'res', 'temporal', 'res', 'res', 'temporal', 'res',
          'res', 'temporal', 'res', 'res', 'temporal', 'res', 'res', 'temporal', 'res']


class ResBlockEmb3D(nn.Module):
    def __init__(self, c, emb, groups=32, dropout=0.1):
        super().__init__()
        self.in_layers = nn.Sequential(nn.GroupNorm(groups, c), nn.SiLU(), nn.Conv3d(c, c, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb, 2 * c))
        self.out_norm = nn.GroupNorm(groups, c)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Conv3d(c, c, 3, padding=1))

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).chunk(2, dim=-1)
        h = self.out_norm(h) * (1 + scale[..., None, None, None]) + shift[..., None, None, None]
        return x + self.out_layers(h)


class TemporalConv(nn.Module):
    def __init__(self, c, k=5, groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(groups, c)
        self.dwconv = nn.Conv3d(c, c, (k, 1, 1), padding=(k // 2, 0, 0), groups=c)
        self.pwconv = nn.Conv3d(c, c, 1)

    def forward(self, x, emb=None):
        return x + self.pwconv(self.dwconv(self.norm(x)))


def _blocks(c, emb):
    return nn.ModuleList([ResBlockEmb3D(c, emb) if k == 'res' else TemporalConv(c) for k in LAYOUT])


class LatentResizer3D(nn.Module):
    def __init__(self, channels=24, width=512, emb=64, groups=32):
        super().__init__()
        self.conv_in = nn.Conv3d(channels, width, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.in_blocks = _blocks(width, emb)
        self.out_blocks = _blocks(width, emb)
        self.norm_out = nn.GroupNorm(groups, width)
        self.conv_out = nn.Conv3d(width, channels, 3, padding=1)
        self.register_buffer('mean', torch.tensor(MEAN).view(1, -1, 1, 1, 1))
        self.register_buffer('std', torch.tensor(STD).view(1, -1, 1, 1, 1))

    @torch.no_grad()
    def forward(self, latent, scale=4.0, size=None):
        """latent: [B,24,T,H,W] (raw H3 latent). Returns the upscaled latent in the same raw space."""
        dt = next(self.conv_in.parameters()).dtype
        x = ((latent.float() - self.mean) / self.std).to(dt)
        emb = self.embed(torch.tensor([[scale - 1.0]], device=x.device, dtype=dt))
        h = self.conv_in(x)
        for b in self.in_blocks:
            h = b(h, emb)
        t, hh, ww = h.shape[-3:]
        out_hw = size if size is not None else (round(hh * scale), round(ww * scale))
        h = F.interpolate(h.float(), size=(t, *out_hw), mode='trilinear', align_corners=False).to(dt)
        for b in self.out_blocks:
            h = b(h, emb)
        out = self.conv_out(F.silu(self.norm_out(h)))
        return (out.float() * self.std + self.mean)


def load(path, device='npu', dtype=torch.bfloat16):
    from safetensors.torch import load_file
    model = LatentResizer3D()
    sd = load_file(str(path))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m not in ('mean', 'std')]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.to(device=device, dtype=dtype).eval(), {'tensors': len(sd)}
