"""Compare the colleague's CPU float64 Dykstra projection with an NPU float32 run of the same math."""
import json, time, numpy as np, torch, torch_npu  # noqa
from PIL import Image
from pathlib import Path
from pixel_consistency import BicubicConsistency
torch.npu.matmul.allow_hf32 = False; torch.npu.conv.allow_hf32 = False
R = Path('/cache/zihaohe/streaming_sr0')
seq = 'bicycle-1'; k = 49
ev = json.loads((R / f'data/events/{seq}/event.json').read_text())
frames = range(17 * k, 17 * k + 22)
low = torch.stack([torch.from_numpy(np.array(Image.open(R / f'data/events/{seq}/lr/{f:06d}.png').convert('RGB')).copy()).permute(2, 0, 1) for f in frames]).float() / 255
rgb = torch.stack([torch.from_numpy(np.array(Image.open(R / f'results/step0_blocks/vae_only/{seq}/block{k:03d}/{f:06d}.png').convert('RGB')).copy()).permute(2, 0, 1) for f in frames]).float() / 255
H, W = rgb.shape[-2:]
t = time.perf_counter(); op64 = BicubicConsistency((H, W), (H // 4, W // 4)); x64, r64 = op64.project(rgb, low, 25); t64 = time.perf_counter() - t
op32 = BicubicConsistency((H, W), (H // 4, W // 4), dtype=torch.float32, device='npu:0')
torch.npu.synchronize(); t = time.perf_counter()
x32, r32 = op32.project(rgb.npu(), low.npu(), 25); torch.npu.synchronize(); t32 = time.perf_counter() - t
d = (x32.cpu().double() - x64.double())
print(json.dumps({'shape': list(rgb.shape), 'cpu64_seconds': t64, 'npu32_seconds': t32, 'max_abs_diff': float(d.abs().max()),
                  'rms_diff': float(d.square().mean().sqrt()), 'u8_max_diff': float((d.abs() * 255).max()),
                  'lr_rmse_64': float(r64.max()), 'lr_rmse_32': float(r32.max())}, indent=1))
