"""Metadata-fixed event panels; no score-based selection."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F
from long_stream import ROOT, DATA, OUT, load

sys.path.insert(0, str(ROOT / 'src'))
from pixel_consistency import BicubicConsistency

SELECTED = {'book-20': [210, 230, 298, 315], 'hat-3': [149, 180, 242, 259],
            'book-19': [0, 200, 400, 599], 'hat-1': [144, 170, 227, 244]}


if __name__ == '__main__':
    torch.set_num_threads(4)
    manifest = json.loads((DATA / 'derived_h3_manifest.json').read_text())
    for video in manifest['videos']:
        sequence = video['sequence']; chosen = SELECTED[sequence]
        records = [video['frames_metadata'][i] for i in chosen]
        hr = torch.stack([load(DATA / f['hr']) for f in records])
        lr = torch.stack([load(DATA / f['lr_x4']) for f in records])
        output = torch.stack([load(OUT / sequence / f'{i:04d}.png') for i in chosen])
        bicubic = F.interpolate(lr, hr.shape[-2:], mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        bicubic = (bicubic * 255).round() / 255
        op = BicubicConsistency(tuple(hr.shape[-2:]), tuple(lr.shape[-2:]))
        with torch.inference_mode():
            base_dc, _ = op.project(bicubic, lr, 25)
            vae_dc, _ = op.project(output, lr, 25)
        methods = [('HR / evaluation only', hr), ('Bicubic + same DC25', base_dc), ('Frozen H3 VAE + same DC25', vae_dc)]
        height, width = hr.shape[-2:]; header = 38
        panel = Image.new('RGB', (3 * width, 4 * (height + header)), 'white')
        draw = ImageDraw.Draw(panel)
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
        except OSError:
            font = ImageFont.load_default(size=21)
        for row, frame in enumerate(chosen):
            for col, (name, values) in enumerate(methods):
                draw.text((col * width + 8, row * (height + header) + 7),
                          f'{name} | local t={frame}', fill='black', font=font)
                image = Image.fromarray(np.uint8(values[row].permute(1, 2, 0).numpy().clip(0, 1) * 255 + .5))
                panel.paste(image, (col * width, row * (height + header) + header))
        panel.save(OUT / sequence / 'event_panel.png')
        preview = panel.copy(); preview.thumbnail((1500, 1500))
        preview.save(OUT / sequence / 'event_panel_preview.jpg', quality=93)
        provenance = {'sequence': sequence, 'selected_local_frames': chosen,
                      'selected_source_frames': [r['source_frame0'] for r in records],
                      'selection': 'Annotation-declared disappearance/return, or fixed 0/200/400/599 scale sequence; no metric-based frame selection',
                      'scope': 'Codec baseline, no learned persistent state',
                      'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (OUT / sequence / 'event_panel.json').write_text(json.dumps(provenance, indent=2))
        print(json.dumps({'sequence': sequence, 'panel': 'complete'}), flush=True)
