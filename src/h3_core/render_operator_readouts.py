"""Render frozen seed101 RGB outputs for raw/DC qualitative inspection."""
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default='results/h3_core');a=p.parse_args();base=Path(a.base);out=base/'operator_function_matched_comparison';out.mkdir(exist_ok=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',15)
    except OSError:font=ImageFont.load_default()
    methods=[('GT',None),('carrier','source_carrier_u0'),('actual H3 family','dense_conditional_H3_pushforward_not_posterior_u0'),('joint CNP','joint_set_cnp_u0'),('mixed','mixed_u0'),('robust inverse','joint_robust_inverse_u0')]
    for sid in ['val_101085','val_101087']:
        panel=Image.new('RGB',(6*256,2*292),(245,245,245));draw=ImageDraw.Draw(panel)
        for ri,(name,title) in enumerate([('operator_function_v1','RAW'),('operator_function_dc_v1','inner DC')]):
            item=torch.load(base/name/'fixed_seed17'/f'{sid}_readouts.pt',map_location='cpu',weights_only=False);gt=item['target_evaluation_only'].float()
            for ci,(label,key) in enumerate(methods):
                x=gt if key is None else item['outputs'][key].float();array=(x[0].permute(1,2,0).clamp(0,1)*255).round().byte().numpy();panel.paste(Image.fromarray(array),(ci*256,ri*292+36));psnr='' if key is None else f" / {float(-10*torch.log10((x-gt).square().mean())):.3f}dB";draw.text((ci*256+5,ri*292+3),title+' '+label,fill=(10,10,10),font=font);draw.text((ci*256+5,ri*292+19),'seed101'+psnr,fill=(10,10,10),font=font)
        panel.save(out/f'{sid}_raw_DC_seed101.png')


if __name__=='__main__':main()
