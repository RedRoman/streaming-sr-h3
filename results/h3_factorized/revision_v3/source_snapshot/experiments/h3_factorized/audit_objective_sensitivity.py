"""Read-only fixed saved overfit endpoints: measured delta dependence."""
import argparse,csv,json,sys,time
from pathlib import Path
import torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_model import build
from revision_utils import batch,Readout
from native_vae import load_vae,decode_last


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--models',required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);p.add_argument('--out',required=True);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;out=Path(args.out);out.mkdir(parents=True,exist_ok=True);cache=Path(args.cache);manifest=json.loads((cache/'manifest.json').read_text());src=[x for x in manifest['items'] if x['split']=='train'];item=torch.load(cache/(src[0]['id']+'.pt'),map_location='cpu',weights_only=False);other=torch.load(cache/(src[1]['id']+'.pt'),map_location='cpu',weights_only=False);inputs,target=batch([item],'cuda');inputs['weights'][:,:,1:]=0;q=target['query_q'][:,0];weight=inputs['weights'][:,3,0].unsqueeze(2);vae=load_vae(args.vae,args.comfy);read=Readout(vae);far=item['classical_inputs']['query_LR'][0:1].cuda().float();mask=torch.zeros_like(target['rgb'][:,:1,0]);mask=torch.zeros((1,1,256,256),device='cuda',dtype=torch.bool);mask[...,32:160,32:160]=True;rows=[];gallery=[('HR evaluation only',target['rgb'][:,0])];start=time.time()
    with torch.no_grad():
        teacher=decode_last(vae,item['classical_inputs']['packet_full_latents'][0][None].cuda().float());teacher=inputs['query_base_rgb'][:,0]+read.historical_band(teacher-inputs['query_base_rgb'][:,0],inputs,0);gallery.extend([('Current LR-VAE',read.far.apply(inputs['query_base_rgb'][:,0],far)),('Actual24 nativeD teacher',read.far.apply(teacher,far))])
        for arm in ['latent_L1_only','actualD_RGB_only']:
            for step in [0,50,150,300]:
                model=build('closed_action').cuda().eval();model.load_state_dict(torch.load(Path(args.models)/arm/f'step{step}.pt',map_location='cpu',weights_only=False)['model'])
                for name,delta in [('actual',inputs['packet_delta'][:,0]),('zero_delta',inputs['packet_delta'][:,0]*0),('shuffled_TRAIN_delta',other['inputs']['packet_delta'][0][None].cuda().float())]:
                    h=model.lift(delta,inputs['packet_proxy'][:,0],inputs['packet_masks'][:,0]);styled=model.core.style(model.core.canonical(h,q),q)*(weight>0);z=model.read(styled,inputs['query_latents'][:,0],q,(weight/8).clamp(0,1));rgb=read.rgb(inputs,z,0)
                    for family,x in [('raw',rgb.clamp(0,1)),('DC',read.far.apply(rgb,far))]:
                        hr=target['rgb'][:,0];err=(x-hr).square();td=teacher if family=='raw' else read.far.apply(teacher,far);terr=(x-td).square();rows.append(dict(arm=arm,step=step,input=name,family=family,HR_single_near_PSNR=float(-10*torch.log10(err[mask.expand_as(err)].mean())),HR_full_PSNR=float(-10*torch.log10(err.mean())),teacher_single_near_PSNR=float(-10*torch.log10(terr[mask.expand_as(terr)].mean()))))
                        if family=='DC' and step==300:gallery.append((arm+' '+name,x))
                del model
        with (out/'sensitivity.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        canvas=Image.new('RGB',(3*256,3*280),'white');draw=ImageDraw.Draw(canvas)
        for j,(name,x) in enumerate(gallery):canvas.paste(Image.fromarray((x[0].cpu().permute(1,2,0).numpy().clip(0,1)*255).round().astype('uint8')),((j%3)*256,(j//3)*280));draw.text(((j%3)*256+3,(j//3)*280+258),name,fill='black')
        canvas.save(out/'fixed300_input_dependence_DC.png');(out/'complete.json').write_text(json.dumps(dict(seconds=time.time()-start,source=src[0]['id'],shuffled_source=src[1]['id'],scope='Read-only of previously saved fixed0/50/150/300; no optimization/no test. Delta0/shuffle retain original proxy+anchor, isolate conditional measured-detail dependence, not generalization.'),indent=2))


if __name__=='__main__':main()
