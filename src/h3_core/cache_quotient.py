"""Rebuild a classical projected LR ledger from actual cached H3 directions.

No native decoder calls and no HR enter these sufficient statistics. A cross
statistic rebases all past likelihoods when independent coarse packets update
their mean. Nuisance bases use only the observed LR, never target appearance.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint,nuisance_project
from h3_data import NaturalCameraSequence
from observation_quotient import ObservationQuotient,SameFootprintCoarseMean


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--splits',nargs='+',default=['train','val']);p.add_argument('--limit',type=int);p.add_argument('--no-nuisance',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True);q=ObservationQuotient();start=time.time()
    manifest={'config':vars(args),'origin_cache':json.loads((Path(args.cache)/'manifest.json').read_text()),'protocol':'A0 observed quotient; same-footprint independent noise writes coarse mean, never additional fine Gram; exact copied IDs write nothing; LR-only appearance Schur; fixed train-only PCA32 span','items':[],'limits':['Pillow simulator vs Torch operator mismatch remains.','Conditional coarse likelihood correlations are not an exact joint posterior.','Classical projection/Schur/statistics are controls, not novelty.']}
    for split in args.splits:
        for path in sorted(Path(args.cache).glob(f'{split}_*.pt'))[:args.limit]:
            dest=out/path.name
            if dest.exists():raise FileExistsError(dest)
            item=torch.load(path,map_location='cpu',weights_only=False);responses=q.null(item['teacher']['decoder_responses'].cuda());base=q.null(item['teacher']['base_rgb'].cuda())
            source=item['source'];sequence=NaturalCameraSequence(source['path'],size=256,frames=39,protocol=source['protocol'],seed=source['sequence_seed'])
            coarse=SameFootprintCoarseMean(source['canonical_rectangle'],(1,3,64,64))
            gram=torch.zeros(32,32,device='cuda',dtype=torch.float64);natural=torch.zeros(32,device='cuda',dtype=torch.float64);cross=torch.zeros(32,3,64,64,device='cuda',dtype=torch.float64)
            grams=[];rhs=[];packet_grams=[];packet_rhs=[];coarse_means=[];packets=[];seen=set();variance=.005**2;far_rms=[]
            for packet in sequence.packets():
                if packet.observation_id in seen:continue
                seen.add(packet.observation_id);coarse.write(packet)
                kwargs=dict(canonical_rectangle=source['canonical_rectangle'],view_rectangle=packet.world_rectangle,view_size=(256,256),lr_size=(64,64))
                def op(value):return camera_footprint(value,**kwargs)
                images=op(responses);coverage=op(torch.ones_like(base[:,:1]))[0,0]
                valid=(coverage>.999)&(coverage<1.001)&(torch.as_tensor(packet.visible_mask.copy(),device='cuda')>0)
                mask=valid[None].expand(3,-1,-1);design=images[:,mask].T.double()
                observed=torch.as_tensor(packet.rgb.copy(),device='cuda').permute(2,0,1).float()/255
                residual=(observed-op(base)[0])[mask].double()
                if not args.no_nuisance:
                    bias=torch.eye(3,device='cuda')[:,:,None,None].expand(-1,-1,64,64)
                    nuisance=torch.cat([bias,bias*observed[None]],0)[:,mask].T.double()
                    design,residual=nuisance_project(design,residual,nuisance,damping=1e-8)
                increment_gram=design.T@design/variance;increment_natural=design.T@residual/variance
                gram+=increment_gram;natural+=increment_natural
                # Each batch row's scalar contraction is a camera adjoint. Then
                # (A0^+)^T maps it to the initial LR coarse coordinates.
                weighted=torch.zeros_like(images);weighted[:,mask]=design.T.float()
                dummy=torch.zeros_like(responses,requires_grad=True)
                adjoint=torch.autograd.grad((op(dummy)*weighted).sum(),dummy)[0].double()
                increment_cross=(q.right.T@adjoint@q.right)/variance;cross+=increment_cross
                current=coarse.mean();current_rhs=natural-torch.einsum('kchw,bchw->k',cross,current.double())
                packet_grams.append(increment_gram.float().cpu());packet_rhs.append((increment_natural-torch.einsum('kchw,bchw->k',increment_cross,current.double())).float().cpu())
                grams.append(gram.float().cpu());rhs.append(current_rhs.float().cpu());coarse_means.append(current[0].cpu())
                packets.append({'rgb':torch.from_numpy(packet.rgb.copy()),'visible_mask':torch.from_numpy(packet.visible_mask.copy()),'world_rectangle':packet.world_rectangle,'index':packet.index,'observation_id':packet.observation_id})
                if packet.index in [0,38]:far_rms.append(float(images.square().mean().sqrt()))
            inputs={**item['inputs'],'grams':torch.stack(grams),'rhs':torch.stack(rhs),'packet_grams':torch.stack(packet_grams),'packet_rhs':torch.stack(packet_rhs),'coarse':torch.stack(coarse_means),'packets':packets}
            packed={'inputs':inputs,'supervision':{'canonical_hr':item['supervision']['canonical_hr']},'source':source,'teacher':{'p0':item['teacher']['p0'],'base_rgb':base.cpu(),'decoder_responses':responses.cpu()},'protocol':{'nuisance':not args.no_nuisance,'no_raw_hr_latent_targets':True,'coarse_rebase':'h_raw - Hcross*ybar','effective_noise':.005}}
            torch.save(packed,dest)
            entry={'id':path.stem,'path':str(dest),'split':split,'source_hash':source['source_hash'],'first_gram_maxabs':float(grams[0].abs().max()),'final_coarse_packets':len(coarse.packet_ids),'far_direction_rms':far_rms,'seconds':time.time()-start}
            manifest['items'].append(entry);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'event':'quotient_cache',**entry}),flush=True)
    print(json.dumps({'event':'complete','items':len(manifest['items']),'seconds':time.time()-start}),flush=True)


if __name__=='__main__':main()
