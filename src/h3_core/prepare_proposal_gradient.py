"""Real VAE preparation for native22 proposal-memory gradient feasibility.

Uses one TRAIN source only. The inference payload contains first-LR-derived
H3 reference, actual near LR innovation and geometry. Supervision is a separate
file, never a condition. Its target keeps each proposal's common measurement
null component and uses train HR only in the newly supportable physical band.
This is a five-step integration audit, not an effectiveness experiment.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from safetensors.torch import save_file

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from camera_operator import camera_footprint
from h3_interface import load_frozen_vae
from observation_quotient import ObservationQuotient


def main():
    p=argparse.ArgumentParser();p.add_argument('--item',required=True);p.add_argument('--proposal',required=True);p.add_argument('--comfy',required=True);p.add_argument('--vae',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(Path(args.comfy).resolve()))
    item=torch.load(args.item,map_location='cpu',weights_only=False);assert item['source']['split']=='train'
    reference=torch.load(Path(args.proposal)/'immutable_identity.pt',map_location='cpu',weights_only=False)['rgb'].cuda()
    packet=item['inputs']['packets'][17];near=packet['rgb'].cuda().permute(2,0,1)[None].float()/255;rect=item['source']['canonical_rectangle'];view=packet['world_rectangle'];q=ObservationQuotient()
    def camera_axis(lo,hi,vlo,vhi):
        u=(torch.arange(256,device='cuda',dtype=torch.float64)+.5)/256*(vhi-vlo)+vlo;coordinate=(u-lo)/(hi-lo)*256-.5;left=coordinate.floor().long();fraction=coordinate-left
        matrix=torch.zeros(256,256,device='cuda',dtype=torch.float64);rows=torch.arange(256,device='cuda')
        for index,value in [(left,1-fraction),(left+1,fraction)]:
            good=(index>=0)&(index<256);matrix[rows[good],index[good]]=value[good]
        forward=q.matrix@matrix;right=forward.T@torch.linalg.solve(forward@forward.T,torch.eye(64,device='cuda',dtype=torch.float64));return forward,right
    wy,ry=camera_axis(rect[1],rect[3],view[1],view[3]);wx,rx=camera_axis(rect[0],rect[2],view[0],view[2])
    def observe(x):return (wy@x.double()@wx.T).float()
    def lift(y):return (ry@y.double()@rx.T).float()
    def new_support(x):return q.null(lift(observe(q.null(x))))
    independent=camera_footprint(reference,rect,view);operator_error=float((observe(reference)-independent).abs().max());assert operator_error<2e-6
    # Remove instantaneous bias/gain from real LR innovation without using HR.
    predicted=observe(reference);innovation=near-predicted
    for channel in range(3):
        design=torch.stack([torch.ones(4096,device='cuda'),near[0,channel].flatten()],1).double();rhs=innovation[0,channel].flatten().double()
        beta=torch.linalg.solve(design.T@design+torch.eye(2,device='cuda',dtype=torch.float64)*1e-6,design.T@rhs);innovation[0,channel]-=(design@beta).reshape(64,64).float()
    current=q.bounded(reference+q.null(lift(innovation)),item['inputs']['coarse'][0:1].cuda())
    # Pixel-space target correction is in the common classical new-evidence
    # support. The jointly unobserved component remains this seed's own X0.
    hr=item['supervision']['canonical_hr'].cuda().permute(2,0,1)[None].float()/255
    raw_target=reference+new_support(hr-reference)
    reference_lock_error=float((q.null(reference-raw_target)-new_support(hr-reference)*-1).abs().max())
    # A supported affine correction can exceed RGB bounds. Use the SAME public
    # far/box DC as all controls before nativeE, record its extra identity change.
    # Do not silently pass out-of-contract pixels to the real foundation VAE.
    target_rgb=q.bounded(raw_target,item['inputs']['coarse'][0:1].cuda())
    assert float(target_rgb.min())>=0 and float(target_rgb.max())<=1
    vae,info=load_frozen_vae(args.vae)
    def encode(rgb):return vae.encode((2*rgb-1)[:,:,None].repeat(1,1,22,1,1).half().clone()).float().cpu().contiguous()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():ref_latent=encode(reference);current_latent=encode(current);target_latent=encode(target_rgb)
    features=F.adaptive_avg_pool2d(innovation,(8,8)).cpu().contiguous();geometry=torch.tensor([[(view[2]-view[0])/(rect[2]-rect[0]),(view[3]-view[1])/(rect[3]-rect[1]),.005,17/38]],dtype=torch.float32)
    save_file({'reference':ref_latent,'current':current_latent,'innovation':features,'geometry':geometry},str(out/'conditioning.safetensors'))
    save_file({'target':target_latent},str(out/'supervision_train_only.safetensors'))
    torch.save({'reference_rgb':reference.cpu(),'current_rgb':current.cpu(),'target_rgb_train_only':target_rgb.cpu()},out/'rgb_audit_train_only.pt')
    result={'status':'complete','scope':'Native22 realH3reference+realnearLR condition/physicalsupporttarget gradient-feasibility only; not quality evidence','source':item['source'],'reference_seed':101,'near_packet_id':packet['observation_id'],'near_index':17,'native_input_context':'22 copies of already-observed canonical memory; distinct cue observations1; no future packet','condition_shape':list(ref_latent.shape),'condition_tokens':448,'operator_maxabs':operator_error,'target_far_measurement_change_rmse':float(q.observe(target_rgb-reference).square().mean().sqrt()),'reference_lock_check_maxabs_before_box':reference_lock_error,'raw_target_range':[float(raw_target.min()),float(raw_target.max())],'native_encoder_target_range':[float(target_rgb.min()),float(target_rgb.max())],'common_box_extra_change_rms':float((target_rgb-raw_target).square().mean().sqrt()),'physical_new_support':'Nfar Pnear Nfar then common64-step far/boxDC; common classical constraints, not a novel projector or exact latent nullspace','target_caveat':'Identity preservation is exact for the raw supported correction, then boxDC can change it (measured); nativeE/D can further drift and must be audited. No inference HR input.','vae':info,'config':vars(args)}
    (out/'preparation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
