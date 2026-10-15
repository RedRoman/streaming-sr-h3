"""CPU-only GT-direction audit of the fixed B historical-band readout.

No decoder, encoder, optimization or inference result. A single evaluation-only
GT direction tests what the hard near-row filter removes. This never changes
the frozen B5 protocol and is not an optimized oracle upper bound.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'h3_factorized'))
from pixel_dc import Projection
from revision_utils import Readout, batch, FP


def spatial(read, error, inputs, qi):
    """Same count/weights/far-null as H, with only its near row projector removed."""
    step=int(inputs['query_steps'][0, qi]);r=read.null(error)
    total=torch.zeros_like(error);count=torch.zeros_like(error[:, :1])
    for j, fp in enumerate(inputs['packet_fp'][0].tolist()):
        weight=inputs['weights'][:, step, j]
        if not bool(weight.any()):
            continue
        top,left=FP[fp];mask=torch.zeros_like(count);mask[...,top:top+128,left:left+128]=1
        weight=F.interpolate(weight, size=(256,256), mode='bilinear', align_corners=False)*mask
        total+=weight*r;count+=weight
    return read.null(total/count.clamp_min(.001)),count


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--phi');args=p.parse_args()
    began=time.monotonic();torch.set_num_threads(4)
    # Avoid constructing the GPU-default Readout or any actual VAE.
    read=object.__new__(Readout);read.vae=None;read.far=Projection(256,4,device='cpu');read.near=Projection(128,2,device='cpu');read.pf=read.far.right@read.far.matrix;read.pn=read.near.right@read.near.matrix
    cache=Path(args.cache);manifest=json.loads((cache/'manifest.json').read_text());sources=[s for s in manifest['items'] if s['split']=='val'];assert len(sources)==8
    rows=[];packet_rows=[]
    with torch.no_grad():
        for source in sources:
            item=torch.load(cache/(source['id']+'.pt'),map_location='cpu',weights_only=False);inputs,targets=batch([item],'cpu');bases={'current_LR_VAE':inputs['query_base_rgb']}
            if args.phi:
                bases['fixed_native_pixel_photo_bridge']=torch.load(Path(args.phi)/(source['id']+'.pt'),map_location='cpu',weights_only=False)['bridge'][None]
            for base_name, allbase in bases.items():
                for qi,step in enumerate(inputs['query_steps'][0].tolist()):
                    base=allbase[:,qi];gt=targets['rgb'][:,qi];error=gt-base;nf=read.null(error);hard=read.historical_band(error,inputs,qi);soft,count=spatial(read,error,inputs,qi);union=(count>0)[0,0]
                    error_energy=0.;row_energy=0.;null_energy=0.
                    for j,fp in enumerate(inputs['packet_fp'][0].tolist()):
                        if not bool(inputs['weights'][:,step,j].any()):continue
                        top,left=FP[fp];crop=nf[...,top:top+128,left:left+128];near_row=read.pn@crop@read.pn.T;near_null=crop-near_row;ee=float(crop.square().sum());re=float(near_row.square().sum());ne=float(near_null.square().sum());error_energy+=ee;row_energy+=re;null_energy+=ne
                        packet_rows.append(dict(source=source['id'],query_step=step,base=base_name,packet=j,footprint=fp,error_energy=ee,near_row_energy=re,near_null_energy=ne,orthogonality_inner=float((near_row*near_null).sum())))
                    for method,correction in [('base',torch.zeros_like(base)),('GT_direction_hard_near_row',hard),('GT_direction_far_null_spatial_support',soft),('GT_direction_full_far_null',nf)]:
                        raw=base+correction;low=item['classical_inputs']['query_LR'][qi:qi+1].float();dc=read.far.apply(raw,low)
                        def metrics(x):
                            err=(x-gt).square()
                            return dict(full_PSNR_db=float(-10*err.mean().clamp_min(1e-12).log10()),union_PSNR_db=float(-10*err[:,:,union].mean().clamp_min(1e-12).log10()))
                        rows.append(dict(source=source['id'],split='val',base=base_name,query_step=step,method=method,raw=metrics(raw),common_DC=metrics(dc),filtered_direction_RMS=float(correction.square().mean().sqrt()),far_measurement_direction_RMS=float(read.far.observe(correction).square().mean().sqrt()),pooled_active_crop_near_null_error_energy_fraction=null_energy/max(error_energy,1e-30),pooled_active_crop_near_row_error_energy_fraction=row_energy/max(error_energy,1e-30),far_null_error_RMS=float(nf.square().mean().sqrt())))
            print(json.dumps(dict(event='historical_band_CPU_audit',source=source['id'],rows=len(rows))),flush=True)
    report=dict(status='complete',rows=rows,packet_rows=packet_rows,actual_D_calls=0,actual_E_calls=0,new_DiT_calls=0,optimizer_steps=0,GPU_used=False,seconds=time.monotonic()-began,source_count=8,scope='Evaluation-only one-pass GT correction directions; not an optimized upper bound, learned output, uncertainty calibration, or change to B5. Overlapping active crops are pooled for an operator energy diagnostic, not independent samples. H and S include weighted blending and outer far-null and are not both orthogonal projectors.',hard_filter='N_far [weighted average of embedded P_near(N_far error)_crop P_near^T] then N_far',spatial_filter='Same actual weights/count/support and far-null, removing only P_near.',camera_maxabs=dict(far=read.far.operator_maxabs,near=read.near.operator_maxabs))
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True);(dest/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(status='complete',seconds=report['seconds'],rows=len(rows))),flush=True)


if __name__=='__main__':main()
