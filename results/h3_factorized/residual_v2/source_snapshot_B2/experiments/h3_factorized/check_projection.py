"""Check physical LR operator, box projection and partial footprint protocol."""
import argparse,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from pixel_dc import Projection,project_visible


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--observations',required=True);p.add_argument('--out',required=True);args=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    x=torch.load(Path(args.cache)/'train_100075.pt',weights_only=False)['inputs']['observation_rgb'].float().cuda();m=torch.load(Path(args.observations)/'train_100075.pt',weights_only=False)['inputs'];far=Projection(256,4);near=Projection(128,2);rows=[]
    for stage in [1,2,4,5,6]:
        projected,rmse=project_visible(x[stage:stage+1],m,stage,far,near);assert torch.isfinite(projected).all() and projected.min()>=0 and projected.max()<=1
        if stage in [2,5]:
            mask=torch.ones_like(projected,dtype=torch.bool);mask[:,:,64:192,64:192]=False;assert torch.equal(projected[mask],x[stage:stage+1][mask]),'Partial measurement modified unseen outside'
        rows.append(dict(stage=stage,actual_measurement_rmse=rmse));assert rmse<.001,rmse
    rejected=False
    try:project_visible(x[3:4],m,3,far,near)
    except AssertionError:rejected=True
    assert rejected,'No-visible query accepted invented projection'
    result=dict(passed=True,far_operator_maxabs=far.operator_maxabs,near_operator_maxabs=near.operator_maxabs,steps=64,rows=rows,near_outside_unchanged=True,no_visible_query_rejected=True)
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
