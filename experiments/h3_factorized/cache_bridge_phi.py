"""Actual frozen native-D fixed phi/base cache; training amortization explicit."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/h3_factorized'))
from revision_utils import batch,Readout
from revision_bridge import attach_native,shared_bridge
from native_vae import load_vae,decode_last,CANONICAL_SHA


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--split',choices=['train','val'],required=True);p.add_argument('--vae',required=True);p.add_argument('--comfy',required=True);a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());sources=[x for x in manifest['items'] if x['split']==a.split];assert len(sources)==(32 if a.split=='train' else 8);vae=load_vae(a.vae,a.comfy);read=Readout(vae);start=time.time();records=[]
    with torch.no_grad():
        for si,source in enumerate(sources):
            path=cache/(source['id']+'.pt');destination=out/(source['id']+'.pt')
            if destination.exists():records.append(torch.load(destination,map_location='cpu',weights_only=False)['metadata']);continue
            item=torch.load(path,map_location='cpu',weights_only=False);inputs,_=batch([item],'cuda');attach_native(inputs,item);torch.cuda.synchronize();t0=time.perf_counter();decoded=[decode_last(vae,z[None]) for z in inputs['packet_native'][0]];torch.cuda.synchronize();dt=time.perf_counter()-t0;t1=time.perf_counter();bridge=shared_bridge(read,inputs,item['classical_inputs']['near_LR'].cuda().float(),decoded);torch.cuda.synchronize();fusion=time.perf_counter()-t1
            # Exact route parity is checked independently before training.
            info=dict(source=source['id'],split=a.split,source_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),native_D_calls=8,native_E_calls=0,native_D_seconds=dt,all7_bridge_fusion_seconds=fusion,decoded_bytes=sum(x.numel()*x.element_size() for x in decoded),bridge_bytes=bridge.numel()*bridge.element_size(),VAE_sha256=CANONICAL_SHA,TF32=False,precision='FP16 nativeD; savedphi/base FP32; camera FP32',scope='Common fixed observed phi, external causal LR controller only; precomputed training scratch, not deployment memory compression')
            torch.save(dict(decoded=torch.cat(decoded).cpu(),bridge=bridge[0].cpu(),metadata=info),destination);records.append(info);live=dict(status='B5_fixed_phi_actualD_cache',pid=os.getpid(),split=a.split,step=si+1,steps=len(sources),foreground=True,seconds=time.time()-start,unix_time=time.time());(out/'LIVE_STATE.json').write_text(json.dumps(live,indent=2));print(json.dumps(live),flush=True)
    (out/f'{a.split}_complete.json').write_text(json.dumps(dict(records=records,actual_D_calls=len(sources)*8,actual_E_calls=0,seconds=time.time()-start,scope='One common cached phi/base per source; all arm costs charge full native/proxy state and actualD phi construction'),indent=2))


if __name__=='__main__':main()
