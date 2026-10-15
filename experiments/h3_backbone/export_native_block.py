"""Export exact existing native BF16 block0 for limited gradient feasibility."""
import argparse
import hashlib
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if not a.output.exists():
        with safe_open(a.model,framework='pt',device='cpu') as f:
            state={k.removeprefix('blocks.0.'):f.get_tensor(k) for k in f.keys() if k.startswith('blocks.0.')}
            state['adaln_t_table']=f.get_tensor('adaln_t_table')
        save_file(state,str(a.output),metadata={'scope':'Only native pretrained block0+time curve for gradient feasibility; not a complete H3 model','source_sha256':'a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4'})
    h=hashlib.sha256()
    with a.output.open('rb') as f:
        while chunk:=f.read(16*1024**2):h.update(chunk)
    result={'path':str(a.output),'bytes':a.output.stat().st_size,'sha256':h.hexdigest(),'scope':'Exact native block0 subset, no method/whole-DiT claim'}
    a.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
