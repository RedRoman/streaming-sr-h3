"""Foreground SSH stream of a 0.77GB exact block subset, never full local model."""
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    started=time.time()
    source=['ssh','-S','/tmp/codex-qfvdn-native.sock','-o','BatchMode=yes','-p','28610','root@connect.bjb2.seetacloud.com']
    path='/root/autodl-tmp/streaming_sr_20261015/external/h3_assets/h3_native_block0.safetensors'
    metadata=json.loads(subprocess.check_output(source+['cat '+path.removesuffix('.safetensors')+'.json'],text=True))
    target='/home/saiadmin/streaming_sr_20261015/external/h3_assets/h3_native_block0.safetensors'
    code='''import sys,json,hashlib,time,os
from pathlib import Path
p=Path(TARGET);partial=p.with_suffix(p.suffix+'.partial');h=hashlib.sha256();total=0;began=time.time();mark=0
with partial.open('wb') as f:
 while block:=sys.stdin.buffer.read(8*1024**2):
  f.write(block);h.update(block);total+=len(block)
  if total-mark>=64*1024**2:
   print(json.dumps({'event':'block_stream_progress','pid':os.getpid(),'bytes':total,'seconds':time.time()-began}),flush=True);mark=total
assert total==EXPECTED_BYTES and h.hexdigest()==EXPECTED_SHA,(total,h.hexdigest())
partial.replace(p)
result={'status':'complete','bytes':total,'sha256':h.hexdigest(),'path':str(p),'pid':os.getpid(),'seconds':time.time()-began}
p.with_suffix('.transfer.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result),flush=True)
'''.replace('TARGET',repr(target)).replace('EXPECTED_BYTES',str(metadata['bytes'])).replace('EXPECTED_SHA',repr(metadata['sha256']))
    src=subprocess.Popen(source+['cat '+path],stdout=subprocess.PIPE)
    dst=subprocess.Popen(['ssh','-o','BatchMode=yes','sai-gpu161','/home/saiadmin/miniconda3/bin/python -u -c '+shlex.quote(code)],stdin=src.stdout,stdout=subprocess.PIPE,text=True)
    src.stdout.close()
    last=None
    for line in dst.stdout:
        print(line,flush=True,end='')
        try:last=json.loads(line)
        except ValueError:pass
    dst_code=dst.wait();src_code=src.wait()
    record={'source':metadata,'source_command':source+['cat '+path],'target':target,'source_exit':src_code,'target_exit':dst_code,'seconds':time.time()-started,'last_output':last}
    Path('results/h3_setup/block0_transfer.json').write_text(json.dumps(record,indent=2)+'\n')
    assert src_code==dst_code==0


if __name__=='__main__':main()
