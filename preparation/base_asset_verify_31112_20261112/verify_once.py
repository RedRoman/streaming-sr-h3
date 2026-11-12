import pathlib,hashlib,json,datetime,time,os
root=pathlib.Path('/cache/zihaohe/omni_openh3_20261111'); out=root/'preparation/base_asset_verify_31112_20261112'; start=time.monotonic();p=pathlib.Path('/cache/zihaohe/streaming_sr0/assets/comfy_h3/minimax_h3_fl2va_pruned_bf16.safetensors');expected='a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4'
with (out/'started.json').open('x') as f:json.dump(dict(state='running',pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
try:
 before=p.stat();assert before.st_size==40225724176;h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 after=p.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino);assert h.hexdigest()==expected
 record=dict(state='completed',bytes=after.st_size,sha256=h.hexdigest(),path=str(p),model_loads=0,NPU_calls=0)
except Exception as e:record=dict(state='failed',error=repr(e),model_loads=0,NPU_calls=0)
record.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
with (out/'result.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
raise SystemExit(0 if record['state']=='completed' else 2)
