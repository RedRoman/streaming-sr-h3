from pathlib import Path
import socket,json,time,datetime,os
out=Path(__file__).parent
r={"schema":"omni.private_tcp_probe.v1","bind":["172.16.0.178",18443],"allowed_client_ip":"172.16.1.52","started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"timeout_seconds":45,"file_serving":False,"pid":os.getpid()}
s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
try:
 s.bind(tuple(r["bind"]));s.listen(1);s.settimeout(45)
 with (out/"tcp_probe_v3_ready.json").open("x") as f:json.dump(r,f,indent=2)
 t=time.monotonic();c,peer=s.accept()
 with c:
  r.update(peer=list(peer),allowed=peer[0]==r["allowed_client_ip"])
  if r["allowed"]:c.sendall(b"OMNI_PRIVATE_PROBE_V1\n")
except Exception as e:r.update(connected=False,error_type=type(e).__name__,error=str(e))
finally:
 s.close();r["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (out/"tcp_probe_v3_server.json").open("x") as f:json.dump(r,f,indent=2)
