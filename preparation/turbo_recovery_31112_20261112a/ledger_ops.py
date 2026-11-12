"""Durable transfer accounting; no networking, no implicit retries."""
import datetime,hashlib,json,pathlib,sys
BASE=pathlib.Path(__file__).resolve().parent
LEDGER=BASE/'recovery_jobs.json'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(rows):
 p=LEDGER.with_suffix('.tmp');p.write_text(json.dumps(rows,indent=2)+'\n');p.replace(LEDGER)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def summary(rows):
 ready=[]
 for part in sorted({r['part'] for r in rows}):
  rr=[x for x in rows if x['part']==part]
  if all(x['download_verified'] for x in rr):ready.append(part)
 return {'utc':now(),'active':[{'session':x['download_session'],'name':x['name']} for x in rows if x['download_started'] and not x['download_verified'] and x.get('state') not in ('FAILED','HASH_FAILED')], 'complete_segments':sum(x['download_verified'] for x in rows),'pending_segments':sum(not x['download_started'] for x in rows),'failed_segments':[x['download_session'] for x in rows if x.get('state') in ('FAILED','HASH_FAILED')],'verified_bytes':sum(x['bytes'] for x in rows if x['download_verified']),'downloaded_bytes':sum(pathlib.Path(x['local_path']).stat().st_size for x in rows if pathlib.Path(x['local_path']).exists()),'ready_parts':ready}
def update(statuses):
 rows=json.loads(LEDGER.read_text());mapping={x['download_session']:x for x in rows}
 for observed in statuses:
  r=mapping[observed['session']];s=observed['text'];r.update(last_status_text=s,last_checked_utc=now());p=pathlib.Path(r['local_path']);r['local_bytes']=p.stat().st_size if p.exists() else 0
  if 'RUNNING' in s:r['state']='RUNNING'
  elif 'FAILED' in s:r['state']='FAILED';r['partial_preserved']=True
  elif 'exit 0' in s or any(x in s for x in (' DONE',' COMPLETED',' FINISHED',' SUCCESS')):
   r['state']='TRANSFER_TERMINAL'
   if not r['download_verified']:
    before=p.stat() if p.exists() else None
    if before is None or before.st_size!=r['bytes']:r['state']='HASH_FAILED';r['verification_error']='wrong size or missing'
    else:
     actual=sha(p);after=p.stat();r['actual_sha256']=actual
     if actual!=r['sha256'] or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):r['state']='HASH_FAILED';r['verification_error']='SHA or stability mismatch'
     else:r.update(download_verified=True,state='VERIFIED',verified_utc=now())
  else:r['state']='UNKNOWN';r['requires_status_inspection']=True
 save(rows);return summary(rows)
def claim():
 rows=json.loads(LEDGER.read_text());s=summary(rows)
 if s['failed_segments']:return {'claim':None,'reason':'failure requires review','summary':s}
 if len(s['active'])>=8:return {'claim':None,'reason':'at max 8','summary':s}
 todo=next((x for x in rows if not x['download_started']),None)
 if todo is None:return {'claim':None,'reason':'no pending','summary':s}
 assert not pathlib.Path(todo['local_path']).exists(),'refusing overwrite'
 todo.update(download_started=True,state='DISPATCH_REQUESTED',dispatch_requested_utc=now());save(rows);return {'claim':todo}
def started(session,result):
 rows=json.loads(LEDGER.read_text());r=next(x for x in rows if x['download_session']==session);r['download_start_result']=result;r['started_utc']=now();r['state']='FAILED' if result.get('isError') else 'RUNNING';save(rows);return {'session':session,'state':r['state']}
if __name__=='__main__':
 action=sys.argv[1]
 if action=='summary':result=summary(json.loads(LEDGER.read_text()))
 elif action=='claim':result=claim()
 else:raise SystemExit('use import for update/start')
 print(json.dumps(result,separators=(',',':')))
