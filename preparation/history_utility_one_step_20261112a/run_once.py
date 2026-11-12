"""Fixed CPU one-step supervisor. Default inspect; execute is a separate call.

Inspection and execution share at most300 active worker seconds. Every worker
has2CPU affinity,4GiB owned-group RSS and10s termination grace. No retries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    p=Path(path);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)

def group_rows(pgid):
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])!=pgid:continue
            used=next((int(line.split()[1])*1024 for line in (p/'status').read_text().splitlines() if line.startswith('VmRSS:')),0)
            rows.append({'pid':int(p.name),'rss_bytes':used})
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return rows

def stop_owned(process,grace=10):
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    until=time.monotonic()+grace
    while time.monotonic()<until:
        process.poll()
        if not group_rows(process.pid):return
        time.sleep(.1)
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait(timeout=5)

def phase_budget(execute,inspection_status=None):
    if not execute:return 90.,0.
    require(inspection_status is not None and inspection_status['complete'] is True
            and inspection_status['phase']=='inspect' and inspection_status['returncode']==0,'successful bounded default inspection required')
    used=inspection_status['active_worker_seconds']
    require(type(used) in (int,float) and 0<=used<300,'invalid/already exhausted total CPU wall budget')
    return 300.-used,float(used)

def validate(path,digest):
    require(sha(path)==digest,'freeze SHA changed');c=read(path)
    require(c['schema']=='rphi.history_utility_one_step_cpu_supervisor.v1','wrong supervisor schema')
    require(c['resource']=={'CPU_threads':2,'wall_seconds':300,'termination_grace_seconds':10,'owned_process_group_RSS_limit_bytes':4*(1<<30)},'resource budget changed')
    require(c['inspection_max_seconds']==90 and c['combined_active_worker_budget'] is True,'inspection/execute shared budget changed')
    require(c['pins'] and all(sha(p)==h for p,h in c['pins'].items()),'runtime pin changed')
    require(c['pins'].get(str(Path(__file__).resolve()))==sha(__file__),'supervisor not pinned')
    for name in ('python','entry','execution_contract'):require(c[name] in c['pins'],'unbound executable/contract: '+name)
    root=Path(c['root']).resolve();outputs=[Path(c[k]) for k in ('out','inspection_out','supervisor_out')]
    require(all(p.is_absolute() and p.parent==root/'runs' for p in outputs) and len(set(outputs))==3,'output outside distinct owned run namespace')
    execution=read(c['execution_contract'])
    require(execution['resource']==c['resource'] and execution['schema']=='rphi.history_utility_one_step_execution.v1','execution budget differs')
    return c

def run(config,freeze_sha,execute=False):
    phase='execute' if execute else 'inspect';base=Path(config['supervisor_out']);out=Path(config['out'] if execute else config['inspection_out'])
    require(not out.exists(),'producer output already exists; no retry/overwrite')
    inspection=None
    if execute:
        inspection=read(base/'inspect/status.json')
        require(inspection['freeze_sha256']==freeze_sha,'default inspection from another freeze')
        report=Path(config['inspection_out'])/'report.json'
        require(sha(report)==inspection['report_sha256'],'inspection report changed')
        data=read(report)
        require(data['complete'] is True and data['status']=='inspected_actual_pixels_no_model'
                and data['model_calls']==0 and data['execution_contract_sha256']==sha(config['execution_contract']),'wrong default inspection')
    limit,prior=phase_budget(execute,inspection)
    status_dir=base/phase;status_dir.mkdir(parents=True,exist_ok=False)
    memory=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    require(memory>=6<<30,'less than6GiB available host memory')
    affinity=sorted(os.sched_getaffinity(0));require(len(affinity)>=2,'twoCPU affinity unavailable');affinity=affinity[:2];os.sched_setaffinity(0,set(affinity))
    env=os.environ.copy();env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    command=[config['python'],config['entry'],'--execution-contract',config['execution_contract'],'--out',str(out)]
    if execute:command.append('--execute')
    status={'schema':config['schema'],'complete':False,'state':'starting','phase':phase,'freeze_sha256':freeze_sha,
        'command':command,'cpu_affinity':affinity,'phase_worker_seconds_cap':limit,'prior_active_worker_seconds':prior,'total_worker_seconds_cap':300,
        'termination_grace_seconds':10,'owned_process_group_RSS_limit_bytes':4<<30,'peak_owned_group_RSS_bytes':0,
        'available_host_memory_before_bytes':memory,'NPU_calls':0,'scope':'owned CPU process only; no wait/retry/other-job changes'}
    write(status_dir/'status.json',status);process=None;started=None
    try:
        with (status_dir/'worker.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=config['root'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.monotonic();status.update(state='running',worker_pid=process.pid,owned_pgid=process.pid);write(status_dir/'status.json',status)
        while process.poll() is None:
            usage=sum(row['rss_bytes'] for row in group_rows(process.pid));status['peak_owned_group_RSS_bytes']=max(status['peak_owned_group_RSS_bytes'],usage)
            if usage>4<<30:raise MemoryError('owned group RSS exceeded4GiB')
            if time.monotonic()-started>=limit:raise TimeoutError('combined inspect+execute active worker budget exhausted')
            time.sleep(.1)
        require(time.monotonic()-started<=limit,'worker completed after active wall budget')
        require(not group_rows(process.pid),'worker left owned descendants')
        report=out/'report.json';require(process.returncode==0 and report.is_file(),'worker failed/no success report')
        evidence=read(report);expected='completed_one_step_engineering' if execute else 'inspected_actual_pixels_no_model'
        require(evidence['complete'] is True and evidence['status']==expected,'wrong worker completion scope')
        if execute:require(evidence['actual']==read(config['execution_contract'])['counts'],'actual scientific operation counts differ')
        status.update(state='completed',complete=True,report_sha256=sha(report),returncode=process.returncode)
    except BaseException as error:
        if process is not None:stop_owned(process,10)
        status.update(state='failed',complete=False,error={'type':type(error).__name__,'message':str(error)},returncode=process.poll() if process else None)
    finally:
        status['active_worker_seconds']=time.monotonic()-started if started is not None else 0.
        status['combined_active_worker_seconds']=prior+status['active_worker_seconds']
        try:
            status['pins_after']={p:sha(p) for p in config['pins']};status['pins_unchanged']=status['pins_after']==config['pins']
            if not status['pins_unchanged']:status.update(state='failed',complete=False,error={'type':'PinChanged','message':'source/contract changed'})
        except BaseException as error:status.update(state='failed',complete=False,error={'type':type(error).__name__,'message':str(error)})
        write(status_dir/'status.json',status)
    return status

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args();c=validate(args.freeze,args.freeze_sha256);status=run(c,args.freeze_sha256,args.execute)
    print(json.dumps({k:status.get(k) for k in ('complete','phase','state','active_worker_seconds','combined_active_worker_seconds','peak_owned_group_RSS_bytes','error')}),flush=True)
    return 0 if status['complete'] else 2
if __name__=='__main__':raise SystemExit(main())
