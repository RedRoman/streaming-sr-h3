"""Capture live process evidence and resumable-byte counts; no job mutations."""

import datetime
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = "/home/saiadmin/streaming_sr_20261015"
COMMANDS = {
    "vae": "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ExitOnForwardFailure=yes -R 127.0.0.1:18799:127.0.0.1:7897 sai-gpu161 '/home/saiadmin/miniconda3/bin/python /home/saiadmin/streaming_sr_20261015/experiments/h3_backbone/download_pinned_ranges.py --asset vae --directory /home/saiadmin/streaming_sr_20261015/external/h3_assets --status /home/saiadmin/streaming_sr_20261015/results/h3_backbone/vae_download.json --proxy http://127.0.0.1:18799 --workers 8'",
    "dit": "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ExitOnForwardFailure=yes -R 127.0.0.1:18794:127.0.0.1:7897 sai-gpu161 '/home/saiadmin/miniconda3/bin/python /home/saiadmin/streaming_sr_20261015/experiments/h3_backbone/download_pinned_ranges.py --asset dit --directory /home/saiadmin/streaming_sr_20261015/external/h3_assets --status /home/saiadmin/streaming_sr_20261015/results/h3_backbone/dit_download.json --proxy http://127.0.0.1:18794 --workers 16'",
}
SESSIONS = {"vae": 84910, "dit": 81647}
REMOTE_CODE = r'''
import json,time,subprocess
from pathlib import Path
root=Path("/home/saiadmin/streaming_sr_20261015")
statuses={}
for name in ["vae","dit"]:
 path=root/"results/h3_backbone"/(name+"_download.json")
 if path.exists(): statuses[name]=json.loads(path.read_text())
processes=[]
for line in subprocess.check_output(["ps","-eo","pid=,ppid=,etime=,stat=,args="],text=True).splitlines():
 row=line.strip().split(None,4)
 if len(row)!=5: continue
 command=row[4]; first=command.split()[0]
 if (first.endswith("python") and "experiments/h3_backbone/download_" in command) or (first=="curl" and ("h3_assets/" in command or "RelayVSR/resolve/main/wan.safetensors" in command)):
  processes.append({"pid":int(row[0]),"ppid":int(row[1]),"etime":row[2],"state":row[3],"command":command})
wan=root/"external/RelayVSR/weights/wan.safetensors"
files={}
for path in (root/"external/h3_assets").glob("*"):
 if path.is_file(): files[str(path)]={"logical_bytes":path.stat().st_size,"allocated_bytes":path.stat().st_blocks*512}
print(json.dumps({"epoch":time.time(),"statuses":statuses,"processes":processes,"files":files,"wan_partial_bytes":wan.stat().st_size if wan.exists() else 0}))
'''


def main():
    remote_command = "/home/saiadmin/miniconda3/bin/python - <<'REMOTE'\n" + REMOTE_CODE + "\nREMOTE"
    snapshot = json.loads(subprocess.check_output(["ssh", "-o", "BatchMode=yes", "sai-gpu161", remote_command], text=True))
    path = ROOT / "results/h3_setup/live_jobs.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous_snapshot = previous.get("snapshot")
    jobs = {}
    for name, session in SESSIONS.items():
        status = snapshot["statuses"].get(name, {})
        process = next((p for p in snapshot["processes"] if p["pid"] == status.get("pid")), None)
        jobs[name] = {"exec_session_id": session, "foreground_command": COMMANDS[name],
                      "live_process_confirmed": process is not None, "process": process,
                      "status": status, "canonical_verified": status.get("status") == "complete"}
        if previous_snapshot and name in previous_snapshot.get("statuses", {}):
            elapsed = snapshot["epoch"] - previous_snapshot["epoch"]
            increase = status.get("complete_bytes", 0) - previous_snapshot["statuses"][name].get("complete_bytes", 0)
            jobs[name]["verified_range_growth_bytes_per_second"] = increase / elapsed if elapsed > 0 else None
            jobs[name]["growth_observation_seconds"] = elapsed
    record = {
        "checked_at_shanghai": datetime.datetime.fromtimestamp(snapshot["epoch"], datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "evidence_rule": "Live status requires matching ps PID now; sparse partial logical file sizes are not downloaded-byte counts. complete_bytes counts only exact completed 206 ranges; final artifact requires full canonical SHA256.",
        "jobs": jobs, "snapshot": snapshot,
        "terminated_jobs": [
            {"exec_session_id": 68086, "status": "stopped_owned_4worker_dit_then_resumed16", "remote_python_pid": 1119123, "exit_code": 255, "verified_preserved_bytes": 3623878656, "reason": "VAE verified complete; controlled SIGTERM of exact owned Python; process absent and reverse port closed before session81647 resumed same ranges"},
            {"exec_session_id": 20008, "status": "stopped_owned_slow_donor_stream", "source_exit": 255, "target_exit": 255,
             "elapsed_seconds": 167.3803, "reason": "1–2MB/s donor SSH stream; replaced with resumable HF mirror ranges; donor tasks untouched"},
            {"exec_session_id": 66311, "status": "stopped_owned_slow_large_hf_ranges", "remote_python_pid": 1113758,
             "reason": "Official large-range throughput fell below small mirror ranges; all owned curl children were terminated; prior .hf-range parts retained"},
            {"exec_session_id": 1443, "status": "failed_not_restarted", "exit_code": 92,
             "observed_download_bytes": 7685209714, "expected_bytes": 10066399024,
             "elapsed_seconds": 1398.266412, "error": "HTTP/2 stream CANCEL", "priority": "RelayVSR comparison; H3 is current priority"},
        ],
        "resumption": "VAE session 84910 completed exit0; poll active DiT session 81647 first. A transient observation failure is not terminal. If a process is actually absent, rerun its exact command only after confirming old tunnel is gone; .mirror.ranges.json resumes completed 64MiB chunks. Never treat preallocated .mirror.partial as a complete model.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    subprocess.run(["ssh", "-o", "BatchMode=yes", "sai-gpu161", "mkdir -p " + REMOTE_ROOT + "/results/h3_setup"], check=True)
    subprocess.run(["scp", str(path), "sai-gpu161:" + REMOTE_ROOT + "/results/h3_setup/live_jobs.json"], check=True)
    print(json.dumps({"checked_at": record["checked_at_shanghai"], "jobs": jobs,
                      "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
