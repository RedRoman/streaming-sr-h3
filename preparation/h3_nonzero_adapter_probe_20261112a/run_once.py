"""One fixed-card6 nonzero adapter probe, with exact admission and two idle checks."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["cards"] != [6] or plan["min_free_mb"] != 60000 or len(plan["steps"]) != 1:
        raise ValueError("fixed card6 / 60000MB / one nonzero adapter step required")
    if plan["steps"][0]["argv"][:5] != ["timeout", "--signal=TERM", "--kill-after=30s", "3600s", "venv/bin/python"]:
        raise ValueError("fixed owned process-group timeout 3600s+30s required")
    root = Path(plan["root"])
    admission = json.loads((root / plan["fresh_admission"]).read_text())
    if (admission.get("schema") != "openh3.nonzero_adapter_admission.v1"
            or admission.get("approved_for_single_nonzero_adapter_probe") is not True
            or admission.get("physical_card") != 6
            or admission.get("queue_sha256") != hashlib.sha256(args.plan.read_bytes()).hexdigest()):
        raise ValueError("fresh root admission must bind this exact card6 nonzero adapter queue")
    sys.path.insert(0, str(root / "code"))
    import takeover_queue as queue
    status = root / plan["status"]
    status.parent.mkdir(parents=True, exist_ok=True)
    with status.with_suffix(".lock").open("a") as status_lock:
        fcntl.flock(status_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.exists():
            raise FileExistsError(status)
        first = queue.inspect()
        if 6 not in first or not queue.available(first[6], 60000):
            queue.save(status, state="not_launched", reason="card6 unavailable; no waiting loop", devices=first)
            return 2
        lock_dir = Path(plan["lock_dir"])
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / "card_6.lock").open("a") as card_lock:
            fcntl.flock(card_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            time.sleep(15)
            second = queue.inspect()
            if 6 not in second or not queue.available(second[6], 60000):
                queue.save(status, state="not_launched", reason="card6 second check failed; no retry", devices=second)
                return 2
            return queue.run_plan(plan, status, 6)


if __name__ == "__main__":
    raise SystemExit(main())
