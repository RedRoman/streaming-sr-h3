"""One fresh admission attempt on fixed card 6. No idle-wait loop or retry."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["cards"] != [6] or len(plan["steps"]) != 1:
        raise ValueError("fixed one-card, one-step probe only")
    root = Path(plan["root"])
    sys.path.insert(0, str(root / "code"))
    import takeover_queue as queue
    status = root / plan["status"]
    status.parent.mkdir(parents=True, exist_ok=True)
    if status.exists():
        raise FileExistsError(status)
    with status.with_suffix(".lock").open("a") as status_lock:
        fcntl.flock(status_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first = queue.inspect()
        if 6 not in first or not queue.available(first[6], plan["min_free_mb"]):
            queue.save(status, state="not_launched", reason="card 6 not available; no wait loop", devices=first)
            return 2
        lock_dir = Path(plan["lock_dir"])
        lock_dir.mkdir(parents=True, exist_ok=True)
        with (lock_dir / "card_6.lock").open("a") as card_lock:
            fcntl.flock(card_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            time.sleep(15)
            second = queue.inspect()
            if 6 not in second or not queue.available(second[6], plan["min_free_mb"]):
                queue.save(status, state="not_launched", reason="second check failed; no retry", devices=second)
                return 2
            return queue.run_plan(plan, status, 6)


if __name__ == "__main__":
    raise SystemExit(main())
