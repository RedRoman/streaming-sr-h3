"""Apply the recorded dependency-metadata patch to the exact observed checkout."""
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).expanduser().resolve()
expected = "19b759a898647cc6de60a2093b2fc86ffebabe36"
actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if actual != expected:
    raise SystemExit(f"Refuse another diffusers revision: {actual}")
path = root / "src/diffusers/dependency_versions_table.py"
source = path.read_text()
old, new = '"peft": "peft>=0.17.0"', '"peft": "peft>=0.14.0"'
if source.count(new) == 1 and old not in source:
    print("Recorded peft metadata patch already applied")
elif source.count(old) == 1 and new not in source:
    path.write_text(source.replace(old, new))
    print("Applied the recorded peft metadata patch")
else:
    raise SystemExit("Refuse unexpected dependency metadata")
