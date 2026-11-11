"""Read-only source/snapshot admission; writes only this preparation's receipt."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

BASE = Path(__file__).resolve().parent
SNAPSHOT = BASE / "snapshot"
sys.path.insert(0, str(SNAPSHOT / "code"))
import materialize_mixed_temporal_pilot as m

EXPECTED = {
    "code/materialize_mixed_temporal_pilot.py": "80fa35dd84e941fb009b276cc704afb07b41088ca2b80ba72604afedc40d1c37",
    "code/tests/test_mixed_temporal_materializer.py": "00333283e676492fd18c45e1a516b8e447389ec63c0916f07a3ac7617b8ee35f",
    "preparation/mixed_temporal_materializer_20261111/execution_contract.json": "a601c5b599be572b140ad6dec96bf5f1caa0d4e5b7e1780c296a9bf4afe54c81",
    "configs/openh3_20261111/mixed_temporal_data_plan_20261111.json": "225723939832d2fa5b29fc6edcca26958e4acfb58e6a9c5886a7e5c94d741dfe",
}
for relative, sha in EXPECTED.items():
    if m.file_sha(SNAPSHOT / relative) != sha:
        raise ValueError("snapshot mismatch: " + relative)
plan_path = SNAPSHOT / "configs/openh3_20261111/mixed_temporal_data_plan_20261111.json"
contract_path = SNAPSHOT / "preparation/mixed_temporal_materializer_20261111/execution_contract.json"
plan, contract = json.loads(plan_path.read_text()), json.loads(contract_path.read_text())
m.validate_plan(plan, contract)
registry, binding = m.load_registry(SNAPSHOT / "results/takeover_20261111/materialized/train_materialized.jsonl", SNAPSHOT / "results/takeover_20261111/materialized/dev_materialized.jsonl", plan)
source = Path("/cache/zihaohe/streaming_sr0/data/real_av/_cand/2BdYaGuVAKo.mp4")
before = source.stat()
source_sha = m.file_sha(source)
after = source.stat()
if source_sha != binding["source_sha256"] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
    raise ValueError("fresh source bytes/identity mismatch")
probe = m.source_probe(source, "ffprobe")
output = Path("/cache/zihaohe/omni_openh3_20261111/runs/mixed_temporal_single_source_smoke_20261111a")
if output.exists():
    raise FileExistsError("no resume/overwrite permitted")
disk = shutil.disk_usage(output.parent)
if disk.free < 2 * 1024**3:
    raise OSError("less than the 2GiB preparation disk margin")
import av
codec = av.codec.Codec("libx264", "w")
libraries = Path(av.__file__).resolve().parent.parent / "av.libs"
binary_bindings = []
if libraries.is_dir():
    for library in sorted(libraries.iterdir()):
        if library.is_file() and ("x264" in library.name or "libavcodec" in library.name or "libavutil" in library.name or "libswscale" in library.name):
            binary_bindings.append({"path": str(library), "bytes": library.stat().st_size, "sha256": m.file_sha(library)})
receipt = {
    "schema": "openh3.mixed_temporal_smoke_preflight.v1", "status": "passed_not_launched",
    "snapshot_files_sha256": EXPECTED, "source": binding, "source_path": str(source),
    "source_bytes": before.st_size, "source_mtime_ns": before.st_mtime_ns,
    "fresh_source_sha256": source_sha, "source_unchanged_during_hash": True,
    "source_probe": probe, "target_specification": m.target_specification(binding, probe, contract),
    "registry": registry, "registry_sha256": m.value_sha(registry),
    "output": str(output), "output_exists": False, "disk_free_bytes": disk.free,
    "software": m.software_receipt("ffmpeg", "ffprobe"),
    "pyav_libx264_encoder": {"name": codec.name, "long_name": codec.long_name},
    "codec_build_binary_bindings": binary_bindings,
    "cpu_tests_final_log_sha256": m.file_sha(BASE / "cpu_tests_final.log"),
    "max_internal_seconds": 600, "max_outer_seconds": 630,
    "memory_budget_mib": 768, "threads": 1,
    "source_natural_cut_status": "unknown_engineering_only", "real_source_rgb_decodes_in_preflight": 0,
}
destination = BASE / "preflight.json"
if destination.exists():
    raise FileExistsError("preflight receipt already exists; retain it")
m.atomic_json(destination, receipt)
print(json.dumps({"status": receipt["status"], "source_sha256": source_sha,
                  "source_bytes": before.st_size, "duration": probe["format"]["duration"],
                  "size_wh": [probe["streams"][0]["width"], probe["streams"][0]["height"]],
                  "disk_free_bytes": disk.free, "preflight_sha256": m.file_sha(destination),
                  "codec_binary_files_hashed": len(binary_bindings), "output": str(output)}))
