"""Read a complete parent recipe while verifying only its used VAE/context.

The full recipe fingerprint is metadata identity, not evidence that absent DiT,
Refiner or Turbo files were inspected. Never pass this partial-verification
record to merge_pretrained: _files_sha256_verified deliberately remains False.
"""
import hashlib
import json
from pathlib import Path

from openh3_assets import NATIVE_SIGMAS, RECIPES, asset_path, sha256

USED_ROLES = ("vae", "context")
SCOPE = "only_used_components_checked; unused_components_metadata_only"


def read_vae_only_manifest(path, *, expected_manifest_sha256):
    path = Path(path).resolve()
    actual_manifest_sha = sha256(path)
    if actual_manifest_sha != expected_manifest_sha256:
        raise ValueError("parent pretrained manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("recipe") not in RECIPES:
        raise ValueError("unsupported parent asset manifest schema/recipe")
    if manifest.get("layout") != "comfyui_minimax_h3":
        raise ValueError("converted ComfyUI parent layout required")
    roles = [component["role"] for component in manifest["components"]]
    if roles != list(RECIPES[manifest["recipe"]]):
        raise ValueError("parent component order/roles disagree with recipe")
    records = {name:manifest[name] for name in ("base","vae","context")}
    records.update({component["role"]:component for component in manifest["components"]})
    if "sharp" in roles:
        if manifest.get("sharp_curve_policy") != "require_zero_B":
            raise ValueError("parent Sharp curve policy missing")
        records["sharp_curve"] = manifest["sharp_curve"]
    # Validate every declared identity without resolving, stat-ing or opening
    # unused asset paths. Path.resolve can itself touch a filesystem/symlink.
    for name, record in records.items():
        digest = record.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("missing/invalid parent asset SHA: " + name)
        if (type(record.get("bytes")) is not int or record["bytes"] < 1 or
                not isinstance(record.get("path"),str) or not record["path"]):
            raise ValueError("invalid parent asset size/path metadata: " + name)
    resolved, checked = {}, {}
    for name in USED_ROLES:
        record = records[name]
        used_path = asset_path(record,path).resolve()
        if not used_path.is_file() or used_path.stat().st_size != record["bytes"]:
            raise ValueError("used asset missing/size mismatch: " + name)
        actual_sha = sha256(used_path)
        if actual_sha != record["sha256"]:
            raise ValueError("used asset actual SHA mismatch: " + name)
        resolved[name] = str(used_path)
        checked[name] = {"path":str(used_path),"bytes":record["bytes"],"sha256":actual_sha,
                         "actual_file_checked":True}
    identity = {"recipe":manifest["recipe"],"layout":manifest["layout"],
        "assets":{name:record["sha256"] for name,record in records.items()},
        "merge":"sum_fp32_then_cast_once","sigma":list(NATIVE_SIGMAS)}
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return {**manifest,"_paths":resolved,"_fingerprint":fingerprint,
        "_manifest_sha256":actual_manifest_sha,"_files_sha256_verified":False,
        "_used_files_sha256_verified":True,
        "_verification_scope":{"scope":SCOPE,"checked":checked,
            "unchecked_roles":[name for name in records if name not in USED_ROLES],
            "unchecked_policy":"metadata only; not resolved, stat-ed, hashed or loaded",
            "parent_recipe_fingerprint_role":"metadata provenance only, not full recipe asset verification",
            "safe_for_DiT_pretrained_merge":False}}
