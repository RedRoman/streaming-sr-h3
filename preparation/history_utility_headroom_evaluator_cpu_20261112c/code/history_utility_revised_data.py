"""Explicit revised P1 cohort over the unchanged v1 engineering numerics.

No MP4 access, decoding, model import or materialization. Every selected source
must have a complete v1-compatible observation receipt and verified NPZ pixels.
The old loader's research gate remains false; revised readiness has its own
namespace and is not a training admission or a claim to complete the old 32/8.

JSON/image references are SHA-bound. Relative evidence paths are resolved under
``evidence_root`` (default: the cohort JSON's directory), never by guessing an
alternative file. Dataset artifact relocation retains v1 ``artifact_root``.
"""
from pathlib import Path
from fractions import Fraction
import json

import numpy as np

import history_utility_data as D

SCHEMA = "openh3.history_utility_revised_cohort.v1"
POLICY_SHA = "1271b6e1abe0519ac34ae5455fe4d0f8afe3a3da26ca7a010b4c6e13716fa277"
LOADER_SHA = "87ad6a83965869846fdeb4dc3abcea800ed0e16ce1819074d6753c8ff481a1d7"
NUMERIC_WINDOW_SHA = "13b6e03f8b5387a101997e1545214a03b43d281454e1d1848a589a57fd9c47ad"
OPERATOR_SHA = "3d08ba296e54fb1827d8ec81ed76f876148fca43cd88896048829cd10af8de3b"
PROFILE_SHA = "9278ec87dc077a0653ee0d52ee1115d7aa88007ae2cad4e535d128d1d1448b06"


def validate_numeric_plan(plan):
    profiles = [p for p in plan["profiles"] if p["profile_id"] in D.PROFILES]
    D.require([p["profile_id"] for p in profiles] == list(D.PROFILES)
              and D.value_sha(profiles) == PROFILE_SHA
              and D.value_sha(plan["operator_contract"]) == OPERATOR_SHA,
              "frozen numerical operator/profile recipe changed")


def bound_path(binding, root):
    D.require(isinstance(binding, dict) and {"path", "sha256"} <= set(binding),
              "explicit path/SHA binding required")
    path = Path(binding["path"])
    path = path if path.is_absolute() else Path(root) / path
    D.require(D.file_sha(path) == D._digest(binding["sha256"]), "bound evidence file changed: " + str(path))
    return path


def bound_json(binding, root):
    return D._json(bound_path(binding, root))


def validate_cohort(cohort, registry):
    """Account for all 40 original identities without changing their roles."""
    D.require(cohort["schema"] == SCHEMA and cohort["policy_sha256"] == POLICY_SHA,
              "wrong revised cohort schema/policy")
    decisions = cohort["decisions"]
    D.require(isinstance(decisions, dict) and set(decisions) == set(registry) and len(registry) == 40,
              "cohort must account for exactly all original40 sources")
    primary = cohort["primary_source_ids"]
    D.require(set(primary) == {"train", "dev"}, "primary split keys differ")
    for sid, row in decisions.items():
        D.require(row.get("source_id", sid) == sid, "decision source key mismatch")
        for key in ("source_sha256", "split"):
            D.require(row[key] == registry[sid][key], "cohort changes original source/role")
        D.require(row["disposition"] in ("primary", "diagnostic", "ineligible"), "unknown disposition")
        if row["disposition"] == "primary":
            D._digest(row["window_key"]); D._digest(row["target_pixel_sha256"])
            D.require(row.get("target_receipt") and row.get("visual_review"), "primary needs numerical and visual evidence")
    for split in ("train", "dev"):
        declared = primary[split]
        expected = {sid for sid, row in decisions.items()
                    if row["split"] == split and row["disposition"] == "primary"}
        D.require(isinstance(declared, list) and len(declared) == len(set(declared))
                  and set(declared) == expected, "exact primary membership differs")
        D.require(bool(expected), "empty primary split")
    return primary


def validate_terminal_audit(cohort, registry, root):
    audit = bound_json(cohort["target_terminal_audit"], root)
    D.require(audit["schema"] == "openh3.history_utility_target_screen_actual_terminal_audit.v1"
              and audit["complete"] is True and audit["source_count"] == 40
              and audit["source_files_actual_SHA_verified"] == 40
              and audit["target_protocol_eligible_count"] == 38 and audit["ineligible_count"] == 2
              and audit["target_files_actual_SHA_and_numpy_pixel_SHA_verified"] == 38,
              "target terminal eligibility audit incomplete")
    rows = audit["source_rows"]
    D.require(len(rows) == 40 and {r["source_id"] for r in rows} == set(registry),
              "target audit membership differs")
    eligible = 0
    for row in rows:
        sid = row["source_id"]; decision = cohort["decisions"][sid]
        for key in ("source_sha256", "split"):
            D.require(row[key] == registry[sid][key], "target audit source/role differs")
        ok = row["state"] == "target_protocol_eligible_continuity_unreviewed"
        eligible += int(ok)
        D.require(ok == (decision["disposition"] != "ineligible"), "cohort/audit eligibility differs")
        if decision["disposition"] == "primary":
            for key in ("window_key", "target_pixel_sha256"):
                D.require(row[key] == decision[key], "primary/audit target identity differs")
            D.require(row["numerical_receipt"]["sha256"] == decision["target_receipt"]["sha256"],
                      "primary numerical receipt not in terminal audit")
    D.require(eligible == 38, "target audit eligible rows disagree with count")
    # These are immutable evidence-file bindings, not a repeat budget audit.
    for key in ("target_engineering_review", "local_target_archive_verification"):
        if key in cohort:
            bound_path(cohort[key], root)
    return {"sha256": cohort["target_terminal_audit"]["sha256"],
            "source_count": 40, "target_protocol_eligible_count": 38,
            "original_pixels_verified_by": "bound prior terminal audit; current dataset verifies observation/target NPZ independently"}


def review_row(document, sid, key):
    """Both keyed decisions and explicit row lists are accepted, without aliases."""
    rows = document[key]
    if isinstance(rows, dict):
        row = rows[sid]
        D.require(row.get("source_id", sid) == sid, "visual review source differs")
        return row
    D.require(isinstance(rows, list), "visual rows must be a map/list")
    found = [row for row in rows if row.get("source_id") == sid]
    D.require(len(found) == 1, "visual source missing/duplicate")
    return found[0]


def validate_primary_evidence(sid, decision, registry, root):
    """Bind reviewed images and original target-only eligibility, no MP4 read."""
    numerical = bound_json(decision["target_receipt"], root)
    D.require(numerical["schema"] == "openh3.history_utility_target_only_receipt.v1"
              and numerical["complete"] is True
              and numerical["status"] == "target_only_complete_not_observation_dataset",
              "primary numerical target receipt incomplete")
    D.require(numerical["source"] == registry[sid]
              and numerical["window_key"] == decision["window_key"]
              and numerical["target"]["pixel_sha256"] == decision["target_pixel_sha256"],
              "primary numerical identity differs")
    D.require(numerical["numerical_window_contract_file_sha256"] == NUMERIC_WINDOW_SHA,
              "original numerical window authority changed")
    spec = numerical["target_specification"]
    D.require(D.value_sha(spec) == decision["window_key"], "target specification/window changed")
    D.require(numerical["target"]["shape"] == list(D.SHAPE) and numerical["target"]["dtype"] == "uint8",
              "target numerical geometry differs")
    decode = numerical["decode"]
    D.require(decode["decoded_target_count"] == 96 and decode["decoded_source_count"] >= 96,
              "primary numerical frame counts ineligible")
    source_stamps = decode["timestamps"]["source"]
    D.require(len(source_stamps) == decode["decoded_source_count"]
              and [x["index"] for x in source_stamps] == list(range(len(source_stamps))),
              "source timestamp history incomplete")
    source_times = [Fraction(x["pts"]) * Fraction(x["time_base"]) for x in source_stamps]
    D.require(all(b > a for a, b in zip(source_times, source_times[1:])), "source timestamps not increasing")
    stamps = decode["timestamps"]["target"]
    D.require(len(stamps) == 96 and [x["index"] for x in stamps] == list(range(96)),
              "target timestamp history incomplete")
    times = [Fraction(x["pts"]) * Fraction(x["time_base"]) for x in stamps]
    D.require(all(times[i] - times[0] == Fraction(i, 24) for i in range(96)),
              "target 24fps timestamp grid changed")
    visual_ref = decision["visual_review"]
    visual = bound_json(visual_ref, root)
    D.require(visual["policy_sha256"] == POLICY_SHA, "visual policy differs")
    row = review_row(visual, sid, visual_ref.get("decision_key", "rows"))
    for key in ("source_sha256", "split", "window_key", "target_pixel_sha256"):
        D.require(row[key] == decision[key], "visual identity differs: " + key)
    D.require(decision["disposition"] == "primary"
              and row["decision"] in ("primary", "primary_continuity_approved", "primary_candidate")
              and row["reviewed_frames"] == [0, 95], "primary visual review incomplete/diagnostic")
    D.require(row["target_file_sha256"] == numerical["target"]["file_sha256"], "visual target file differs")
    images = row["images_reviewed"]
    D.require(isinstance(images, list) and len(images) >= 2, "missing visual image bindings")
    names = {Path(x["path"]).name for x in images}
    D.require({"target96.png", "candidate_pairs.png"} <= names, "all96-frame and ranked-pair images required")
    artifacts = numerical["diagnostics"]["artifacts"]
    for name in ("all_frame_sheet", "ranked_candidate_pairs"):
        expected = artifacts[name]
        matches = [x for x in images if x["path"] == expected["path"] and x["sha256"] == expected["sha256"]]
        D.require(len(matches) == 1, "visual images differ from target numerical diagnostics")
    for image in images:
        bound_path(image, root)
    return {"target_receipt_sha256": decision["target_receipt"]["sha256"],
            "visual_review_sha256": visual_ref["sha256"], "reviewed_images": len(images),
            "target": numerical["target"], "target_specification": spec}


def validate_source_authority(entry, receipt, *, cohort, cohort_sha256, decision, evidence, root):
    """New artifacts carry authority; old complete artifacts use a new sidecar."""
    is_new = receipt.get("producer_schema") == "openh3.history_utility_revised_materialization.v1"
    if is_new:
        D.require("revised_authority" not in entry and receipt["revised_cohort_sha256"] == cohort_sha256,
                  "new receipt cohort/authority namespace differs")
        authority = receipt["execution_authority"]
        kind = "new_materialization"
    else:
        D.require("revised_authority" in entry, "legacy receipt requires new reuse authority sidecar")
        sidecar = bound_json(entry["revised_authority"], root)
        D.require(sidecar["schema"] == "openh3.history_utility_revised_reuse_admission.v1"
                  and sidecar["original_receipt_unchanged"] is True
                  and sidecar["target_pixels_unchanged"] is True, "reuse authority flags missing")
        D.require(sidecar["reused_receipt"]["sha256"] == entry["sha256"]
                  and sidecar["source"] == receipt["source"]
                  and sidecar["window_key"] == decision["window_key"]
                  and sidecar["target_pixel_sha256"] == decision["target_pixel_sha256"],
                  "reuse sidecar does not bind unchanged selected receipt")
        bound_path(sidecar["reused_receipt"], root)
        authority = sidecar["execution_authority"]
        kind = "reused_complete"
    D.require(authority["schema"] == "openh3.history_utility_revised_materialization_authority.v1",
              "wrong revised execution authority schema")
    execution = bound_json(authority["execution_contract"], root)
    D.require(execution["schema"] == "openh3.history_utility_revised_materialization_execution.v1"
              and D.value_sha(execution) == authority["execution_contract_value_sha256"],
              "new execution authority changed")
    for ref in (authority["cohort"], execution["files"]["cohort"]):
        D.require(ref["sha256"] == cohort_sha256 and bound_json(ref, root) == cohort,
                  "authority binds another revised cohort")
    numerical = execution["files"]["numerical_window_contract"]
    D.require(numerical["sha256"] == NUMERIC_WINDOW_SHA, "original numerical namespace differs")
    bound_path(numerical, root)
    D.require(authority["input_target_receipt"]["sha256"] == decision["target_receipt"]["sha256"],
              "authority binds another target receipt")
    bound_path(authority["input_target_receipt"], root)
    D.require(authority["input_target"] == evidence["target"], "authority original target artifact differs")
    return kind, authority


def validate_new_metadata(source, authority, cohort_sha256):
    for name, row in source["artifacts"].items():
        path = source["root"] / (name + ".npz") if source["root"] else Path(row["path"])
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"].item()))
        D.require(metadata.get("execution_authority") == authority
                  and metadata.get("revised_cohort_sha256") == cohort_sha256,
                  "new NPZ lacks exact revised cohort/execution authority")


def inspect_cohort_evidence(*, cohort, plan, train_manifest, dev_manifest, evidence_root=None):
    """Pre-materialization evidence only; do not claim a usable dataset."""
    D.require(D.file_sha(D.__file__) == LOADER_SHA, "frozen v1 loader source changed")
    path = Path(cohort["path"])
    if not path.is_absolute():
        path = (Path(evidence_root) if evidence_root is not None else Path.cwd()) / path
    root = Path(evidence_root) if evidence_root is not None else path.resolve().parent
    document = bound_json({**cohort, "path": str(path)}, root)
    numeric_plan = D._json(plan); validate_numeric_plan(numeric_plan)
    registry = D.load_source_registry(numeric_plan, train_manifest, dev_manifest)
    primary = validate_cohort(document, registry)
    terminal = validate_terminal_audit(document, registry, root)
    evidence = {sid: validate_primary_evidence(sid, document["decisions"][sid], registry, root)
                for sid in sorted(primary["train"] + primary["dev"])}
    return {"schema": "openh3.history_utility_revised_cohort_evidence_inspection.v1",
            "complete": True, "cohort_sha256": cohort["sha256"],
            "registry_sha256": D.value_sha(registry), "original_registry_counts": {"train": 32, "dev": 8},
            "primary_source_ids": primary, "actual_selected_counts": {s: len(primary[s]) for s in primary},
            "terminal_audit": terminal, "primary_evidence": evidence,
            "visual_files_and_numerical_receipts_SHA_verified": True,
            "current_target_NPZ_read": False, "observation_NPZ_read": False,
            "revised_dataset_ready": False, "legacy_research_ready": False,
            "training_admitted": False, "model_calls": 0, "decode_calls": 0,
            "scope": "Cohort/role/numerical-receipt/image evidence only; observation materialization and actual dataset pixel validation remain pending."}


class RevisedHistoryUtilityDataset:
    def __init__(self, receipts, *, cohort, plan, train_manifest, dev_manifest,
                 split, evidence_root=None, cache_bytes=512 << 20):
        D.require(split in ("train", "dev"), "invalid split")
        D.require(D.file_sha(D.__file__) == LOADER_SHA, "frozen v1 loader source changed")
        cohort_path = Path(cohort["path"])
        if not cohort_path.is_absolute():
            cohort_path = (Path(evidence_root) if evidence_root is not None else Path.cwd()) / cohort_path
        self.evidence_root = Path(evidence_root) if evidence_root is not None else cohort_path.resolve().parent
        self.cohort = bound_json({**cohort, "path": str(cohort_path)}, self.evidence_root)
        self.cohort_sha256 = cohort["sha256"]
        numeric_plan = D._json(plan)
        validate_numeric_plan(numeric_plan)
        self.registry = D.load_source_registry(numeric_plan, train_manifest, dev_manifest)
        primary = validate_cohort(self.cohort, self.registry)
        terminal = validate_terminal_audit(self.cohort, self.registry, self.evidence_root)
        # Exact membership, not a silently usable partial revised dataset.
        selected, normalized = [], []
        for entry in receipts:
            D.require(set(entry) <= {"path", "sha256", "artifact_root", "revised_authority"},
                      "unknown revised receipt binding field")
            path = bound_path(entry, self.evidence_root)
            receipt = D._json(path)
            selected.append(receipt["source"]["source_id"])
            current = {**entry, "path": str(path.resolve())}
            if "artifact_root" in current and not Path(current["artifact_root"]).is_absolute():
                current["artifact_root"] = str((self.evidence_root / current["artifact_root"]).resolve())
            normalized.append(current)
        D.require(len(selected) == len(set(selected)) and set(selected) == set(primary[split]),
                  "materialized receipt membership must equal revised primary split")
        self.evidence = {sid: validate_primary_evidence(sid, self.cohort["decisions"][sid],
                         self.registry, self.evidence_root) for sid in sorted(primary[split])}
        numeric_entries = [{k: v for k, v in e.items() if k != "revised_authority"} for e in normalized]
        self._numeric = D.HistoryUtilityDataset(numeric_entries, plan=plan, train_manifest=train_manifest,
            dev_manifest=dev_manifest, split=split, scope="engineering", verify_pixels=True, cache_bytes=cache_bytes)
        D.require(self._numeric.research_ready is False, "do not relabel old research-ready dataset")
        self.split, self.source_ids = split, list(self._numeric.source_ids)
        self.authorities = {}
        for entry in normalized:
            sid = D._json(entry["path"])["source"]["source_id"]
            old = self._numeric.sources[sid]["receipt"]
            decision, evidence = self.cohort["decisions"][sid], self.evidence[sid]
            D.require(old["window_key"] == decision["window_key"]
                      and old["target"]["pixel_sha256"] == decision["target_pixel_sha256"]
                      and old["target_specification"] == evidence["target_specification"],
                      "observation receipt is not the visually selected target/window")
            D.require(old.get("scientific_continuity_approved") is False,
                      "old observation receipt continuity must remain unchanged/false")
            kind, authority = validate_source_authority(entry, old, cohort=self.cohort,
                cohort_sha256=self.cohort_sha256, decision=decision, evidence=evidence, root=self.evidence_root)
            if kind == "new_materialization":
                validate_new_metadata(self._numeric.sources[sid], authority, self.cohort_sha256)
            self.authorities[sid] = {"kind": kind, "execution_authority": authority,
                "sidecar": entry.get("revised_authority"), "receipt_sha256": entry["sha256"]}
        self.research_ready = False
        self.revised_cohort_ready = True
        self.binding = {"schema": "openh3.history_utility_revised_dataset.v1", "split": split,
            "cohort_sha256": self.cohort_sha256, "policy_sha256": POLICY_SHA,
            "original_registry_sha256": D.value_sha(self.registry), "original_registry_counts": {"train": 32, "dev": 8},
            "primary_source_ids": primary, "actual_selected_counts": {s: len(primary[s]) for s in primary},
            "source_ids": self.source_ids, "legacy_dataset_sha256": self._numeric.dataset_sha256,
            "legacy_binding": self._numeric.binding, "legacy_research_ready": False,
            "revised_cohort_ready": True, "readiness_scope": "this exact primary split only; train/dev separately verified",
            "original_32_8_completion_claim": False, "frozen_test_sources": 0,
            "development_selection_used": True, "training_admitted": False,
            "visual_evidence": self.evidence, "numeric_loader_sha256": LOADER_SHA,
            "source_authorities": self.authorities, "target_terminal_audit": terminal,
            "operator_contract_sha256": OPERATOR_SHA, "profile_recipes_sha256": PROFILE_SHA,
            "model_calls": 0, "decode_calls": 0, "training_steps": 0}
        self.dataset_sha256 = D.value_sha(self.binding)

    def schedule(self, role="student", steps=1000):
        D.require(type(steps) is int and steps == 1000, "revised research schedule is exactly1000 steps")
        numeric = self._numeric.schedule(role, steps)
        result = {"schema": "openh3.history_utility_revised_schedule.v1",
                  "dataset_sha256": self.dataset_sha256, "cohort_sha256": self.cohort_sha256,
                  "role": role, "steps": 1000, "batch_size": 16, "seed": D.SEEDS[role],
                  "legacy_research_ready": False, "revised_cohort_ready": True,
                  "training_admitted": False, "numeric_schedule": numeric}
        result["schedule_sha256"] = D.value_sha(result)
        return result

    def validate_schedule(self, schedule):
        D.require(schedule == self.schedule(schedule["role"], schedule["steps"]),
                  "revised schedule differs from exact cohort/numeric recipe")

    def sample(self, row):
        return self._numeric.sample(row)

    def batch(self, rows):
        return self._numeric.batch(rows)

    def evaluation_rows(self):
        return self._numeric.evaluation_rows()


def assert_revised_pair(train, dev):
    D.require(train.split == "train" and dev.split == "dev"
              and train.revised_cohort_ready and dev.revised_cohort_ready,
              "both exact revised splits must be verified")
    D.require(train.cohort_sha256 == dev.cohort_sha256 and train.registry == dev.registry,
              "revised train/dev cohort or original registry differs")
    for key in ("source_id", "source_sha256", "source_group_id"):
        a = {train.registry[sid][key] for sid in train.source_ids}
        b = {dev.registry[sid][key] for sid in dev.source_ids}
        D.require(not a & b, "revised source identity crosses train/dev")
    return {"revised_pair_ready": True, "training_admitted": False,
            "legacy_research_ready": False, "cohort_sha256": train.cohort_sha256,
            "train_sources": len(train.source_ids), "dev_sources": len(dev.source_ids)}
