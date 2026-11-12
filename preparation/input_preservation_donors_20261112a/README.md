# Single benefit-map donor metadata

Prepared only. Reuses the fixed student16000row JSON; no new schedule, dataset decoding, GT/label inspection or model. Source/test bytes are frozen in source_freeze.json. All actual source/group/window/crop/control metadata stays attached through row SHA.

Unit: one entire1×6×6 benefit map per row, including both-history-absent rows. Stratum: exact(profile_id,severity_phase,effective_availability2), where effective=true_availability*external_mask. This differs intentionally from the old per-valid-slot map: no lag dimension, no raw external-pattern separation after effective support is identical.

Seed20261112. Canonical compact sorted UTF-8 JSON descriptor keys. Order source groups by SHA256([seed,key,"group",group_id]); order rows within each contiguous group by SHA256([seed,key,"unit",row_index]); tie-break by raw group/row identity. Rotate once by largest group count. No resampling, coarsening, or label-based ordering. If any stratum has largest_group>half, record infeasible groups/rows, publish no partial usable entries, complete=false, both head arms blocked. No automatic continuation.

A separate validator rechecks exact row hashes, every recipient/donor metadata item, all16000 recipient IDs and donor IDs exactly once, same stratum, different source AND group, nonself, and all00rows. It does not rely on the construction rotation to prove these invariants.

Runtime requires explicit root admission{path,sha256}: approved:true,scope:input_preservation_donor_metadata_only,execution_payload_sha256=canonical execution excluding admission,entry_sha256,out=absolute fresh output directory. Default inspect reads proposal+manifest and code SHA only, not the14.9MB student schedule. Actual execute reads/authenticates the original schedule and writes input_preservation_donor_map.json + manifest.json into a new output. Complete metadata readiness never grants student training.

Default:
`/cache/zihaohe/omni_openh3_20261111/venv/bin/python /cache/zihaohe/omni_openh3_20261111/preparation/input_preservation_donors_20261112a/code/prepare_input_preservation_donors.py --execution-contract /cache/zihaohe/omni_openh3_20261111/preparation/input_preservation_donors_20261112a/execution_contract.json --out /cache/zihaohe/omni_openh3_20261111/runs/input_preservation_donors_20261112a_inspect`

Admitted actual run uses the same command with `--out /cache/zihaohe/omni_openh3_20261111/runs/input_preservation_donors_20261112a --execute`. Linux-only actual bound:1CPU/2GiB address space/120s/32MiB output; no subprocesses/model or NPU imports. Failure/infeasibility is preserved with nonzero exit. Old frozen files are not modified.
