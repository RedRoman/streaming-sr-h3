# One TRAIN-only weighting diagnostic — prepared, not admitted

Only the class weights change from the accepted Huber probe. No logistic loss or binary-target conversion is used. The superseded logistic draft was never tested on actual data or admitted.

CLI: `python probe.py --execution-contract execution_contract.json --execution-contract-sha256 SHA --out NEW_DIR [--execute]`. Default is metadata/split inspection with zero cache-array reads. Execution inherits the frozen Linux180s/2CPU/4GiB address-space guard and8MiB-per-JSON bound. No automatic wait, retry, larger budget, or further experiment. Root alone controls admission/SSH.

Only legacy_probe.py, probe.py, test_probe.py and the exact execution contract are needed at runtime. The legacy reader is copied byte-for-byte; old metadata, split and score identities must match. Its cache reader opens only the accepted19TRAIN NPZs and decompresses raw_features/normalized_benefit. Recomputed feature/label and old held-out score hashes must equal the completed earlier probe before a new weighted fit. No teacher/head model, Torch, NPU, DEV, RGB metrics or LPIPS calls.

Report old/new source-macro balanced accuracy, negative falseK1, positive recall and AUC with denominator counts; same unweighted Huber loss remains directly comparable. Weighted training objective and weighted constant are separately named. Logits/predictions are not physical utility or calibrated probabilities. All outcomes are descriptive, never deployment admission. After this single diagnostic, pause this head branch while H3 and genuinely different information/data work proceed. See scientific_plan.json for the pre-result interpretation and limitations.
