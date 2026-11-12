Prepared only; no execution admission exists.

CPU metadata/SHA inspection command (no codec/model):

```sh
/cache/zihaohe/omni_openh3_20261111/venv/bin/python /cache/zihaohe/omni_openh3_20261111/preparation/history_utility_revised_materializer_20261112a/run_once.py --freeze /cache/zihaohe/omni_openh3_20261111/preparation/history_utility_revised_materializer_20261112a/freeze.json --freeze-sha256 96f9b75c7c9dcf23bad21d03003a16bd2d0d49ffae98ddd7fd5c8c5d793581d8
```

Actual execution requires a new root-reviewed admission with schema openh3.history_utility_revised_materialization_admission.v1; exact freeze/contract/entry/inspection SHA, approved=true and host31984. Only then add --execute --admission <absolute path> to the same supervisor command. Never call a source worker directly.

Budget: one CPU, fresh sequential source workers in one newly owned PG,768MiB actual live PG RSS,3600s+30s cleanup,6GiB free before each source,30GiB new output accounting checked per source. Disk is not a filesystem quota. No overwrite/retry. Three complete sources reference unchanged artifacts;22 sources each bridge exact cached target pixels then3profiles×2codec replays.0source MP4 decodes.

11 local CPU tests passed3.287s; fullshape synthetic NPZ artifacts use stub codec output, so this is interface verification only. Old3source engineering timings71.4–75.9s include one target decode+six codec passes each and do not guarantee new cohort throughput.
