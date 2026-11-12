# Fixed public Refiner asset preparation

This single authorized download prepares only the byte-identical Refiner missing on host 31112. It uses the public immutable revision `c2e7b89b1800123c49a0f620a1932c01d8952256` in `Kijai/MiniMax-H3-experimental`, via the already checked HTTPS mirror and its `cas-bridge.xethub.hf.co` object redirect. The expected file is 5,868,409,512 bytes with SHA256 `4bf30291343eef5e51ee867ba37dd31bce3133d1da1ace1a245a7130b88818c2`.

The source review is `results/takeover_20261111/public_asset_access_review_31112_20261112.json`. Its HEAD success demonstrates availability at that time, not completed download or transfer speed. No Turbo or other object is downloaded by this task.

`launch_once.py` claims this directory exclusively and starts only the owned downloader, with a 7200-second external timeout and a 10-second termination cleanup grace. The downloader has its own 7200-second alarm and 30-second socket timeout. It makes one attempt, follows only the previously observed two-host HTTPS chain, validates revision/link-SHA/link-size metadata, and limits written bytes to the fixed object size. Existing partial/ready files are refused. No automatic retries or resume are permitted.

`status.json` tracks the owned PID, byte count and phase every 15 seconds during transfer; `launch_receipt.json` records the supervisor PID/process group. On terminal completion or handled failure, `receipt.json` preserves the result. Failures keep the partial file. Success requires both streaming SHA and a second full on-disk SHA read, then publishes `.ready` with an exclusive hard link and unlinks `.partial`. This move deliberately avoids POSIX rename's silent-overwrite behavior. No manifest, shared asset, runtime or model is changed; no NPU call is made. Publishing `.ready` alone does not establish model readiness or quality.

`launch_command.txt` contains the exact single-line command. It must not be invoked again after the directory has been claimed; read the existing status instead.
