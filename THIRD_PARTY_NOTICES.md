# Third-party sources and assets

This snapshot does not assign a new blanket license to upstream code, model weights, datasets or reference material. Original applicable licenses and notices continue to govern them. No new open-source license is granted by this private archival upload.

- MiniMax-H3: https://github.com/MiniMax-AI/MiniMax-H3 and https://huggingface.co/MiniMaxAI/MiniMax-H3 . Foundation-model weights are excluded.
- ComfyUI: https://github.com/comfyanonymous/ComfyUI . The inspected upstream excerpts in `research/*upstream.py` retain their upstream origin and licensing; the external checkout is excluded. See the project's source hashes and research notes for the inspected versions.
- BSDS500: official Berkeley source and split metadata are recorded in the experiment report and `data/bsds_manifest.json`; image dataset excluded.
- DAVIS: dataset terms are retained under `data/real_video/source_metadata/`. The research records use the archive's CC BY-NC 4.0 terms; dataset images excluded except limited result panels.
- LaSOT: source/revision/license evidence is recorded in `research/long_video/SOURCES.md` and dataset manifests; video frames excluded.
- Other cited methods and implementations are attributed in `research/sources_and_method.md` and individual experiment reports.

- RelayVSR: https://github.com/kopperx/RelayVSR . Used as an external baseline; its downloaded implementation and model weights are excluded. The protocol records the evaluated commit and conditioning differences.
