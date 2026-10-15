"""Foreground official/causal RelayVSR comparison on the existing H3 LR cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/relay_baseline/cohort'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit(gpu):
    for attempt in range(3):
        text = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu),
            '--query-gpu=memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        used, free, utilization = map(int, text.split(','))
        if used < 512 and free > 60000 and utilization == 0:
            return text
        if attempt < 2:
            time.sleep(1)
    raise RuntimeError('GPU no longer idle; no foreign process modified: ' + text)


def verify(folder, case, protocol):
    meta = json.loads((folder / 'measurement.json').read_text())
    assert meta['status'] == 'complete' and meta['output_frames'] == 22
    assert meta['source_status'] == '' and meta['commit'] == 'e856773350f53036d825404c9ea9228e6dffadca'
    assert meta['explicit_cuda_memory_fraction'] == .8
    assert meta['max_frames'] == 22 and meta['window_size'] == 16 and meta['seed'] == 42
    png = sorted((folder / 'png').glob('*.png'))
    assert [p.name for p in png] == [f'{i:05d}.png' for i in range(22)]
    for path in png:
        with Image.open(path) as image:
            assert image.size == (case['shape_hw'][1], case['shape_hw'][0])
            assert image.mode == 'RGB'
    rows = [json.loads(x) for x in (folder / 'measurement.frames.jsonl').read_text().splitlines()]
    assert [r['index'] for r in rows] == list(range(22))
    if protocol == 'official':
        assert meta['wan_keyframe_indices'] == [0, 15, 21]
        assert meta['max_lookahead_frames'] == 15
    else:
        assert meta['wan_keyframe_indices'] == [0, 15]
        assert all(r['last_consumed_input_index'] == r['index'] for r in rows)
    record = {'case': case['id'], 'protocol': protocol, 'source': case['sequence'],
              'split': case['split'], 'resolution': case['resolution'],
              'output_sha256': [sha(path) for path in png],
              'observed_lr_sha256': case['observed_lr_sha256'], 'frames': 22,
              'max_lookahead_frames': meta['max_lookahead_frames'],
              'wan_keyframe_indices': meta['wan_keyframe_indices'],
              'measurement_sha256': sha(folder / 'measurement.json'),
              'only_lr_input': True, 'verified': True}
    (folder / 'verification.json').write_text(json.dumps(record, indent=2))
    return record


def main(gpu):
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((ROOT / 'results/real_video_vae/input_manifest.json').read_text())
    cases = sorted([c for c in inputs['cases'] if c['split'] in ['train', 'val']],
                   key=lambda c: (c['resolution'] != '256', c['sequence']))
    assert len(cases) == 12 and len({c['sequence'] for c in cases}) == 6
    protocols = ['official', 'causal_repeat_start']
    config = {'frames': 22, 'window_size': 16, 'seed': 42, 'fps_nominal': 24,
              'protocols': protocols, 'cases': cases, 'fit_or_parameter_search': False,
              'primary_backbone': 'H3; Relay is a frozen external baseline',
              'fairness': 'Same observed LR and final DC. Different pretrained models, reference budgets, waiting protocols and training data; not equal-compute or architectural ablation.',
              'evaluation_unit': '6 source videos (4train/2val), 2 spatial contexts per source',
              'official_commit': 'e856773350f53036d825404c9ea9228e6dffadca',
              'observer_source_sha256': sha(ROOT / 'experiments/relay_baseline/measure_official.py'),
              'causal_source_sha256': sha(ROOT / 'experiments/relay_baseline/causal_stream.py')}
    (OUT / 'protocol.json').write_text(json.dumps(config, indent=2))
    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = str(gpu)
    completed = []
    for protocol in protocols:
        for case in cases:
            folder = OUT / protocol / case['id']; folder.mkdir(parents=True, exist_ok=True)
            observed = [sha(ROOT / 'data/real_video' / p) for p in case['lr_paths']]
            assert observed == case['observed_lr_sha256']
            (folder / 'input_provenance.json').write_text(json.dumps(case, indent=2))
            if (folder / 'measurement.json').exists():
                # Only verified completed cases may be reused; partial cases stop here.
                completed.append(verify(folder, case, protocol))
                print(json.dumps({'reused_verified': case['id'], 'protocol': protocol}), flush=True)
                continue
            admission = admit(gpu)
            command = ['external/relay_venv/bin/python', '-u',
                'experiments/relay_baseline/measure_official.py', '--repository', 'external/RelayVSR',
                '--input', str(Path('data/real_video') / Path(case['lr_paths'][0]).parent),
                '--fps', '24', '--max-frames', '22', '--window-size', '16', '--protocol', protocol,
                '--wan-checkpoint', 'external/RelayVSR/weights/wan.safetensors',
                '--flash-checkpoint', 'data/flash.safetensors', '--memory-fraction', '0.8',
                '--output', str(folder / 'output_x4.mp4'), '--png-dir', str(folder / 'png'),
                '--metrics', str(folder / 'measurement.json')]
            (folder / 'launch.json').write_text(json.dumps({'command': command, 'gpu': gpu,
                'admission': admission, 'started_unix': time.time()}, indent=2))
            (OUT / 'LIVE_STATE.json').write_text(json.dumps({'phase': 'foreground_inference',
                'pid': os.getpid(), 'case': case['id'], 'protocol': protocol,
                'completed_cases': len(completed), 'total_cases': 24, 'gpu': gpu,
                'last_update_unix': time.time()}, indent=2))
            with (folder / 'run.log').open('w') as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            completed.append(verify(folder, case, protocol))
            print(json.dumps({'completed': case['id'], 'protocol': protocol,
                              'count': len(completed)}), flush=True)
    (OUT / 'all_outputs_verified.json').write_text(json.dumps({'passed': True,
        'case_count': len(completed), 'png_count': 528, 'cases': completed}, indent=2))
    (OUT / 'LIVE_STATE.json').write_text(json.dumps({'phase': 'inference_complete',
        'cases': 24, 'png_count': 528, 'gpu_released_after_parent_exit': gpu}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gpu', required=True, type=int); a = p.parse_args()
    main(a.gpu)
