"""Strict small evidence archive; NPZ/images/videos/weights never included."""
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(out, config, source):
    out, source = Path(out), Path(source)
    files = {}
    for name in ('status.json', 'admission.json', 'producer_status.json', 'worker.log', 'dataset_audit/audit.json'):
        path = out / name
        if path.is_file(): files['audit/' + name] = path
    run = Path(config['producer_run'])
    for name in ('complete.json', 'failure.json', 'train_receipt_index.json', 'dev_receipt_index.json'):
        path = run / name
        if path.is_file(): files['producer/' + name] = path
    complete = run / 'complete.json'
    if complete.is_file():
        report = json.loads(complete.read_text())
        for row in report.get('source_results', []):
            sid = row['source_id']
            if '/' in sid or '..' in sid or not sid: raise ValueError('unsafe source ID')
            files['receipts/' + sid + '/complete.json'] = Path(row['entry']['path'])
            if 'revised_authority' in row['entry']:
                files['receipts/' + sid + '/reuse_admission.json'] = Path(row['entry']['revised_authority']['path'])
    for name in ('audit_terminal.py', 'finalize_once.py', 'small_archive.py', 'test_terminal.py',
                 'execution_plan.json', 'freeze.json', 'README.md', 'cpu_receipt.json',
                 'code/history_utility_data.py', 'code/history_utility_revised_data.py'):
        path = source / name
        if path.is_file(): files['source/' + name] = path
    for key in ('producer_contract', 'producer_freeze', 'producer_admission'):
        files['input/' + key + '.json'] = Path(config[key]['path'])
    records, total = [], 0
    for name, path in files.items():
        if path.is_symlink() or not path.is_file() or path.suffix not in ('.json', '.py', '.log', '.md'):
            raise ValueError('non-small/unsafe evidence: ' + str(path))
        total += path.stat().st_size
        if total > 32 << 20: raise ValueError('small evidence exceeds32MiB')
        records.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    manifest = {'schema': 'openh3.small_evidence_manifest.v1', 'files': records,
                'NPZ_included': False, 'images_video_weights_included': False, 'model_calls': 0}
    mp = out / 'MANIFEST.json'
    with mp.open('x') as f: json.dump(manifest, f, indent=2); f.write('\n')
    archive = out / 'small_evidence.tar.gz'
    with archive.open('xb') as f:
        with tarfile.open(fileobj=f, mode='w:gz') as tar:
            tar.add(mp, arcname='MANIFEST.json', recursive=False)
            for row in records:
                path = files[row['path']]
                if sha(path) != row['sha256']: raise ValueError('evidence changed while packing')
                tar.add(path, arcname=row['path'], recursive=False)
    with tarfile.open(archive) as tar:
        if len(tar.getmembers()) != len(records) + 1: raise ValueError('archive member count differs')
        for row in records:
            member = tar.getmember(row['path'])
            if not member.isfile() or member.size != row['bytes']: raise ValueError('archive member differs')
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest() != row['sha256']: raise ValueError('archive hash differs')
    receipt = {'path': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
               'payloads': len(records), 'payload_bytes': total, 'manifest_sha256': sha(mp),
               'NPZ_included': False, 'download_count_by_this_program': 0}
    with (out / 'archive_receipt.json').open('x') as f: json.dump(receipt, f, indent=2); f.write('\n')
    return receipt
