import json
from pathlib import Path
import tempfile
import unittest

from finalize_rphi_stage2 import inspect_states, main


class Stage2WaiterTests(unittest.TestCase):
    def test_missing_completed_and_failed_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('a.json', 'b.json')]
            self.assertEqual(inspect_states(paths)[0], 'waiting')
            paths[0].write_text(json.dumps({'state': 'completed'}))
            self.assertEqual(inspect_states(paths)[0], 'waiting')
            paths[1].write_text(json.dumps({'state': 'completed'}))
            self.assertEqual(inspect_states(paths)[0], 'completed')
            paths[1].write_text(json.dumps({'state': 'failed'}))
            self.assertEqual(inspect_states(paths)[0], 'failed')

    def test_failure_preserves_upstream_and_prevents_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream = root / 'native_queue_status.json'
            original = json.dumps({'state': 'failed', 'step': 'training'})
            upstream.write_text(original)
            self.assertEqual(main(['--run-dir', directory]), 2)
            self.assertEqual(upstream.read_text(), original)
            self.assertEqual(json.loads((root / 'finalize_status.json').read_text())['state'], 'failed')
            with self.assertRaises(FileExistsError):
                main(['--run-dir', directory])


if __name__ == '__main__':
    unittest.main()
