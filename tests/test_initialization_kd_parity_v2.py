"""Synthetic attempt routing; never starts a worker or reads real assets."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_initialization_kd_eval_parity as runner


class AttemptTests(unittest.TestCase):
    def test_only_declared_delta(self):
        old, new = runner.load_spec(), runner.load_spec('v2')
        expected = copy.deepcopy(old)
        expected['run_id'] = old['run_id'].replace('_v1', '_v2')
        expected['output_namespace'] = old['output_namespace'].replace('_v1', '_v2')
        expected['launch_command'] += ' --attempt v2'
        self.assertEqual(new, expected)
        with self.assertRaises(ValueError):
            runner.load_spec('v3')

    def test_parent_forwards_attempt_and_refuses_reuse(self):
        spec = runner.load_spec('v2')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner, 'load_spec', return_value=spec) as load, \
                    patch.object(runner, '_source_gate', return_value='synthetic'), \
                    patch.object(runner, '_supervise') as supervise, \
                    patch.object(sys, 'argv', ['parity', '--attempt', 'v2']):
                runner.main()
                load.assert_called_once_with('v2')
                command = supervise.call_args.kwargs['command']
                self.assertEqual(command[-4:], ['--attempt', 'v2', '--worker', str(root / spec['output_namespace'])])
                manifest = json.loads((root / spec['output_namespace'] / 'launch_manifest.json').read_text())
                self.assertEqual(manifest['config'], spec)
                with self.assertRaises(FileExistsError):
                    runner.main()
                self.assertEqual(supervise.call_count, 1)

    def test_worker_uses_v2_spec_and_output(self):
        spec = runner.load_spec('v2')
        output = runner.ROOT / spec['output_namespace']
        with patch.object(runner, 'load_spec', return_value=spec) as load, \
                patch.object(runner, '_source_gate', return_value='synthetic'), \
                patch.object(runner, 'bind_worker', return_value={'source_commit': 'synthetic'}) as bind, \
                patch.object(runner, 'worker') as worker, \
                patch.object(sys, 'stdin') as stdin, \
                patch.object(sys, 'argv', ['parity', '--attempt', 'v2', '--worker', str(output)]):
            stdin.isatty.return_value = False
            stdin.readline.return_value = 'synthetic-token\n'
            runner.main()
            load.assert_called_once_with('v2')
            self.assertEqual(bind.call_args.args[:3], (output, output, spec))
            worker.assert_called_once_with(spec, output, {'source_commit': 'synthetic'})


if __name__ == '__main__':
    unittest.main()
