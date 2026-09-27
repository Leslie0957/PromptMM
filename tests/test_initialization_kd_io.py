"""Synthetic publication checks; no assets, data splits, model or GPU."""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_initialization_kd_interaction import _json


class PublicationTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows file-sharing regression')
    def test_real_reader_lock_then_release(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'telemetry.json'
            _json(target, {'old': True})
            temporary = target.with_name(target.name + '.tmp')
            temporary.write_text('{}')
            with target.open('rb') as reader:
                with self.assertRaises(OSError) as caught:
                    temporary.replace(target)
                self.assertIn(caught.exception.winerror, (5, 32, 33))
                timer = threading.Timer(0.1, reader.close)
                timer.start()
                try:
                    _json(target, {'new': True})
                finally:
                    timer.join()
            self.assertEqual(json.loads(target.read_text()), {'new': True})
            self.assertFalse(temporary.exists())

    def test_persistent_denial_is_bounded_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'telemetry.json'
            _json(target, {'old': True})
            old = target.read_bytes()
            error = PermissionError('persistent Windows denial')
            error.winerror = 5
            with patch.object(Path, 'replace', side_effect=error) as replace, \
                    patch('run_initialization_kd_interaction.time.sleep') as sleep:
                with self.assertRaises(PermissionError):
                    _json(target, {'new': True})
            self.assertEqual(replace.call_count, 11)
            self.assertEqual(sleep.call_count, 10)
            self.assertEqual(target.read_bytes(), old)
            self.assertEqual(json.loads(target.with_name(target.name + '.tmp').read_text()), {'new': True})

    def test_other_errors_propagate_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'telemetry.json'
            for error in (PermissionError('non-Windows denial'), OSError('disk failure')):
                with patch.object(Path, 'replace', side_effect=error) as replace, \
                        patch('run_initialization_kd_interaction.time.sleep') as sleep:
                    with self.assertRaises(OSError):
                        _json(target, {})
                self.assertEqual(replace.call_count, 1)
                sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
