"""Metadata navigation must not break the diagnostic's sealed role boundary."""
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_experiment_navigation import ROOT, json_record


class SealedNavigationChecks(unittest.TestCase):
    def test_sealed_payload_is_never_opened(self):
        for name in ("sealed_lock.json", "SEALED_LOCK.JSON"):
            path = ROOT / "exp" / "synthetic-navigation-fixture" / name
            with mock.patch.object(Path, "read_text", side_effect=AssertionError("sealed payload opened")):
                record = json_record(path, 123)
            self.assertEqual(record["read_status"], "sealed_payload_not_loaded")
            self.assertEqual(record["bytes"], 123)
            self.assertIsNone(record["status"])

    def test_normal_record_fields_and_read_cap_remain(self):
        path = ROOT / "exp" / "synthetic-navigation-fixture" / "report.json"
        with mock.patch.object(Path, "read_text", return_value='{"status":"completed","seed":2022,"profile":"fixture"}') as reader:
            record = json_record(path, 80)
            reader.assert_called_once()
        self.assertEqual((record["status"], record["seed"], record["profile"]), ("completed", 2022, "fixture"))
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("oversized payload opened")):
            capped = json_record(path, 33 * 1024 * 1024)
        self.assertIn("capped", capped["read_status"])


if __name__ == "__main__":
    unittest.main()
