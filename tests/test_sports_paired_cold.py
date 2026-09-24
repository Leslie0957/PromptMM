import sys
from pathlib import Path

import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from sports_paired_cold import PairedColdSession
from td_distill_model_no_projection import TDDistillNoProjectionModel


class PairedColdTest(unittest.TestCase):
    def test_exact_initial_and_triplet_replay(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            full_model = TDDistillNoProjectionModel(4, 5, 2, 2)
            full = PairedColdSession(tmp_path, 7, 'full', 2, 2, 3)
            initial = full.apply_initial(full_model)
            samples = [([0, 1, 2], [1, 2, 3], [4, 4, 4]),
                       ([1, 2, 3], [2, 3, 4], [0, 0, 0])]
            for epoch in range(2):
                for batch in range(2):
                    self.assertEqual(full.sample(epoch, batch, lambda b=batch: samples[b]), samples[batch])
            tape = full.finish()
            image_model = TDDistillNoProjectionModel(4, 5, 2, 2, item_head_names=('image',))
            image = PairedColdSession(tmp_path, 7, 'image_matched', 2, 2, 3)
            self.assertEqual(image.apply_initial(image_model), initial)
            self.assertTrue(torch.equal(full_model.user_id_embedding.weight, image_model.user_id_embedding.weight))
            self.assertTrue(torch.equal(full_model.item_id_embedding.weight, image_model.item_id_embedding.weight))
            for epoch in range(2):
                for batch in range(2):
                    replay = image.sample(epoch, batch, lambda: self.fail('Replay sampled new triplets'))
                    self.assertEqual([[int(x) for x in part] for part in replay],
                                     [list(part) for part in samples[batch]])
                    del replay
            self.assertEqual(image.finish(), tape)
            with self.assertRaises(FileExistsError):
                PairedColdSession(tmp_path, 7, 'full', 2, 2, 3)


if __name__ == '__main__':
    unittest.main()
