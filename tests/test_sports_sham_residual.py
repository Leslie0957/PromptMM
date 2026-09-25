import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
sys.path.insert(0, str(ROOT / 'tools'))
from sports_sham_residual import geometry_error, sham_text
from run_sports_sham_residual_three_seed import gradient_scale


class ShamResidualTest(unittest.TestCase):
    def test_geometry_determinism_and_degenerate_rows(self):
        generator = torch.Generator().manual_seed(17)
        image = torch.randn((40, 64), generator=generator)
        text = torch.randn((40, 64), generator=generator) * 2.0
        image[0] = 0
        text[1] = 0
        text[2] = 3 * F.normalize(image[2], dim=0)
        sham, valid = sham_text(image, text, 9022022)
        again, valid_again = sham_text(image, text, 9022022)
        different, _ = sham_text(image, text, 9022023)
        self.assertTrue(torch.equal(sham, again))
        self.assertTrue(torch.equal(valid, valid_again))
        self.assertTrue(torch.equal(sham[~valid], text[~valid]))
        self.assertFalse(torch.equal(sham[valid], different[valid]))
        result = geometry_error(image, text, sham, valid)
        self.assertGreater(result['valid_rows'], 30)
        self.assertLess(max(result[key] for key in result if key.startswith('max_')), 2e-5)

    def test_zero_update_gradient_check(self):
        generator = torch.Generator().manual_seed(23)
        initial = {'user': torch.randn((5, 64), generator=generator),
                   'item': torch.randn((8, 64), generator=generator)}
        image = torch.randn((8, 64), generator=generator)
        text = torch.randn((8, 64), generator=generator)
        sham, _ = sham_text(image, text, 9022022)
        batches = [([0, 1, 2, 3], [0, 1, 2, 3], [4, 5, 6, 7]),
                   ([1, 2, 3, 4], [3, 4, 5, 6], [0, 0, 1, 2])]
        before = {k: v.clone() for k, v in initial.items()}
        result = gradient_scale(initial, image, text, sham, batches)
        self.assertEqual(result['batches'], 2)
        self.assertGreater(result['sham_over_real_rms'], 0)
        for key in initial:
            self.assertTrue(torch.equal(initial[key], before[key]))


if __name__ == '__main__':
    unittest.main()
