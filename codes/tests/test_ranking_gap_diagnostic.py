import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from initialization_kd_fast_eval import exact_topk, reference_topk
from ranking_gap_diagnostic import decompose, thirds


class RankingGapTests(unittest.TestCase):
    def test_filter_tie_and_partition(self):
        scores = np.array([.5, .9, .9, .7, .5, .9], dtype=np.float32)
        self.assertEqual(exact_topk(scores, {1}, 3), [2, 5, 3])
        self.assertEqual(exact_topk(scores, {1}, 3), reference_topk(scores, {1}, 3))
        for start in (0, 2, 4):
            self.assertEqual(exact_topk(scores[start:start + 2], set(), 1),
                             reference_topk(scores[start:start + 2], set(), 1))

    def test_decomposition_and_groups(self):
        val = csr_matrix((np.ones(4), ([0, 0, 1, 1], [0, 2, 1, 3])), shape=(2, 6))
        teacher = np.array([[0, 1], [1, 2]], dtype=np.int32)
        student = np.array([[2, 1], [3, 2]], dtype=np.int32)
        arrays, means, user_rows, item_rows = decompose(
            teacher, student, val, np.array([0, 1]),
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8))
        self.assertAlmostEqual(means['g'], .5)
        self.assertAlmostEqual(means['l'], .5)
        self.assertAlmostEqual(means['N'], 0)
        self.assertAlmostEqual(sum(row['G_contribution'] for row in item_rows), means['g'])
        self.assertEqual(sum(row['users'] for row in user_rows), 2)
        self.assertEqual(arrays['teacher_only_user'].tolist(), [1, 1])
        self.assertEqual(thirds(np.array([2, 0, 2, 1, 0, 1])).tolist(), [2, 0, 2, 1, 0, 1])


if __name__ == '__main__':
    unittest.main()
