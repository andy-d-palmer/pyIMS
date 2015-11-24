import unittest

import numpy as np

from ..image_measures.level_sets_measure import measure_of_chaos, _nan_to_zero


class MeasureOfChaosTest(unittest.TestCase):
    def test__nan_to_zero_with_ge_zero(self):
        ids = (
            np.zeros(1),
            np.ones(range(1, 10)),
            np.arange(1024 * 1024)
        )
        for id_ in ids:
            before = id_.copy()
            _nan_to_zero(id_)
            np.testing.assert_array_equal(before, id_)

    def test__nan_to_zero_with_negatives(self):
        negs = (
            np.array([-1]),
            -np.arange(1, 1024 * 1024 + 1).reshape((1024, 1024)),
            np.linspace(0, -20, 201)
        )
        for neg in negs:
            sh = neg.shape
            _nan_to_zero(neg)
            np.testing.assert_array_equal(neg, np.zeros(sh))

if __name__ == '__main__':
    unittest.main()
