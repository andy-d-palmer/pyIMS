import unittest

import numpy as np

from ..image_measures.level_sets_measure import measure_of_chaos, _nan_to_zero, _quantile_threshold


class MeasureOfChaosTest(unittest.TestCase):
    def test_measure_of_chaos(self):
        np.random.seed(0)
        a = np.random.normal(loc=0.5, scale=0.01, size=64).reshape((8, 8))
        a[0, :-1] = 0
        a[1, 0] = 0
        a[3:6, :] = 0
        a[6, :4] = 0
        self.assertEqual(0.03625, measure_of_chaos(a, 200)[0])

    def test__nan_to_zero_with_ge_zero(self):
        ids = (
            np.zeros(1),
            np.ones(range(1, 10)),
            np.arange(1024 * 1024)
        )
        for id_ in ids:
            before = id_.copy()
            notnull = _nan_to_zero(id_)
            np.testing.assert_array_equal(before, id_)
            np.testing.assert_array_equal(notnull, before != 0)

    def test__nan_to_zero_with_negatives(self):
        negs = (
            np.array([-1]),
            np.array([np.nan])
            - np.arange(1, 1024 * 1024 + 1).reshape((1024, 1024)),
            np.linspace(0, -20, 201)
        )
        for neg in negs:
            sh = neg.shape
            expected_notnull = np.zeros(sh).astype(np.bool_)
            actual_notnull = _nan_to_zero(neg)
            np.testing.assert_array_equal(neg, np.zeros(sh))
            np.testing.assert_array_equal(actual_notnull, expected_notnull)

    def test__nan_to_zero_with_mixed(self):
        test_cases = (
            (np.array([-1, np.nan, 1e6, -1e6]), np.array([0, 0, 1e6, 0])),
            (np.arange(-2, 7).reshape((3, 3)), np.array([[0, 0, 0], np.arange(1, 4), np.arange(4, 7)])),
        )
        for input, expected in test_cases:
            _nan_to_zero(input)
            np.testing.assert_array_equal(input, expected)

    def test__nan_to_zero_with_empty(self):
        in_ = None
        self.assertRaises(AttributeError, _nan_to_zero, in_)
        self.assertIs(in_, None)

        in_ = []
        self.assertRaises(TypeError, _nan_to_zero, in_)
        self.assertEqual(in_, [])

        in_ = np.array([])
        notnull = _nan_to_zero(in_)
        self.assertSequenceEqual(in_, [])
        self.assertSequenceEqual(notnull, [])

    def test__quantile_threshold_ValueError(self):
        test_cases = (
            (np.arange(0), np.arange(0, dtype=np.bool_), -37),
            (np.arange(0), np.arange(0, dtype=np.bool_), -4.4),
            (np.arange(0), np.arange(0, dtype=np.bool_), 101)
        )
        for args in test_cases:
            self.assertRaises(ValueError, _quantile_threshold, *args)

    def test__quantile_threshold_trivial(self):
        test_cases = (
            ((np.arange(10), np.ones(10, dtype=np.bool_), 100), (np.arange(10), 9)),
            (
                (np.arange(101, dtype=np.float32), np.ones(101, dtype=np.bool_), 100. / 3),
                (np.concatenate((np.arange(34), np.repeat(100. / 3, 67))), 100. / 3),
            ),
            ((np.arange(20), np.repeat([True, False], 10), 100), (np.concatenate((np.arange(10), np.repeat(9, 10))), 9)),
        )
        for args, expected in test_cases:
            im_in = args[0]
            im_expected, q_expected = expected
            q_actual = _quantile_threshold(im_in, *args[1:])
            self.assertAlmostEqual(q_expected, q_actual, delta=1e-7)
            np.testing.assert_array_almost_equal(im_in, im_expected, decimal=6)



        if __name__ == '__main__':
            unittest.main()
