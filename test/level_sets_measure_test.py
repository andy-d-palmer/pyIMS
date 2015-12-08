import itertools
import unittest

import numpy as np

from ..image_measures.level_sets_measure import measure_of_chaos, _nan_to_zero, _quantile_threshold, _interpolate, \
    _level_sets, _measure


class MeasureOfChaosTest(unittest.TestCase):
    def test_measure_of_chaos(self):
        np.random.seed(0)
        a = np.random.normal(loc=0.5, scale=0.01, size=64).reshape((8, 8))
        a[0, :-1] = 0
        a[1, 0] = 0
        a[3:6, :] = 0
        a[6, :4] = 0
        self.assertEqual(0.03625, measure_of_chaos(a, 200))

    def test_measure_of_chaos_ValueError(self):
        valid_ims = (np.ones((3, 3)),)
        valid_nlevelss = (4,)
        valid_interps = (True,)
        valid_q_vals = (99.,)
        invalid_nlevelss = (0, -3)
        invalid_interps = ('foo', 7)
        invalid_q_vals = (-7, 108.3)
        test_cases = itertools.chain(itertools.product(valid_ims, valid_nlevelss, valid_interps, invalid_q_vals),
                                     itertools.product(valid_ims, valid_nlevelss, invalid_interps, valid_q_vals),
                                     itertools.product(valid_ims, invalid_nlevelss, valid_interps, valid_q_vals))
        for args in test_cases:
            self.assertRaises(ValueError, measure_of_chaos, *args)

    def test_measure_of_chaos_does_not_overwrite(self):
        im_before = np.linspace(0, 1, 100).reshape((10, 10))
        im_after = np.copy(im_before)
        measure_of_chaos(im_after, 10, True, 50., overwrite=False)
        np.testing.assert_array_equal(im_before, im_after)

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
            np.array([np.nan]),
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
        for input_, expected in test_cases:
            _nan_to_zero(input_)
            np.testing.assert_array_equal(input_, expected)

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
            (
                (np.arange(20), np.repeat([True, False], 10), 100),
                (np.concatenate((np.arange(10), np.repeat(9, 10))), 9)
            ),
        )
        for args, expected in test_cases:
            im_in = args[0]
            im_expected, q_expected = expected
            q_actual = _quantile_threshold(im_in, *args[1:])
            self.assertAlmostEqual(q_expected, q_actual, delta=1e-7)
            np.testing.assert_array_almost_equal(im_in, im_expected, decimal=6)

    def test__interpolate(self):
        im_in = np.arange(900, dtype=np.float32).reshape((30, 30))
        im_in[2, 3] = np.nan
        notnull = im_in > 0

        im_out = _interpolate(im_in, notnull)

        np.testing.assert_array_almost_equal(im_in[notnull], im_out[notnull])
        self.assertAlmostEqual(im_out[0, 0], 0)
        self.assertAlmostEqual(im_out[2, 3], 63)

    def test__level_sets_ValueError(self):
        self.assertRaises(ValueError, _level_sets, np.arange(5), 0)

    def test__level_sets(self):
        test_cases = _make_level_sets_cases()
        for args, expected in test_cases:
            actual = _level_sets(*args)
            np.testing.assert_array_equal(actual, expected)

    def test__measure_ValueError(self):
        invalid_num_objs = ([], [-1], [2, -1], [2, -4, -1], [0], [1, 2, 3, 0, 1, 2], [0.999, 20])
        valid_num_objs = (range(5, 0, -1),)
        invalid_sum_notnulls = (-2.7, -1, 0,)
        valid_sum_notnulls = (15,)
        invalid_combinations = itertools.chain(itertools.product(invalid_num_objs, valid_sum_notnulls),
                                               itertools.product(valid_num_objs, invalid_sum_notnulls),
                                               itertools.product(invalid_num_objs, invalid_sum_notnulls))
        for num_objs, sum_notnull in invalid_combinations:
            self.assertRaises(ValueError, _measure, num_objs, sum_notnull)
        for num_objs, sum_notnull in itertools.product(valid_num_objs, valid_sum_notnulls):
            _measure(num_objs, sum_notnull)

    def test__measure_trivial(self):
        test_cases = (
            ((range(5), 1), 3),
            ((np.nan, 1), np.nan),
            ((range(5), np.nan), np.nan),
            (([1.1, 2.2, 3.3], 1), 5. / 3),
            ((range(5), .5), 6),
        )


def _make_level_sets_cases():
    nlevelss = (2, 5, 500)
    # for each number of levels, insert one object per level into the matrix, such that it will be dilated to a 3x3
    # square and then eroded to a single pixel
    for nlevels in nlevelss:
        # test only vertical extension:
        # . . .
        # 0 0 0
        # 1 1 1
        # 0 0 0
        # . . .
        im = np.zeros((nlevels * 4 + 3, 3))
        for i in range(nlevels):
            r = 4 * i + 1
            im[(r, r, r), (0, 1, 2)] = 1 - float(i) / nlevels
        yield ((im, nlevels), np.arange(nlevels, 0, -1))

        # test mainly vertical extension but surround with sufficient zeros
        # . . . . .
        # 0 0 0 0 0
        # 0 1 1 1 0
        # 0 0 0 0 0
        # . . . . .
        im = np.zeros((nlevels * 4 + 4, 7))
        for i in range(nlevels):
            r = 4 * i + 2
            im[(r, r, r), (2, 3, 4)] = 1 - float(i) / nlevels
        yield ((im, nlevels), np.arange(nlevels, 0, -1))

        # test both vertical and horizontal extension with surrounding zeros
        # . . . . . . .
        # 0 0 0 0 0 0 0
        # 0 0 0 1 0 0 0
        # 0 0 0 1 0 0 0
        # 0 0 1 0 1 0 0
        # 0 0 0 0 0 0 0
        # . . . . . . .
        im = np.zeros((nlevels * 6 + 6, 7))
        for i in range(nlevels):
            r = 6 * i + 3
            im[(r - 1, r, r + 1, r + 1), (3, 3, 2, 4)] = 1 - float(i) / nlevels
        yield ((im, nlevels), np.arange(nlevels, 0, -1))

    # non-monotonic case where an objects splits into two in the second level and one of them disappears in the highest
    # level
    im = np.zeros((9, 5))
    im[2, 1:4] = 1
    im[4, 1:4] = 0.4
    im[6, 1:4] = 0.6
    yield ((im, 3), [1, 2, 1])


if __name__ == '__main__':
    unittest.main()
