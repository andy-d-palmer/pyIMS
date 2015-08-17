__author__ = 'intsco'

import cPickle
from engine.pyIMS.image_measures.level_sets_measure import measure_of_chaos_dict
from unittest import TestCase
import unittest
from os.path import join, realpath, dirname


class MeasureOfChaosDictTest(TestCase):

    def setUp(self):
        self.rows, self.cols = 65, 65
        self.input_fn = join(dirname(realpath(__file__)), 'data/measure_of_chaos_dict_test_input.pkl')
        with open(self.input_fn) as f:
            self.input_data = cPickle.load(f)

    def testMOCBoundaries(self):
        for img_d in self.input_data:
            if len(img_d) > 0:
                assert 0 <= measure_of_chaos_dict(img_d, self.rows, self.cols) <= 1

    def testEmptyInput(self):
        # print measure_of_chaos_dict({}, self.cols, self.cols)
        self.assertRaises(Exception, measure_of_chaos_dict, {}, self.cols, self.cols)
        self.assertRaises(Exception, measure_of_chaos_dict, None, self.cols, self.cols)
        self.assertRaises(Exception, measure_of_chaos_dict, (), self.cols, self.cols)
        self.assertRaises(Exception, measure_of_chaos_dict, [], self.cols, self.cols)

    def testMaxInputDictKeyVal(self):
        max_key_val = self.rows * self.cols - 1
        self.assertRaises(Exception, measure_of_chaos_dict, {max_key_val + 10: 1}, self.rows, self.cols)


if __name__ == '__main__':
    unittest.main()
