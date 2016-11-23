
import unittest
import tensorflow as tf
import numpy as np


class TestMath(unittest.TestCase):
    def test_scatter_cols_errors(self):
        # TODO
        pass

    def test_scatter_cols(self):
        def test(params, indices, num_cols, pad_elem, dtype, true_output):

            #scatter_columns_module = tf.load_op_library('/home/nash/.dot/modules/50_dot-module-libspn/tmp/tensorflow/tensorflow/core/user_ops/scatter_columns.so')
            scatter_columns_module = tf.load_op_library('./scatter_columns.so')

            with self.subTest(params=params, indices=indices,
                              num_cols=num_cols, pad_elem=pad_elem,
                              dtype=dtype):
                if dtype == bool:
                    row2 = row3 = 1
                else:
                    row2 = 2
                    row3 = 3
                p1d = tf.constant(params, dtype=dtype)
                p2d1 = tf.constant(np.array([np.array(params)]), dtype=dtype)
                p2d2 = tf.constant(np.array([np.array(params),
                                             np.array(params) * row2,
                                             np.array(params) * row3]), dtype=dtype)

                op1d = scatter_columns_module.scatter_columns(p1d, indices, num_cols, pad_elem)
                op2d1 = scatter_columns_module.scatter_columns(p2d1, indices, num_cols, pad_elem)
                op2d2 = scatter_columns_module.scatter_columns(p2d2, indices, num_cols, pad_elem)

                with tf.Session() as sess:
                    out1d = sess.run(op1d)
                    out2d1 = sess.run(op2d1)
                    out2d2 = sess.run(op2d2)

                # print ("out1d: \n", out1d)
                # print ("out2d1: \n", out2d1)
                # print ("out2d2: \n", out2d2)

                np.testing.assert_array_almost_equal(out1d, true_output)
                self.assertEqual(dtype.as_numpy_dtype, out1d.dtype)

                true_output_2d1 = [np.array(true_output)]
                np.testing.assert_array_almost_equal(out2d1, true_output_2d1)
                self.assertEqual(dtype.as_numpy_dtype, out2d1.dtype)

                true_output_2d2 = [np.array(true_output),
                                   np.array(true_output) * row2,
                                   np.array(true_output) * row3]
                np.testing.assert_array_almost_equal(out2d2, true_output_2d2)
                self.assertEqual(dtype.as_numpy_dtype, out2d2.dtype)


        pad_elem = 0

        # Single column output
        # float
        test([10],
             [0],
             1,
             pad_elem,
             tf.float32,
             [10.0])
        test([10],
             [0],
             1,
             pad_elem,
             tf.float64,
             [10.0])
        # int
        test([1111],
             [0],
             1,
             pad_elem,
             tf.int32,
             [1111])
        test([111111],
             [0],
             1,
             pad_elem,
             tf.int64,
             [111111])
        # bool
        test([True],
             [0],
             1,
             False,
             tf.bool,
             [True])

        # Multi-column output, single-column input
        test([10],
             [1],
             4,
             pad_elem,
             tf.float32,
             [0.0, 10.0, 0.0, 0.0])
        test([10],
             [0],
             4,
             pad_elem,
             tf.float64,
             [10.0, 0.0, 0.0, 0.0])
        # int
        test([1111],
             [2],
             5,
             pad_elem,
             tf.int32,
             [0, 0, 1111, 0, 0])
        test([111111],
             [4],
             5,
             pad_elem,
             tf.int64,
             [0, 0, 0, 0, 111111])
        # bool
        test([True],
             [3],
             5,
             False,
             tf.bool,
             [False, False, False, True, False])

        # Multi-column output, multi-column input
        test([0.1101, 0.2202, 0.3303, 0.4404, 0.5505, 0.6606],
             [5, 3, 9, 1, 8, 6],
             10,
             pad_elem,
             tf.float32,
             [0.0, 0.4404, 0.0, 0.2202, 0.0, 0.1101, 0.6606, 0.0, 0.5505, 0.3303])
        test([0.11001, 0.22002, 0.33003, 0.44004, 0.55005, 0.66006],
             [5, 3, 9, 1, 8, 6],
             10,
             pad_elem,
             tf.float64,
             [0.0, 0.44004, 0.0, 0.22002, 0.0, 0.11001, 0.66006, 0.0, 0.55005, 0.33003])
        # int
        test([111, 222, 333, 444, 555, 666],
             [7, 1, 5, 2, 0, 4],
             10,
             pad_elem,
             tf.int32,
             [555, 222, 444, 0, 666, 333, 0, 111, 0, 0])
        test([111111, 222222, 333333, 444444, 555555, 666666],
             [7, 1, 5, 2, 3, 4],
             10,
             pad_elem,
             tf.int64,
             [0, 222222, 444444, 555555, 666666, 333333, 0, 111111, 0, 0])


if __name__ == '__main__':
    unittest.main()
