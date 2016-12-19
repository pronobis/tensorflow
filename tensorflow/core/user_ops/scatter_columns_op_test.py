import unittest
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes

class TestMath(unittest.TestCase):
    def test_scatter_cols_errors(self):
        # TODO
        pass

    scatter_columns_module = tf.load_op_library('./scatter_columns.so')

    def test_scatter_cols(self):
        ops.RegisterShape("ScatterColumns")(common_shapes.call_cpp_shape_fn)

        def test(params, indices, num_cols, pad_elem, dtype, true_output):

            with self.subTest(params=params, indices=indices,
                              num_cols=num_cols, pad_elem=pad_elem,
                              dtype=dtype):
                if dtype == bool:
                    row1 = row2 = row3 = 1
                else:
                    row1 = 1
                    row2 = 0
                    row3 = -1

                p1d = tf.constant(params, dtype=dtype)
                p2d1 = tf.constant(np.array([np.array(params)]), dtype=dtype)
                p2d2 = tf.constant(np.array([np.array(params) * row1,
                                             np.array(params) * row2,
                                             np.array(params) * row3]), dtype=dtype)

                op1d = self.scatter_columns_module.scatter_columns(p1d, indices, num_cols, pad_elem)
                op2d1 = self.scatter_columns_module.scatter_columns(p2d1, indices, num_cols, pad_elem)
                op2d2 = self.scatter_columns_module.scatter_columns(p2d2, indices, num_cols, pad_elem)

                with tf.Session() as sess:
                    out1d = sess.run(op1d)
                    out2d1 = sess.run(op2d1)
                    out2d2 = sess.run(op2d2)

                np.testing.assert_array_almost_equal(out1d, true_output)
                self.assertEqual(dtype.as_numpy_dtype, out1d.dtype)

                true_output_2d1 = [np.array(true_output)]
                np.testing.assert_array_almost_equal(out2d1, true_output_2d1)
                self.assertEqual(dtype.as_numpy_dtype, out2d1.dtype)

                r_1 = np.array(true_output)
                r_2 = np.array(true_output)
                r_3 = np.array(true_output)
                ind = np.array(indices)

                r_1[ind] = r_1[ind] * row1
                r_2[ind] = r_2[ind] * row2
                r_3[ind] = r_3[ind] * row3

                true_output_2d2 = [r_1,
                                   r_2,
                                   r_3]
                np.testing.assert_array_almost_equal(out2d2, true_output_2d2)
                self.assertEqual(dtype.as_numpy_dtype, out2d2.dtype)


        float_val = 1.23456789
        int_val = 123456789
        int_32_upper = 2147483647
        int_64_upper = 9223372036854775807

        pad_elem = 333

        # Single column output
        # float
        test([float_val],
             [0],
             1,
             pad_elem,
             tf.float32,
             [float_val])
        test([float_val],
             [0],
             1,
             pad_elem,
             tf.float64,
             [float_val])

        # int
        test([int_32_upper],
             [0],
             1,
             pad_elem,
             tf.int32,
             [int_32_upper])
        test([int_64_upper],
             [0],
             1,
             pad_elem,
             tf.int64,
             [int_64_upper])

        # bool
        test([True],
             [0],
             1,
             False,
             tf.bool,
             [True])

        # Multi-column output, single-column input
        test([float_val],
             [1],
             4,
             pad_elem,
             tf.float32,
             [pad_elem, float_val, pad_elem, pad_elem])
        test([float_val],
             [0],
             4,
             pad_elem,
             tf.float64,
             [float_val, pad_elem, pad_elem, pad_elem])

        # int
        test([int_32_upper],
             [2],
             5,
             pad_elem,
             tf.int32,
             [pad_elem, pad_elem, int_32_upper, pad_elem, pad_elem])
        test([int_64_upper],
             [4],
             5,
             pad_elem,
             tf.int64,
             [pad_elem, pad_elem, pad_elem, pad_elem, int_64_upper])

        # bool
        test([True],
             [3],
             5,
             False,
             tf.bool,
             [False, False, False, True, False])

        # Multi-column output, multi-column input
        # float
        # No consecutive padded columns
        test([float_val, float_val*2, float_val*3, float_val*4, float_val*5, float_val*6],
             [5, 3, 9, 1, 8, 6],
             10,
             pad_elem,
             tf.float32,
             [pad_elem, float_val*4, pad_elem, float_val*2, pad_elem, float_val, float_val*6, pad_elem, float_val*5, float_val*3])
        # Consecutive padded columns in the end
        test([float_val, float_val*2, float_val*3, float_val*4, float_val*5, float_val*6],
             [5, 3, 9, 1, 8, 6],
             15,
             pad_elem,
             tf.float64,
             [pad_elem, float_val*4, pad_elem, float_val*2, pad_elem, float_val, float_val*6, pad_elem, float_val*5, float_val*3, pad_elem, pad_elem, pad_elem, pad_elem, pad_elem])

        # int
        # Consecutive padded columns in the beginning
        test([int_val, int_val*2, int_val*3, int_val*4, int_val*5, int_val*6, int_val*7, int_val*8, int_val*9],
             [7, 14, 5, 9, 10, 11, 6, 3, 8],
             15,
             pad_elem,
             tf.int32,
             [pad_elem, pad_elem, pad_elem, int_val*8, pad_elem, int_val*3, int_val*7, int_val, int_val*9, int_val*4, int_val*5, int_val*6, pad_elem, pad_elem, int_val*2])
        # Consecutive padded columns in the middle
        test([int_val, int_val*2, int_val*3, int_val*4, int_val*5, int_val*6, int_val*7, int_val*8, int_val*9],
             [13, 8, 4, 1, 2, 3, 11, 0, 9],
             15,
             pad_elem,
             tf.int64,
             [int_val*8, int_val*4, int_val*5, int_val*6, int_val*3, pad_elem, pad_elem, pad_elem, int_val*2, int_val*9, pad_elem, int_val*7, pad_elem, int_val, pad_elem])

        # bool
        # No padded columns
        test([True, False, False, True],
             [2, 1, 3, 0],
             4,
             False,
             tf.bool,
             [True, False, True, False])
        # Consecutive padded columns in the beginning, middle and end
        test([True, False, False, True],
             [5, 11, 3, 9],
             15,
             False,
             tf.bool,
             [False, False, False, False, False, True, False, False, False, True, False, False, False, False, False])



if __name__ == '__main__':
    unittest.main()
