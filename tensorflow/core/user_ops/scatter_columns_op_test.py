import unittest
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes

class TestMath(tf.test.TestCase):
    scatter_columns_module = tf.load_op_library('./scatter_columns.so')
    num_cols = 1000
    num_rows = 30000

    def testEmptyIndices(self):
      with self.test_session():
        params = [0, 1, 2]
        indices = tf.constant([], dtype=tf.int32)
        num_cols = 10
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("Indices cannot be empty."):
            with tf.Session() as sess:
                sess.run(scatter)

    def testScalarParams(self):
      with self.test_session():
        params = 10
        indices = [1, 2, 3]
        num_cols = 10
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("Params must be at least a vector."):
            with tf.Session() as sess:
                sess.run(scatter)

    def testScalarIndices(self):
      with self.test_session():
        params = [1, 2, 3]
        indices = 1
        num_cols = 10
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("Indices must be a vector, but it is a: 0D Tensor."):
            with tf.Session() as sess:
                sess.run(scatter)

    def test3DParams(self):
      with self.test_session():
        params = [[[0, 1, 2]]]
        indices = [1, 2, 3]
        num_cols = 10
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("Params must be 1D or 2D but it is: 3D."):
            with tf.Session() as sess:
                sess.run(scatter)

    def test2DIndices(self):
      with self.test_session():
        params = [[0, 1, 2]]
        indices = [[1, 2, 3]]
        num_cols = 10
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("Indices must be a vector, but it is a: 2D Tensor."):
            with tf.Session() as sess:
                sess.run(scatter)

    def testVectorOutNumCols(self):
      with self.test_session():
        params = [[0, 1, 2]]
        indices = [1, 2, 3]
        num_cols = [4]
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("out_num_cols must be a scalar, but it is a: 1D Tensor"):
            with tf.Session() as sess:
                sess.run(scatter)

    def test2DPadElem(self):
      with self.test_session():
        params = [[0, 1, 2]]
        indices = [1, 2, 3]
        num_cols = 5
        pad_elem = [[0]]
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("pad_elem must be a scalar, but it is a: 2D Tensor"):
            with tf.Session() as sess:
                sess.run(scatter)

    def testNegativeIndices(self):
      with self.test_session():
        params = [0, 1, 2]
        indices = [2, -1, 0]
        num_cols = 6
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("Indices\(1\): -1 is not in range \(0, 6\]."):
            with tf.Session() as sess:
                sess.run(scatter)

    def testBadIndices(self):
      with self.test_session():
        params = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.float64)
        indices = tf.constant([2, 1, 10, 6, 5], dtype=tf.int32)
        num_cols = 7
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("Indices\(2\): 10 is not in range \(0, 7\]."):
            with tf.Session() as sess:
                sess.run(scatter)

    def testDuplicateIndices(self):
      with self.test_session():
        params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
        indices = tf.constant([0, 1, 2, 2, 4], dtype=tf.int32)
        num_cols = 5
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("Indices cannot contain duplicates. Total no. of indices: 5 != no. of unique indices: 4"):
            with tf.Session() as sess:
                sess.run(scatter)

    def testWrongOutNumCols(self):
      with self.test_session():
        params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
        indices = tf.constant([4, 3, 2, 1, 0], dtype=tf.int32)
        num_cols = 4
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("out_num_cols: 4 must be >= size of the indexed dimension of params: 5"):
            with tf.Session() as sess:
                sess.run(scatter)

    def testIncorrectIndicesSize(self):
      with self.test_session():
        params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float64)
        indices = tf.constant([11, 10, 9, 8], dtype=tf.int32)
        num_cols = 12
        pad_elem = 0
        scatter = self.scatter_columns_module.scatter_columns(params, indices, num_cols, pad_elem)
        with self.assertRaisesOpError("Size of indices: 4 and the indexed dimension of params - 5 - must be the same."):
            with tf.Session() as sess:
                sess.run(scatter)

    def test_scatter_cols(self):
        ops.RegisterShape("ScatterColumns")(common_shapes.call_cpp_shape_fn)

        def test(params, indices, num_cols, pad_elem, dtype, true_output, large_case=False):

            with self.subTest(params=params, indices=indices,
                              num_cols=num_cols, pad_elem=pad_elem,
                              dtype=dtype, large_case=large_case):
                if dtype == bool:
                    row1 = row2 = row3 = 1
                else:
                    row1 = 1
                    row2 = 0
                    row3 = -1

                p1d = tf.constant(params, dtype=dtype)
                p2d1 = tf.constant(np.array([np.array(params)]), dtype=dtype)

                if not large_case:
                    p2d2 = tf.constant(np.array([np.array(params) * row1,
                                                 np.array(params) * row2,
                                                 np.array(params) * row3]), dtype=dtype)
                else:
                    params_matrix = np.empty([self.num_rows, self.num_cols])
                    params_row = np.array(params)
                    for i in range(0, self.num_rows):
                        params_matrix[i,:] = params_row * (i+1)
                    p2d2 = tf.constant(params_matrix, dtype=dtype)

                    # For testing only the overhead time
                    #p2d2 = tf.constant(params, dtype=dtype)

                ind_32 = tf.constant(indices, dtype=tf.int32)
                ind_64 = tf.constant(indices, dtype=tf.int64)

                op1d = self.scatter_columns_module.scatter_columns(p1d, ind_64, num_cols, pad_elem)
                op2d1 = self.scatter_columns_module.scatter_columns(p2d1, ind_32, num_cols, pad_elem)
                op2d2 = self.scatter_columns_module.scatter_columns(p2d2, ind_64, num_cols, pad_elem)

                with tf.Session() as sess:
                    out1d = sess.run(op1d)
                    out2d1 = sess.run(op2d1)
                    out2d2 = sess.run(op2d2)

                np.testing.assert_array_almost_equal(out1d, true_output)
                self.assertEqual(dtype.as_numpy_dtype, out1d.dtype)

                true_output_2d1 = [np.array(true_output)]
                np.testing.assert_array_almost_equal(out2d1, true_output_2d1)
                self.assertEqual(dtype.as_numpy_dtype, out2d1.dtype)

                if not large_case:
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
                else:
                    params_matrix = np.empty([self.num_rows, self.num_cols*2])
                    true_output_row = np.array(true_output)
                    ind = np.array(indices)
                    for i in range(0, self.num_rows):
                        params_matrix[i,:] = true_output_row
                        params_matrix[i,ind] = true_output_row[ind] * (i+1)
                    true_output_2d2 = params_matrix

                    # For testing only the overhead time
                    #true_output_2d2 = true_output

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

        # Large case for performance test
        true_output = list(np.arange(self.num_cols, 0, -0.5))
        true_output[1:self.num_cols*2:2] = list(np.full((self.num_cols), pad_elem, np.int64))
        test(list(range(1, self.num_cols+1)), # [1, 2, 3, ..., n-1, n]
             list(range((self.num_cols*2)-2, -1, -2)), # [2n-2, n-4, n-6, ..., 2, 0]
             self.num_cols*2,
             pad_elem,
             tf.int64,
             true_output, # [n, pad_elem, n-1, pad_elem, n-2, ..., 2, pad_elem, 1, pad_elem]
             True)


if __name__ == '__main__':
    unittest.main()
