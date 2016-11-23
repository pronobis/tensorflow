
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



# scatter_columns_module = tf.load_op_library('./scatter_columns.so')
#
# sess = tf.Session()
#
# with tf.Session(''):
#     print("Scattered columns: \n", sess.run(scatter_columns_module.scatter_columns([[0.0100001, 0.0200001, 0.0300001, 0.0400001, 0.0500001, 0.0600001],
#                                                                                  [0.1100001, 0.1200001, 0.1300001, 0.1400001, 0.1500001, 0.1600001],
#                                                                                  [0.2100001, 0.2200001, 0.2300001, 0.2400001, 0.2500001, 0.2600001],
#                                                                                  [0.3100001, 0.3200001, 0.3300001, 0.3400001, 0.3500001, 0.3600001],
#                                                                                  [0.4100001, 0.4200001, 0.4300001, 0.4400001, 0.4500001, 0.4600001]], [5, 1, 2, 3, 4, 0], 10, 1.23)))
#                                                                                  #[0.4101, 0.4201, 0.43000001, 0.44000001, 0.45000001, 0.46000000]], [2, 3, 4, 3, 2, 0, 1, 2, 3, 4, 5, 2, 4, 2, 3, 4])))
#
# with tf.Session(''):
#     print("Scattered columns: \n", sess.run(scatter_columns_module.scatter_columns([0.0100001, 0.0200001, 0.0300001, 0.0400001, 0.0500001, 0.0600001], [5, 1, 2, 3, 4, 0], 10, 1.23)))
#
#
# with tf.Session(''):
#     print("Scattered columns: \n", sess.run(scatter_columns_module.scatter_columns([[101101101101101101, 102102102102102102, 103103103103103103, 104104104104104104, 105105105105105105, 106106106106106106],
#                                                                                  [111111111111111111, 112112112112112112, 113113113113113113, 114114114114114114, 115115115115115115, 116116116116116116],
#                                                                                  [121121121121121121, 122122122122122122, 123123123123123123, 124124124124124124, 125125125125125125, 126126126126126126],
#                                                                                  [131131131131131131, 132132132132132132, 133133133133133133, 134134134134134134, 135135135135135135, 136136136136136136],
#                                                                                  [141141141141141141, 142142142142142142, 143143143143143143, 144144144144144144, 145145145145145145, 146146146146146146]], [0, 1, 2, 3, 4, 5], 10, 123)))
#
# with tf.Session(''):
#     print("Scattered columns: \n", sess.run(scatter_columns_module.scatter_columns([101101101101101101, 102102102102102102, 103103103103103103, 104104104104104104, 105105105105105105, 106106106106106106], [5, 4, 3, 2, 1, 0], 10, 123)))
#
# with tf.Session(''):
#     print("Scattered columns: \n", sess.run(scatter_columns_module.scatter_columns([[True, True, True, True, False, False, False, False],
#                                                                                  [True, True, False, False, True, True, False, False],
#                                                                                  [True, False, True, False, True, False, True, False]], [1, 3, 5, 7, 6, 4, 2, 0], 10, False)))
#
# with tf.Session(''):
#     print("Scattered columns: \n", sess.run(scatter_columns_module.scatter_columns([True, True, True, True, False, False, False, False], [1, 3, 5, 7, 6, 4, 2, 0], 10, False)))
#
#
# params = [1, 2, 3]
# indices = [9, 7, 5]
# num_cols = 10
# pad_elem = 123.456
# dtype = tf.float32
# true_output = [3.0, 2.0, 1.0]
#
# p1d = tf.constant(params, dtype=dtype)
# p2d1 = tf.constant(np.array([np.array(params)]), dtype=dtype)
# p2d2 = tf.constant(np.array([np.array(params), np.array(params) * 2, np.array(params) * 3]), dtype=dtype)
#
# indices = np.array(indices, dtype=np.int32)
#
#
# with tf.Session(''):
#     print("Scattered columns: \n", sess.run(scatter_columns_module.scatter_columns(p1d, indices, num_cols, pad_elem)))
#
#
# with tf.Session(''):
#     print("Scattered columns: \n", sess.run(scatter_columns_module.scatter_columns(p2d1, indices, num_cols, pad_elem)))
#
# with tf.Session(''):
#     print("Scattered columns: \n", sess.run(scatter_columns_module.scatter_columns(p2d2, indices, num_cols, pad_elem)))
