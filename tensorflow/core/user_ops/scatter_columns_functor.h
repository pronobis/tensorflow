#ifndef TENSORFLOW_USEROPS_SCATTER_COLUMNS_FUNCTOR_H_
#define TENSORFLOW_USEROPS_SCATTER_COLUMNS_FUNCTOR_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

using namespace std;

namespace tensorflow {
  typedef Eigen::ThreadPoolDevice CPUDevice;

  namespace functor {

    //--Helper method to copy using memcpy()--//
    template <typename T, typename IndT>
    IndT CountAndCopy(const typename TTypes<T>::ConstMatrix& params,
                      const IndT& out_num_cols,
                      const std::vector<IndT>& out_indices,
                      const std::vector<int>& cons_pad_cols,
                      const gtl::InlinedVector<T, 4>& pad_elem_vec,
                      const int64& params_rows,
                      typename TTypes<T>::Matrix& output) {

      //--Mem-copy columns, bunching consecutive padding columns together, one row at a time--//
      for(int row = 0; row < params_rows; row++ )
      {
        for(int col = 0; col < out_num_cols;)
        {
          //--If not the final copy--//
          if (col + 1 < out_num_cols)
          {
            //--Prefetch the next destination (output) memory address--//
            port::prefetch<port::PREFETCH_HINT_T0>(&output(row, (col + 1)));

            //--If the next column is not a padding column--//
            if(out_indices[col+1] >= 0)
            {
              //--Prefetch the next source (params) memory address--//
              port::prefetch<port::PREFETCH_HINT_T0>(&params(row, out_indices[col+1]));
            }
          }

          if(out_indices[col] >= 0)
          {
            //--Mem-copy a single non-padding element from params tensor--//
            memcpy(&output(row, col), &params(row, out_indices[col]), sizeof(T));
            ++col;
          }
          else
          {
            //--Mem-copy columns of padding elements (per row) from padding element vector--//
            memcpy(&output(row, col), &pad_elem_vec[0], (cons_pad_cols[col] * sizeof(T)));
            col += cons_pad_cols[col];
          }
        }
      }

      return -1;
    }

    template <typename T, typename IndT>
    struct ScatterColumnsFunctorCPU {
      int64 operator()(const typename TTypes<T>::ConstMatrix& params,
                       const IndT& out_num_cols,
                       const std::vector<IndT>& out_indices,
                       const std::vector<int>& cons_pad_cols,
                       const gtl::InlinedVector<T, 4>& pad_elem_vec,
                       const int64& params_rows,
                       typename TTypes<T>::Matrix& output) {

        return CountAndCopy<T, IndT>(params, out_num_cols, out_indices, cons_pad_cols, pad_elem_vec, params_rows, output);
      }
    };

    template <typename Device, typename T, typename IndT>
    struct ScatterColumnsFunctor {
      int64 operator()(const Device& dvc,
                       const typename TTypes<T>::ConstMatrix& params,
                       const IndT& out_num_cols,
                       const std::vector<IndT>& out_indices,
                       const std::vector<int>& cons_pad_cols,
                       const gtl::InlinedVector<T, 4>& pad_elem_vec,
                       const int64& params_rows,
                       typename TTypes<T>::Matrix& output) {
        return ScatterColumnsFunctorCPU<T, IndT>()(params, out_num_cols, out_indices, cons_pad_cols, pad_elem_vec, params_rows, output);
      }
    };

  }  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_USEROPS_SCATTER_COLUMNS_FUNCTOR_H_
