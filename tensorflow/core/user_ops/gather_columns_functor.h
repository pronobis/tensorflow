#ifndef TENSORFLOW_USEROPS_GATHER_COLUMNS_FUNCTOR_H_
#define TENSORFLOW_USEROPS_GATHER_COLUMNS_FUNCTOR_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "../kernels/bounds_check.h" //--TODO: Currently <#include "tensorflow/core/kernels/bounds_check.h"> gives an error. Need to check and fix.
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
  typedef Eigen::ThreadPoolDevice CPUDevice;

  namespace functor {

    //--Helper method to copy using memcpy()--//
    template <typename T, typename IndT>
    IndT CountAndCopy(typename TTypes<T>::ConstMatrix params,
                            typename TTypes<IndT>::ConstFlat indices,
                            int64 params_rows,
                            int64 params_cols,
                            typename TTypes<T>::Matrix output) {

      const int64 indices_size = indices.dimension(0);

      std::vector<int> cons_cols_counter(indices_size, 1);

      if(indices_size > 1)
      {
        //--Group consecutive columns together--//
        //--E.g.:     params_size = 10
        //--          indices = [7, 8, 9, 2, 0, 4, 5, 3, 1, 5, 6, 7]
        //--cons_cols_counter = [3, 2, 1, 1, 1, 2, 1, 1, 1, 3, 2, 1]

        int cols;
        for(int c=0; c < indices_size; c++)
        {
          //--Check indices[i] ∈ (0, params_cols]--//
          if (!FastBoundsCheck(indices(c), params_cols))
          {
            return c;
          }

          cols = 1;
          if(c + 1 < indices_size)
          {
            while(indices(c)+cols == indices(c+cols))
            {
              cols++;
              if(c + 1 >= indices_size)
              {
                break;
              }
            }
          }

          while(cols > 1)
          {
            cons_cols_counter[c++] = cols--;
          }
        }
      }
      else //--indices_size == 1--//
      {
        cons_cols_counter[0] = 1;
      }

      //--Mem-copy columns, bunching consecutive columns together, one row at a time--//
      for(int row = 0; row < params_rows; row++ )
      {
        for(int col=0; col < indices_size;)
        {
          //--If not final iteration--//
          if (col + 1 < indices_size)
          {
            //--Prefetch the next source (params_matrix) and destination (output_matrix) memory addresses--//
            port::prefetch<port::PREFETCH_HINT_T0>(&output(row, col + cons_cols_counter[col]));
            port::prefetch<port::PREFETCH_HINT_T0>(&params(row, indices(col + cons_cols_counter[col])));
          }

          //--Mem-copy column(s)--//
          memcpy(&output(row, col), &params(row, indices(col)), (cons_cols_counter[col] * sizeof(T)));
          col += cons_cols_counter[col];
        }
      }

      return -1;
    }

    template <typename T, typename IndT>
    struct GatherColumnsFunctorCPU {
      int64 operator()(typename TTypes<T>::ConstMatrix params,
                       typename TTypes<IndT>::ConstFlat indices,
                       int64 params_rows,
                       int64 params_cols,
                       typename TTypes<T>::Matrix output) {

        return CountAndCopy<T, IndT>(params, indices, params_rows, params_cols, output);
      }
    };

    template <typename Device, typename T, typename IndT>
    struct GatherColumnsFunctor {
      int64 operator()(const Device& dvc, typename TTypes<T>::ConstMatrix params,
                       typename TTypes<IndT>::ConstFlat indices,
                       int64 params_rows,
                       int64 params_cols,
                       typename TTypes<T>::Matrix output) {
        return GatherColumnsFunctorCPU<T, IndT>()(params, indices, params_rows, params_cols, output);
      }
    };

  }  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_USEROPS_GATHER_COLUMNS_FUNCTOR_H_
