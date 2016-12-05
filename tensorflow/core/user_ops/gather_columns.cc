#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "../kernels/bounds_check.h" //--TODO: Currently <#include "tensorflow/core/kernels/bounds_check.h"> gives an error. Need to check and fix.
#include "tensorflow/core/platform/prefetch.h"

using namespace tensorflow;
using namespace std;

//--TODO: shape inference--//
REGISTER_OP("GatherColumns")
.Input("params: T")
.Input("indices: IndT")
.Output("columns: T")
.Attr("T: type")
.Attr("IndT: {int32,int64}");

template <typename T, typename IndT>
class GatherColumnsOp : public OpKernel {
public:
  explicit GatherColumnsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    const DataType data_t = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<IndT>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({data_t, index_t}, {data_t}));
  }

  void Compute(OpKernelContext* ctx) override {

    //--Grab the input tensor - params--//
    const Tensor& params = ctx->input(0);

    //--Grab the input tensor - indices--//
    const Tensor& indices = ctx->input(1);
    auto ind_flat = indices.flat<IndT>();

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
                errors::InvalidArgument("params must be at least a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be a vector, but it is a: ", indices.dims(), "D Tensor."));

    const TensorShape& params_shape(params.shape());

    OP_REQUIRES(
          ctx, params_shape.dims() <= 2,
          errors::InvalidArgument("params must be 1D or 2D but it is: ", params_shape.dims(), "D"));

    TensorShape output_shape(params_shape);

    int64 params_rows;
    int64 params_cols;
    const int64 indices_size = indices.dim_size(0);

    OP_REQUIRES(ctx, indices_size > 0,
                errors::InvalidArgument("indices cannot be empty."));

    if(params_shape.dims() == 1)
    {
      params_rows = 1;
      params_cols = params.dim_size(0);

      //--Set output tensor dims--//
      output_shape.set_dim(0, indices_size);
    }
    else if(params_shape.dims() == 2)
    {
      params_rows = params.dim_size(0);
      params_cols = params.dim_size(1);

      //--Set output tensor dims--//
      output_shape.set_dim(0, params_rows);
      output_shape.set_dim(1, indices_size);
    }

    //--Create an output tensor--//
    Tensor* output = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    //--Single column tensor, indices must include it, so just copy params tensor to output tensor--//
    if(params_cols == 1)
    {
      auto output_flat = output->flat<T>();
      auto params_flat = params.flat<T>();

      memcpy(&output_flat(0), &params_flat(0), (params_rows * sizeof(T)));
      return;
    }

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
        OP_REQUIRES(
              ctx, FastBoundsCheck(ind_flat(c), params_cols),
              errors::InvalidArgument("indices(", c, "): ", ind_flat(c), " is not in range (0, ", params_cols, "]."));

        cols = 1;
        if(c + 1 < indices_size)
        {
          while(ind_flat(c)+cols == ind_flat(c+cols))
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

    auto output_tensor = output->shaped<T, 2>({params_rows, indices_size});
    auto params_tensor = params.shaped<T, 2>({params_rows, params_cols});

    //--Mem-copy columns, bunching consecutive columns together, one row at a time--//
    for(int row = 0; row < params_rows; row++ )
    {
      for(int col=0; col < indices_size;)
      {
        //--If not final iteration--//
        if (col + 1 < indices_size)
        {
          //--Prefetch the next source (params_matrix) and destination (output_matrix) memory addresses--//
          port::prefetch<port::PREFETCH_HINT_T0>(&output_tensor(row, col + cons_cols_counter[col]));
          port::prefetch<port::PREFETCH_HINT_T0>(&params_tensor(row, ind_flat(col + cons_cols_counter[col])));
        }

        //--Mem-copy column(s)--//
        memcpy(&output_tensor(row, col), &params_tensor(row, ind_flat(col)), (cons_cols_counter[col] * sizeof(T)));
        col += cons_cols_counter[col];
      }
    }
  }
};

#define REGISTER_GATHERCOLUMNS_INT32(type) \
  REGISTER_KERNEL_BUILDER(Name("GatherColumns") \
  .Device(DEVICE_CPU) \
  .TypeConstraint<type>("T") \
  .TypeConstraint<int32>("IndT"), \
  GatherColumnsOp<type, int32>)

TF_CALL_ALL_TYPES(REGISTER_GATHERCOLUMNS_INT32);

#undef REGISTER_GATHERCOLUMNS_INT32


#define REGISTER_GATHERCOLUMNS_INT64(type) \
  REGISTER_KERNEL_BUILDER(Name("GatherColumns") \
  .Device(DEVICE_CPU) \
  .TypeConstraint<type>("T") \
  .TypeConstraint<int64>("IndT"), \
  GatherColumnsOp<type, int64>)

TF_CALL_ALL_TYPES(REGISTER_GATHERCOLUMNS_INT64);

#undef REGISTER_GATHERCOLUMNS_INT64


