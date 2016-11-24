#include "tensorflow/core/framework/op.h"
//#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/prefetch.h"

using namespace tensorflow;
using namespace std;


REGISTER_OP("GatherColumns")
    .Input("params: T")
    .Input("indices: Index")
    .Output("columns: T")
    .Attr("T: type")
    .Attr("Index: {int32,int64}");

template <typename T>
class GatherColumnsOp : public OpKernel {
 public:
  explicit GatherColumnsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {

    //--Grab the input tensor - params--//
    const Tensor& params = ctx->input(0);

    //--Grab the input tensor - indices--//
    const Tensor& indices = ctx->input(1);
    auto ind_flat = indices.flat<int32>();

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
                errors::InvalidArgument("params must be at least a vector"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be a vector, but it a: ", indices.dims(), "D Tensor."));

    const TensorShape& params_shape(params.shape());

    OP_REQUIRES(
        ctx, params_shape.dims() <= 2,
        errors::InvalidArgument("params must be 1D or 2D but it is: ", params_shape.dims(), "D"));

    TensorShape output_shape(params_shape);

    int64 params_rows;
    int64 params_cols;
    int64 indices_size= indices.dim_size(0);

    OP_REQUIRES(ctx, indices_size > 0,
                errors::InvalidArgument("indices cannot be a empty."));

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

    //--Check indices[i] ∈ (0, params_cols], ∀i--//
    for(int64 i=0; i < indices_size; i++)
    {
      //--TODO: Should look for a more optimal way to do this, or maybe there is a TF macro for this--//
      OP_REQUIRES(
        ctx, ind_flat(i) >= 0 && ind_flat(i) < params_cols,
        errors::InvalidArgument("indices(", i, "): ", ind_flat(i), " is not in range (0, ", params_cols, "]."));
    }

    //--Create an output tensor--//
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

    //--Single column tensor, indices must include it, so just copy param tensor to output tensor--//
    if(params_cols == 1)
    {
      auto output_flat = output_tensor->flat<T>();
      auto params_flat = params.flat<T>();

      memcpy(&output_flat(0), &params_flat(0), (params_rows * sizeof(T)));
      return;
    }

    //--TODO: Check if using gtl::InlinedVector<T, N> has an advantage over std::vector<int>--//
    std::vector<int> cons_cols_counter;

    if(indices_size > 1)
    {
      //--Group consecutive columns together--//
      for(int c=0; c < indices_size; c++)
      {
        int cols = 1;
        if(c + 1 < indices_size)
        {
          while(ind_flat(c)+1 == ind_flat(c+1))
          {
            cols++;
            c++;
            if(c + 1 >= indices_size)
            {
              break;
            }
          }
        }
        cons_cols_counter.push_back(cols);
      }
      //--TODO: Check if SUM(cons_cols_counter) == indices_size--//
    }
    else
    {
      cons_cols_counter.push_back(1);
    }

    int cons_cols_counter_size = cons_cols_counter.size();

    if(params_shape.dims() == 1)
    {
      auto output_flat = output_tensor->flat<T>();
      auto params_flat = params.flat<T>();

      //--Mem-copy columns, bunching consecutive columns together--//
      for(int i=0, col=0; i < cons_cols_counter_size; i++)
      {
        //--If not final iteration--//
        if (i + 1 < cons_cols_counter_size)
        {
          //--Prefetch the next source (params_flat) and destination (output_flat) memory addresses--//
          port::prefetch<port::PREFETCH_HINT_T0>(&output_flat(col + cons_cols_counter[i]));
          port::prefetch<port::PREFETCH_HINT_T0>(&params_flat(ind_flat(col + cons_cols_counter[i])));
        }

        //--Mem-copy column(s)--//
        memcpy(&output_flat(col), &params_flat(ind_flat(col)), (cons_cols_counter[i] * sizeof(T)));
        col += cons_cols_counter[i];
      }
    }
    else if(params_shape.dims() == 2)
    {
      auto output_matrix = output_tensor->matrix<T>();
      auto params_matrix = params.matrix<T>();

      //--Mem-copy columns, bunching consecutive columns together, one row at a time--//
      for(int row = 0; row < params_rows; row++ )
      {
        for(int i=0, col=0; i < cons_cols_counter_size; i++)
        {
          //--If not final iteration--//
          if (i + 1 < cons_cols_counter_size)
          {
            //--Prefetch the next source (params_matrix) and destination (output_matrix) memory addresses--//
            port::prefetch<port::PREFETCH_HINT_T0>(&output_matrix(row, col + cons_cols_counter[i]));
            port::prefetch<port::PREFETCH_HINT_T0>(&params_matrix(row, ind_flat(col + cons_cols_counter[i])));
          }

          //--Mem-copy column(s)--//
          memcpy(&output_matrix(row, col), &params_matrix(row, ind_flat(col)), (cons_cols_counter[i] * sizeof(T)));
          col += cons_cols_counter[i];
        }
      }
    }
  }
};

#define REGISTER_GATHERCOLUMNS(type) \
  REGISTER_KERNEL_BUILDER(Name("GatherColumns") \
                              .Device(DEVICE_CPU) \
                              .TypeConstraint<type>("T"), \
                          GatherColumnsOp<type>)

TF_CALL_ALL_TYPES(REGISTER_GATHERCOLUMNS);

#undef REGISTER_GATHERCOLUMNS


