#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/prefetch.h"

using namespace tensorflow;
using namespace std;

REGISTER_OP("ScatterColumns")
.Input("params: T")
.Input("indices: IndT")
.Input("out_num_cols: IndT")
.Input("pad_elem: T")
.Output("columns: T")
.Attr("T: type")
.Attr("IndT: {int32,int64}");

template <typename T, typename IndT>
class ScatterColumnsOp : public OpKernel {
public:
  explicit ScatterColumnsOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {

    //--Grab the input tensor - params--//
    const Tensor& params = ctx->input(0);

    //--Grab the input tensor - indices--//
    const Tensor& indices = ctx->input(1);
    auto ind_flat = indices.flat<IndT>();

    //--Grab the input - out_num_cols--//
    const Tensor& out_num_cols_tensor = ctx->input(2);

    //--Grab the input - pad_elem--//
    const Tensor& pad_elem_tensor = ctx->input(3);

    //--Check and convert out_num_cols into scalar--//
    IndT out_num_cols = out_num_cols_tensor.scalar<IndT>()();

    //--Check and convert pad_elem into scalar--//
    T pad_elem = pad_elem_tensor.scalar<T>()();

    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
                errors::InvalidArgument("params must be at least a vector"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("indices must be a vector, but it is a: ", indices.dims(), "D Tensor."));

    int64 indices_size = indices.dim_size(0);

    OP_REQUIRES(ctx, indices_size > 0,
                errors::InvalidArgument("indices cannot be a empty."));

    const TensorShape& params_shape(params.shape());

    OP_REQUIRES(ctx, params_shape.dims() <= 2,
                errors::InvalidArgument("params must be 1D or 2D but it is: ", params_shape.dims(), "D"));


    TensorShape output_shape(params_shape);

    int64 params_rows;
    int64 params_cols;

    if(params_shape.dims() == 1)
    {
      params_rows = 1;
      params_cols = params.dim_size(0);

      //--Set output tensor dims--//
      output_shape.set_dim(0, out_num_cols);
    }
    else if(params_shape.dims() == 2)
    {
      params_rows = params.dim_size(0);
      params_cols = params.dim_size(1);

      //--Set output tensor dims--//
      output_shape.set_dim(0, params_rows);
      output_shape.set_dim(1, out_num_cols);
    }

    OP_REQUIRES(ctx, out_num_cols >= params_cols,
                errors::InvalidArgument("out_num_cols: ", out_num_cols,
                                        " must be >= size of the indexed dimension of params: ", params_cols));

    OP_REQUIRES(ctx, indices_size == params_cols,
                errors::InvalidArgument("Size of indices: ", indices_size,
                                        " and the indexed dimension of params: ", params_cols, " must be the same."));

    //--Check indices[i] ∈ (0, out_num_cols], ∀i--//
    for(int i=0; i < indices_size; i++)
    {
      //--TODO: Should look for a more optimal way to do this, or maybe there is a TF macro for this--//
      OP_REQUIRES(ctx, ind_flat(i) >= 0 && ind_flat(i) < out_num_cols,
                  errors::InvalidArgument("indices(", i, "): ", ind_flat(i), " is not in range (0, ", out_num_cols, "]."));
    }

    unordered_set<IndT> unique_ind(&ind_flat(0), &ind_flat(indices_size));

    OP_REQUIRES(ctx, unique_ind.size() == indices_size,
                errors::InvalidArgument("indices cannot contain duplicates.",
                                        " Total no. of indices: ", indices_size,
                                        " != no. of unique indices: ", unique_ind.size()));

    //--Create an output tensor--//
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

    //--Arrange output indices--//
    vector<IndT> out_indices(out_num_cols, -1); //--Here '-1' refers to padding column(s)--//
    for(IndT i=0; i<indices_size; i++)
    {
      out_indices[ind_flat(i)] = i;
    }

    //--Vector containing padding elements. Size of this vector = maximum no. of consecutive padding columns in the output tensor--//
    gtl::InlinedVector<T, 4> pad_elem_vec; //--TODO: What is 'N' in InlinedVector<T, N> templete?--//

    if(params_shape.dims() == 1)
    {
      auto output_flat = output_tensor->flat<T>();
      auto params_flat = params.flat<T>();

      //--Mem-copy columns, bunching consecutive padding columns together--//
      for(int col = 0; col < out_num_cols;)
      {
        //--If not the final copy--//
        if (col + 1 < out_num_cols)
        {
          //--Prefetch the next destination (output_matrix) memory address--//
          port::prefetch<port::PREFETCH_HINT_T0>(&output_flat(col + 1));

          //--If the next column is not a padding column--//
          if(out_indices[col+1] >= 0)
          {
            //--Prefetch the next source (params_matrix) memory address--//
            port::prefetch<port::PREFETCH_HINT_T0>(&params_flat(out_indices[col+1]));
          }
        }

        if(out_indices[col] >= 0)
        {
          //--Mem-copy a single non-padding element from params tensor--//
          memcpy(&output_flat(col), &params_flat(out_indices[col]), sizeof(T));
          ++col;
        }
        else
        {
          int cons_pad_elem = 0;
          CountConsecutivePaddedElements(col, cons_pad_elem, out_num_cols, out_indices, pad_elem_vec, pad_elem);

          //--Mem-copy columns of padding elements (per row) from padding element vector--//
          memcpy(&output_flat(col), &pad_elem_vec[0], (cons_pad_elem * sizeof(T)));
          col += cons_pad_elem;
        }
      }
    }
    else if(params_shape.dims() == 2)
    {
      auto output_matrix = output_tensor->matrix<T>();
      auto params_matrix = params.matrix<T>();

      //--Mem-copy columns, bunching consecutive padding columns together, one row at a time--//
      for(int row = 0; row < params_rows; row++ )
      {
        for(int col = 0; col < out_num_cols;)
        {
          //--If not the final copy--//
          if (col + 1 < out_num_cols)
          {
            //--Prefetch the next destination (output_matrix) memory address--//
            port::prefetch<port::PREFETCH_HINT_T0>(&output_matrix(row, (col + 1)));

            //--If the next column is not a padding column--//
            if(out_indices[col+1] >= 0)
            {
              //--Prefetch the next source (params_matrix) memory address--//
              port::prefetch<port::PREFETCH_HINT_T0>(&params_matrix(row, out_indices[col+1]));
            }
          }

          if(out_indices[col] >= 0)
          {
            //--Mem-copy a single non-padding element from params tensor--//
            memcpy(&output_matrix(row, col), &params_matrix(row, out_indices[col]), sizeof(T));
            ++col;
          }
          else
          {
            int cons_pad_elem = 0;
            CountConsecutivePaddedElements(col, cons_pad_elem, out_num_cols, out_indices, pad_elem_vec, pad_elem);

            //--Mem-copy columns of padding elements (per row) from padding element vector--//
            memcpy(&output_matrix(row, col), &pad_elem_vec[0], (cons_pad_elem * sizeof(T)));
            col += cons_pad_elem;
          }
        }
      }
    }
  }

private:
  inline void CountConsecutivePaddedElements(const int col,
                                             int& cons_pad_elem,
                                             const IndT& out_num_cols,
                                             const vector<IndT>& out_indices,
                                             gtl::InlinedVector<T, 4>& pad_elem_vec,
                                             const T pad_elem)
  {
    int c = col;

    //--Count no. of consecutive padding elements--//
    do
    {
      ++cons_pad_elem;
      ++c;

      if(c >= out_num_cols)
      {
        break;
      }
    }while(out_indices[c] < 0);

    //--If size of padding element vector is less than the current count
    //--of consecutive padding elements, then increase it accordingly--//
    while(pad_elem_vec.size() < cons_pad_elem)
    {
      pad_elem_vec.push_back(pad_elem);
    }
  }
};


#define REGISTER_SCATTERCOLUMNS_INT32(type) \
  REGISTER_KERNEL_BUILDER(Name("ScatterColumns") \
  .Device(DEVICE_CPU) \
  .TypeConstraint<type>("T") \
  .TypeConstraint<int32>("IndT"), \
  ScatterColumnsOp<type, int32>)

TF_CALL_ALL_TYPES(REGISTER_SCATTERCOLUMNS_INT32);

#undef REGISTER_SCATTERCOLUMNS_INT32


#define REGISTER_SCATTERCOLUMNS_INT64(type) \
  REGISTER_KERNEL_BUILDER(Name("ScatterColumns") \
  .Device(DEVICE_CPU) \
  .TypeConstraint<type>("T") \
  .TypeConstraint<int64>("IndT"), \
  ScatterColumnsOp<type, int64>)

TF_CALL_ALL_TYPES(REGISTER_SCATTERCOLUMNS_INT64);

#undef REGISTER_SCATTERCOLUMNS_INT64


