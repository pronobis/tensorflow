#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/user_ops/scatter_columns_functor.h"

using namespace tensorflow;

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

REGISTER_OP("ScatterColumns")
.Input("params: T")
.Input("indices: IndT")
.Input("out_num_cols: IndT")
.Input("pad_elem: T")
.Output("columns: T")
.Attr("T: type")
.Attr("IndT: {int32,int64}")
.SetShapeFn([](InferenceContext* ctx) {
  ShapeHandle params_shape;
  TF_RETURN_IF_ERROR(ctx->WithRankAtLeast(ctx->input(0), 1, &params_shape));
  TF_RETURN_IF_ERROR(ctx->WithRankAtMost(ctx->input(0), 2, &params_shape));

  ShapeHandle unused_shape;
  TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 1, &unused_shape)); //--indices--//
  TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(2), 0, &unused_shape)); //--out_num_cols--//
  TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(3), 0, &unused_shape)); //--pad_elem--//

  DimensionHandle out_last_dim;
  TF_RETURN_IF_ERROR(ctx->MakeDimForScalarInput(2, &out_last_dim));

  ShapeHandle out_shape;
  TF_RETURN_IF_ERROR(ctx->ReplaceDim(params_shape, -1, out_last_dim, &out_shape));
  ctx->set_output(0, out_shape);

  return Status::OK();
});

template <typename Device, typename T, typename IndT>
class ScatterColumnsOp : public OpKernel {
public:
  explicit ScatterColumnsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    const DataType data_t = DataTypeToEnum<T>::v();
    const DataType index_t = DataTypeToEnum<IndT>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({data_t, index_t, index_t, data_t}, {data_t}));
  }

  void Compute(OpKernelContext* ctx) override {

    //--Grab the input tensor - params--//
    const Tensor& params = ctx->input(0);

    //--Grab the input tensor - indices--//
    const Tensor& indices = ctx->input(1);
    auto indices_flat = indices.flat<IndT>();

    //--Grab the input - out_num_cols--//
    const Tensor& out_num_cols_tensor = ctx->input(2);

    //--Grab the input - pad_elem--//
    const Tensor& pad_elem_tensor = ctx->input(3);


    OP_REQUIRES(ctx, TensorShapeUtils::IsVectorOrHigher(params.shape()),
                errors::InvalidArgument("Params must be at least a vector."));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices.shape()),
                errors::InvalidArgument("Indices must be a vector, but it is a: ", indices.dims(), "D Tensor."));

    //--Check and convert out_num_cols into scalar--//
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(out_num_cols_tensor.shape()),
                errors::InvalidArgument("out_num_cols must be a scalar, but it is a: ", out_num_cols_tensor.dims(), "D Tensor."));
    IndT out_num_cols = out_num_cols_tensor.scalar<IndT>()();

    //--Check and convert pad_elem into scalar--//
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(pad_elem_tensor.shape()),
                errors::InvalidArgument("pad_elem must be a scalar, but it is a: ", pad_elem_tensor.dims(), "D Tensor."));
    T pad_elem = pad_elem_tensor.scalar<T>()();

    int64 indices_size = indices.dim_size(0);

    OP_REQUIRES(ctx, indices_size > 0,
                errors::InvalidArgument("Indices cannot be empty."));

    const TensorShape& params_shape(params.shape());

    OP_REQUIRES(ctx, params_shape.dims() <= 2,
                errors::InvalidArgument("Params must be 1D or 2D but it is: ", params_shape.dims(), "D."));


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
                                        " and the indexed dimension of params - ", params_cols, " - must be the same."));

    unordered_set<IndT> unique_ind(&indices_flat(0), &indices_flat(indices_size));

    OP_REQUIRES(ctx, unique_ind.size() == indices_size,
                errors::InvalidArgument("Indices cannot contain duplicates.",
                                        " Total no. of indices: ", indices_size,
                                        " != no. of unique indices: ", unique_ind.size()));

    //--Create an output tensor--//
    Tensor* output = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    auto output_tensor = output->shaped<T, 2>({params_rows, out_num_cols});
    auto params_tensor = params.shaped<T, 2>({params_rows, params_cols});

    functor::ScatterColumnsFunctor<Device, T, IndT> functor;
    int64 bad_i = functor(ctx->eigen_device<Device>(),
                          params_tensor,
                          indices_flat,
                          out_num_cols,
                          pad_elem,
                          params_rows,
                          output_tensor);

    OP_REQUIRES(ctx, bad_i < 0,
                errors::InvalidArgument("Indices(", bad_i, ") is not in range (0, ", out_num_cols, "]."));
  }
};


#define REGISTER_SCATTERCOLUMNS_ALL(dev, type, index_type) \
  REGISTER_KERNEL_BUILDER(Name("ScatterColumns") \
  .Device(DEVICE_##dev) \
  .TypeConstraint<type>("T") \
  .TypeConstraint<index_type>("IndT"), \
  ScatterColumnsOp<dev##Device, type, index_type>)

#define REGISTER_SCATTERCOLUMNS_ALL_INDICES(dev, type) \
  REGISTER_SCATTERCOLUMNS_ALL(dev, type, int32);      \
  REGISTER_SCATTERCOLUMNS_ALL(dev, type, int64)

#define REGISTER_SCATTERCOLUMNS_CPU(type) REGISTER_SCATTERCOLUMNS_ALL_INDICES(CPU, type)

//--Registration of CPU implementations--//
TF_CALL_ALL_TYPES(REGISTER_SCATTERCOLUMNS_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_SCATTERCOLUMNS_CPU);

#undef REGISTER_SCATTERCOLUMNS_CPU
#undef REGISTER_SCATTERCOLUMNS_ALL_INDICES
#undef REGISTER_SCATTERCOLUMNS_ALL
