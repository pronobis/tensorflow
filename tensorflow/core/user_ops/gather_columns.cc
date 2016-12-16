#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "gather_columns_functor.h"

using namespace tensorflow;
using namespace std;

//--TODO: shape inference--//
REGISTER_OP("GatherColumns")
.Input("params: T")
.Input("indices: IndT")
.Output("columns: T")
.Attr("T: type")
.Attr("IndT: {int32,int64}");

template <typename Device, typename T, typename IndT>
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

    auto output_tensor = output->shaped<T, 2>({params_rows, indices_size});
    auto params_tensor = params.shaped<T, 2>({params_rows, params_cols});
    auto indices_flat = indices.flat<IndT>();

    functor::GatherColumnsFunctor<Device, T, IndT> functor;
    int64 bad_i = functor(ctx->eigen_device<Device>(),
                          params_tensor,
                          indices_flat,
                          params_rows,
                          params_cols,
                          output_tensor);

    OP_REQUIRES(ctx, bad_i < 0,
                errors::InvalidArgument("indices(", bad_i, "): ", indices_flat(bad_i),
                                        " is not in range (0, ", params_cols, "]."));
  }
};

#define REGISTER_GATHERCOLUMNS_INT32(type) \
  REGISTER_KERNEL_BUILDER(Name("GatherColumns") \
  .Device(DEVICE_CPU) \
  .TypeConstraint<type>("T") \
  .TypeConstraint<int32>("IndT"), \
  GatherColumnsOp<CPUDevice, type, int32>)

TF_CALL_ALL_TYPES(REGISTER_GATHERCOLUMNS_INT32);

#undef REGISTER_GATHERCOLUMNS_INT32


#define REGISTER_GATHERCOLUMNS_INT64(type) \
  REGISTER_KERNEL_BUILDER(Name("GatherColumns") \
  .Device(DEVICE_CPU) \
  .TypeConstraint<type>("T") \
  .TypeConstraint<int64>("IndT"), \
  GatherColumnsOp<CPUDevice, type, int64>)

TF_CALL_ALL_TYPES(REGISTER_GATHERCOLUMNS_INT64);

#undef REGISTER_GATHERCOLUMNS_INT64


