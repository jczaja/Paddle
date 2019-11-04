/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::reorder;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::sum;

template <typename T>
class EltwiseAddMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* z = ctx.Output<Tensor>("Out");
    const T* x_data = x->data<T>();
    const T* y_data = y->data<T>();
    T* z_data = z->mutable_data<T>(ctx.GetPlace());

    int axis = ctx.Attr<int>("axis");

    auto x_dims = x->dims();
    auto y_dims_untrimed = y->dims();
    auto z_dims = z->dims();

    mkldnn::stream astream(mkldnn_engine);

    // Execute default elementwise_add operator when
    // broadcast operations need to performed.
    if (x_dims != y_dims_untrimed) {
      MKLDNNMemoryFormat x_format = platform::GetMKLDNNFormat(x->get_mkldnn_mem_desc());
      Tensor _x;
      auto src_x_tz = framework::vectorize(x_dims);
      MKLDNNMemoryFormat format;
      mkldnn::memory::data_type in_type = platform::MKLDNNGetDataType<T>();

      if ((src_x_tz.size() == 3 &&
           x_format != (format = MKLDNNMemoryFormat::ncw)) ||
          (src_x_tz.size() == 4 &&
           x_format != (format = MKLDNNMemoryFormat::nchw)) ||
          (src_x_tz.size() == 5 &&
           x_format != (format = MKLDNNMemoryFormat::ncdhw))) {
        _x.Resize(x_dims);

        auto out_md = mkldnn::memory::desc(src_x_tz, in_type, {});

        const std::string key =
            platform::CreateKey(src_x_tz, x_format, platform::GetMKLDNNFormat(out_md), in_type);

        platform::ReorderMKLDNNHandler handler(src_x_tz, x->type(), in_type,
                                               dev_ctx, mkldnn_engine, key);

        auto user_x_memory_p = handler.AcquireSrcMemory(
            x->get_mkldnn_mem_desc(), paddle::platform::to_void_cast(x_data));

        auto x_memory_p =
            handler.AcquireDstMemory(&_x, out_md, ctx.GetPlace());

        auto x_reorder = handler.AcquireReorder(x_memory_p, user_x_memory_p);

        x_reorder->execute(astream, *user_x_memory_p, *x_memory_p);
        astream.wait();
      } else {
        format = x_format;
        _x.ShareDataWith(*x);
      }

      auto sum_func = [](T a, T b) -> T { return a + b; };

      TransformFunctor<decltype(sum_func), T,
                       paddle::platform::CPUDeviceContext, T>
          functor(
              &_x, y, z,
              ctx.template device_context<paddle::platform::CPUDeviceContext>(),
              sum_func);

      axis = (axis == -1 ? x_dims.size() - y_dims_untrimed.size() : axis);
      PADDLE_ENFORCE(axis >= 0 && axis < x_dims.size(),
                     "Axis should be in range [0, x_dims)");

      auto y_dims = trim_trailing_singular_dims(y_dims_untrimed);
      axis = (y_dims.size() == 0) ? x_dims.size() : axis;

      int pre, n, post;
      get_mid_dims(x_dims, y_dims, axis, &pre, &n, &post);

      if (post == 1) {
        functor.RunRowWise(n, pre);
      } else {
        functor.RunMidWise(n, pre, post);
      }
      auto z_md = mkldnn::memory::desc(src_x_tz, in_type, format); 
      z->set_mkldnn_mem_desc(z_md);
    } else {
      PADDLE_ENFORCE_EQ(x->layout(), DataLayout::kMKLDNN,
                        "Wrong layout set for X tensor");
      PADDLE_ENFORCE_EQ(y->layout(), DataLayout::kMKLDNN,
                        "Wrong layout set for Y tensor");
      MKLDNNMemoryFormat x_format = platform::GetMKLDNNFormat(x->get_mkldnn_mem_desc());
      MKLDNNMemoryFormat y_format = platform::GetMKLDNNFormat(y->get_mkldnn_mem_desc());

      auto src_x_tz = framework::vectorize(x_dims);
      auto src_y_tz = framework::vectorize(y_dims_untrimed);
      auto dst_tz = framework::vectorize(z_dims);

      std::vector<float> scales = {1.0f, 1.0f};

      const std::string key =
          platform::CreateKey(src_x_tz, ctx.op().Output("Out"));

      platform::SumMKLDNNHandler handler(dev_ctx, mkldnn_engine, key);

      auto src_x_memory = handler.AcquireSrcMemory(
          {{src_x_tz}, platform::MKLDNNGetDataType<T>(), x_format},
          paddle::platform::to_void_cast(x_data));

      auto src_y_memory = handler.AcquireSecondSrcMemory(
          {{src_y_tz}, platform::MKLDNNGetDataType<T>(), y_format},
          paddle::platform::to_void_cast(y_data));

      auto dst_md = memory::desc({dst_tz}, platform::MKLDNNGetDataType<T>(),
                                 MKLDNNMemoryFormat::any);

      auto sum_pd = handler.AcquireSumPrimitiveDescriptor(
          {src_x_memory, src_y_memory}, scales, dst_md);

      auto dst_memory = handler.AcquireDstMemoryFromPrimitive(z_data);

      auto sum_prim = handler.AcquireSum();

      sum_prim->execute(astream, {{MKLDNN_ARG_MULTIPLE_SRC, *src_x_memory},
                                  {MKLDNN_ARG_MULTIPLE_SRC + 1, *src_y_memory},
                                  {MKLDNN_ARG_DST, *dst_memory}});
      astream.wait();

      z->set_mkldnn_mem_desc(dst_memory->get_desc());
    }
  }
};

template <typename T>
class EltwiseAddMKLDNNGradKernel : public ElemwiseGradKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    using Tensor = framework::Tensor;

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));
    int axis = ctx.Attr<int>("axis");
    // skip out, x, y,
    // dout length is larger or equal than dx, dy.
    auto* out = dout;
    auto *x = dout, *y = dout;

    if (dx != nullptr && dy != nullptr && dx->dims() == dy->dims()) {
      if (dx->dims() == dy->dims()) {
        auto blas = math::GetBlas<paddle::platform::CPUDeviceContext, T>(ctx);
        if (dx) {
          blas.VCOPY(dout->numel(), dout->data<T>(),
                     dx->mutable_data<T>(ctx.GetPlace()));
          dx->set_mkldnn_mem_desc(dout->get_mkldnn_mem_desc());
        }

        if (dy) {
          blas.VCOPY(dout->numel(), dout->data<T>(),
                     dy->mutable_data<T>(ctx.GetPlace()));
          dy->set_mkldnn_mem_desc(dout->get_mkldnn_mem_desc());
        }
      }
    } else {
      // Execute default kernel when broadcast is needed
      ElemwiseExplicitGradCompute<paddle::platform::CPUDeviceContext, T,
                                  IdentityGrad<T>, IdentityGrad<T>>(
          ctx, *x, *y, *out, *dout, axis, dx, dy, IdentityGrad<T>(),
          IdentityGrad<T>());
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(elementwise_add, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::EltwiseAddMKLDNNKernel<float>)

REGISTER_OP_KERNEL(elementwise_add_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::EltwiseAddMKLDNNGradKernel<float>)
