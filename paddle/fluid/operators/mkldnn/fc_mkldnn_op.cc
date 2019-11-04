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

#include <mkldnn/include/mkldnn_types.h>
#include <memory>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using framework::LoDTensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

template <typename T>
class FCPrimitiveFactory {
 public:
  explicit FCPrimitiveFactory(const mkldnn::engine& engine) : engine_(engine) {}

  inner_product_forward CreateFcPrimitive(const LoDTensor* input,
                                          const Tensor* weights,
                                          const Tensor* bias, LoDTensor* output,
                                          const ExecutionContext& ctx) {
    RecomputeOutputDims(ctx, input, weights, output);
    if (fc_) {
      UpdateDataPointers(ctx, output, input);
      return *fc_;
    }
    auto src_desc = input->get_mkldnn_mem_desc();
    input_ = CreateMemory(src_desc, input);

    weights_ = TransposeWeights(weights);
    if (src_desc.data.ndims == 4) {
      weights_ = CreateFourDimWeightsMemory(input, weights);
    }

    auto dst_desc = CreateMemDescriptor(output, MKLDNNMemoryFormat::any);

    fc_ = CreateFcPrimitive(*input_, *weights_, dst_desc, bias, output, ctx);
    return *fc_;
  }

  void Execute() {
    mkldnn::stream astream(engine_);
    if (bias_) {
      fc_->execute(astream, {{MKLDNN_ARG_SRC, *input_},
                             {MKLDNN_ARG_WEIGHTS, *weights_},
                             {MKLDNN_ARG_BIAS, *bias_},
                             {MKLDNN_ARG_DST, *output_}});
    } else {
      fc_->execute(astream, {{MKLDNN_ARG_SRC, *input_},
                             {MKLDNN_ARG_WEIGHTS, *weights_},
                             {MKLDNN_ARG_DST, *output_}});
    }
    astream.wait();
  }

 private:
  void UpdateDataPointers(const ExecutionContext& ctx, Tensor* out,
                          const Tensor* in) {
    input_->set_data_handle(const_cast<T*>(in->data<T>()));
    output_->set_data_handle(out->mutable_data<T>(ctx.GetPlace()));
    //Why this??
//    if (out->format() == MKLDNNMemoryFormat::undef) {
//      auto output_format = platform::GetMKLDNNFormat(*output_);
 //     out->set_format((MKLDNNMemoryFormat)output_format);
//    }
  }

  MKLDNNMemoryFormat MatchWeightFormat(MKLDNNMemoryFormat fmt) {
    using format = MKLDNNMemoryFormat;
    switch (fmt) {
      case format::nChw16c:
        return format::aBcd16b;
      case format::nChw8c:
        return format::aBcd8b;
      case format::nchw:
        return format::oihw;
      default:
        return format::undef;
    }
  }

  mkldnn::memory Reorder(const memory::desc& src_desc,
                         const memory::desc& dst_desc, void* src_data) {
    auto src_mem = memory(src_desc, engine_, src_data);
    auto dst_mem = memory(dst_desc, engine_);

    auto reorder = mkldnn::reorder(src_mem, dst_mem);
    mkldnn::stream astream(engine_);
    reorder.execute(astream, src_mem, dst_mem);
    astream.wait();

    return dst_mem;
  }

  static mkldnn::memory::desc CreateMemDescriptor(
      const std::vector<int64_t>& dims, MKLDNNMemoryFormat format) {
    return platform::MKLDNNMemDesc(dims, platform::MKLDNNGetDataType<T>(),
                                   format);
  }

  static mkldnn::memory::desc CreateMemDescriptor(const Tensor* tensor,
                                                  MKLDNNMemoryFormat format) {
    auto dims = framework::vectorize(tensor->dims());
    return CreateMemDescriptor(dims, format);
  }

  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc,
                              const Tensor* tensor) {
    return memory(desc, engine_, to_void_cast<T>(tensor->data<T>()));
  }

  mkldnn::memory TransposeWeights(const Tensor* weights) {
    auto dims = framework::vectorize(weights->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc = CreateMemDescriptor(dims, MKLDNNMemoryFormat::io);
    auto dst_desc = CreateMemDescriptor(dims, MKLDNNMemoryFormat::oi);
    return Reorder(src_desc, dst_desc, to_void_cast<T>(weights->data<T>()));
  }

  inner_product_forward CreateFcPrimitive(const memory& src_memory,
                                          const memory& weights_memory,
                                          const memory::desc& dst_desc,
                                          const Tensor* bias, Tensor* output,
                                          const ExecutionContext& ctx) {
    const auto weights_desc = weights_memory.get_desc();
    const auto src_desc = src_memory.get_desc();
    if (bias) {
      auto bias_desc = bias->get_mkldnn_mem_desc();
      bias_ = CreateMemory(bias_desc, bias);
      auto fc_prim_desc =
          CreateFcPrimDesc(src_desc, weights_desc, bias_desc, dst_desc);

      output_ = CreateDstMemory(fc_prim_desc, ctx, output);

      return inner_product_forward(fc_prim_desc);
    } else {
      auto fc_prim_desc = CreateFcPrimDesc(src_desc, weights_desc, dst_desc);

      output_ = CreateDstMemory(fc_prim_desc, ctx, output);

      return inner_product_forward(fc_prim_desc);
    }
  }

  mkldnn::inner_product_forward::primitive_desc CreateFcPrimDesc(
      const mkldnn::memory::desc& input_desc,
      const mkldnn::memory::desc& weights_desc,
      const mkldnn::memory::desc& bias_desc,
      const mkldnn::memory::desc& dst_desc) {
    auto fc_desc =
        inner_product_forward::desc(prop_kind::forward_scoring, input_desc,
                                    weights_desc, bias_desc, dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, engine_);
  }

  mkldnn::inner_product_forward::primitive_desc CreateFcPrimDesc(
      const mkldnn::memory::desc& input_desc,
      const mkldnn::memory::desc& weights_desc,
      const mkldnn::memory::desc& dst_desc) {
    auto fc_desc = inner_product_forward::desc(prop_kind::forward, input_desc,
                                               weights_desc, dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, engine_);
  }

  mkldnn::memory CreateFourDimWeightsMemory(const Tensor* input,
                                            const Tensor* weights) {
    auto input_dims = framework::vectorize(input->dims());
    auto weight_dims = framework::vectorize(weights->dims());
    auto dims = {weight_dims[1], input_dims[1], input_dims[2], input_dims[3]};

    auto dst_format = MatchWeightFormat(paddle::platform::GetMKLDNNFormat(input->get_mkldnn_mem_desc()));
    auto src_desc = CreateMemDescriptor(dims, MKLDNNMemoryFormat::oihw);
    auto dst_desc = CreateMemDescriptor(dims, dst_format);

    return Reorder(src_desc, dst_desc, weights_->get_data_handle());
  }

  mkldnn::memory CreateDstMemory(
      const mkldnn::inner_product_forward::primitive_desc& fc_prim_desc,
      const ExecutionContext& ctx, Tensor* output) {
    auto dst_desc = fc_prim_desc.dst_desc();
    auto buffer_size = dst_desc.get_size();
    T* output_data = output->mutable_data<T>(ctx.GetPlace(), buffer_size);
    output->set_mkldnn_mem_desc(dst_desc);
    return memory(dst_desc, engine_, to_void_cast<T>(output_data));
  }

  void RecomputeOutputDims(const ExecutionContext& ctx, const LoDTensor* input,
                           const Tensor* w, LoDTensor* output) {
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    std::vector<int64_t> output_dims;
    FCOutputSize(input->dims(), w->dims(), output_dims, in_num_col_dims);
    output->Resize(framework::make_ddim(output_dims));
    output->set_lod(input->lod());
  }

 private:
  const mkldnn::engine& engine_;
  boost::optional<memory> bias_;
  boost::optional<memory> input_;
  boost::optional<memory> output_;
  boost::optional<memory> weights_;
  boost::optional<inner_product_forward> fc_;
};

template <typename T>
std::shared_ptr<FCPrimitiveFactory<T>> GetPrimitiveFactory(
    const MKLDNNDeviceContext& dev_ctx, const ExecutionContext& ctx,
    const Tensor* input, const Tensor* weights,
    const mkldnn::engine& mkldnn_engine) {
  const std::string key = platform::CreateKey(
      framework::vectorize(weights->dims()), ctx.op().Output("Out"));

  auto prim_creator =
      std::static_pointer_cast<FCPrimitiveFactory<T>>(dev_ctx.GetBlob(key));
  if (prim_creator == nullptr) {
    prim_creator = std::make_shared<FCPrimitiveFactory<T>>(mkldnn_engine);
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

template <typename T>
class FCMKLDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto input = ctx.Input<LoDTensor>("Input");
    auto w = ctx.Input<Tensor>("W");
    auto bias = ctx.Input<Tensor>("Bias");
    auto output = ctx.Output<LoDTensor>("Out");

    auto prim_creator =
        GetPrimitiveFactory<T>(dev_ctx, ctx, input, w, mkldnn_engine);
    auto fc = prim_creator->CreateFcPrimitive(input, w, bias, output, ctx);
    prim_creator->Execute();

    output->set_mkldnn_mem_desc(fc.get_primitive_desc().dst_desc());
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_KERNEL(fc, MKLDNN, ::paddle::platform::CPUPlace,
                   paddle::operators::FCMKLDNNOpKernel<float>);
