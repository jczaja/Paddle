/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <typeindex>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {

namespace platformx {

inline mkldnn::memory::desc MKLDNNMemDesc(const std::vector<int>& dims,
                                          mkldnn::memory::data_type data_type,
                                          mkldnn::memory::format format) {
  mkldnn::memory::dims tz = dims;
  return mkldnn::memory::desc({tz}, data_type, format);

}

inline mkldnn::memory::format MKLDNNFormatForSize(

    size_t dims_size, mkldnn::memory::format data_format) {

  if (dims_size == 1) {

    return mkldnn::memory::format::x;

  } else if (dims_size == 2) {

    return mkldnn::memory::format::nc;

  } else if (dims_size == 3) {

    if (data_format == mkldnn::memory::format::nchw) {

      return mkldnn::memory::format::ncw;

    } else if (data_format == mkldnn::memory::format::nhwc) {

      return mkldnn::memory::format::nwc;

    }

  } else if (dims_size == 5) {

    if (data_format == mkldnn::memory::format::nchw) {

      return mkldnn::memory::format::ncdhw;

    } else if (data_format == mkldnn::memory::format::nhwc) {

      return mkldnn::memory::format::ndhwc;

    }

  }

  return data_format;

}

inline void Reorder(const mkldnn::memory& src, const mkldnn::memory& dst) {

  auto reorder_prim = mkldnn::reorder(src, dst);

  std::vector<mkldnn::primitive> pipeline;

  pipeline.push_back(reorder_prim);

  mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

}

template <typename Type>

mkldnn::memory::data_type MKLDNNGetDataType() {

  return mkldnn::memory::data_type::data_undef;

}



template <>

inline mkldnn::memory::data_type MKLDNNGetDataType<float>() {

  return mkldnn::memory::data_type::f32;

}

template <>

inline mkldnn::memory::data_type MKLDNNGetDataType<int32_t>() {

  return mkldnn::memory::data_type::s32;

}

template <>

inline mkldnn::memory::data_type MKLDNNGetDataType<int8_t>() {

  return mkldnn::memory::data_type::s8;

}

template <>

inline mkldnn::memory::data_type MKLDNNGetDataType<uint8_t>() {

  return mkldnn::memory::data_type::u8;

}

}

namespace framework {

class LoDTensor;

class Tensor {
#ifdef PADDLE_WITH_MKLDNN

 public:
  inline mkldnn::memory::format format() const { return format_; }

  inline void set_format(const mkldnn::memory::format format) {
    format_ = format;
  }

 template<typename T>

  void Print(const std::string& tensor_name, const mkldnn::engine& engine) const {

    std::cout << tensor_name << std::endl;

    const T* data = this->data<T>();

    auto src_tz = paddle::framework::vectorize<int>(dims_);
    
    int output_num = 4096*2048;
    if(const char* env_p = std::getenv("OUT_NUMB"))
        output_num = atoi(env_p);

    std::unique_ptr<mkldnn::memory> dst_mem;
    if (layout_ == DataLayout::kMKLDNN) { 

      auto src_md = platformx::MKLDNNMemDesc({src_tz}, platformx::MKLDNNGetDataType<T>(), format_);

      auto src_pd = mkldnn::memory::primitive_desc(src_md, engine);

      mkldnn::memory src_mem = mkldnn::memory(src_pd, (T*)data);



      auto dst_md = platformx::MKLDNNMemDesc({src_tz}, platformx::MKLDNNGetDataType<T>(),

        platformx::MKLDNNFormatForSize(src_tz.size(), mkldnn::memory::format::nchw));

      auto dst_pd = mkldnn::memory::primitive_desc(dst_md, engine);

      dst_mem.reset(new mkldnn::memory(dst_pd));

      platformx::Reorder(src_mem, *dst_mem);

      data = (T*)dst_mem->get_data_handle();

    } 

    output_num = output_num < this->numel() ? output_num : this->numel(); 

    for(int i = 0; i < output_num; i++) {
      std::cout << std::to_string(data[i]) << std::endl;
    }

  }

 protected:
  /**
   * @brief the detail format of memory block which have layout as kMKLDNN
   *
   * @note MKLDNN lib support various memory format like nchw, nhwc, nChw8C,
   *       nChw16c, etc. For a MKLDNN memory block, layout will be set as
   *       DataLayout::kMKLDNN meanwhile detail memory format will be kept in
   *       this field.
   */

  mkldnn::memory::format format_ = mkldnn::memory::format::format_undef;
#endif

 public:
  template <typename T, size_t D, int MajorType, typename IndexType>
  friend struct EigenTensor;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenMatrix;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenVector;

 public:
  Tensor() : type_(proto::VarType::FP32), offset_(0) {}

  explicit Tensor(const proto::VarType::Type&);

  /*! Return a pointer to mutable memory block. */
  template <typename T>
  T* data();

  /*! Return a pointer to constant memory block. */
  template <typename T>
  const T* data() const;

  inline bool IsInitialized() const;

  /**
   * @brief   Return a pointer to mutable memory block.
   * @note    If not exist, then allocation.
   */
  template <typename T>
  T* mutable_data(platform::Place place, size_t requested_size = 0);

  void* mutable_data(platform::Place place, proto::VarType::Type type,
                     size_t requested_size = 0);

  void* mutable_data(platform::Place place, size_t requested_size = 0);

  /**
   * @brief     Return a pointer to mutable memory block.
   *
   * @param[in] dims           The dimensions of the memory block.
   * @param[in] place          The place of the memory block.
   * @param[in] requested_size The size of the block in bytes.
   *
   * @note      If not exist, then allocation.
   */
  template <typename T>
  T* mutable_data(DDim dims, platform::Place place, size_t requested_size = 0);

  /*! Return the dimensions of the memory block. */
  const DDim& dims() const;

  /*! Return the numel of the memory block. */
  int64_t numel() const;

  /*! Resize the dimensions of the memory block. */
  Tensor& Resize(const DDim& dims);

  /*! The internal of two tensors share the same memory block. */
  Tensor& ShareDataWith(const Tensor& src);

  /**
   * @brief  Return a sub-tensor of the given tensor.
   *
   * @param[in] begin_idx   The index of the start row(inclusive) to slice.
   *                        The index number begins from 0.
   * @param[in] end_idx     The index of the end row(exclusive) to slice.
   *                        The index number begins from 0.
   */
  Tensor Slice(int64_t begin_idx, int64_t end_idx) const;

  platform::Place place() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_, "Tensor not initialized yet when Tensor::place() is called.");
    return holder_->place();
  }

  proto::VarType::Type type() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_, "Tensor not initialized yet when Tensor::type() is called.");
    return type_;
  }

  // memory size returns the holding memory size in byte.
  size_t memory_size() const;

  void check_memory_size() const;

  DataLayout layout() const { return layout_; }

  void set_layout(const DataLayout layout) { layout_ = layout; }

  void clear() {
    holder_ = nullptr;
    offset_ = 0;
  }

  void ShareBufferWith(const Tensor& tensor) {
    holder_ = tensor.holder_;
    offset_ = tensor.offset_;
  }

  const std::shared_ptr<memory::Allocation>& Holder() const { return holder_; }
  size_t offset() const { return offset_; }

  std::shared_ptr<memory::Allocation> MoveMemoryHolder() {
    return std::move(holder_);
  }

  void ResetHolder(std::shared_ptr<memory::Allocation> holder);

 private:
  /*! holds the memory block if allocated. */
  std::shared_ptr<memory::Allocation> holder_;
  proto::VarType::Type type_;
  /**
   * @brief points to elements dimensions.
   *
   * @note dims_ do not indicate the memory block size.
   */

  DDim dims_;

  /**
   * @brief the layout of memory block, default is NHWC.
   *
   * @note the memory allocation order, describe how weight/data is stored
   *       For example, in 4-D Tensor(rank=4), there are three commonly
   *       used layout. They are
   *            NCHW, NHWC, CHWN.
   *       N,C,H,W for respectively the batch size, the number of
   *       feature maps, the height.
   */
  // Fix me: here just change the default layout to kNCHW
  // it doesn't fix the real issue, i.e. feeder should set up tensor layout
  // according to actual input data
  DataLayout layout_ = DataLayout::kNCHW;

  /**
   * @brief   A PlaceHolder may be shared by more than one tensor.
   *
   * @note    Some of them may be slices of the others. So the offset_
   *          is introduced here to indicate the byte offset between
   *          PlaceHolder::ptr_ and where the tensor data really begins.
   */
  size_t offset_;
};

}  // namespace framework
}  // namespace paddle

#include "paddle/fluid/framework/tensor_impl.h"
