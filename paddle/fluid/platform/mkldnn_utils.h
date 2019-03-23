/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <mkldnn.h>
#include <string>
#include <vector>

namespace paddle {
namespace platform {

/////////// PROFILING /////
#include <x86intrin.h>

#define INIT_PERF() static RtdscHelper rtdsc_helper
#define MAKE_PERF_VAR() unsigned long long perf = 0; (void) perf
#define BEGIN() perf = __rdtsc()
#define END(name) rtdsc_helper.AddMeasurement(name, __rdtsc() - perf)
#define BEGIN_OVERALL() unsigned long long overall = __rdtsc()
#define END_OVERALL() rtdsc_helper.AddMeasurement("Overall", __rdtsc() - overall)

class RtdscHelper
{
using uint64 = unsigned long long;
public:
  void AddMeasurement(std::string name, uint64 time) {
    if(m_Measurements.find(name) != m_Measurements.end()) {
      m_Measurements[name].first += time;
      m_Measurements[name].second++;
    }
    else
      m_Measurements[name] = {time, 1};
  }

  void PrintResults() {
    std::cout << "Bn measurement results" << std::endl;
    auto width = std::setw(20);
    std::cout << std::left << width << "Name"
                           << width << "Avg Time"
                           << width << "Ratio" << std::endl;
    auto overall_m = m_Measurements["Overall"];
    auto overall = overall_m.first / (double) overall_m.second;
    for(auto const& m : m_Measurements) {
      auto average = m.second.first / (double) m.second.second;
      std::cout << std::left << width << m.first
                             << width << average
                             << width << average / overall << std::endl;
    }
    std::cout << "------------------------" << std::endl;
  }

  ~RtdscHelper() {
    PrintResults();
  }
private:
  std::map<std::string, std::pair<uint64, unsigned>> m_Measurements; // name, time, count
};
// TODO(jczaja): Move This to mkldnn_reuse.h
static unsigned long long pdGetHash(const mkldnn::memory::dims& operand_dims) {
  
  unsigned long long hash_sum = 0;
  
  for (size_t i = 0; i < operand_dims.size(); ++i) {
    hash_sum = (hash_sum << 10) + (unsigned long)operand_dims[i];
  }

  return hash_sum;
}


inline std::shared_ptr<mkldnn::memory::primitive_desc> create_prim_desc_from_dims(
    const std::vector<int>& ltz, mkldnn::memory::format fmt,
    mkldnn::memory::data_type data_type = mkldnn::memory::data_type::f32) {

        INIT_PERF();
        MAKE_PERF_VAR();
        BEGIN_OVERALL();

        BEGIN();
   // Make hash of PD
   auto key = std::to_string(pdGetHash(ltz) * static_cast<unsigned long long>(data_type) * static_cast<unsigned long long>(fmt));

        END("Key gen");
        BEGIN();
   // If there is PD registered then return reference to it
   auto& pool = platform::DeviceContextPool::Instance();
   auto place = paddle::platform::CPUPlace();
   auto dev_ctx = dynamic_cast<platform::MKLDNNDeviceContext*>(pool.Get(place));

        END("context get");
        BEGIN();
   auto mpd = std::static_pointer_cast<mkldnn::memory::primitive_desc>(dev_ctx->GetBlob(key));
   // if there is no PD then create one
   if (mpd == nullptr) {
     mkldnn_memory_desc_t mem_fmt;

     mem_fmt.primitive_kind = mkldnn_memory;
     mem_fmt.ndims = ltz.size();
     for (unsigned int i = 0; i < ltz.size(); ++i) {
       mem_fmt.dims[i] = ltz[i];  // logical dimensions (nchw format,
                                  // regardless physical layout)
     }
     mem_fmt.data_type = static_cast<mkldnn_data_type_t>(data_type);
     mem_fmt.format = static_cast<mkldnn_memory_format_t>(fmt);

     unsigned int total_stride = 1;
     for (int i = ltz.size() - 1; i >= 0; --i) {
       mem_fmt.layout_desc.blocking.padding_dims[i] =
           ltz[i];  // logical dimensions (nchw format, regardless physical
                    // layout)
       mem_fmt.layout_desc.blocking.block_dims[i] = 1;
       mem_fmt.layout_desc.blocking.offset_padding_to_data[i] = 0;  // no offset
       mem_fmt.layout_desc.blocking.strides[0][i] = total_stride;
       mem_fmt.layout_desc.blocking.strides[1][i] = 1;
       total_stride *= ltz[i];
     }
     auto& cpu_engine = dev_ctx->GetEngine();
     mem_fmt.layout_desc.blocking.offset_padding = 0;  // no initial offset
     mpd = std::make_shared<mkldnn::memory::primitive_desc>(mem_fmt, cpu_engine);
     dev_ctx->SetBlob(key,mpd);
   } 
   END("create/get mpd");
   END_OVERALL();

  return mpd;
}

inline mkldnn::memory::primitive_desc create_prim_desc_from_format(
    const std::vector<int>& ltz, const mkldnn::memory::format format,
    const mkldnn::memory::data_type data_type) {
  auto md = mkldnn::memory::desc({ltz}, data_type, format);
  auto& pool = platform::DeviceContextPool::Instance();
  auto place = paddle::platform::CPUPlace();
  auto dev_ctx = dynamic_cast<platform::MKLDNNDeviceContext*>(pool.Get(place));
  PADDLE_ENFORCE_NOT_NULL(dev_ctx, "Could not get valid device");
  auto& cpu_engine = dev_ctx->GetEngine();
  return mkldnn::memory::primitive_desc(md, cpu_engine);
}

inline mkldnn::memory::format mkldnn_fmt(int rank) {
  switch (rank) {
    case 5:
      return mkldnn::memory::format::ncdhw;
    case 4:
      return mkldnn::memory::format::nchw;
    case 3:
      return mkldnn::memory::format::ncw;
    case 2:
      return mkldnn::memory::format::nc;
    case 1:
      return mkldnn::memory::format::x;
    default:
      return mkldnn::memory::format::blocked;
  }
}

}  // namespace platform
}  // namespace paddle
