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

#include "paddle/fluid/framework/data_transform.h"

#include "paddle/fluid/framework/data_device_transform.h"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/data_type_transform.h"

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace framework {

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
////////////////////////
static void PassTensorData(Tensor *from, Tensor *to) {
  to->ShareDataWith(*from);
  *from = Tensor();
}

void TransformData(const OpKernelType &expected_kernel_type,
                   const OpKernelType &kernel_type_for_var,
                   const Tensor &input_tensor, Tensor *output_tensor) {
  bool transformed = false;
  Tensor in;
  in.ShareDataWith(input_tensor);
  Tensor out;
  DataLayout lin = kernel_type_for_var.data_layout_;
  DataLayout lout = expected_kernel_type.data_layout_;

  // do layout transform
  if (NeedTransformLayout(lout, lin)) {
    if (lin == DataLayout::kMKLDNN || lout == DataLayout::kMKLDNN) {
      PADDLE_ENFORCE(
          !(lin == DataLayout::kMKLDNN && lout == DataLayout::kMKLDNN),
          "No layout transform needed between two MKLDNN OPKernels");

      if (lin != DataLayout::kMKLDNN && lout == DataLayout::kMKLDNN) {
#ifdef PADDLE_WITH_MKLDNN
        // Case1 - transform from Non-MKLDNN OPKernel to MKLDNN OPKernel
        // Just set layout/format. No real transform occur
        INIT_PERF();
        MAKE_PERF_VAR();
        BEGIN_OVERALL();

        BEGIN();
        out.ShareDataWith(input_tensor);
        END("ShareDataWith");
        BEGIN();
        auto out_mem_pd = paddle::platform::create_prim_desc_from_dims(
            paddle::framework::vectorize2int(out.dims()),
            paddle::platform::mkldnn_fmt(out.dims().size()),
            paddle::framework::ToMKLDNNDataType(in.type()));
        END("create_prim_desc_from_dims");
        BEGIN();

        out.set_mkldnn_prim_desc(out_mem_pd);
        END("set_mkldnn_prim_desc");
        END_OVERALL();
#endif
      } else {
        // Case2 - transfrom from MKLDNN OPKernel to Non-MKLDNN OPKernel
        // Do transform via MKLDNN lib
        TransDataLayoutFromMKLDNN(kernel_type_for_var, expected_kernel_type, in,
                                  &out);
      }
    } else {
      // Case3 - transfrom between Non-MKLDNN OPKernels
      TransDataLayout(kernel_type_for_var, expected_kernel_type, in, &out);
    }
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do data type transform
  if (expected_kernel_type.data_type_ != kernel_type_for_var.data_type_) {
    TransDataType(kernel_type_for_var, expected_kernel_type, in, &out);
    transformed = true;
    PassTensorData(&out, &in);
  }

  // do device transform
  if (!platform::is_same_place(kernel_type_for_var.place_,
                               expected_kernel_type.place_)) {
    TransDataDevice(in, expected_kernel_type.place_, &out);
    transformed = true;
    PassTensorData(&out, &in);
  }

  PADDLE_ENFORCE(transformed, "No transform is applied, please check!");
  // get output data
  output_tensor->ShareDataWith(in);
}

void SetTensorToVariable(const Variable &in_var, const Tensor &tensor,
                         Variable *out_var) {
  if (in_var.IsType<LoDTensor>()) {
    auto &in_lod_tensor = in_var.Get<LoDTensor>();
    auto *tran_lod_tensor = out_var->GetMutable<LoDTensor>();
    tran_lod_tensor->set_lod(in_lod_tensor.lod());
    tran_lod_tensor->set_layout(in_lod_tensor.layout());
    tran_lod_tensor->ShareDataWith(tensor);
  } else if (in_var.IsType<SelectedRows>()) {
    auto &in_selected_rows = in_var.Get<SelectedRows>();
    auto *trans_selected_rows = out_var->GetMutable<SelectedRows>();
    trans_selected_rows->set_height(in_selected_rows.height());
    trans_selected_rows->set_rows(in_selected_rows.rows());
    trans_selected_rows->mutable_value()->ShareDataWith(tensor);
  } else {
    PADDLE_THROW("unknown var type");
  }
}

}  // namespace framework
}  // namespace paddle
