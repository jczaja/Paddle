// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/mkldnn/mkldnn_inplace_pass.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/framework/details/var_handle.h"

namespace paddle {
namespace framework {
namespace ir {


void MKLDNNInPlacePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  GraphPatternDetector gpd;
  patterns::MKLDNNInPlace mkldnn_inplace{gpd.mutable_pattern(),
                                                       "mkldnn_inplace"};
  mkldnn_inplace();

  
  int found_inplace_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Handle MKL-DNN In-Place pass";

    GET_IR_NODE_FROM_SUBGRAPH(mkldnn_outplace_op, mkldnn_outplace_op, mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(mkldnn_outplace_in, mkldnn_outplace_in, mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(mkldnn_outplace_out, mkldnn_outplace_out, mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, mkldnn_inplace);


    if((mkldnn_outplace_op->Op()->HasAttr("use_mkldnn") == false)  || (boost::get<bool>(mkldnn_outplace_op->Op()->GetAttr("use_mkldnn")) == false)) {
      VLOG(4) << "do not perform mkl-dnn inplace";
      return;
    }


    // TODO(jczaja): Check inferer
//    auto &infer_inplace = OpInfoMap::Instance().Get(mkldnn_outplace_op->Op()->Type()).infer_inplace_;
//    if (!infer_inplace) {
//      VLOG(4) << "do not perform mkl-dnn inplace: missing InplaceInferer";
//      return;
//    }

    // TODO(jczaja): Enable more ops
    if (mkldnn_outplace_op->Op()->Type() != "softmax") {
      VLOG(4) << "Curently works for softmax only. TODO(jczaja): support other ops";
      return;
    }

   // Iterate over all nodes  that are ops
   // and check if in-place to be var is part of inputs
   // if positive then do not perform inplace
    for (const Node* n : graph->Nodes()) {
      if (n->IsOp()) {
        // Avoid searchin in op that is to be inplace
        if ((n->id() != mkldnn_outplace_op->id()) ) {
          auto* op = n->Op();
          auto inputs = op->Inputs();
          auto in_place_input = mkldnn_outplace_in->Name();
          for(auto& it : inputs) {
            if (it.second == in_place_input) {
             VLOG(4) << "MKL-DNN in-place pass: in-place var cannot be an input to more than one operator";
             return;
            }
          }
        }
      }
    }

    //TODO(jczaja): iterate over outputs searching for output with origianal name
    // then rename it
    mkldnn_outplace_out->RenameVar(mkldnn_outplace_in->Name());
    mkldnn_outplace_op->Op()->SetOutput("Out",
              std::vector<std::string>({mkldnn_outplace_out->Name()}));

    //TODO(jczaja): do properly
    next_op->Op()->SetInput("X", std::vector<std::string>({mkldnn_outplace_out->Name()}));

        

    found_inplace_count++;
    VLOG(4) << "MKL-DNN InPlace applied!"; 
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mkldnn_inplace_pass, paddle::framework::ir::MKLDNNInPlacePass);
