// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <boost/logic/tribool.hpp>

namespace paddle {
namespace framework {
namespace ir {

class MKLDNNInplacePassTest {
 private:
  void SetOp(ProgramDesc* prog, const std::string& type,
             const std::string& name, const std::vector<std::string>& inputs,
             const std::vector<std::string>& outputs,
             boost::tribool use_mkldnn) {
    auto* op = prog->MutableBlock(0)->AppendOp();

    op->SetType(type);

    if (!boost::indeterminate(use_mkldnn))
      op->SetAttr("use_mkldnn", use_mkldnn);

    if (type == "conv2d") {
      op->SetAttr("name", name);
      op->SetInput("Input", {inputs[0]});
      op->SetInput("Filter", {inputs[1]});
      op->SetInput("Bias", {inputs[2]});
    } else if (type == "relu") {
      op->SetInput("X", inputs);
    } else if (type == "softmax") {
      op->SetAttr("axis", -1);
      op->SetInput("X", inputs);
    } else if (type == "elementwise_add") {
      op->SetInput("X", {inputs[0]});
      op->SetInput("Y", {inputs[1]});
    } else {
      FAIL() << "Unexpected operator type.";
    }
    op->SetOutput("Out", {outputs[0]});
  }

  ProgramDesc BuildProgramDesc(const std::string& mkldnn_enabled_op) {
    ProgramDesc prog;

    for (auto& v :
         std::vector<std::string>({"a", "weights", "bias", "f", "g", "h", "i", "j"})) {
      auto* var = prog.MutableBlock(0)->Var(v);
      var->SetType(proto::VarType::SELECTED_ROWS);
      if (v == "weights" || v == "bias") {
        var->SetPersistable(true);
      }
    }

    SetOp(&prog, "conv2d", "conv1",
          std::vector<std::string>({"a", "weights", "bias"}),
          std::vector<std::string>({"f"}), boost::indeterminate);
    SetOp(&prog, "relu", "relu1", std::vector<std::string>({"f"}),
          std::vector<std::string>({"g"}), mkldnn_enabled_op.compare("relu") == 0);
    SetOp(&prog, "softmax", "softmax1", std::vector<std::string>({"g"}),
          std::vector<std::string>({"h"}), mkldnn_enabled_op.compare("softmax") == 0);
    SetOp(&prog, "elementwise_add", "elementwise_add1", std::vector<std::string>({"h","i"}),
          std::vector<std::string>({"j"}), mkldnn_enabled_op.compare("elementwise_add") == 0);

    return prog;
  }

  Scope* CreateParamScope() {
    auto param_scope = new Scope();
//    AddVarToScope(param_scope, "bias_1", {3});
//    AddVarToScope(param_scope, "scale", {3});
//    AddVarToScope(param_scope, "mean", {3});
//    AddVarToScope(param_scope, "variance", {3});
//    AddVarToScope(param_scope, "filters", {3, 3, 2, 2});
    return param_scope;
  }


 public:
  void MainTest(const std::string& mkldnn_enabled_op,
                unsigned expected_use_mkldnn_true_count) {
    auto prog = BuildProgramDesc(mkldnn_enabled_op);

    std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
    graph->Set("__param_scope__", CreateParamScope());
    auto pass = PassRegistry::Instance().Get("mkldnn_inplace_pass");

    graph.reset(pass->Apply(graph.release()));

    unsigned use_mkldnn_true_count = 0;
    std::unordered_map<std::string, std::string> input_names;
    std::unordered_map<std::string, std::string> output_names;
    input_names["softmax"] = "X";
    output_names["softmax"] = "Out";
    input_names["batch_norm"] = "X";
    output_names["batch_norm"] = "Y";
    input_names["layer_norm"] = "X";
    output_names["layer_norm"] = "Y";

    VLOG(3) << DebugString(graph);

    for (auto* node : graph->Nodes()) {
      if (node->IsOp()) {
        auto* op = node->Op();
        if (op->Type() == mkldnn_enabled_op ) {
          auto ins = op->Inputs();
          auto outs = op->Outputs(); 
          // Input and output are the same var
          if (ins[input_names[mkldnn_enabled_op]] == outs[output_names[mkldnn_enabled_op]]) {
            ++use_mkldnn_true_count;
          }
        }
      }
    }

    EXPECT_EQ(use_mkldnn_true_count, expected_use_mkldnn_true_count);
  }

};

TEST(MKLDNNInplacePass, inplace_softmax) {
  // softmax to be mkl-dnn enabled and made in-place
  MKLDNNInplacePassTest().MainTest("softmax", 1);
}


}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(mkldnn_inplace_pass);
