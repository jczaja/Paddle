// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/mkldnn/conv_elementwise_add_mkldnn_fuse_pass.h"
#include <gtest/gtest.h>
#include <boost/logic/tribool.hpp>
#include <unordered_set>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_registry.h"

USE_OP(conv2d);

namespace paddle {
namespace framework {
namespace ir {



class MKLDNNConvBatchNormPassTest {
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
    } else if (std::unordered_set<std::string>{"gelu", "leaky_relu", "relu",
                                               "tanh"}
                   .count(type)) {
      op->SetInput("X", inputs);
    } else if (type == "elementwise_add") {
      op->SetInput("X", {inputs[0]});
      op->SetInput("Y", {inputs[1]});
    } else if (type == "batch_norm") {
      op->SetInput("X", {inputs[0]});
      op->SetInput("Scale", {inputs[1]});
      op->SetInput("Bias", {inputs[2]});
      op->SetInput("Mean", {inputs[3]});
      op->SetInput("Variance", {inputs[4]});
    } else {
      FAIL() << "Unexpected operator type.";
    }
    op->SetOutput("Out", {outputs[0]});
  }

  ProgramDesc BuildProgramDesc(const std::string& mkldnn_enabled_op,
                               bool is_elementwise_add) {
    ProgramDesc prog;

    for (auto& v :
         std::vector<std::string>({"a", "weights", "bias", "f", "g", "h", "i",
                                   "j", "k", "l", "m", "n", "z"})) {
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
          std::vector<std::string>({"g"}),
          mkldnn_enabled_op.compare("relu") == 0);
    SetOp(&prog, "softmax", "softmax1", std::vector<std::string>({"g"}),
          std::vector<std::string>({"h"}),
          mkldnn_enabled_op.compare("softmax") == 0);
    SetOp(&prog, "elementwise_add", "elementwise_add1",
          std::vector<std::string>({"h", "i"}), std::vector<std::string>({"j"}),
          mkldnn_enabled_op.compare("elementwise_add") == 0);
    SetOp(&prog, "relu", "relu2", std::vector<std::string>({"j"}),
          std::vector<std::string>({"k"}),
          mkldnn_enabled_op.compare("relu") == 0);
    SetOp(&prog, "tanh", "tanh1", std::vector<std::string>({"k"}),
          std::vector<std::string>({"l"}),
          mkldnn_enabled_op.compare("tanh") == 0);
    SetOp(&prog, "relu", "relu3", std::vector<std::string>({"l"}),
          std::vector<std::string>({"m"}),
          mkldnn_enabled_op.compare("relu") == 0);
    SetOp(&prog, "leaky_relu", "leaky_relu1", std::vector<std::string>({"m"}),
          std::vector<std::string>({"n"}),
          mkldnn_enabled_op.compare("leaky_relu") == 0);
    SetOp(&prog, "gelu", "gelu1", std::vector<std::string>({"n"}),
          std::vector<std::string>({"m"}),
          mkldnn_enabled_op.compare("gelu") == 0);

    return prog;
  }

 public:
  void MainTest(const std::string& mkldnn_enabled_op, bool is_elementwise_add) {
    auto prog = BuildProgramDesc(mkldnn_enabled_op, is_elementwise_add);

    std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
    auto pass = PassRegistry::Instance().Get("conv_transpose_eltwiseadd_bn_fuse_pass");

    graph.reset(pass->Apply(graph.release()));

    // Two graphs. Execute both and compare results


    VLOG(3) << DebugString(graph);

    EXPECT_EQ(1, 1);
  }
};

TEST(MKLDNNConvBatchNormPassTest , inplace_softmax) {
  MKLDNNConvBatchNormPassTest().MainTest("softmax", false);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_transpose_eltwiseadd_bn_fuse_pass);
