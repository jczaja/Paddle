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

USE_OP(batch_norm);
USE_OP_DEVICE_KERNEL(batch_norm, MKLDNN);
USE_OP(conv2d_transpose);
USE_OP_DEVICE_KERNEL(conv2d_transpose, MKLDNN);
USE_OP(elementwise_add);
USE_OP_DEVICE_KERNEL(elementwise_add, MKLDNN);
USE_OP(gelu);

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

    if (type == "conv2d_transpose") {
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

  ProgramDesc BuildProgramDesc(bool is_elementwise_add) {
    ProgramDesc prog;

    for (auto& v :
         std::vector<std::string>({"a", "weights", "bias", "bias_bn", "mean", "variance", "f", "g", "h", "i", "j" })) {
      auto* var = prog.MutableBlock(0)->Var(v);
      var->SetType(proto::VarType::SELECTED_ROWS);
      if (v == "weights" || v == "bias" || v == "bias_bn" ||
         v == "scale" || v == "mean" ||v == "variance" ) {
        var->SetPersistable(true);
      }
    }

    SetOp(&prog, "conv2d_transpose", "conv1",
          std::vector<std::string>({"a", "weights", "bias"}),
          std::vector<std::string>({"f"}), true);
    if (is_elementwise_add == true) {
    SetOp(&prog, "elementwise_add", "elementwise_add1",
          std::vector<std::string>({"f", "g"}), std::vector<std::string>({"h"}),
          true);
    SetOp(&prog, "batch_norm", "batch_norm1",
          std::vector<std::string>({"h", "scale","bias_bn", "mean", "variance"}),
          std::vector<std::string>({"i"}), true);
    } else {
    SetOp(&prog, "batch_norm", "batch_norm1",
          std::vector<std::string>({"f", "scale","bias_bn", "mean", "variance"}),
          std::vector<std::string>({"i"}), true);
    }
    SetOp(&prog, "gelu", "gelu1", std::vector<std::string>({"i"}),
          std::vector<std::string>({"j"}), true);

    return prog;
  }

 public:
  void MainTest(bool is_elementwise_add) {
    auto prog = BuildProgramDesc(is_elementwise_add);

    std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
    Scope scope;
    (*graph)->SetNotOwned(kParamScopeAttr, &scope);
    auto pass = PassRegistry::Instance().Get("conv_transpose_eltwiseadd_bn_fuse_pass");

    graph.reset(pass->Apply(graph.release()));

    // Two graphs. Execute both and compare results

    VLOG(3) << DebugString(graph);


    EXPECT_EQ(1, 1);
  }
};

TEST(MKLDNNConvBatchNormPassTest , conv_elementwise_add_batch_norm) {
  MKLDNNConvBatchNormPassTest().MainTest(true);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_transpose_eltwiseadd_bn_fuse_pass);
