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
#include <random>
#include <unordered_set>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/platform/place.h"

USE_OP(batch_norm);
USE_OP_DEVICE_KERNEL(batch_norm, MKLDNN);
USE_OP(conv2d_transpose);
USE_OP_DEVICE_KERNEL(conv2d_transpose, MKLDNN);
USE_OP(elementwise_add);
USE_OP_DEVICE_KERNEL(elementwise_add, MKLDNN);
USE_OP(gelu);
USE_OP_DEVICE_KERNEL(gelu, MKLDNN);

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
      op->SetOutput("Output", {outputs[0]});
      op->SetAttr("is_test", true);
      op->SetAttr("strides", std::vector<int>(2,2));
    } else if (std::unordered_set<std::string>{"gelu", "leaky_relu", "relu",
                                               "tanh"}
                   .count(type)) {
      op->SetInput("X", inputs);
      op->SetOutput("Out", {outputs[0]});
    } else if (type == "elementwise_add") {
      op->SetAttr("axis", static_cast<int>(1));
      op->SetInput("X", {inputs[0]});
      op->SetInput("Y", {inputs[1]});
      op->SetOutput("Out", {outputs[0]});
    } else if (type == "batch_norm") {
      op->SetAttr("is_test", true);
      op->SetAttr("epsilon", static_cast<float>(1e-5));
      op->SetInput("X", {inputs[0]});
      op->SetInput("Scale", {inputs[1]});
      op->SetInput("Bias", {inputs[2]});
      op->SetInput("Mean", {inputs[3]});
      op->SetInput("Variance", {inputs[4]});
      op->SetOutput("Y", {outputs[0]});
      op->SetOutput("MeanOut", {outputs[1]});
      op->SetOutput("VarianceOut", {outputs[2]});
      op->SetOutput("SavedMean", {outputs[3]});
      op->SetOutput("SavedVariance", {outputs[4]});
    } else {
      FAIL() << "Unexpected operator type.";
    }
  }

  ProgramDesc BuildProgramDesc(bool is_elementwise_add) {
    ProgramDesc prog;

    for (auto& v :
         std::vector<std::string>({"a", "weights", "bias", "bias_bn", 
         "scale", "mean", "variance", "saved_mean", "saved_variance",
          "f", "g", "h", "i", "j" })) {
      auto* var = prog.MutableBlock(0)->Var(v);
      var->SetType(proto::VarType::LOD_TENSOR);
      if (v == "weights" || v == "bias" || v == "bias_bn" ||
          v == "scale" || v == "mean" ||v == "variance" ){//||
          //v == "a" || v == "j" || v == "g") {
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
          std::vector<std::string>({"i", "mean", "variance",
           "saved_mean", "saved_variance"}), true);
    } else {
    SetOp(&prog, "batch_norm", "batch_norm1",
          std::vector<std::string>({"f", "scale","bias_bn", "mean", "variance"}),
          std::vector<std::string>({"i", "mean", "variance out",
           "saved_mean", "saved_variance"}), true);
    }
    SetOp(&prog, "gelu", "gelu1", std::vector<std::string>({"i"}),
          std::vector<std::string>({"j"}), true);

    return prog;
  }

  void FillTensorWithFixedData(Tensor *tnsr, float value, platform::CPUPlace place) {
    float* ptr = tnsr->mutable_data<float>(place);
    for (int i = 0; i < tnsr->numel(); ++i) {
      ptr[i] = value; 
    }
  }

  void FillTensorWithRandomData(Tensor* tnsr, float lowb, float upb, platform::CPUPlace place)  {
    float* ptr = tnsr->mutable_data<float>(place);
    // Initialize input data
    std::uniform_real_distribution<float> dist(static_cast<float>(lowb),
                                           static_cast<float>(upb));
    std::mt19937 engine;
    for (int i = 0; i < tnsr->numel(); ++i) {
      ptr[i] = dist(engine);
    }
  }

  void CompareTensors(Tensor* tensor1, Tensor* tensor2) {
    // check dims
    for (int i=0; i< tensor1->numel(); ++i) {
      EXPECT_NEAR(tensor1->data<float>()[i], tensor2->data<float>()[i], 1e-3);
    }
  }

 public:
  void MainTest(bool is_elementwise_add) {
    auto base_prog = BuildProgramDesc(is_elementwise_add);

    std::unique_ptr<ir::Graph> graph(new ir::Graph(base_prog));
    Scope scope;
    auto place = paddle::platform::CPUPlace();
    NaiveExecutor exe{place};

    auto pass = PassRegistry::Instance().Get("conv_transpose_eltwiseadd_bn_fuse_pass");
    graph->SetNotOwned(kParamScopeAttr, &scope);

    auto& prog = graph->OriginProgram();

    exe.CreateVariables(prog, 0, true, &scope);
    exe.CreateVariables(prog, 0, false, &scope);

    exe.Prepare(&scope, prog, 0, false);

    std::cout << GenScopeTreeDebugInfo(&scope);

    auto* a_tensor = exe.FindTensor("a");
    auto* weights_tensor = exe.FindTensor("weights");
    auto* bias_tensor = exe.FindTensor("bias");
    auto* g_tensor = exe.FindTensor("g");

    // Batch Norm
    auto* bias_bn_tensor = exe.FindTensor("bias_bn"); //shift
    auto* scale_tensor = exe.FindTensor("scale");
    auto* mean_tensor = exe.FindTensor("mean");
    auto* variance_tensor = exe.FindTensor("variance");

    // mb1_ic24oc24_ih8oh16kh2sh2dh0ph0_iw80ow160kw2sw2dw0pw0 deconv
    a_tensor->Resize({1, 24, 160, 160});
    weights_tensor->Resize({24, 24, 2, 2});
    bias_tensor->Resize({24});
    g_tensor->Resize({24});

    bias_bn_tensor->Resize({24});
    scale_tensor->Resize({24});
    mean_tensor->Resize({24});
    variance_tensor->Resize({24});

    FillTensorWithFixedData(a_tensor,1.0f,place);
    FillTensorWithFixedData(g_tensor,1.0f,place);
    FillTensorWithFixedData(weights_tensor,1.0f,place);
    FillTensorWithFixedData(bias_tensor,1.0f,place);
    FillTensorWithFixedData(bias_bn_tensor,1.0f,place);
    FillTensorWithFixedData(scale_tensor,1.0f,place);
    FillTensorWithFixedData(mean_tensor,1.0f,place);
    FillTensorWithFixedData(variance_tensor,1.0f,place);

    exe.Run();

    // Get result without IR passes applied
    auto* j_tensor = exe.FindTensor("j");
    Tensor no_ir_result;
    TensorCopy(*j_tensor, place, &no_ir_result);

    graph.reset(pass->Apply(graph.release()));

    // Get Program from graph
    ProgramDesc optimized_prog;
    auto graph2program_pass = paddle::framework::ir::PassRegistry::Instance().Get("graph_to_program_pass");
    graph2program_pass->SetNotOwned<paddle::framework::ProgramDesc>("program", &optimized_prog);
    graph2program_pass->Apply(graph.release());

    exe.Prepare(&scope, optimized_prog, 0, false);
    exe.Run();







    // Two graphs. Execute both and compare results
    CompareTensors(&no_ir_result,j_tensor);

    VLOG(3) << DebugString(graph);
  }
};

TEST(MKLDNNConvBatchNormPassTest , conv_elementwise_add_batch_norm) {
  MKLDNNConvBatchNormPassTest().MainTest(true);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_transpose_eltwiseadd_bn_fuse_pass);
USE_PASS(graph_to_program_pass);
