/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/tests/api/tester_helper.h"

DEFINE_string(infer_shape, "", "data shape file");
DEFINE_int32(sample, 20, "number of sample");

namespace paddle {
namespace inference {
namespace analysis {

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
};

Record ProcessALine(const std::string &line, const std::string &shape_line) {
  VLOG(3) << "process a line";
  std::vector<std::string> columns;

  Record record;
  std::vector<std::string> data_strs;
  split(line, ' ', &data_strs);
  for (auto &d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(shape_line, ' ', &shape_strs);
  for (auto &s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  return record;
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model", FLAGS_infer_model + "/params");
  cfg->DisableGpu();
  cfg->SwitchIrDebug();
  cfg->SwitchSpecifyInputNames(false);
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              const std::string &line, const std::string &shape_line) {
  auto record = ProcessALine(line, shape_line);

  PaddleTensor input;
  input.shape = record.shape;
  input.dtype = PaddleDType::FLOAT32;
  size_t input_size = record.data.size() * sizeof(float);
  input.data.Resize(input_size);
  memcpy(input.data.data(), record.data.data(), input_size);
  std::vector<PaddleTensor> input_slots;
  input_slots.assign({input});
  (*inputs).emplace_back(input_slots);
}

void compare(int cache_capacity = 1) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  // cfg.EnableMKLDNN();
  cfg.SetMkldnnCacheCapacity(cache_capacity);

  AnalysisConfig cfg2;
  SetConfig(&cfg2);
  // cfg2.EnableMKLDNN();
  cfg2.SetMkldnnCacheCapacity(cache_capacity);

  // we will use two predictors (same model)
  auto first_predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto second_predictor = CreatePaddlePredictor<AnalysisConfig>(cfg2);
  std::vector<std::thread> threads;
  std::vector<std::vector<PaddleTensor>> ref_outputs;
  std::vector<std::vector<PaddleTensor>> outputs;
  std::vector<std::vector<PaddleTensor>> input_slots_all;

  std::ifstream file(FLAGS_infer_data);
  std::ifstream infer_file(FLAGS_infer_shape);
  std::vector<std::string> lines;
  std::vector<std::string> shape_lines;

  std::mutex access_mutex;

  // Let's work with 4 samples
  auto num_samples = 4;
  ref_outputs.resize(num_samples);
  outputs.resize(num_samples);
  lines.resize(num_samples);
  shape_lines.resize(num_samples);

  // compute sequenctilly and then
  // compare to multi-threaded / multi-instanced
  // prediction
  for (int i = 0; i < num_samples; ++i) {
    std::getline(file, lines[i]);
    std::getline(infer_file, shape_lines[i]);
    SetInput(&input_slots_all, lines[i], shape_lines[i]);
    if (i % 2) {
      first_predictor->Run(input_slots_all[i], &ref_outputs[i],
                           FLAGS_batch_size);
    } else {
      second_predictor->Run(input_slots_all[i], &ref_outputs[i],
                            FLAGS_batch_size);
    }
  }

  file.close();
  infer_file.close();

  // Start three threads , wait for completition of first one
  // and then start fourth one
  for (int i = 0; i < num_samples; i++) {
    threads.emplace_back([&, i]() {
      std::ifstream tfile(FLAGS_infer_data);
      std::ifstream tinfer_file(FLAGS_infer_shape);
      std::string tline;
      std::string tshape_line;
      std::vector<PaddleTensor> toutput;

      // Get i'th line
      for (int j = 0; j <= i; ++j) {
        std::getline(tfile, tline);
        std::getline(tinfer_file, tshape_line);
      }
      std::vector<std::vector<PaddleTensor>> tinput_slots_all;
      SetInput(&tinput_slots_all, tline, tshape_line);

      if (i % 2) {
        first_predictor->Run(tinput_slots_all[0], &toutput, FLAGS_batch_size);
      } else {
        second_predictor->Run(tinput_slots_all[0], &toutput, FLAGS_batch_size);
      }
      tfile.close();
      tinfer_file.close();

      // Copy output to shared data
      std::lock_guard<std::mutex> guard(access_mutex);
      for (size_t j = 0; j < toutput.size(); ++j) {
        outputs[i].push_back(toutput[j]);
      }
    });
    if (i == 2) {
      threads[0].join();
    }
  }

  // Wait for remaining threads
  std::for_each(std::next(threads.begin(), 1), threads.end(),
                [](std::thread &t) { t.join(); });
  threads.clear();

  // Compare results
  for (size_t i = 0; i < ref_outputs.size(); ++i) {
    for (size_t j = 0; j < ref_outputs[i].size(); ++j) {
      for (size_t k = 0; k < ref_outputs[i][j].data.length() / sizeof(float);
           ++k) {
        EXPECT_NEAR(static_cast<float *>(outputs[i][j].data.data())[k],
                    static_cast<float *>(ref_outputs[i][j].data.data())[k],
                    1e-4);
      }
    }
  }
}

#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_detect, profile_mkldnn) { compare(10 /*cache_capacity */); }
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
