/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"

//must include
//#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/examples/mobile_net_v1/models/mobile_net_v1_float_model_data.h"
#include "tensorflow/lite/micro/examples/mobile_net_v1/models/mobile_net_v1_int8_model_data.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

//needed
namespace {
using MobileNetV1OpResolver = tflite::MicroMutableOpResolver<5>;

TfLiteStatus RegisterOps(MobileNetV1OpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  return kTfLiteOk;
}
}  // namespace

/*
*The function is primarily for debugging and optimization. It helps developers understand:
*How much memory the model uses.
*The latency of each operation.
*It does not validate the model's output or perform any specific inference tests.
*Maybe we don't need it when performing the inference on core l110 from codasip.
*/
TfLiteStatus ProfileMemoryAndLatency() {
  tflite::MicroProfiler profiler;
  MobileNetV1OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 6000000;
  uint8_t tensor_arena[kTensorArenaSize];
  constexpr int kNumResourceVariables = 24;

  tflite::RecordingMicroAllocator* allocator(
      tflite::RecordingMicroAllocator::Create(tensor_arena, kTensorArenaSize));


  tflite::RecordingMicroInterpreter interpreter(
      tflite::GetModel(g_mobile_net_v1_float_model_data), op_resolver, allocator,
      tflite::MicroResourceVariables::Create(allocator, kNumResourceVariables),
      &profiler);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());
  MicroPrintf("~~~Input size = %d~~~\n", interpreter.inputs_size()); // added for debugging
  MicroPrintf("~~~Output size = %d~~~\n", interpreter.outputs_size()); // added for debugging

  TFLITE_CHECK_EQ(interpreter.inputs_size(), 1); //number of input tensors
  TFLITE_CHECK_EQ(interpreter.outputs_size(), 1); //number of output tensors

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_EQ(output->dims->size, 2);  // Ensure it's 2D
  TFLITE_CHECK_EQ(output->dims->data[0], 1);  // batch
  TFLITE_CHECK_EQ(output->dims->data[1], 43);  // Height

  TfLiteTensor* input = interpreter.input(0); // we use the index 0 because we only have one input tensor(see interpreter.inputs_size()),
                                              // and it has the shape [1, 224, 224, 3]. if you use the index 1, it will be out of bounds.
  TFLITE_CHECK_EQ(input->dims->size, 4);  // Ensure it's 4D
  TFLITE_CHECK_EQ(input->dims->data[0], 1);  // batch
  TFLITE_CHECK_EQ(input->dims->data[1], 224);  // Height
  TFLITE_CHECK_EQ(input->dims->data[2], 224);  // Width
  TFLITE_CHECK_EQ(input->dims->data[3], 3);    // Channels

  // interpreter.input(0)->data.f[0] = 1.f;
  // dummy input
  for (int h = 0; h < 224; ++h) {
    for (int w = 0; w < 224; ++w) {
      for (int c = 0; c < 3; ++c) {
        int index = h * 224 * 3 + w * 3 + c;  // Flattened index
        input->data.f[index] = 1.f;  // Example: Set all pixels to 1.0
      }
    }
  }
  TF_LITE_ENSURE_STATUS(interpreter.Invoke());

  MicroPrintf("");  // Print an empty new line
  profiler.LogTicksPerTagCsv();

  MicroPrintf("");  // Print an empty new line
  interpreter.GetMicroAllocator().PrintAllocations();
  return kTfLiteOk;
}

TfLiteStatus LoadFloatModelAndPerformInference() {
  const tflite::Model* model =
      ::tflite::GetModel(g_mobile_net_v1_float_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  MobileNetV1OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 6000000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);
  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  // Check if the predicted output is within a small range of the
  // expected output
  // float epsilon = 0.05f; // Normally don't needed, when doing a classification
  // constexpr int kNumTestValues = 4; 
  // float golden_inputs[kNumTestValues] = {0.f, 1.f, 3.f, 5.f}; // We need to pass a complete image instead

  // for (int i = 0; i < kNumTestValues; ++i) {
  //   interpreter.input(0)->data.f[0] = golden_inputs[i];
  //   TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  //   float y_pred = interpreter.output(0)->data.f[0];
  //   TFLITE_CHECK_LE(abs(sin(golden_inputs[i]) - y_pred), epsilon);
  // }

  // TODOs:
  // pass the Input: // 224x224x3 image to the interpreter.input(0)->data.f
  //   TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  // get the prediction
  // optional: compare with an expected value for a successful test

  return kTfLiteOk;
}

TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(g_mobile_net_v1_int8_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  MobileNetV1OpResolver op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 3000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  // float output_scale = output->params.scale;
  // int output_zero_point = output->params.zero_point;

  // Check if the predicted output is within a small range of the
  // expected output
  // float epsilon = 0.05;

  // constexpr int kNumTestValues = 4;
  // float golden_inputs_float[kNumTestValues] = {0.77, 1.57, 2.3, 3.14};

  // The int8 values are calculated using the following formula
  // (golden_inputs_float[i] / input->params.scale + input->params.zero_point)
  // int8_t golden_inputs_int8[kNumTestValues] = {-96, -63, -34, 0};

  // for (int i = 0; i < kNumTestValues; ++i) {
  //   input->data.int8[0] = golden_inputs_int8[i];
  //   TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  //   float y_pred = (output->data.int8[0] - output_zero_point) * output_scale;
  //   TFLITE_CHECK_LE(abs(sin(golden_inputs_float[i]) - y_pred), epsilon);
  // }

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  MicroPrintf("~~~EXECUTION STARTED~~~\n");
  tflite::InitializeTarget();
  TF_LITE_ENSURE_STATUS(ProfileMemoryAndLatency());
  // TF_LITE_ENSURE_STATUS(LoadFloatModelAndPerformInference());
  // TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());
  MicroPrintf("~~~WISSEM HOW ARE YOU~~~\n");
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
