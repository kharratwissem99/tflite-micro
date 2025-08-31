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


#include "tensorflow/lite/micro/examples/mobile_net_v1/models/mobile_net_v1_int8_model_data.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "input_data.h"
#include "labels.h"  // Generated from label.json
#include "map_labels.h"  // to map the predicted labels

namespace {
using MobileNetV1OpResolver2 = tflite::MicroMutableOpResolver<6>;

TfLiteStatus RegisterOps2(MobileNetV1OpResolver2& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  return kTfLiteOk;
}


}  // namespace


TfLiteStatus LoadQuantModelAndPerformInference() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model =
      ::tflite::GetModel(g_mobile_net_v1_int8_model_data);
  TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

  MobileNetV1OpResolver2 op_resolver;
  TF_LITE_ENSURE_STATUS(RegisterOps2(op_resolver));

  // Arena size just a round number. The exact arena usage can be determined
  // using the RecordingMicroInterpreter.
  constexpr int kTensorArenaSize = 450000;
  uint8_t tensor_arena[kTensorArenaSize];

  tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                       kTensorArenaSize);

  TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

  TfLiteTensor* input = interpreter.input(0);
  TFLITE_CHECK_NE(input, nullptr);

  TfLiteTensor* output = interpreter.output(0);
  TFLITE_CHECK_NE(output, nullptr);

  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  for (int h = 0; h < 224; ++h) {
    for (int w = 0; w < 224; ++w) {
      for (int c = 0; c < 3; ++c) {
        int index = h * 224 * 3 + w * 3 + c;  // Flattened index
        input->data.uint8[index] = g_image_array[index] / input->params.scale + input->params.zero_point; //should be an integer??
      }
    }
  }

  TF_LITE_ENSURE_STATUS(interpreter.Invoke());
  TFLITE_CHECK_NE(output, nullptr);
  TFLITE_CHECK_EQ(output->dims->size, 2);  // Ensure it's 2D
  TFLITE_CHECK_EQ(output->dims->data[0], 1);  // batch
  TFLITE_CHECK_EQ(output->dims->data[1], 43);  // Height

  int max_index = 0;
  float max_value = 0.0f;
  // Print the output values
  for (int i = 0; i < 43; ++i) {
    if ((output->data.uint8[i] - output_zero_point) * output_scale > max_value) {
      max_index = i;
      max_value = (output->data.uint8[i] - output_zero_point) * output_scale;
    }
    MicroPrintf("Output[%d]: %f\n", i, static_cast<double>((output->data.uint8[i] - output_zero_point) * output_scale));
  }

  MicroPrintf("Predicted index: %d\n", g_labels_array[max_index]);
  MicroPrintf("Prediction: %s\n", g_map_labels_array[g_labels_array[max_index]]);

  return kTfLiteOk;
}

int main(int argc, char* argv[]) {
  MicroPrintf("~~~EXECUTION STARTED~~~\n");
  TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference());
  MicroPrintf("~~~ALL TESTS PASSED~~~\n");
  return kTfLiteOk;
}
