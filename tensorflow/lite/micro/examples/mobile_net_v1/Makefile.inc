MOBILE_NET_V1_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/mobile_net_v1_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/input_data.cc
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/labels.cc

MOBILE_NET_V1_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/mobile_net_v1_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/input_data.cc
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/labels.cc

MOBILE_NET_V1_HDRS :=

MOBILE_NET_V1_GENERATOR_INPUTS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/models/mobile_net_v1_float.tflite \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/models/mobile_net_v1_int8.tflite

MOBILE_NET_V1_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/models/mobile_net_v1_float_model_data.cc \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/models/mobile_net_v1_int8_model_data.cc

MOBILE_NET_V1_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/models/mobile_net_v1_float_model_data.h \
$(GENERATED_SRCS_DIR)$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/mobile_net_v1/models/mobile_net_v1_int8_model_data.h

# Tests loading and running the sine model.
$(eval $(call microlite_test,mobile_net_v1_test,\
$(MOBILE_NET_V1_TEST_SRCS),,$(MOBILE_NET_V1_GENERATOR_INPUTS)))

# Builds a standalone binary.
$(eval $(call microlite_test,mobile_net_v1,\
$(MOBILE_NET_V1_SRCS),,$(MOBILE_NET_V1_GENERATOR_INPUTS)))

# Add sources and headers generated from $(MOBILE_NET_V1_GENERATOR_INPUTS).
MOBILE_NET_V1_SRCS += $(MOBILE_NET_V1_GENERATED_SRCS)
MOBILE_NET_V1_HDRS += $(MOBILE_NET_V1_GENERATED_HDRS)

list_mobile_net_v1_example_sources:
	@echo $(MOBILE_NET_V1_SRCS)

list_mobile_net_v1_example_headers:
	@echo $(MOBILE_NET_V1_HDRS)
