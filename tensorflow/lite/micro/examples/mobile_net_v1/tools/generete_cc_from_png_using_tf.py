import tensorflow as tf
from PIL import Image
import numpy as np

# Load the .png image
# image = Image.open("/home/wissemkharrat/repos/tflite-micro/tensorflow/lite/micro/examples/mobile_net_v1/testdata/stopsign.png").convert("RGB")

img = tf.keras.utils.load_img("/home/wissemkharrat/repos/tflite-micro/tensorflow/lite/micro/examples/mobile_net_v1/testdata/image.png", target_size=(224, 224))
# Resize the image to 224x224 (or the required size for your model)
# image = image.resize((224, 224))
test_image = np.expand_dims(img, axis=0).astype(np.float32)
# Convert the image to a NumPy array
# image_array = np.array(image, dtype=np.float32)
print(test_image)

# Normalize the pixel values to [0.0, 1.0] (for float models)
image_array /= 255.0 # needed later

# Flatten the array (if needed) and save it as a binary file
image_array = test_image.flatten()
print(image_array)
print(len(image_array))

with open("/home/wissemkharrat/repos/tflite-micro/tensorflow/lite/micro/examples/mobile_net_v1/input_data.cc", "w") as f:
    f.write("#include \"input_data.h\"\n\n")
    f.write("alignas(16) const float g_image_array[] = {")
    for i in range(len(image_array)):
        f.write(f"{image_array[i]}, ")
    f.write("};\n")

with open("/home/wissemkharrat/repos/tflite-micro/tensorflow/lite/micro/examples/mobile_net_v1/input_data.h", "w") as f:
    f.write("#ifndef INPUT_DATA_H_\n")
    f.write("#define INPUT_DATA_H_\n\n")
    f.write("extern const float g_image_array[];\n\n")
    f.write("#endif  // INPUT_DATA_H_\n")
