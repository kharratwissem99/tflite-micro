from PIL import Image
import numpy as np

# Load the .png image
image = Image.open("/home/wissemkharrat/repos/tflite-micro/tensorflow/lite/micro/examples/mobile_net_v1/testdata/stopsign.png").convert("RGB")

# Resize the image to 224x224 (or the required size for your model)
image = image.resize((224, 224))

print(image.size)
# Convert the image to a NumPy array
image_array = np.array(image, dtype=np.float32)
print(image_array)

# Normalize the pixel values to [0.0, 1.0] (for float models)
# image_array /= 255.0 # needed later

# Flatten the array (if needed) and save it as a binary file
image_array = image_array.flatten()
print(len(image_array))

# with open("input_data.cc", "w") as f:
#     f.write("alignas(16) const float g_stopsign[] = {")
#     for i in range(len(image_array)):
#         f.write(f"{image_array[i]}, ")
#     f.write("};\n")
