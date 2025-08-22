import json

with open("/home/wissemkharrat/repos/tflite-micro/tensorflow/lite/micro/examples/mobile_net_v1/label.json") as f:
    class_indices = json.load(f)
indices_class = dict((j,i) for i,j in class_indices.items())
print("indices_class:", indices_class)

# # Load label 
# with open('map_labels.json') as f:
#     labels = json.load(f)

with open("/home/wissemkharrat/repos/tflite-micro/tensorflow/lite/micro/examples/mobile_net_v1/labels.cc", "w") as f:
    f.write("#include \"labels.h\"\n\n")
    f.write("alignas(16) const unsigned char g_labels_array[] = {")
    for i in range(len(indices_class)):
        if i == len(indices_class) - 1:
            f.write(f"{indices_class[i]}")
        else:
            f.write(f"{indices_class[i]}, ")
    f.write("};\n")

with open("/home/wissemkharrat/repos/tflite-micro/tensorflow/lite/micro/examples/mobile_net_v1/labels.h", "w") as f:
    f.write("#ifndef LABELS_H_\n")
    f.write("#define LABELS_H_\n\n")
    f.write("extern const unsigned char g_labels_array[];\n\n")
    f.write("#endif  // LABELS_H_\n")
