# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import onnx
import onnxruntime as ort
from PIL import Image
import numpy as np

def display_onnx_model_info(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Extract the graph from the model
    graph = model.graph

    # Function to display tensor info
    def display_tensor_info(tensor):
        print(f"Name: {tensor.name}")
        shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]
        print(f"Shape: {shape}")
        print(f"Data Type: {onnx.TensorProto.DataType.Name(tensor.type.tensor_type.elem_type)}")

    # Display input information
    print("Model Inputs:")
    for input_tensor in graph.input:
        display_tensor_info(input_tensor)
        print()

    # Display output information
    print("Model Outputs:")
    for output_tensor in graph.output:
        display_tensor_info(output_tensor)
        print()

model_path = "end2end.onnx"
display_onnx_model_info(model_path)


def preprocess_image(image_path, input_shape):
    image = Image.open(image_path)
    print((input_shape[2], input_shape[3]))
    image = image.resize((input_shape[2], input_shape[3]))
    image_data = np.array(image).astype(np.float32)
    image_data = image_data / 255.0

    # image_data = np.transpose(image_data, (2, 0, 1))  # HWC to CHW
    if image_data.ndim == 2:  # Grayscale image
        # Add a channel dimension and repeat the single channel three times
        image_data = np.expand_dims(image_data, axis=0)  # Shape (H, W, 1)
        image_data = np.repeat(image_data, 3, axis=0)    # Shape (H, W, 3)
        image_data = np.expand_dims(image_data, axis=0)  # Shape (H, W, 1)

    elif image_data.shape[2] == 1:  # Handle case where image has single channel
        image_data = np.repeat(image_data, 3, axis=0)    # Shape (H, W, 3)
        image_data = np.expand_dims(image_data, axis=0)  # Shape (H, W, 1)
    print(image_data.shape)
    return image_data

def run_model(model_path, image_data):
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    outputs = ort_session.run([output_name], {input_name: image_data})
    return outputs

def main():
    model_path = "end2end.onnx"
    image_path = "test_image.png"
    model = onnx.load(model_path)
    graph = model.graph

    input_shape = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
    image_data = preprocess_image(image_path, input_shape)

    outputs = run_model(model_path, image_data)
    out = np.array(outputs)
    print(out[0][0])
    print(out.shape)
    # print("Model Outputs:")
    # for output in outputs:
    #     print(output)

if __name__ == "__main__":
    main()
