import onnx

INPUT_MODEL = "end2end.onnx"
OUTPUT_MODEL = "end2end_2.onnx"


def main():
    onnx_model = onnx.load(INPUT_MODEL)

    for node in onnx_model.graph.node:
        # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#tensorrt-plugins-support
        if "TRTBatchedNMS" in node.name:
            node.domain = "trt.plugins"

    onnx.save(onnx_model, OUTPUT_MODEL)


if __name__ == "__main__":
    main()
