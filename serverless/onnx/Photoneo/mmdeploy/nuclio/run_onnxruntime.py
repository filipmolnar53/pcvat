import time

import cv2
import onnxruntime as ort
from onnxruntime import SessionOptions

from cuda import print_cuda_memory_usage
from rtmdet_tools import draw_predictions, postprocess_results, preprocess_image

# ort.set_default_logger_severity(0)

IMAGE = "/home/molnar/github/cvat/serverless/onnx/Photoneo/mmdeploy/nuclio/2024-01-26_scan_10.png"
ONNX_MODEL = (
    "/home/molnar/github/cvat/serverless/onnx/Photoneo/mmdeploy/nuclio/end2end.onnx"
    # "models/rtmdet-inst.onnx"
    # "models/rtmdet-inst_trt.onnx"
)
import numpy as np

def main():
    #read and preprocess image (handle)
    image = cv2.imread(IMAGE)
    t = time.perf_counter()
    image_input, real_shape = preprocess_image(image)
    print(f"Pre: {(time.perf_counter() - t) * 1000:0.1f} ms")

    providers = [('CPUExecutionProvider', {})]

    assert providers[0][0] in ort.get_available_providers()

    t = time.perf_counter()
    sess_options = SessionOptions()
    if providers[0][0] in ("OpenVINOExecutionProvider",):
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(ONNX_MODEL, providers=providers, sess_options=sess_options)
    sess.disable_fallback()
    if sess.get_providers()[0] != providers[0][0]:
        raise RuntimeError(f"Error creating {providers[0][0]}")
    print(f"Init: {(time.perf_counter() - t) * 1000:0.1f} ms")

    for _ in range(5):
        t = time.perf_counter()
        dets, labels, masks = sess.run(["dets", "labels", "masks"], {"input": image_input})
        print(f"Model: {(time.perf_counter() - t) * 1000:0.1f} ms")

    t = time.perf_counter()
    scores, labels, boxes, masks = postprocess_results(dets, labels, masks, image.shape[:2], real_shape)
    ints = masks[0].astype(np.uint8)
    app = [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE, cv2.CHAIN_APPROX_TC89_KCOS, cv2.CHAIN_APPROX_TC89_L1]
    l = []
    for a in app:

        contours, _ = cv2.findContours(ints, cv2.RETR_TREE, a)
        coords = []

        for obj in contours:
            l.append(len(obj))
            # for point in obj:
            #     coords.append(int(point[0][0]))
            #     coords.append(int(point[0][1]))


    print(f"Post: {(time.perf_counter() - t) * 1000:0.1f} ms")

    print_cuda_memory_usage()

    print(f"scores: {len(scores)}")
    print(f"scores values: {scores}")
    print(f"boxes: {boxes}")
    print(f"masks: {masks.shape}")
    draw_predictions(image, "/home/molnar/github/cvat/serverless/onnx/Photoneo/mmdeploy/nuclio/rtmdet-inst_output.png", scores, labels, boxes, masks)


if __name__ == "__main__":
    main()
