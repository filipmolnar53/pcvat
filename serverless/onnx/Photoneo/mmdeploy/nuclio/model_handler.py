# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import SessionOptions

from cuda import print_cuda_memory_usage
from rtmdet_tools import draw_predictions, postprocess_results, preprocess_image

class ModelHandler:
    def __init__(self, context, labels, types):
        context.logger.info(f"Initializing")
        self.labels = labels
        self.types = types
        self.ONNX_MODEL = (
            "end2end.onnx"
        )
        self.providers = [('CPUExecutionProvider', {})]
        context.logger.info(f"ort.providers {ort.get_available_providers()}")

        assert self.providers[0][0] in ort.get_available_providers()

        self.sess_options = SessionOptions()
        if self.providers[0][0] in ("OpenVINOExecutionProvider",):
            self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.sess = ort.InferenceSession(self.ONNX_MODEL, providers=self.providers, sess_options=self.sess_options)
        self.sess.disable_fallback()
        if self.sess.get_providers()[0] != self.providers[0][0]:
            raise RuntimeError(f"Error creating {self.providers[0][0]}")

    def handle(self, IMAGE, context):
        context.logger.info(f"in handle: {IMAGE}")
        # image = cv2.imread(IMAGE)
        image = np.array(IMAGE)

        # Step 4: Convert the RGB format to BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        context.logger.info(f"image: {image}")

        image_input, real_shape = preprocess_image(image)
        for _ in range(5):
            dets, labels, masks = self.sess.run(["dets", "labels", "masks"], {"input": image_input})

        scores, labels, boxes, masks = postprocess_results(dets, labels, masks, image.shape[:2], real_shape)
        context.logger.info(f"scores: {scores}")
        context.logger.info(f"\n\n")
        context.logger.info(f"boxes: {boxes}")
        context.logger.info(f"\n\n")
        result = {
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
            "masks": masks,
        }
        return result