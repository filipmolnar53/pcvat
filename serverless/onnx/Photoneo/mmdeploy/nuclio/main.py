import base64
import io
import json

import yaml
from model_handler import ModelHandler
from PIL import Image
import cv2
import numpy as np

def init_context(context):
    context.logger.info("Init context...  0%")

    # Read labels
    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    types = {item['id']: item['type'] for item in json.loads(labels_spec)}
    context.logger.info(f"labels: {labels}")
    context.logger.info(f"types: {types}")

    # Read the DL model
    model = ModelHandler(context, labels, types)

    context.user_data.model = model

    context.logger.info(f"model: {model}")
    context.logger.info(f"model type: {type(model)}")
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("handling our ONNX model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB

    features = context.user_data.model.handle(image, context)
    results = []
    for box, score, label, mask in zip(features['boxes'], features['scores'], features['labels'], features['masks']):
        context.logger.info(f"labelIDX: {label}, label: {context.user_data.model.labels[label]}")
        context.logger.info(f"mask: {mask}")

        # ints = masks[0].astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        coords = []

        for obj in contours:

            for point in obj:
                coords.append(int(point[0][0]))
                coords.append(int(point[0][1]))

        results.append({
            "confidence": str(float(score)),
            "label": context.user_data.model.labels[label],
            "points": coords,
            # "points": box.tolist(),
            "type": context.user_data.model.types[label],
        })

    results.append({
        "confidence": "0.8",
        "label": "kp",
        "points": [100, 200],
        "type": "points",
    })
    context.logger.info("json.dumps(results)==============================================")
    # context.logger.info(json.dumps(results))
    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)

