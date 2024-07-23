import warnings

import cv2
import numpy as np

from draw import COLORS_24, COLORS_60, draw_mask, draw_text, draw_transparent_rectangle


def draw_predictions(image, path, scores, labels, boxes, masks):
    results = image.copy()
    colors = COLORS_60 if len(scores) > 24 else COLORS_24
    assert len(scores) == len(labels) == len(boxes) == len(masks)
    for score, label, box, mask, color in zip(scores, labels, boxes, masks, colors):
        left, top, right, bottom = box
        box = [int(left), int(top), int(right), int(bottom)]

        draw_transparent_rectangle(results, box, color, 0.4, 3)

        draw_mask(results, mask, color, box, draw_contours=True)

        # if point is not None:
        #     cv2.circle(image, point, radius=3, color=color, thickness=-1, lineType=cv2.LINE_AA)

        font_scale_box = 0.45
        alpha_font = 0.4
        local_scale = np.clip(min((box[2] - box[0]) / 50, (box[3] - box[1]) / 50), 1.25, 2) * 0.5
        color = (1 - alpha_font) * 255 + alpha_font * np.array(color, dtype=np.float32)
        draw_text(
            results,
            f"{label}|{score:0.3f}",
            origin=np.array(box[:2]).astype(np.int32),
            color=color.tolist(),
            font_scale=local_scale * font_scale_box,
        )

    cv2.imwrite(str(path), results)


def resize(image, max_size: int):
    h, w = image.shape[:2]
    if h > w:
        new_h, new_w = max_size, int(max_size * w / h)
    else:
        new_h, new_w = int(max_size * h / w), max_size

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def preprocess_image(image):
    target_max_size = 640
    mean_bgr = [103.53, 116.28, 123.675]
    std_bgr = [57.375, 57.12, 58.395]

    image = resize(image, 640)
    real_shape = image.shape[:2]
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_normalized = image.astype(np.float32)
    image_normalized -= mean_bgr
    image_normalized /= std_bgr

    image_normalized = cv2.copyMakeBorder(
        image_normalized,
        top=0,
        bottom=target_max_size - image_normalized.shape[0],
        left=0,
        right=target_max_size - image_normalized.shape[1],
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    assert image_normalized.shape == (640, 640, 3)

    return np.transpose(image_normalized, (2, 0, 1))[None].astype(np.float32), real_shape


def postprocess_results(
        dets: np.ndarray, labels: np.ndarray, masks: np.ndarray, target_shape: tuple[int, int],
        real_shape: tuple[int, int]
):
    assert len(target_shape) == len(real_shape) == 2

    scores = dets[0, :, 4].copy()
    limit = np.argmin(scores >= 0.5) or len(scores)
    if limit < len(scores):
        warnings.warn("Model predicts objects with `score < 0.5` -> ignoring them.")
    scores = scores[:limit].copy()
    labels = labels[0, :limit].astype(np.int32)

    boxes = dets[0, :limit, :4].reshape(-1, 2, 2) * (np.array(target_shape[::-1]) / np.array(real_shape[::-1]))
    boxes = np.clip(boxes, 0, np.array(target_shape[::-1]))
    boxes = boxes.reshape(-1, 4).astype(np.int32)

    masks = (masks[0, :limit, : real_shape[0], : real_shape[1]] > 0.5).view(np.uint8)
    masks_full = np.empty((masks.shape[0],) + target_shape, dtype=np.uint8)
    for i, mask in enumerate(masks):
        masks_full[i] = cv2.resize(np.ascontiguousarray(mask), target_shape[::-1])
    masks = masks_full.view(bool)

    return scores, labels, boxes, masks
