import cv2
import numpy as np


def _generate_colors(n_per_channel):
    x = np.linspace(0, 255, n_per_channel, dtype=np.uint8)
    colors = np.array(np.meshgrid(x, x, x), dtype=np.float32).T.reshape(-1, 3)
    colors = colors[~(colors == colors[:, 0, None]).all(axis=1)]  # filter shades of gray
    return colors


COLORS_24 = _generate_colors(3)
COLORS_60 = _generate_colors(4)

_rng = np.random.default_rng()


def random_colors(size: int, maximum=255) -> np.ndarray:
    palette = COLORS_60 if size > 24 else COLORS_24
    idx = _rng.choice(len(palette), size=size, replace=size > 60)
    ret = palette[idx] * maximum
    return ret


def draw_transparent_rectangle(image, box, color, alpha, thickness=1):
    x1, y1, x2, y2 = box
    t = thickness
    a = alpha
    color = np.array(color)

    image[y1 : y2 + 1, x1 : x1 + t] = (1 - a) * image[y1 : y2 + 1, x1 : x1 + t] + a * color
    image[y1 : y2 + 1, x2 - t + 1 : x2 + 1] = (1 - a) * image[y1 : y2 + 1, x2 - t + 1 : x2 + 1] + a * color
    image[y1 : y1 + t, x1 + t : x2 - t + 1] = (1 - a) * image[y1 : y1 + t, x1 + t : x2 - t + 1] + a * color
    image[y2 - t + 1 : y2 + 1, x1 + t : x2 - t + 1] = (1 - a) * image[
        y2 - t + 1 : y2 + 1, x1 + t : x2 - t + 1
    ] + a * color


def draw_mask(
    image,
    mask,
    color,
    box,
    alpha=0.4,
    draw_contours=False,
    inverse_mask=False,
    mask2=None,
    full_res_mask=True,
):
    assert mask.dtype == bool

    x_min, y_min, x_max, y_max = box
    image_crop = image[y_min:y_max, x_min:x_max]
    if full_res_mask:
        overlay = mask[y_min:y_max, x_min:x_max]
    else:
        overlay = mask
    if mask2 is not None:
        if full_res_mask:
            overlay = overlay * mask2[y_min:y_max, x_min:x_max]
        else:
            overlay = overlay * mask2
    if inverse_mask:
        overlay = ~overlay
    image_crop[overlay] = np.float32(1 - alpha) * image_crop[overlay] + np.float32(alpha) * np.array(
        color, dtype=np.float32
    )

    if draw_contours:
        assert full_res_mask

        if isinstance(color, np.ndarray):
            color = color.tolist()
        contours, _ = cv2.findContours(mask.view(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(contour, 1, True) for contour in contours]
        cv2.drawContours(image, contours, -1, color, 1, lineType=cv2.LINE_AA)


def draw_text(image, label, origin, color, font_scale, align_center=False):
    size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    align_offset = (-size[0] // 2 if align_center else 0, 0)
    cv2.rectangle(
        image,
        origin + align_offset,
        origin + size + (0, 3) + align_offset,
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        label,
        origin + (0, size[1]) + (0, 1) + align_offset,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        1,
        cv2.LINE_AA,
    )
