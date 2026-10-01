"""Task-specific prompt generation for DocRes.

DocRes is conditioned on a 3-channel task prompt concatenated to the RGB input
(6 channels total). Each restoration task derives that prompt from the input
image with a cheap classical CV step, exactly as in the original repo.
"""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def get_base_coord(height, width):
    """Normalized identity sampling grid, channel order (x, y)."""
    coord_y = np.tile(np.arange(height).reshape(height, 1), (1, width)).astype(np.float32)
    coord_x = np.tile(np.arange(width).reshape(1, width), (height, 1)).astype(np.float32)
    return np.concatenate(
        (np.expand_dims(coord_x, -1), np.expand_dims(coord_y, -1)), -1
    )


def _estimate_background(img, kernel=21):
    """Background estimate via dilation + large median filter, per channel."""
    planes = cv2.split(img)
    backgrounds = []
    for plane in planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        backgrounds.append(cv2.medianBlur(dilated, kernel))
    return cv2.merge(backgrounds)


def deblur_prompt(img):
    """Sobel high-frequency map, replicated to 3 channels."""
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)
    high_freq = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    high_freq = cv2.cvtColor(high_freq, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(high_freq, cv2.COLOR_GRAY2BGR)


def appearance_prompt(img):
    """Normalized local-contrast map against the estimated background."""
    h, w = img.shape[:2]
    scaled = cv2.resize(img, (1024, 1024))
    planes = cv2.split(scaled)
    result_norm_planes = []
    for plane in planes:
        dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(plane, bg)
        result_norm_planes.append(
            cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_8UC1)
        )
    result_norm = cv2.merge(result_norm_planes)
    return cv2.resize(result_norm, (w, h))


def deshadow_prompt(img):
    """Background illumination estimate, used as the deshadowing prompt."""
    h, w = img.shape[:2]
    scaled = cv2.resize(img, (1024, 1024))
    bg_imgs = _estimate_background(scaled)
    return cv2.resize(bg_imgs, (w, h))


def sauvola_binarization(image, n1=51, n2=51, k1=0.3, k2=0.3, default=True):
    """Two-pass Sauvola binarization. Returns (binary_image, T2_threshold)."""
    if default:
        n1 = int(0.05 * min(image.shape[0], image.shape[1]))
        n1 += n1 % 2 == 0
        n2 = int(0.1 * min(image.shape[0], image.shape[1]))
        n2 += n2 % 2 == 0
        k1 = 0.5
        k2 = 0.5
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else np.copy(image)

    t1 = threshold_sauvola(gray, window_size=n1, k=k1)
    max_val, min_val = np.amax(gray), np.amin(gray)
    c = t1.astype(np.float32)
    c[gray > t1] = (gray[gray > t1] - t1[gray > t1]) / (max_val - t1[gray > t1])
    c[gray <= t1] = 0
    new_in = np.copy((c * 255.0).astype(np.uint8))

    t2 = threshold_sauvola(new_in, window_size=n2, k=k2)
    binary = np.copy(gray)
    binary[new_in <= t2] = 0
    binary[new_in > t2] = 255
    return binary, t2


def binarization_prompt(img):
    """Sauvola threshold + Sobel gradient + binary map, stacked as 3 channels."""
    result, thresh = sauvola_binarization(img)
    thresh = thresh.astype(np.uint8)
    result = np.where(result > 155, 255, 0).astype(np.uint8)

    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    abs_x = cv2.convertScaleAbs(grad_x)
    abs_y = cv2.convertScaleAbs(grad_y)
    high_freq = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
    high_freq = cv2.cvtColor(high_freq, cv2.COLOR_BGR2GRAY)
    return np.concatenate(
        (
            np.expand_dims(thresh, -1),
            np.expand_dims(high_freq, -1),
            np.expand_dims(result, -1),
        ),
        -1,
    )
