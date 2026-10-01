"""Sliding-window tiling so large scans fit in memory.

Ported from the original `data/preprocess/crop_merge_image.py`, reduced to the
forward-only path: split -> model -> weighted merge. Overlapping windows are
blended with a linear weight ramp so seams are not visible.
"""

import cv2
import numpy as np


def split_img(img, size_x, size_y, strides):
    max_y, max_x = img.shape[:2]
    border_y = 0
    if max_y % size_y != 0:
        border_y = size_y - (max_y % size_y)
        img = cv2.copyMakeBorder(img, border_y, 0, 0, 0, cv2.BORDER_REPLICATE)
    border_x = 0
    if max_x % size_x != 0:
        border_x = size_x - (max_x % size_x)
        img = cv2.copyMakeBorder(img, 0, 0, border_x, 0, cv2.BORDER_REPLICATE)

    max_y, max_x = img.shape[:2]
    parts = []
    curr_y = 0
    while curr_y + size_y <= max_y:
        curr_x = 0
        while curr_x + size_x <= max_x:
            parts.append(img[curr_y:curr_y + size_y, curr_x:curr_x + size_x])
            curr_x += strides
        curr_y += strides
    return parts, border_x, border_y, max_x, max_y


def _weight_map(max_y, max_x, size, strides):
    index = int(size / strides)
    weight = np.ones(shape=(max_y, max_x))
    weight[0:strides] = index
    weight[-strides:] = index
    weight[:, 0:strides] = index
    weight[:, -strides:] = index

    i = 0
    for j in range(1, index + 1):
        weight[0:strides, i:i + strides] = j
        weight[i:i + strides, 0:strides] = j
        weight[0:strides, -strides:] = j
        if i == 0:
            weight[0:strides, -strides:] = j
        else:
            weight[0:strides, -strides - i:-i] = j
        weight[-strides:, i:i + strides] = j
        if i == 0:
            weight[-strides:, 0:strides] = j
        else:
            weight[-strides - i:-i, 0:strides] = j
        if i == 0:
            weight[-strides:, -strides:] = j
        else:
            weight[-strides - i:-i, -strides:] = j
            weight[-strides:, -strides - i:-i] = j
        i += strides

    for i in range(strides, max_y - strides, strides):
        for j in range(strides, max_x - strides, strides):
            weight[i:i + strides, j:j + strides] = weight[i][0] * weight[0][j]
    return weight


def combine_imgs(border_x, border_y, imgs, max_y, max_x, size_x, size_y, strides):
    if len(imgs[0].shape) == 2:
        new_img = np.zeros(shape=(max_y, max_x))
        weight = 1 / _weight_map(max_y, max_x, size_x, strides)
    else:
        new_img = np.zeros(shape=(max_y, max_x, imgs[0].shape[-1]))
        weight = np.tile((1 / _weight_map(max_y, max_x, size_x, strides)).reshape(max_y, max_x, 1),
                         (1, 1, imgs[0].shape[-1]))

    curr_y = 0
    i = 0
    while curr_y + size_y <= max_y:
        curr_x = 0
        while curr_x + size_x <= max_x:
            new_img[curr_y:curr_y + size_y, curr_x:curr_x + size_x] += \
                weight[curr_y:curr_y + size_y, curr_x:curr_x + size_x] * imgs[i]
            i += 1
            curr_x += strides
        curr_y += strides

    new_img = new_img[border_y:, border_x:]
    return np.clip(new_img, 0, 255).astype(np.uint8)


def stride_integral(img, stride=8):
    """Pad bottom/right to a multiple of `stride`.

    Returns (padded, original_h, original_w) so callers can crop the model
    output back with `out[:h, :w]`. The original repo padded the same way but
    cropped from the top-left, which shifted the image and clipped the real
    bottom-right content; this version crops the correct edges.
    """
    h, w = img.shape[:2]
    pad_h = (-h) % stride
    pad_w = (-w) % stride
    if pad_h or pad_w:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return img, h, w
